import entmax
import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import sparsemax, entmax15


class MultiLabelLinearNN(nn.Module):
    def __init__(self, input_dim, output_dim, activation="softmax", dropout_rate=0.2):
        super(MultiLabelLinearNN, self).__init__()
        self.activation_type = activation

        fc1_dim = min(512, max(64, input_dim // 2))
        fc2_dim = max(32, fc1_dim // 2)

        print("fc1_dim", fc1_dim)
        print("fc2_dim", fc2_dim)

        # self.fc = nn.Sequential(
        #     nn.Linear(input_dim, fc1_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(fc1_dim, fc2_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout_rate),
        #     nn.Linear(fc2_dim, output_dim)
        # )

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x, targets=None):
        logits = self.fc(x)

        # Apply chosen sparse activation
        if self.activation_type == "sparsemax":
            probs = sparsemax(logits, dim=1)
        elif self.activation_type == "entmax15":
            probs = entmax15(logits, dim=1)
        elif self.activation_type == "softmax":
            probs = F.softmax(logits, dim=1)
        else:
            raise ValueError(f"Unknown activation type: {self.activation_type}")

        loss = None
        if targets is not None:
            # KL divergence expects log probs
            log_probs = probs.clamp(min=1e-8).log()
            valid = targets.sum(dim=1) > 0  # skip all-zero (neutral) targets
            if valid.any():
                loss = F.kl_div(log_probs[valid], targets[valid], reduction="batchmean")

        return probs, loss


class MultiLabelLinearNNShallow(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultiLabelLinearNNShallow, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x, targets=None):
        logits = self.fc(x)

        probs = F.softmax(logits, dim=1)

        loss = None
        if targets is not None:
            # KL divergence expects log probs
            log_probs = probs.clamp(min=1e-8).log()
            valid = targets.sum(dim=1) > 0  # skip all-zero (neutral) targets
            if valid.any():
                loss = F.kl_div(log_probs[valid], targets[valid], reduction="batchmean")

        return probs, loss


class MultiLabelLinearNNSuperShallow(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MultiLabelLinearNNSuperShallow, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x, targets=None):
        logits = self.fc(x)

        probs = F.softmax(logits, dim=1)

        loss = None
        if targets is not None:
            # KL divergence expects log probs
            log_probs = probs.clamp(min=1e-8).log()
            valid = targets.sum(dim=1) > 0  # skip all-zero (neutral) targets
            if valid.any():
                loss = F.kl_div(log_probs[valid], targets[valid], reduction="batchmean")

        return probs, loss


class MultiLabelRNN(nn.Module):

    def __init__(self, input_dim, output_dim, activation="softmax"):
        super().__init__()

        hidden_dim = min(512, max(128, input_dim // 4))
        fc_dim = min(256, max(64, hidden_dim))

        print("hidden_dim", hidden_dim)
        print("fc_dim", fc_dim)

        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, output_dim),
        )

        # Store activation type
        self.activation_type = activation

    def forward(self, x, targets=None):
        # x: [B, T, D]
        _, h_n = self.rnn(x)  # h_n: [2, B, H]
        h_n = h_n.permute(1, 0, 2).reshape(x.size(0), -1)  # [B, 2*H]
        logits = self.fc(h_n)  # [B, C]

        # Apply sparse activation
        if self.activation_type == "sparsemax":
            probs = sparsemax(logits, dim=1)
        elif self.activation_type == "entmax15":
            probs = entmax15(logits, dim=1)
        elif self.activation_type == "softmax":
            probs = F.softmax(logits, dim=1)
        else:
            raise ValueError(f"Unknown activation type: {self.activation_type}")

        # Compute loss if targets are provided
        loss = None
        if targets is not None:
            log_probs = probs.clamp(min=1e-8).log()
            valid = targets.sum(dim=1) > 0  # avoid all-zero (neutral) targets
            if valid.any():
                loss = F.kl_div(log_probs[valid], targets[valid], reduction="batchmean")

        return probs, loss
