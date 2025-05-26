import entmax
import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import sparsemax, entmax15


class MultiLabelLinearNN(nn.Module):
    def __init__(self, input_dim, output_dim, activation="entmax15", dropout_rate=0.2):
        super(MultiLabelLinearNN, self).__init__()
        self.activation_type = activation

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, output_dim)
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

class MultiLabelRNN(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),  # ← note the *2 here
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, targets=None):
        # x: [B, T, D]
        _, h_n = self.rnn(x)  # h_n: [2, B, H] for bidirectional
        h_n = h_n.permute(1, 0, 2).reshape(x.size(0), -1)  # [B, 2*H]
        logits = self.fc(h_n)  # [B, C]
        log_probs = self.log_softmax(logits)

        loss = None
        if targets is not None:
            loss = F.kl_div(log_probs, targets, reduction="batchmean")

        probs = torch.exp(log_probs)
        return probs, loss
