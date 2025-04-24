import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the model with LogSoftmax
class MultiLabelSoftmaxNN(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.2):
        super(MultiLabelSoftmaxNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout Regularization
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Another Dropout Layer
            nn.Linear(64, output_dim),
            nn.LogSoftmax(dim=1)  # LogSoftmax instead of Softmax
        )

    def forward(self, x, targets=None):
        log_probs = self.fc(x)

        loss = None
        if targets is not None:
            loss = F.kl_div(log_probs, targets, reduction="batchmean")

        probs = torch.exp(log_probs)
        return probs, loss