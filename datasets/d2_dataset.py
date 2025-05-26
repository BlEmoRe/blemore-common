import torch
from torch.utils.data import Dataset

class D2Dataset(Dataset):
    """ Simplified Dataset for 2D static features with filename access """

    def __init__(self, X, labels, filenames, scaler=None):
        """
        X: np.ndarray [N, D]
        labels: np.ndarray [N, C]
        filenames: list[str] of length N
        scaler: optional sklearn-like scaler (fit on X)
        """
        if not (len(X) == len(labels) == len(filenames)):
            raise ValueError("X, labels, and filenames must have the same length")

        self.X = scaler.transform(X) if scaler else X
        self.labels = labels
        self.filenames = filenames

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y  # include filename in the returned tuple

    @property
    def input_dim(self):
        return self.X.shape[1]

    @property
    def output_dim(self):
        return self.labels.shape[1]