import os
import numpy as np
import torch
from torch.utils.data import Dataset


class D3Dataset(Dataset):
    """ PyTorch Dataset for sequence-based video emotion recognition """

    def __init__(self, filenames, labels, encoding_dir, max_seq_len=400):
        self.encoding_dir = encoding_dir
        self.max_seq_len = max_seq_len  # optionally pad/truncate

        # Filter out missing files
        valid = []
        for i, fname in enumerate(filenames):
            path = os.path.join(encoding_dir, f"{fname}.npy")
            if os.path.isfile(path):
                valid.append(i)

        self.filenames = [filenames[i] for i in valid]
        self.labels = labels[valid]

        if len(self.filenames) == 0:
            raise RuntimeError("No valid video files found in the encoding directory.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        video_path = os.path.join(self.encoding_dir, f"{self.filenames[idx]}.npy")
        x = np.load(video_path)  # shape: [T, D]

        if self.max_seq_len:
            x = self._pad_or_truncate(x)

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

    def _pad_or_truncate(self, x):
        T, D = x.shape
        if T > self.max_seq_len:
            return x[:self.max_seq_len]
        elif T < self.max_seq_len:
            pad = np.zeros((self.max_seq_len - T, D), dtype=x.dtype)
            return np.concatenate([x, pad], axis=0)
        return x

    @property
    def input_dim(self):
        for fname in self.filenames:
            try:
                example = np.load(os.path.join(self.encoding_dir, f"{fname}.npy"))
                return example.shape[1]
            except FileNotFoundError:
                continue
        raise RuntimeError("No valid files found for determining input dimension.")

    @property
    def output_dim(self):
        return self.labels.shape[1]

    @property
    def sequence_length(self):
        for fname in self.filenames:
            try:
                example = np.load(os.path.join(self.encoding_dir, f"{fname}.npy"))
                return example.shape[0]
            except FileNotFoundError:
                continue
        raise RuntimeError("No valid files found for determining sequence length.")