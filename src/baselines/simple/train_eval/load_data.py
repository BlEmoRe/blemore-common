import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import json

from src.baselines.simple.config_simple_baseline import VECTOR_TRAINING_SET_PATH, LABEL_MAPPING_PATH

class EmotionDataset(Dataset):
    """ Custom PyTorch Dataset for Emotion Classification """

    def __init__(self, X, y, original_indices=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.original_indices = original_indices

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    @property
    def input_dim(self):
        return self.X.shape[1]  # Number of features

    @property
    def output_dim(self):
        return self.y.shape[1]  # Number of labels (5 in your case)

    @property
    def labels(self):
        return self.y  # Full label tensor

    @property
    def indices(self):
        return self.original_indices


def load_data(fold_id: int = 0):
    """Loads and preprocesses training and validation data for a specific fold."""

    data = np.load(VECTOR_TRAINING_SET_PATH)

    X, y, folds, indices = data["X"], data["y"], data["folds"], data["indices"]

    # Select train and val based on fold
    is_val = folds == fold_id
    is_train = ~is_val

    X_train, y_train = X[is_train], y[is_train]
    X_val, y_val = X[is_val], y[is_val]
    val_indices = indices[is_val]

    # Normalize features based on training set only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Wrap in datasets
    train_dataset = EmotionDataset(X_train_scaled, y_train)
    val_dataset = EmotionDataset(X_val_scaled, y_val, original_indices=val_indices)

    return train_dataset, val_dataset


def get_index2emotion():
    # Load JSON file
    with open(LABEL_MAPPING_PATH, "r") as f:
        emotion2index = json.load(f)

    index2emotion = {int(v): k for k, v in emotion2index.items()}

    return index2emotion