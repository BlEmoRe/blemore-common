from torch.utils.data import Dataset
import torch
import os
import numpy as np


class SubsampledVideoDataset(Dataset):
    """ Dataset that expands each video into its subsamples """

    def __init__(self, filenames, labels, data_dir):
        """
        filenames: list of filenames (without .npy).
        labels: list of label vectors, same order as filenames.
        data_dir: directory where .npy files are stored.
        """
        self.features = []
        self.labels = []
        self.filenames = []
        self.data_dir = data_dir
        self.filename_list = []
        self.label_list = []

        for filename, label in zip(filenames, labels):
            file_path = os.path.join(self.data_dir, f"{filename}.npy")
            subsample_features = np.load(file_path)  # (N_subsamples, feature_dim)

            num_subsamples = subsample_features.shape[0]

            self.features.append(subsample_features)
            self.labels.extend([label] * num_subsamples)
            self.filenames.extend([filename] * num_subsamples)

            self.filename_list.append(filename)  # to store the filename once
            self.label_list.append(label)        # to store the label once

        self.features = np.vstack(self.features)  # (total_subsamples, feature_dim)
        self.labels = np.stack(self.labels)        # (total_subsamples, num_classes)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

    @property
    def input_dim(self):
        return self.features.shape[1]

    @property
    def output_dim(self):
        return self.labels.shape[1]

