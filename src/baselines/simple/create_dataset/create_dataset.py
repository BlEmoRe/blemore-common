import os
import json
import pandas as pd
import numpy as np

from src.baselines.simple.config_simple_baseline import feature_columns

def create_label_mapping(df, save_path):
    # Build label mapping (excluding 'neu')
    unique_labels = sorted(df["emotion_1"].unique())
    unique_labels.remove("neu")
    label_to_index = {label: i for i, label in enumerate(unique_labels)}

    # Save label mapping
    with open(save_path, "w") as f:
        json.dump(label_to_index, f, indent=4)

    return label_to_index


def create_y(df, label_to_index):
    # Create label matrix
    n_samples = df.shape[0]
    n_labels = len(label_to_index)
    y = np.zeros((n_samples, n_labels), dtype=float)

    # populate label matrix
    for i in range(n_samples):
        emotion_1, emotion_2 = df["emotion_1"].iloc[i], df["emotion_2"].iloc[i]
        salience_1, salience_2 = df["emotion_1_salience"].iloc[i], df["emotion_2_salience"].iloc[i]
        mix = df["mix"].iloc[i]

        if emotion_1 not in label_to_index:
            continue  # skip "neu"

        if mix == 0:
            y[i, label_to_index[emotion_1]] = 1
        elif mix == 1:
            y[i, label_to_index[emotion_1]] = salience_1 / 100
            y[i, label_to_index[emotion_2]] = salience_2 / 100
        else:
            raise ValueError("Invalid mix value")

    return y

def create_dataset(df, save_folder, train=True):
    # Load training data
    label_mapping_path = os.path.join(save_folder, "label_mapping.json")
    dataset_path = os.path.join(save_folder, "dataset.npz")

    label_to_index = create_label_mapping(df, label_mapping_path)

    y = create_y(df, label_to_index)
    # Store original indices **before** shuffling
    original_indices = df.index.to_numpy()
    # Extract features
    X = df.loc[:, df.columns.str.contains('|'.join(feature_columns))].values

    folds = None
    if train:
        folds = df["fold"].values
        perm = np.random.permutation(len(df))
        X, y, original_indices, folds = X[perm], y[perm], original_indices[perm], folds[perm]


    # Create the data dictionary
    data_dict = {
        'X': X,
        'y': y,
        'indices': original_indices,
        'folds': folds
    }

    # Save using unpacked dictionary
    np.savez_compressed(dataset_path, **data_dict)

    return data_dict, label_to_index
