import numpy as np
from sklearn.preprocessing import StandardScaler

from datasets.d2_dataset import D2Dataset
from datasets.d3_dataset import D3Dataset
from datasets.subsample_dataset import SubsampledVideoDataset
from utils.standardization import create_transform


def extract_files_and_labels(df, labels, mask):
    files = df.loc[mask, "filename"].tolist()
    subset_labels = labels[mask.to_numpy()]
    return files, subset_labels


def get_validation_split(df, labels, fold_id):
    train_mask = df["fold"] != fold_id
    val_mask = df["fold"] == fold_id

    train_files, train_labels = extract_files_and_labels(df, labels, train_mask)
    val_files, val_labels = extract_files_and_labels(df, labels, val_mask)

    return (train_files, train_labels), (val_files, val_labels)


def prepare_split_2d(df, labels, fold_id, filepath):
    # Load full feature data and filenames
    data = np.load(filepath)
    X = data["X"]

    all_filenames = data["filenames"]

    # Get train/val filename lists and corresponding labels
    (train_files, train_labels), (val_files, val_labels) = get_validation_split(
        df=df,
        labels=labels,
        fold_id=fold_id,
    )

    # Map filenames to indices in the X matrix
    name_to_idx = {name: i for i, name in enumerate(all_filenames)}
    train_idx = [name_to_idx[f] for f in train_files]
    val_idx = [name_to_idx[f] for f in val_files]

    # Subset data
    X_train = X[train_idx]
    X_val = X[val_idx]
    filenames_train = [all_filenames[i] for i in train_idx]
    filenames_val = [all_filenames[i] for i in val_idx]

    # Fit scaler on training data
    scaler = StandardScaler()
    scaler.fit(X_train)
    # Transform training and validation data
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    # Create datasets
    train_dataset = D2Dataset(X=X_train, labels=train_labels, filenames=filenames_train)
    val_dataset = D2Dataset(X=X_val, labels=val_labels, filenames=filenames_val)

    return train_dataset, val_dataset


def prepare_split_subsampled(df, labels, fold_id, data_dir):
    """
    data_dir: directory containing .npy files for each video.
    """
    (train_videos, train_labels), (val_videos, val_labels) = get_validation_split(
        df=df,
        labels=labels,
        fold_id=fold_id,
    )

    # Initialize datasets
    train_dataset = SubsampledVideoDataset(filenames=train_videos, labels=train_labels, data_dir=data_dir)
    val_dataset = SubsampledVideoDataset(filenames=val_videos, labels=val_labels, data_dir=data_dir)

    # Standard Scaler: Fit only on train, transform both
    scaler = StandardScaler()
    scaler.fit(train_dataset.features)

    # Apply scaler
    train_dataset.features = scaler.transform(train_dataset.features)
    val_dataset.features = scaler.transform(val_dataset.features)

    return train_dataset, val_dataset