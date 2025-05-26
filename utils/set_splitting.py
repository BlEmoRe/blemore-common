import numpy as np
from sklearn.preprocessing import StandardScaler

from datasets.d2_dataset import D2Dataset
from datasets.d3_dataset import D3Dataset
from utils.standardization import create_transform


def filter_basic_samples(filenames, labels, mix_flags):
    mask = np.array(mix_flags) == 0
    return [f for f, keep in zip(filenames, mask) if keep], labels[mask]


def extract_files_and_labels(df, labels, mask, only_basic=False):
    files = df.loc[mask, "filename"].tolist()
    subset_labels = labels[mask.to_numpy()]
    mix = df.loc[mask, "mix"].tolist()

    if only_basic:
        files, subset_labels = filter_basic_samples(files, subset_labels, mix)

    return files, subset_labels


def get_split_files_and_labels(df, labels, fold_id, only_basic=False):
    train_mask = df["fold"] != fold_id
    val_mask = df["fold"] == fold_id

    train_files, train_labels = extract_files_and_labels(df, labels, train_mask, only_basic)
    val_files, val_labels = extract_files_and_labels(df, labels, val_mask, only_basic)

    return (train_files, train_labels), (val_files, val_labels)


def prepare_split_3d(df, labels, fold_id, encoding_folder, only_basic=False):
    (train_files, train_labels), (val_files, val_labels) = get_split_files_and_labels(
        df=df,
        labels=labels,
        fold_id=fold_id,
        only_basic=only_basic
    )
    scaler = create_transform(train_files, encoding_folder)
    train_dataset = D3Dataset(filenames=train_files, labels=train_labels, encoding_dir=encoding_folder, scaler=scaler)
    val_dataset = D3Dataset(filenames=val_files, labels=val_labels, encoding_dir=encoding_folder, scaler=scaler)
    return train_dataset, val_dataset


def prepare_split_2d(df, labels, fold_id, filepath, only_basic=False):
    # Load full feature data and filenames
    data = np.load(filepath)
    X = data["X"]

    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)

    all_filenames = data["filenames"]

    # Get train/val filename lists and corresponding labels
    (train_files, train_labels), (val_files, val_labels) = get_split_files_and_labels(
        df=df,
        labels=labels,
        fold_id=fold_id,
        only_basic=only_basic
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