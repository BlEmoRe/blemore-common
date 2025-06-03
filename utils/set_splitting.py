import numpy as np
from sklearn.preprocessing import StandardScaler
from datasets.d2_dataset import D2Dataset
from datasets.subsample_dataset import SubsampledVideoDataset


def get_validation_split(df, labels, fold_id):
    """Split filenames and labels into train/val sets based on fold."""
    train_mask = df["fold"] != fold_id
    val_mask = ~train_mask

    train_files = df.loc[train_mask, "filename"].tolist()
    val_files = df.loc[val_mask, "filename"].tolist()

    train_labels = labels[train_mask.to_numpy()]
    val_labels = labels[val_mask.to_numpy()]

    return (train_files, train_labels), (val_files, val_labels)


def prepare_split_2d(train_files, train_labels, val_files, val_labels, filepath):
    data = np.load(filepath)
    X = data["X"]
    filenames = data["filenames"]

    # Map filenames to indices
    name_to_idx = {name: i for i, name in enumerate(filenames)}
    train_idx = [name_to_idx[f] for f in train_files]
    val_idx = [name_to_idx[f] for f in val_files]

    # Subset and scale
    X_train = X[train_idx]
    X_val = X[val_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    train_dataset = D2Dataset(X=X_train, labels=train_labels, filenames=train_files)
    val_dataset = D2Dataset(X=X_val, labels=val_labels, filenames=val_files)

    return train_dataset, val_dataset


def prepare_split_subsampled(df, labels, fold_id, data_dir):
    (train_files, train_labels), (val_files, val_labels) = get_validation_split(df, labels, fold_id)

    train_dataset = SubsampledVideoDataset(filenames=train_files, labels=train_labels, data_dir=data_dir)
    val_dataset = SubsampledVideoDataset(filenames=val_files, labels=val_labels, data_dir=data_dir)

    # Scale
    scaler = StandardScaler()
    train_dataset.features = scaler.fit_transform(train_dataset.features)
    val_dataset.features = scaler.transform(val_dataset.features)

    return train_dataset, val_dataset