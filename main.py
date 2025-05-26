import argparse
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import torch

from config import ROOT_DIR, LABEL_TO_INDEX
from datasets.d2_dataset import D2Dataset
from datasets.d3_dataset import D3Dataset
from model.models import MultiLabelRNN, MultiLabelLinearNN
from post_processing import grid_search_thresholds

from trainer import Trainer
import matplotlib.pyplot as plt

from utils.standardization import create_transform

# from utils.standardization import compute_train_stats

hparams = {
    "batch_size": 32,
    "max_seq_len": None,  # Set to None for no padding/truncation
    "learning_rate": 0.0005,
    "num_epochs": 100,
    "weight_decay": 1e-4,
    # "dropout": 0.5,
    # "hidden_dim": 128,
    # "num_layers": 2,
}

def filter_basic_samples(filenames, labels, mix_flags):
    mask = np.array(mix_flags) == 0
    return [f for f, keep in zip(filenames, mask) if keep], labels[mask]


def create_labels(records):
    labels = np.zeros((len(records), len(LABEL_TO_INDEX)), dtype=float)

    for i, record in enumerate(records):
        e1 = record["emotion_1"]
        e2 = record["emotion_2"]
        s1 = record["emotion_1_salience"]
        s2 = record["emotion_2_salience"]
        mix = record["mix"]

        if e1 not in LABEL_TO_INDEX:
            continue  # skip neutral

        if mix == 0:
            labels[i, LABEL_TO_INDEX[e1]] = 1
        elif mix == 1:
            labels[i, LABEL_TO_INDEX[e1]] = s1 / 100
            labels[i, LABEL_TO_INDEX[e2]] = s2 / 100
        else:
            raise ValueError(f"Invalid mix value: {mix}")
    return labels


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



def main():
    argparser = argparse.ArgumentParser()
    encoder = argparser.add_argument("--encoder", type=str, default="videomae", help="Encoder to use")
    only_basic = argparser.add_argument("--only_basic", action="store_true", help="Use only basic emotion samples")
    model = argparser.add_argument("--model", type=str, default="", help="Path to the model checkpoint")

    args = argparser.parse_args()

    # args.only_basic = True
    # args.model = os.path.join(ROOT_DIR, "data/baselines/simple/checkpoints", "epoch-200.pt")

    # paths
    train_metadata = "/home/tim/Work/quantum/data/blemore/train_metadata.csv"
    test_metadata = "/home/tim/Work/quantum/data/blemore/test_metadata.csv"

    encoding_paths_3d = {
        "openface": "/home/tim/Work/quantum/data/blemore/encoded_videos/openface_npy/",
        "imagebind": "/home/tim/Work/quantum/data/blemore/encoded_videos/ImageBind/",
        "clip": "/home/tim/Work/quantum/data/blemore/encoded_videos/CLIP_npy/",
        "dinov2": "/home/tim/Work/quantum/data/blemore/encoded_videos/DINOv2_reshaped/",
        "videoswintransformer": "/home/tim/Work/quantum/data/blemore/encoded_videos/VideoSwinTransformer/",
        "videomae": "/home/tim/Work/quantum/data/blemore/encoded_videos/VideoMAEv2_reshaped/",
    }

    encoding_paths_2d = {
        "openface": "/home/tim/Work/quantum/data/blemore/encoded_videos/static_data/openface_static_features.npz",
        "imagebind": "/home/tim/Work/quantum/data/blemore/encoded_videos/ImageBind/",
        "clip": "/home/tim/Work/quantum/data/blemore/encoded_videos/static_data/clip_static_features.npz",
        "dinov2": "/home/tim/Work/quantum/data/blemore/encoded_videos/DINOv2_reshaped/",
        "videoswintransformer": "/home/tim/Work/quantum/data/blemore/encoded_videos/VideoSwinTransformer/",
        "videomae": "/home/tim/Work/quantum/data/blemore/encoded_videos/static_data/videomae_static_features.npz",
    }


    # if args.encoder == "openface":
    encoding_folder = encoding_paths_2d[args.encoder]

    print(f"Using encoder: {args.encoder}")
    print(f"Encoding folder: {encoding_folder}")

    train_df = pd.read_csv(train_metadata)
    train_records = train_df.to_dict(orient="records")
    train_labels = create_labels(train_records)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summary_rows = []

    folds = [0, 1, 2, 3, 4]
    activations = ["entmax15", "softmax", "sparsemax"]
    for fold_id in folds:
        for activation in activations:
            print(f"Fold: {fold_id}, Activation: {activation}")
            # train_dataset, val_dataset = prepare_split_3d(train_df, train_labels, fold_id, encoding_folder, args.only_basic)
            train_dataset, val_dataset = prepare_split_2d(train_df, train_labels, fold_id, encoding_folder, args.only_basic)

            print(f"Fold {fold_id}:")
            print(f"Train dataset size: {len(train_dataset)}")
            print(f"Validation dataset size: {len(val_dataset)}")

            print("Train dataset filenames:", len(train_dataset.filenames))
            print("Validation dataset filenames:", len(val_dataset.filenames))

            train_loader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=hparams["batch_size"], shuffle=False)

            # model = MultiLabelRNN(input_dim=train_dataset.input_dim, output_dim=train_dataset.output_dim)
            model = MultiLabelLinearNN(input_dim=train_dataset.input_dim, output_dim=train_dataset.output_dim, activation=activation)

            if args.model != '':
                model.load_state_dict(torch.load(args.model))

            optimizer = torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"],
                                         weight_decay=hparams["weight_decay"])
            model.to(device)

            trainer = Trainer(model=model, optimizer=optimizer, data_loader=train_loader,
                              epochs=hparams["num_epochs"], valid_data_loader=val_loader)

            res = trainer.train()

            best_epoch = max(res, key=lambda r: 0.5 * r["best_acc_presence"] + 0.5 * r["best_acc_salience"])

            summary_rows.append({
                "fold": fold_id,
                "activation": activation,
                "epoch": best_epoch["epoch"],
                "val_loss": best_epoch["val_loss"],
                "acc_presence": best_epoch["best_acc_presence"],
                "acc_salience": best_epoch["best_acc_salience"],
                "alpha": best_epoch["best_alpha"],
                "beta": best_epoch["best_beta"],
            })

            df = pd.DataFrame(res)

            # Plotting
            fig, axs = plt.subplots(2, 2, figsize=(12, 8))

            # Losses
            axs[0, 0].plot(df["epoch"], df["train_loss"], label="Train Loss")
            if "val_loss" in df.columns and df["val_loss"].notna().any():
                axs[0, 0].plot(df["epoch"], df["val_loss"], label="Validation Loss")
            axs[0, 0].set_title("Loss per Epoch")
            axs[0, 0].set_xlabel("Epoch")
            axs[0, 0].set_ylabel("Loss")
            axs[0, 0].legend()

            # Accuracy presence
            axs[0, 1].plot(df["epoch"], df["best_acc_presence"], label="Accuracy Presence", color='green')
            axs[0, 1].set_title("Best Accuracy (Presence)")
            axs[0, 1].set_xlabel("Epoch")
            axs[0, 1].set_ylabel("Accuracy")
            axs[0, 1].legend()

            # Accuracy salience
            axs[1, 0].plot(df["epoch"], df["best_acc_salience"], label="Accuracy Salience", color='orange')
            axs[1, 0].set_title("Best Accuracy (Salience)")
            axs[1, 0].set_xlabel("Epoch")
            axs[1, 0].set_ylabel("Accuracy")
            axs[1, 0].legend()

            # Alpha & Beta
            axs[1, 1].plot(df["epoch"], df["best_alpha"], label="Alpha", linestyle='--')
            axs[1, 1].plot(df["epoch"], df["best_beta"], label="Beta", linestyle='-.')
            axs[1, 1].set_title("Best Alpha & Beta")
            axs[1, 1].set_xlabel("Epoch")
            axs[1, 1].legend()

            plt.tight_layout()
            plt.show()
            break

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.groupby("activation")[["acc_presence", "acc_salience"]].mean())




if __name__ == "__main__":
    main()
