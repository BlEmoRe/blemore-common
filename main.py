import argparse
import os

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch

from config import ROOT_DIR, LABEL_TO_INDEX
from datasets.d3_dataset import D3Dataset
from model.models import MultiLabelRNN
from post_processing import grid_search_thresholds

from trainer import Trainer
import matplotlib.pyplot as plt

# STANDARDIZE THE DATA FOR CHRISTS’S SAKE!!!!!!



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


def prepare_split(df, labels, fold_id, encoding_folder, only_basic=False):
    train_mask = df["fold"] != fold_id
    val_mask = df["fold"] == fold_id

    def extract_subset(mask):
        files = df.loc[mask, "filename"].tolist()
        subset_labels = labels[mask.to_numpy()]
        mix = df.loc[mask, "mix"].tolist()
        if only_basic:
            files, subset_labels = filter_basic_samples(files, subset_labels, mix)
        return files, subset_labels

    train_files, train_labels = extract_subset(train_mask)
    val_files, val_labels = extract_subset(val_mask)

    train_dataset = D3Dataset(filenames=train_files, labels=train_labels, encoding_dir=encoding_folder)
    val_dataset = D3Dataset(filenames=val_files, labels=val_labels, encoding_dir=encoding_folder)

    return train_dataset, val_dataset


def main():
    argparser = argparse.ArgumentParser()
    encoder = argparser.add_argument("--encoder", type=str, default="imagebind", help="Encoder to use")
    only_basic = argparser.add_argument("--only_basic", action="store_true", help="Use only basic emotion samples")
    model = argparser.add_argument("--model", type=str, default="", help="Path to the model checkpoint")

    args = argparser.parse_args()

    # args.only_basic = True
    # args.model = os.path.join(ROOT_DIR, "data/baselines/simple/checkpoints", "epoch-200.pt")

    # paths
    train_metadata = "/home/tim/Work/quantum/data/blemore/train_metadata.csv"
    test_metadata = "/home/tim/Work/quantum/data/blemore/test_metadata.csv"

    encoding_paths = {
        "openface": "/home/tim/Work/quantum/data/blemore/encoded_videos/openface_npy/",
        "imagebind": "/home/tim/Work/quantum/data/blemore/encoded_videos/ImageBind/",
        "clip": "/home/tim/Work/quantum/data/blemore/encoded_videos/CLIP_npy/",
        "dinov2": "/home/tim/Work/quantum/data/blemore/encoded_videos/DINOv2/",
        "videoswintransformer": "/home/tim/Work/quantum/data/blemore/encoded_videos/VideoSwinTransformer/",
    }

    # if args.encoder == "openface":
    encoding_folder = encoding_paths[args.encoder]

    print(f"Using encoder: {args.encoder}")
    print(f"Encoding folder: {encoding_folder}")

    train_df = pd.read_csv(train_metadata)
    train_records = train_df.to_dict(orient="records")
    train_labels = create_labels(train_records)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folds = [0, 1, 2, 3, 4]
    for fold_id in folds:
        train_dataset, val_dataset = prepare_split(train_df, train_labels, fold_id, encoding_folder, args.only_basic)

        print(f"Fold {fold_id}:")
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")

        print("Train dataset filenames:", len(train_dataset.filenames))
        print("Validation dataset filenames:", len(val_dataset.filenames))

        train_loader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=hparams["batch_size"], shuffle=False)

        model = MultiLabelRNN(input_dim=train_dataset.input_dim, output_dim=train_dataset.output_dim)

        if args.model != '':
            model.load_state_dict(torch.load(args.model))

        optimizer = torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"],
                                     weight_decay=hparams["weight_decay"])
        model.to(device)

        trainer = Trainer(model=model, optimizer=optimizer, data_loader=train_loader,
                          epochs=hparams["num_epochs"], valid_data_loader=val_loader)

        res = trainer.train()

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





if __name__ == "__main__":
    main()
