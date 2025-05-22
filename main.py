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



# STANDARDIZE THE DATA FOR CHRISTS’S SAKE!!!!!!



hparams = {
    "batch_size": 32,
    "max_seq_len": None,  # Set to None for no padding/truncation
    "learning_rate": 0.0005,
    "num_epochs": 500,
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


def train(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs, loss = model(X_batch, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def predict(data_loader, model, device):
    model.eval()
    y_pred_list = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_pred, _ = model(X_batch)
            y_pred_list.append(y_pred.cpu())
    y_pred_tensor = torch.cat(y_pred_list, dim=0)
    return y_pred_tensor.numpy()



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

        for epoch in range(hparams["num_epochs"]):

            total_loss = train(model, train_loader, optimizer, device, epoch)
            print(f"Epoch [{epoch + 1}/{hparams['num_epochs']}], Loss: {total_loss / len(train_loader):.4f}")

            if epoch != 0 and epoch % 10 == 0:
                model.eval()
                all_preds = []
                val_loss = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                        preds, loss = model(X_batch, y_batch)
                        val_loss += loss.item()
                        all_preds.append(preds.cpu().numpy())

                all_preds = np.concatenate(all_preds, axis=0)
                val_filenames = val_dataset.filenames  # already filtered for missing files

                # Grid search for best thresholds
                best_alpha, best_beta, best_acc_presence, best_acc_salience = grid_search_thresholds(val_filenames,
                                                                                                     all_preds,
                                                                                                     presence_weight=0.5)

                print(f"Epoch {epoch + 1}/{hparams['num_epochs']}, "
                      f"Validation Loss: {val_loss / len(val_loader)}"
                      f", Presence Accuracy: {best_acc_presence:.4f}, "
                      f"Salience Accuracy: {best_acc_salience:.4f}")

            if epoch != 0 and epoch % 100 == 0:
                save_path = os.path.join(ROOT_DIR, "data/baselines/simple/checkpoints", f"epoch-{epoch}.pt")
                torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    main()
