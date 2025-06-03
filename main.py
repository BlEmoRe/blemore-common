import pandas as pd
from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter
from model.models import ConfigurableLinearNN
from post_processing import grid_search_thresholds, probs2dict, get_top_2_predictions

from trainer import Trainer
from utils.create_soft_labels import create_labels
from utils.generic_accuracy.accuracy_funcs import acc_presence_total, acc_salience_total
from utils.set_splitting import prepare_split_2d, get_validation_split
import os
import numpy as np

hparams = {
    "batch_size": 32,
    "max_seq_len": None,  # Set to None for no padding/truncation
    "learning_rate": 5e-5,
    "num_epochs": 5,
    "weight_decay": 1e-3,
}


def select_model_type(model_type, train_dataset):
    if model_type == "Linear":
        model = ConfigurableLinearNN(input_dim=train_dataset.input_dim, output_dim=train_dataset.output_dim, n_layers=0)
    elif model_type == "MLP_256":
        model = ConfigurableLinearNN(input_dim=train_dataset.input_dim, output_dim=train_dataset.output_dim, n_layers=1,
                                     hidden_dim=256)
    elif model_type == "MLP_512":
        model = ConfigurableLinearNN(input_dim=train_dataset.input_dim, output_dim=train_dataset.output_dim, n_layers=1,
                                     hidden_dim=512)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model


def train_and_test(model_type, encoding_folder, train_df, train_labels, test_df, test_labels, alpha, beta, device):
    train_files = train_df.filename.tolist()
    test_files = test_df.filename.tolist()

    train_dataset, test_dataset = prepare_split_2d(train_files, train_labels, test_files, test_labels, encoding_folder)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"Input dimension: {train_dataset.input_dim}")
    print(f"Output dimension: {train_dataset.output_dim}")

    train_loader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=hparams["batch_size"], shuffle=False)
    model = select_model_type(model_type, train_dataset)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"],
                                    weight_decay=hparams["weight_decay"])

    model.to(device)

    trainer = Trainer(model=model, optimizer=optimizer,
                      data_loader=train_loader, epochs=hparams["num_epochs"],
                      subsample_aggregation=False)

    trainer.train()

    all_probs = []
    all_logits = []

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            probs, logits, loss = model(data)

            all_probs.append(probs.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    test_filenames = test_loader.dataset.filenames

    top_2_probs = get_top_2_predictions(all_probs)

    final_preds = probs2dict(top_2_probs, test_filenames, alpha, beta)

    acc_presence = acc_presence_total(final_preds)
    acc_salience = acc_salience_total(final_preds)

    print(f"Test Accuracy Presence: {acc_presence:.4f}, Salience: {acc_salience:.4f}")



def main():
    data_folder = "/home/tim/Work/quantum/data/blemore/"

    # paths
    train_metadata = os.path.join(data_folder, "train_metadata.csv")
    test_metadata = os.path.join(data_folder, "test_metadata.csv")

    encoding_paths_2d = {
        "openface": os.path.join(data_folder, "encoded_videos/static_data/openface_static_features.npz"),
        "imagebind": os.path.join(data_folder, "encoded_videos/static_data/imagebind_static_features.npz"),
        "clip": os.path.join(data_folder, "encoded_videos/static_data/clip_static_features.npz"),
        "videoswintransformer": os.path.join(data_folder,
                                             "encoded_videos/static_data/videoswintransformer_static_features.npz"),
        "videomae": os.path.join(data_folder, "encoded_videos/static_data/videomae_static_features.npz"),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(train_metadata)
    train_records = train_df.to_dict(orient="records")
    train_labels = create_labels(train_records)

    test_df = pd.read_csv(test_metadata)
    test_records = test_df.to_dict(orient="records")
    test_labels = create_labels(test_records)

    encoders = ["imagebind", "videomae", "videoswintransformer", "openface", "clip"]
    model_types = ["Linear", "MLP_256", "MLP_512"]
    folds = [0, 1, 2, 3, 4]

    summary_rows = []

    for encoder in encoders:

        # Pick encoding path
        encoding_folder = encoding_paths_2d[encoder]

        for model_type in model_types:
            for fold_id in folds:
                print(f"\nRunning encoder={encoder}, model={model_type}, fold={fold_id}")

                # Dataset split
                (train_files_, train_labels_), (val_files_, val_labels_) = get_validation_split(train_df, train_labels, fold_id)

                train_dataset, val_dataset = prepare_split_2d(train_files_, train_labels_, val_files_, val_labels_,
                                                              encoding_folder)

                print(f"Train dataset size: {len(train_dataset)}")
                print(f"Validation dataset size: {len(val_dataset)}")
                print(f"Input dimension: {train_dataset.input_dim}")
                print(f"Output dimension: {train_dataset.output_dim}")

                train_loader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=hparams["batch_size"], shuffle=False)

                model = select_model_type(model_type, train_dataset)

                optimizer = torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"],
                                             weight_decay=hparams["weight_decay"])
                model.to(device)

                trainer = Trainer(model=model, optimizer=optimizer,
                                  data_loader=train_loader, epochs=hparams["num_epochs"],
                                  valid_data_loader=val_loader,
                                  subsample_aggregation=False)

                log_dir = f"runs/{encoder}_{model_type}_fold{fold_id}"
                writer = SummaryWriter(log_dir=log_dir)
                res = trainer.train(writer=writer)
                writer.close()

                best_epoch = max(res, key=lambda r: 0.5 * r["best_acc_presence"] + 0.5 * r["best_acc_salience"])

                summary_rows.append({
                    "encoder": encoder,
                    "model": model_type,
                    "fold": fold_id,
                    "epoch": best_epoch["epoch"],
                    "val_loss": best_epoch["val_loss"],
                    "acc_presence": best_epoch["best_acc_presence"],
                    "acc_salience": best_epoch["best_acc_salience"],
                    "alpha": best_epoch["best_alpha"],
                    "beta": best_epoch["best_beta"],
                })

            # Convert collected summary rows to DataFrame (for the current encoder + model_type)
            summary_df_fold = pd.DataFrame(summary_rows)

            # Filter for current encoder and model_type
            filtered = summary_df_fold[
                (summary_df_fold["encoder"] == encoder) & (summary_df_fold["model"] == model_type)]

            # Compute mean alpha and beta
            alpha = filtered["alpha"].mean()
            beta = filtered["beta"].mean()

            print(f"Selected alpha: {alpha:.4f}, beta: {beta:.4f} for encoder={encoder}, model={model_type}")

            train_and_test(
                           model_type,
                           encoding_folder,
                           train_df,
                           train_labels,
                           test_df,
                           test_labels,
                           alpha,
                           beta,
                           device)

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df)
    summary_df.to_csv("validation_summary.csv", index=False)

    print("\nFold-averaged results:")
    print(summary_df.groupby(["encoder", "model"])[["acc_presence", "acc_salience"]].mean())


if __name__ == "__main__":
    main()
