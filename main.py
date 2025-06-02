import pandas as pd
from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter
from model.models import MultiLabelRNN, MultiLabelLinearNN, MultiLabelLinearNNShallow, MultiLabelLinearNNSuperShallow

from trainer import Trainer
from utils.create_soft_labels import create_labels
from utils.set_splitting import prepare_split_2d
import os

hparams = {
    "batch_size": 32,
    "max_seq_len": None,  # Set to None for no padding/truncation
    "learning_rate": 0.0001,  # Use the HEAD value
    "num_epochs": 100,
    "weight_decay": 1e-3,
}

def main():
    data_folder = "/home/tim/Work/quantum/data/blemore/"

    # paths
    train_metadata = os.path.join(data_folder, "train_metadata.csv")
    test_metadata = os.path.join(data_folder, "test_metadata.csv")

    encoding_paths_2d = {
        "openface": os.path.join(data_folder, "encoded_videos/static_data/openface_static_features.npz"),
        "imagebind": os.path.join(data_folder, "encoded_videos/static_data/imagebind_static_features.npz"),
        "clip": os.path.join(data_folder, "encoded_videos/static_data/clip_static_features.npz"),
        "videoswintransformer": os.path.join(data_folder, "encoded_videos/static_data/videoswintransformer_static_features.npz"),
        "videomae": os.path.join(data_folder, "encoded_videos/static_data/videomae_static_features.npz"),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(train_metadata)
    train_records = train_df.to_dict(orient="records")
    train_labels = create_labels(train_records)

    encoders = ["imagebind", "clip", "videoswintransformer", "videomae", "openface"]
    models = ["shallow", "very_shallow", "deep"]
    folds = [0, 1, 2, 3, 4]

    summary_rows = []

    for model_type in models:
        for encoder in encoders:
            for fold_id in folds:
                print(f"\nRunning encoder={encoder}, model={model_type}, fold={fold_id}")

                # Pick encoding path
                encoding_folder = encoding_paths_2d[encoder]

                # Dataset split
                train_dataset, val_dataset = prepare_split_2d(train_df, train_labels, fold_id, encoding_folder)

                train_loader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=hparams["batch_size"], shuffle=False)

                # Model selection
                if model_type == "deep":
                    model = MultiLabelLinearNN(input_dim=train_dataset.input_dim,
                                               output_dim=train_dataset.output_dim,
                                               activation="softmax")
                elif model_type == "shallow":
                    model = MultiLabelLinearNNShallow(input_dim=train_dataset.input_dim,
                                                      output_dim=train_dataset.output_dim)
                elif model_type == "very_shallow":
                    model = MultiLabelLinearNNSuperShallow(input_dim=train_dataset.input_dim,
                                                           output_dim=train_dataset.output_dim)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")

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

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df)
    summary_df.to_csv("validation_summary.csv", index=False)

    print("\nFold-averaged results:")
    print(summary_df.groupby(["encoder", "model"])[["acc_presence", "acc_salience"]].mean())

if __name__ == "__main__":
    main()
