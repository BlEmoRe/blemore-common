import pandas as pd
from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter
from model.models import MultiLabelRNN, MultiLabelLinearNN

from trainer import Trainer

from utils.create_soft_labels import create_labels
from utils.set_splitting import prepare_split_3d, prepare_split_2d, prepare_split_subsampled
import os

hparams = {
    "batch_size": 512,
    "max_seq_len": None,  # Set to None for no padding/truncation
    "learning_rate": 0.0005,
    "num_epochs": 200,
    "weight_decay": 1e-3,
}


def main():
    data_folder = "/home/tim/Work/quantum/data/blemore/"
    # data_folder = "/home/user/Work/quantum/data/blemore/"

    # paths
    train_metadata = os.path.join(data_folder, "train_metadata.csv")
    test_metadata = os.path.join(data_folder, "test_metadata.csv")

    encoding_paths_3d = {
        "dinov2": os.path.join(data_folder, "encoded_videos/dynamic_data/DINOv2_first_component/"),
        "videoswintransformer": os.path.join(data_folder, "encoded_videos/dynamic_data/VideoSwinTransformer/"),
        "videomae": os.path.join(data_folder, "encoded_videos/dynamic_data/VideoMAEv2_reshaped/"),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(train_metadata)
    train_records = train_df.to_dict(orient="records")
    train_labels = create_labels(train_records)

    encoders = ["dinov2", "videomae", "videoswintransformer"]
    folds = [0, 1, 2, 3, 4]

    summary_rows = []

    for encoder in encoders:
        for fold_id in folds:
            print(f"\nRunning encoder={encoder}, fold={fold_id}")

            # Pick encoding path
            encoding_folder = encoding_paths_3d[encoder]

            train_dataset, val_dataset = prepare_split_subsampled(train_df, train_labels, fold_id, encoding_folder)

            print(f"Train dataset size: {len(train_dataset)}")
            print(f"Validation dataset size: {len(val_dataset)}")

            print(f"Input dimension: {train_dataset.input_dim}")
            print(f"Output dimension: {train_dataset.output_dim}")

            train_loader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=hparams["batch_size"], shuffle=False)

            model = MultiLabelLinearNN(input_dim=train_dataset.input_dim,
                                           output_dim=train_dataset.output_dim,
                                           activation="softmax")

            optimizer = torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"],
                                         weight_decay=hparams["weight_decay"])
            model.to(device)

            trainer = Trainer(model=model, optimizer=optimizer,
                              data_loader=train_loader, epochs=hparams["num_epochs"],
                              valid_data_loader=val_loader, subsample_aggregation=True)

            log_dir = f"runs/{encoder}_fold{fold_id}"
            writer = SummaryWriter(log_dir=log_dir)
            res = trainer.train(writer=writer)

            writer.close()

            best_epoch = max(res, key=lambda r: 0.5 * r["best_acc_presence"] + 0.5 * r["best_acc_salience"])

            summary_rows.append({
                "encoder": encoder,
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
