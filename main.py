import pandas as pd
from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter
from model.models import MultiLabelRNN, MultiLabelLinearNN

from trainer import Trainer

from utils.create_soft_labels import create_labels
from utils.set_splitting import prepare_split_3d, prepare_split_2d


# 🔧 1. Add early stopping or checkpointing
# Let validation metrics decide the epoch — not fixed num_epochs.
#
# 🔧 2. Add stronger regularization
# Increase Dropout to 0.3–0.5
#
# Add weight_decay = 1e-3 or 1e-2
#
# 🔧 3. Normalize your label vectors
# Ensure they sum to 1 (or a fixed target mass for blends) — if they don’t, KLDivLoss can behave unpredictably.
#
# 🔧 4. Replace KL loss with JS divergence, MSE, or even cosine distance
# KL is harsh on small differences. Try:
# TODO: MSE loss seems useless when tried in a quick experiment...
# python
# Edit
# F.mse_loss(preds, targets)  # Or cosine similarity
# To see if loss trends behave differently.


hparams = {
    "batch_size": 32,
    "max_seq_len": None,  # Set to None for no padding/truncation
    "learning_rate": 0.0005,
    "num_epochs": 200,
    "weight_decay": 1e-2,
}


def main():
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
        "imagebind": "/home/tim/Work/quantum/data/blemore/encoded_videos/static_data/imagebind_static_features.npz",
        "clip": "/home/tim/Work/quantum/data/blemore/encoded_videos/static_data/clip_static_features.npz",
        "dinov2": "/home/tim/Work/quantum/data/blemore/encoded_videos/static_data/dinov2_static_features.npz",
        "videoswintransformer": "/home/tim/Work/quantum/data/blemore/encoded_videos/static_data/videoswintransformer_static_features.npz",
        "videomae": "/home/tim/Work/quantum/data/blemore/encoded_videos/static_data/videomae_static_features.npz",
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(train_metadata)
    train_records = train_df.to_dict(orient="records")
    train_labels = create_labels(train_records)

    encoders = ["openface", "imagebind", "clip", "dinov2", "videoswintransformer", "videomae"]
    # models = ["rnn", "linear"]
    models = ["linear"]
    folds = [0, 1, 2, 3, 4]

    summary_rows = []

    for model_type in models:
        for encoder in encoders:
            for fold_id in folds:
                print(f"\nRunning encoder={encoder}, model={model_type}, fold={fold_id}")

                # Pick encoding path
                encoding_folder = encoding_paths_3d[encoder] if model_type == "rnn" else encoding_paths_2d[encoder]

                # Dataset split
                if model_type == "rnn":
                    train_dataset, val_dataset = prepare_split_3d(train_df, train_labels, fold_id, encoding_folder)
                else:
                    train_dataset, val_dataset = prepare_split_2d(train_df, train_labels, fold_id, encoding_folder)

                train_loader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=hparams["batch_size"], shuffle=False)

                # Instantiate model
                if model_type == "rnn":
                    model = MultiLabelRNN(input_dim=train_dataset.input_dim,
                                          output_dim=train_dataset.output_dim,
                                          activation="softmax")  # or "entmax15"
                else:
                    model = MultiLabelLinearNN(input_dim=train_dataset.input_dim,
                                               output_dim=train_dataset.output_dim,
                                               activation="softmax")

                optimizer = torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"],
                                             weight_decay=hparams["weight_decay"])
                model.to(device)

                trainer = Trainer(model=model, optimizer=optimizer,
                                  data_loader=train_loader, epochs=hparams["num_epochs"],
                                  valid_data_loader=val_loader)

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
