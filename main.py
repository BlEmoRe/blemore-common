import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.models import ConfigurableLinearNN
from post_processing import get_top_2_predictions, probs2dict
from trainer import Trainer
from utils.create_soft_labels import create_labels
from utils.generic_accuracy.accuracy_funcs import acc_presence_total, acc_salience_total
from utils.set_splitting import prepare_split_2d, get_validation_split, prepare_split_subsampled
from utils.subsample_utils import aggregate_subsamples

# --- Global Settings ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hparams = {
    "batch_size": 32,
    "learning_rate": 5e-5,
    "num_epochs": 5,
    "weight_decay": 1e-3,
}


def select_model(model_type, input_dim, output_dim):
    if model_type == "Linear":
        return ConfigurableLinearNN(input_dim=input_dim, output_dim=output_dim, n_layers=0)
    elif model_type == "MLP_256":
        return ConfigurableLinearNN(input_dim=input_dim, output_dim=output_dim, n_layers=1, hidden_dim=256)
    elif model_type == "MLP_512":
        return ConfigurableLinearNN(input_dim=input_dim, output_dim=output_dim, n_layers=1, hidden_dim=512)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_one_fold(train_dataset, val_dataset, model_type, log_dir, subsample_aggregation=False):
    train_loader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hparams["batch_size"], shuffle=False)

    model = select_model(model_type, train_dataset.input_dim, train_dataset.output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"], weight_decay=hparams["weight_decay"])
    model.to(device)

    trainer = Trainer(model=model, optimizer=optimizer,
                      data_loader=train_loader, epochs=hparams["num_epochs"],
                      valid_data_loader=val_loader, subsample_aggregation=subsample_aggregation)

    writer = SummaryWriter(log_dir=log_dir)
    res = trainer.train(writer=writer)
    writer.close()

    best_epoch = max(res, key=lambda r: 0.5 * r["best_acc_presence"] + 0.5 * r["best_acc_salience"])
    return best_epoch


def train_and_test(train_dataset, test_dataset, model_type, alpha, beta, subsample_aggregation):
    train_loader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=hparams["batch_size"], shuffle=False)

    model = select_model(model_type, train_dataset.input_dim, train_dataset.output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"], weight_decay=hparams["weight_decay"])
    model.to(device)

    trainer = Trainer(model=model, optimizer=optimizer,
                      data_loader=train_loader, epochs=hparams["num_epochs"],
                      subsample_aggregation=subsample_aggregation)

    trainer.train()

    all_probs = []
    all_logits = []
    model.eval()
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            probs, logits, _ = model(data)
            all_probs.append(probs.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)

    test_filenames = test_loader.dataset.filenames

    if subsample_aggregation:
        test_filenames, all_probs = aggregate_subsamples(test_filenames, all_logits)

    top_2_probs = get_top_2_predictions(all_probs)
    final_preds = probs2dict(top_2_probs, test_filenames, alpha, beta)

    acc_presence = acc_presence_total(final_preds)
    acc_salience = acc_salience_total(final_preds)

    return acc_presence, acc_salience


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

    encoding_paths_3d = {
        "videoswintransformer": os.path.join(data_folder, "encoded_videos/dynamic_data/VideoSwinTransformer/"),
        "videomae": os.path.join(data_folder, "encoded_videos/dynamic_data/VideoMAEv2_reshaped/"),
    }


    train_df = pd.read_csv(train_metadata)
    train_labels = create_labels(train_df.to_dict(orient="records"))

    test_df = pd.read_csv(test_metadata)
    test_labels = create_labels(test_df.to_dict(orient="records"))

    encoders_2d = ["imagebind", "videomae", "videoswintransformer", "openface", "clip"]
    encoders_3d = ["videoswintransformer", "videomae"]

    model_types = ["Linear", "MLP_256", "MLP_512"]

    folds = [0, 1, 2, 3, 4]

    summary_rows = []
    test_summary_rows = []

    subsample_aggregation = False

    for mode in ["3d"]:
        if mode == "2d":
            encoders = encoders_2d
            encoding_paths = encoding_paths_2d
            hparams["batch_size"] = 32
            subsample_aggregation = False
        else:
            encoders = encoders_3d
            encoding_paths = encoding_paths_3d
            hparams["batch_size"] = 512
            subsample_aggregation = True

        for encoder in encoders:
            encoding_path = encoding_paths[encoder]

            for model_type in model_types:
                fold_results = []

                for fold_id in folds:
                    print(f"\nRunning encoder={encoder}, model={model_type}, fold={fold_id}, mode={mode}")
                    if mode == "2d":
                        # Prepare 2D dataset
                        (train_files, train_labels_fold), (val_files, val_labels) = get_validation_split(train_df, train_labels, fold_id)
                        train_dataset, val_dataset = prepare_split_2d(train_files, train_labels_fold, val_files, val_labels, encoding_path)
                    else:
                        train_dataset, val_dataset = prepare_split_subsampled(train_df, train_labels, fold_id, encoding_path)

                    log_dir = f"runs/{encoder}_{model_type}_fold{fold_id}, mode={mode}"
                    best_epoch = train_one_fold(train_dataset, val_dataset, model_type, log_dir, subsample_aggregation)
                    best_epoch.update({"encoder": encoder, "model": model_type, "fold": fold_id, "mode": mode})

                    summary_rows.append(best_epoch)
                    fold_results.append(best_epoch)

                # Compute average alpha and beta over folds
                fold_df = pd.DataFrame(fold_results)
                alpha_mean = fold_df["best_alpha"].mean()
                beta_mean = fold_df["best_beta"].mean()

                print(f"Selected alpha: {alpha_mean:.4f}, beta: {beta_mean:.4f} for encoder={encoder}, model={model_type}, mode={mode}")

                # Train on full train set and evaluate on test set
                train_files = train_df.filename.tolist()
                test_files = test_df.filename.tolist()
                train_dataset, test_dataset = prepare_split_2d(train_files, train_labels, test_files, test_labels, encoding_path)

                acc_presence, acc_salience = train_and_test(train_dataset, test_dataset, model_type, alpha_mean, beta_mean)

                print(f"Test Accuracy Presence: {acc_presence:.4f}, Salience: {acc_salience:.4f}")

                # Save test results
                test_summary_rows.append({
                    "mode": mode,
                    "encoder": encoder,
                    "model": model_type,
                    "alpha": alpha_mean,
                    "beta": beta_mean,
                    "test_acc_presence": acc_presence,
                    "test_acc_salience": acc_salience,
                })

    # Save validation results
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("validation_summary.csv", index=False)
    print("\nValidation Summary:")
    print(summary_df)

    # Save test results
    test_summary_df = pd.DataFrame(test_summary_rows)
    test_summary_df.to_csv("test_summary.csv", index=False)
    print("\nTest Summary:")
    print(test_summary_df)

    # Fold-averaged validation results
    print("\nFold-averaged Validation Results:")
    print(summary_df.groupby(["encoder", "model"])[["best_acc_presence", "best_acc_salience"]].mean())

    # Encoder/model-averaged test results
    print("\nAveraged Test Results:")
    print(test_summary_df.groupby(["encoder", "model"])[["test_acc_presence", "test_acc_salience"]].mean())



if __name__ == "__main__":
    main()
