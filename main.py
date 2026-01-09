import os
import json
from collections import Counter

import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.model import ConfigurableLinearNN
from model.post_process import get_top_2_predictions, probs2dict
from trainer.trainer import Trainer
from utils.create_soft_labels import create_labels
from utils.generic_accuracy.accuracy_funcs import acc_presence_total, acc_salience_total
from utils.set_splitting import prepare_split_2d, get_validation_split

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
    "learning_rate": 5e-6,
    "num_epochs": 200,
    "weight_decay": 1e-3,
}

data_folder = "/home/tim/Work/quantum/data/blemore/"

train_metadata_path = os.path.join(data_folder, "train_metadata.csv")
test_metadata_path = os.path.join(data_folder, "test_metadata.csv")

encoding_paths = {
    # vision
    "openface": os.path.join(data_folder, "encoded_videos/static_data/openface_static_features.npz"),
    "imagebind": os.path.join(data_folder, "encoded_videos/static_data/imagebind_static_features.npz"),
    "clip": os.path.join(data_folder, "encoded_videos/static_data/clip_static_features.npz"),
    "videoswintransformer": os.path.join(data_folder,
                                         "encoded_videos/static_data/videoswintransformer_static_features.npz"),
    "videomae": os.path.join(data_folder, "encoded_videos/static_data/videomae_static_features.npz"),

    # audio
    "wavlm": os.path.join(data_folder, "encoded_videos/static_data/wavlm_static_features.npz"),
    "hubert": os.path.join(data_folder, "encoded_videos/static_data/hubert_static_features.npz"),

    # fused
    "imagebind_wavlm": os.path.join(data_folder, "encoded_videos/static_data/fused/imagebind_wavlm_fused.npz"),
    "imagebind_hubert": os.path.join(data_folder, "encoded_videos/static_data/fused/imagebind_hubert_fused.npz"),
    "videomae_wavlm": os.path.join(data_folder, "encoded_videos/static_data/fused/videomae_wavlm_fused.npz"),
    "videomae_hubert": os.path.join(data_folder, "encoded_videos/static_data/fused/videomae_hubert_fused.npz"),

    # multimodal
    "hicmae": os.path.join(data_folder, "encoded_videos/static_data/hicmae_static_features.npz"),
}


def select_model(model_type, input_dim, output_dim):
    if model_type == "Linear":
        return ConfigurableLinearNN(input_dim=input_dim, output_dim=output_dim, model_type= model_type, n_layers=0)
    elif model_type == "MLP_256":
        return ConfigurableLinearNN(input_dim=input_dim, output_dim=output_dim, model_type= model_type, n_layers=1, hidden_dim=256)
    elif model_type == "MLP_512":
        return ConfigurableLinearNN(input_dim=input_dim, output_dim=output_dim, model_type= model_type, n_layers=1, hidden_dim=512)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_one_fold(train_dataset, val_dataset, model_type, log_dir, save_prefix):
    train_loader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hparams["batch_size"], shuffle=False)

    model = select_model(model_type, train_dataset.input_dim, train_dataset.output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"], weight_decay=hparams["weight_decay"])
    model.to(device)

    trainer = Trainer(model=model, optimizer=optimizer,
                      data_loader=train_loader, epochs=hparams["num_epochs"],
                      valid_data_loader=val_loader, subsample_aggregation=False)

    writer = SummaryWriter(log_dir=log_dir)
    best_epoch, best_model_path = trainer.train(writer=writer, save_prefix=save_prefix)
    writer.close()
    return best_epoch, best_model_path


def train_and_test_from_scratch(train_dataset, test_dataset, model_type, alpha, beta, encoder):
    train_loader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=hparams["batch_size"], shuffle=False)

    model = select_model(model_type, train_dataset.input_dim, train_dataset.output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"], weight_decay=hparams["weight_decay"])
    model.to(device)

    trainer = Trainer(model=model, optimizer=optimizer,
                      data_loader=train_loader, epochs=100,
                      subsample_aggregation=False)

    trainer.train()

    return evaluate_model(model, test_loader, alpha, beta, encoder)


def evaluate_model(model, test_loader, alpha, beta, encoder):
    all_probs = []
    model.eval()
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            probs, _, _ = model(data)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    top_2_probs = get_top_2_predictions(all_probs)
    test_filenames = test_loader.dataset.filenames

    final_preds = probs2dict(top_2_probs, test_filenames, alpha, beta)

    with open("data/{}_test_predictions.json".format(encoder), "w") as f:
        json.dump(final_preds, f, indent=4)

    acc_presence = acc_presence_total(final_preds)
    acc_salience = acc_salience_total(final_preds)

    return acc_presence, acc_salience

def run_validation(train_df, train_labels, encoders, model_types):
    folds = [0, 1, 2, 3, 4]

    summary_rows = []
    for encoder in encoders:
        encoding_path = encoding_paths[encoder]

        for model_type in model_types:
            fold_results = []
            for fold_id in folds:
                print(f"\nRunning encoder={encoder}, model={model_type}, fold={fold_id}")
                (train_files, train_labels_fold), (val_files, val_labels) = get_validation_split(train_df, train_labels, fold_id)
                train_dataset, val_dataset = prepare_split_2d(train_files, train_labels_fold, val_files, val_labels, encoding_path)

                log_dir = f"runs/{encoder}_{model_type}_fold{fold_id}"
                save_prefix = f"{encoder}_{model_type}_fold{fold_id}"
                best_epoch, _ = train_one_fold(train_dataset, val_dataset, model_type, log_dir, save_prefix)

                best_epoch.update({"encoder": encoder, "model": model_type, "fold": fold_id})
                summary_rows.append(best_epoch)
                fold_results.append(best_epoch)

    # Save validation results
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv("validation_summary.csv", index=False)
    print("\nValidation Summary:")
    print(summary_df)

    return summary_df


def run_test(train_df, train_labels, test_df, test_labels, encoders, model_types, use_best_model_from_val=True, use_fold_id=None):
    test_summary_rows = []

    # Load validation summary
    summary_df = pd.read_csv("data/validation_summary_hicmae.csv")

    for encoder in encoders:
        encoding_path = encoding_paths[encoder]

        for model_type in model_types:
            # Filter for encoder and model type
            fold_df = summary_df[(summary_df["encoder"] == encoder) & (summary_df["model"] == model_type)]

            # Select alpha and beta from the best fold
            best_row = fold_df.loc[
                (0.5 * fold_df["best_acc_presence"] + 0.5 * fold_df["best_acc_salience"]).idxmax()
            ]
            alpha_best = best_row["best_alpha"]
            beta_best = best_row["best_beta"]
            fold_id = best_row["fold"]

            print(f"Selected alpha: {alpha_best:.4f}, beta: {beta_best:.4f} for encoder={encoder}, model={model_type}")

            # Train on full train set and evaluate on test set
            train_files = train_df.filename.tolist()
            test_files = test_df.filename.tolist()
            train_dataset, test_dataset = prepare_split_2d(train_files, train_labels, test_files, test_labels, encoding_path)

            if use_best_model_from_val:

                if use_fold_id is not None:
                    fold_id = use_fold_id

                # Use best model from validation
                best_model_path = f"checkpoints/{encoder}_{model_type}_fold{fold_id}_best.pth"
                print(f"Loading model from {best_model_path}")

                checkpoint = torch.load(best_model_path, map_location=device)
                model = select_model(checkpoint['model_type'], checkpoint['input_dim'], checkpoint['output_dim'])
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(device)
                test_loader = DataLoader(test_dataset, batch_size=hparams["batch_size"], shuffle=False)

                acc_presence, acc_salience = evaluate_model(model, test_loader, alpha_best, beta_best, encoder)
            else:
                acc_presence, acc_salience = train_and_test_from_scratch(train_dataset, test_dataset, model_type, alpha_best, beta_best, encoder)

            print(f"Test Accuracy Presence: {acc_presence:.4f}, Salience: {acc_salience:.4f}")
            print(f"Test Accuracy Presence: {acc_presence:.4f}, Salience: {acc_salience:.4f}")

            # Save test results
            test_summary_rows.append({
                "encoder": encoder,
                "model": model_type,
                "alpha": alpha_best,
                "beta": beta_best,
                "test_acc_presence": acc_presence,
                "test_acc_salience": acc_salience,
            })

    # Save test results
    test_summary_df = pd.DataFrame(test_summary_rows)
    test_summary_df.to_csv("test_summary.csv", index=False)
    print("\nTest Summary:")
    print(test_summary_df)

    # Encoder/model-averaged test results
    print("\nAveraged Test Results:")
    print(test_summary_df.groupby(["encoder", "model"])[["test_acc_presence", "test_acc_salience"]].mean())


def main(do_val=True, do_test=False):
    vision_encoders = ["imagebind", "videomae", "videoswintransformer", "openface", "clip"]
    audio_encoders = ["wavlm", "hubert"]
    encoder_fusions = ["imagebind_wavlm", "imagebind_hubert", "videomae_wavlm", "videomae_hubert"]
    encoders = vision_encoders + audio_encoders + encoder_fusions

    model_types = ["Linear", "MLP_256", "MLP_512"]

    if do_val:
        train_df = pd.read_csv(train_metadata_path)
        train_labels = create_labels(train_df.to_dict(orient="records"))

        run_validation(train_df, train_labels, encoders, model_types)

    if do_test:
        test_df = pd.read_csv(test_metadata_path)
        test_labels = create_labels(test_df.to_dict(orient="records"))

        run_test(train_df, train_labels, test_df, test_labels, encoders, model_types, use_best_model_from_val=False)

if __name__ == "__main__":
    main(do_val=True, do_test=False)
