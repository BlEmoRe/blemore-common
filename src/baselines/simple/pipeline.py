import os
import pandas as pd
from config import ROOT_DIR
from src.baselines.simple.create_dataset.aggregate_data import aggregate
from src.baselines.simple.create_dataset.create_dataset import create_dataset
from src.baselines.simple.train_eval.cross_validation import cross_validate
from src.baselines.simple.train_eval.test_set_validation import train_and_evaluate


def cross_validation_pipeline():
    raw_openface_files_path = "/media/tim/Seagate Hub/mixed_emotion_challenge/openface_files/train"
    metadata_path = os.path.join(ROOT_DIR, "data/train_metadata.csv")

    save_folder = os.path.join(ROOT_DIR, "data/baselines/simple/train")
    os.makedirs(save_folder, exist_ok=True)

    # df = aggregate(metadata_path, raw_openface_files_path, save_folder)

    df = pd.read_csv(os.path.join(save_folder, "aggregated_openface.csv"))

    dataset, label2index = create_dataset(df, save_folder, train=True)
    index2label = {v: k for k, v in label2index.items()}

    cross_validate(dataset, df, index2label)


def test_pipeline():
    # Paths
    raw_openface_train_path = "/media/tim/Seagate Hub/mixed_emotion_challenge/openface_files/train"
    raw_openface_test_path = "/media/tim/Seagate Hub/mixed_emotion_challenge/openface_files/test"

    train_metadata_path = os.path.join(ROOT_DIR, "data/train_metadata.csv")
    test_metadata_path = os.path.join(ROOT_DIR, "data/test_metadata.csv")

    train_save_folder = os.path.join(ROOT_DIR, "data/baselines/simple/train")
    test_save_folder = os.path.join(ROOT_DIR, "data/baselines/simple/test")
    os.makedirs(train_save_folder, exist_ok=True)
    os.makedirs(test_save_folder, exist_ok=True)

    # Aggregate OpenFace features if not already aggregated
    # train_df = aggregate(train_metadata_path, raw_openface_train_path, train_save_folder)
    # test_df = aggregate(test_metadata_path, raw_openface_test_path, test_save_folder)

    train_df = pd.read_csv(os.path.join(train_save_folder, "aggregated_openface.csv"))
    test_df = pd.read_csv(os.path.join(test_save_folder, "aggregated_openface.csv"))

    # Create datasets
    train_dataset, label2index = create_dataset(train_df, train_save_folder, train=True)
    test_dataset, _ = create_dataset(test_df, test_save_folder, train=False)

    index2label = {v: k for k, v in label2index.items()}

    # Train on full train set and evaluate on test set
    train_and_evaluate(train_dataset, test_dataset, test_df, index2label)



if __name__ == "__main__":
    # cross_validation_pipeline()
    test_pipeline()

