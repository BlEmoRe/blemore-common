import os
import pandas as pd
from config import ROOT_DIR
from src.baselines.simple.create_dataset.aggregate_data import aggregate
from src.baselines.simple.create_dataset.create_dataset import create_dataset
from src.baselines.simple.train_eval.cross_validation import cross_validate


def cross_validation_pipeline():
    raw_openface_files_path = "/media/tim/Seagate Hub/mixed_emotion_challenge/openface_files/train"
    metadata_path = os.path.join(ROOT_DIR, "data/train_metadata.csv")

    save_folder = os.path.join(ROOT_DIR, "data/baselines/simple/train")
    os.makedirs(save_folder, exist_ok=True)

    # df = aggregate(raw_openface_files_path, metadata_path, save_folder)

    df = pd.read_csv(os.path.join(save_folder, "aggregated_openface.csv"))

    dataset, label2index = create_dataset(df, save_folder, train=True)
    index2label = {v: k for k, v in label2index.items()}

    cross_validate(dataset, df, index2label)


def test_pipeline():
    raw_openface_files_path = "/media/tim/Seagate Hub/mixed_emotion_challenge/openface_files/test"
    metadata_path = os.path.join(ROOT_DIR, "data/test_metadata.csv")

    save_folder = os.path.join(ROOT_DIR, "data/baselines/simple/test")
    os.makedirs(save_folder, exist_ok=True)

    df = aggregate(raw_openface_files_path, metadata_path, save_folder)

    dataset, label2index = create_dataset(df, save_folder, train=False)
    index2label = {v: k for k, v in label2index.items()}




if __name__ == "__main__":
    cross_validation_pipeline()

