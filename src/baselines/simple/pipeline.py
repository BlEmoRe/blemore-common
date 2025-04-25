import os
import pandas as pd
from config import ROOT_DIR
from src.baselines.simple.create_dataset.aggregate_data import aggregate
from src.baselines.simple.create_dataset.create_dataset import create_dataset
from src.baselines.simple.train_eval.cross_validation import cross_validate


def main():
    raw_openface_files_path = "/media/tim/Seagate Hub/mixed_emotion_challenge/openface_files/train"
    metadata_path = os.path.join(ROOT_DIR, "data/train_metadata.csv")

    save_folder = os.path.join(ROOT_DIR, "data/baselines/simple/train")
    os.makedirs(save_folder, exist_ok=True)

    # df = aggregate(raw_openface_files_path, metadata_path, save_folder)

    df = pd.read_csv(os.path.join(save_folder, "aggregated_openface.csv"))

    data_dict, label_to_index = create_dataset(df, save_folder, train=True)

    cross_validate()





