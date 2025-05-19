import os
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
from tqdm import tqdm

from src.baselines.simple.config_simple_baseline import feature_columns
from src.baselines.simple.create_dataset.openface_operations import (
    get_success_ratio,
    get_ok_confidence_ratio,
    interpolate_openface,
)

def convert_openface_to_npy(raw_openface_dir, save_dir,
                             confidence_threshold=0.85, good_frames_ratio_threshold=0.85):
    os.makedirs(save_dir, exist_ok=True)
    input_files = glob(os.path.join(raw_openface_dir, "*.csv"))

    valid_filenames = []

    for path in tqdm(input_files):
        filename = Path(path).stem
        df = pd.read_csv(path)

        if df.empty:
            continue

        sr = get_success_ratio(df)
        cr = get_ok_confidence_ratio(df, confidence_threshold)

        if sr < good_frames_ratio_threshold or cr < good_frames_ratio_threshold:
            print(f"Throwing away {filename}")
            continue

        if sr < 1 or cr < 1:
            print(f"Interpolating {filename}")
            df = interpolate_openface(df, confidence_threshold)

        features = df[feature_columns].to_numpy().astype(np.float32)
        out_path = os.path.join(save_dir, f"{filename}.npy")
        np.save(out_path, features)
        valid_filenames.append(filename)

    return valid_filenames


def main():
    openface_raw_dir = Path("/home/tim/Work/quantum/data/blemore/encoded_videos/openface/")
    openface_npy_dir = Path("/home/tim/Work/quantum/data/blemore/encoded_videos/openface_npy/")

    convert_openface_to_npy(openface_raw_dir, openface_npy_dir)


if __name__ == "__main__":
    main()