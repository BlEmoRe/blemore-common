import os
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
from tqdm import tqdm

from feature_extraction.openface_operations.config_openface import feature_columns
from feature_extraction.openface_operations.openface_helpers import (
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

        sr = get_success_ratio(df)
        cr = get_ok_confidence_ratio(df, confidence_threshold)

        if good_frames_ratio_threshold <= sr < 1 or good_frames_ratio_threshold <= cr < 1:
            print(f"Interpolating {filename}")
            print(f"Success ratio: {sr:.2f}, Confidence ratio: {cr:.2f}")
            df = interpolate_openface(df, confidence_threshold)

        if df.empty:
            num_frames = 1  # default to 1 timestep, or estimate from metadata if needed
            features = np.zeros((num_frames, len(feature_columns)), dtype=np.float32)
        else:
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