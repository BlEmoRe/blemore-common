import os
import pandas as pd
from pathlib import Path

from glob import glob

from src.baselines.simple.openface_operations.config_openface import feature_columns
from src.baselines.simple.openface_operations.openface_helpers import get_success_ratio, interpolate_openface, \
    get_ok_confidence_ratio


def aggregate(metadata_path, raw_openface_files_path, save_folder):
    """
    This function aggregates the OpenFace data by calculating the mean, 20th, 50th, 80th percentiles and IQR for each file.

    :param metadata_path: filename mapping to metadata and labels
    :param raw_openface_files_path: folder with openface files
    :param save_folder: where to save output (optional)
    :return:
    """

    save_path = os.path.join(save_folder, "aggregated_openface.csv")

    df_metadata = pd.read_csv(metadata_path)
    files_glob = glob(raw_openface_files_path + "/*.csv")

    confidence_threshold = 0.85
    good_frames_ratio_threshold = 0.85

    interpolated = []
    thrown_away = []
    empty = []

    agg = []

    for idx, path  in enumerate(files_glob):

        filename = Path(path).stem

        item = {"filename": filename}

        df = pd.read_csv(path)

        if df.shape[0] == 0:
            print("filename has no data: ", filename)
            empty.append(filename)
            continue

        # Ensure quality of the data
        # If the data quality is not good enough, the data is thrown away or interpolated
        success_ratio = get_success_ratio(df)
        ok_confidence_ratio = get_ok_confidence_ratio(df, confidence_threshold)
        if success_ratio >= good_frames_ratio_threshold and ok_confidence_ratio >= good_frames_ratio_threshold:
            if success_ratio < 1 or ok_confidence_ratio < 1:
                print(f"INTERPOLATION: Data quality ok but not perfect for {filename}. has success ratio of {success_ratio} and confidence ratio of {ok_confidence_ratio}")
                print(f"Interpolating {filename}")
                df = interpolate_openface(df, confidence_threshold)
                interpolated.append(filename)
        else:
            print(f"DISCARD: Data quality not good enough for {filename}. has success ratio of {success_ratio} and confidence ratio of {ok_confidence_ratio}")
            print(f"Throwing away {filename}")
            thrown_away.append(filename)
            continue

        means = df[feature_columns].mean()
        item.update({f"{col}_mean": mean_val for col, mean_val in means.items()})

        quantile_20 = df[feature_columns].quantile(0.2)
        item.update({f"{col}_20th": val for col, val in quantile_20.items()})

        quantile_50 = df[feature_columns].quantile(0.5)
        item.update({f"{col}_50th": val for col, val in quantile_50.items()})

        quantile_80 = df[feature_columns].quantile(0.8)
        item.update({f"{col}_80th": val for col, val in quantile_80.items()})

        iqr_values = quantile_80 - quantile_20
        item.update({f"{col}_iqr": val for col, val in iqr_values.items()})

        agg.append(item)

        if idx % 100 == 0:
            print(f"Processed {idx+1}/{len(files_glob)} files...")
            print(f"Interpolated: {len(interpolated)}")
            print(f"Thrown away: {len(thrown_away)}")
            print(f"Empty: {len(empty)}")

    print(f"\nProcessed {len(files_glob)} files...")
    print(f"Interpolated: {len(interpolated)}")
    print(f"Thrown away: {len(thrown_away)}")
    print(f"Empty: {len(empty)}")

    df = pd.DataFrame(agg)

    df_out = pd.merge(df, df_metadata, on="filename", how="inner")

    df_out.to_csv(save_path, index=False)

    return df_out








