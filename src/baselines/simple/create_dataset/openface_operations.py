import numpy as np
from src.baselines.simple.config_openface import feature_columns


def get_success_ratio(df):
    success = df["success"].values
    n_rows = df.shape[0]
    ratio_successful = (success == 1).sum() / n_rows
    return ratio_successful


def get_ok_confidence_ratio(df, confidence_threshold=0.85):
    confidence = df["confidence"].values
    n_rows = df.shape[0]
    ratio_high_conf = (confidence >= confidence_threshold).sum() / n_rows
    return ratio_high_conf


def interpolate_openface(df, confidence_threshold=0.85):
    """
    :param confidence_threshold: interpolate frames with confidence below this threshold
    :param df: DataFrame containing OpenFace data
    :return: DataFrame with interpolated values
    """
    # iterate over feature columns
    for x in feature_columns:
        df.loc[(df["success"] != 1) |
               (df["confidence"] < confidence_threshold), x] = np.nan

    # interpolate
    df[feature_columns] = df[feature_columns].interpolate(method="linear")

    # drop rows that couldn't be interpolated
    df.dropna(subset=feature_columns, inplace=True)

    return df