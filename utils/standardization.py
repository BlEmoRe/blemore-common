import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Only used for visualization purposes
def standardize_by_group(x, groups):
    """
    :param x: a matrix with shape (observations, features)
    :param groups: np Array
    :return: scaled x matrix
    """
    for group in np.unique(groups):
        scaler = StandardScaler()

        rows = np.where(groups == group)[0]
        x[rows] = scaler.fit_transform(x[rows])
    return x


# Used in training / testing settings
def create_transform(filenames, directory):
    x_array = []

    for fn in filenames:
        x = np.load(os.path.join(directory, f"{fn}.npy"))
        x_array.append(x)
    X = np.concatenate(x_array, axis=0)
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


def main():
    enc_path = "/home/tim/Work/quantum/data/blemore/encoded_videos/openface_npy/"
    train_metadata = "/home/tim/Work/quantum/data/blemore/train_metadata.csv"

    train_metadata = "/home/tim/Work/quantum/data/blemore/train_metadata.csv"
    df = pd.read_csv(train_metadata)
    files = df["filename"].tolist()

    scaler = create_transform(files, enc_path)


if __name__ == "__main__":
    main()

