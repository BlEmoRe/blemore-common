import os
import json
import pandas as pd
import numpy as np

from config import ROOT_DIR
from src.benchmarks.simple.constants import feature_columns

train_path = os.path.join(ROOT_DIR, "data/benchmarks/simple/agg_openface_data.csv")
label_map_path = os.path.join(ROOT_DIR, "data/benchmarks/simple/emotion_label_mapping_probabilistic.json")

# Load training data first to create label mapping
df_train = pd.read_csv(train_path)

unique_labels = sorted(df_train["emotion_1"].unique())
unique_labels.remove("neu")
label_to_index = {label: i for i, label in enumerate(unique_labels)}

# Save label mapping as a JSON file
with open(label_map_path, "w") as f:
    json.dump(label_to_index, f, indent=4)

df = pd.read_csv(train_path)

n = df.shape[0]
y = np.zeros((n, len(unique_labels)), dtype=float)

for i in range(n):
    emotion_1 = df["emotion_1"].iloc[i]
    emotion_2 = df["emotion_2"].iloc[i]
    mix = df["mix"].iloc[i]
    emotion_1_salience = df["emotion_1_salience"].iloc[i]
    emotion_2_salience = df["emotion_2_salience"].iloc[i]
    if emotion_1 not in label_to_index:
        continue
    if mix == 0:
        y[i, label_to_index[emotion_1]] = 1
    elif mix == 1:
        y[i, label_to_index[emotion_1]] = emotion_1_salience / 100
        y[i, label_to_index[emotion_2]] = emotion_2_salience / 100
    else:
        raise ValueError("Invalid mix value")

# Store original indices **before** shuffling
original_indices = df.index.to_numpy()

# Extract features
X = df.loc[:, df.columns.str.contains('|'.join(feature_columns))].values

folds = df["fold"].values
perm = np.random.permutation(len(df))
X, y, original_indices, folds = X[perm], y[perm], original_indices[perm], folds[perm]

# Save the processed data
np.savez_compressed(os.path.join(ROOT_DIR, f"data/benchmarks/simple/train_data_probabilistic.npz"),
                    X=X,
                    y=y,
                    indices=original_indices,
                    folds=folds)
