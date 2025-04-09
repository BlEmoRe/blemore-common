import os
import json
import pandas as pd
import numpy as np

from src.baselines.simple.config_simple_baseline import feature_columns

from src.baselines.simple.config_simple_baseline import AGGREGATED_OPENFACE_PATH, LABEL_MAPPING_PATH, VECTOR_TRAINING_SET_PATH

# Load training data
df = pd.read_csv(AGGREGATED_OPENFACE_PATH)

# Build label mapping (excluding 'neu')
unique_labels = sorted(df["emotion_1"].unique())
unique_labels.remove("neu")
label_to_index = {label: i for i, label in enumerate(unique_labels)}

# Save label mapping
with open(LABEL_MAPPING_PATH, "w") as f:
    json.dump(label_to_index, f, indent=4)

# Create label matrix
n_samples = df.shape[0]
n_labels = len(unique_labels)
y = np.zeros((n_samples, n_labels), dtype=float)

# populate label matrix
for i in range(n_samples):
    emotion_1, emotion_2 = df["emotion_1"].iloc[i], df["emotion_2"].iloc[i]
    salience_1, salience_2 = df["emotion_1_salience"].iloc[i], df["emotion_2_salience"].iloc[i]
    mix = df["mix"].iloc[i]

    if emotion_1 not in label_to_index:
        continue # skip "neu"

    if mix == 0:
        y[i, label_to_index[emotion_1]] = 1
    elif mix == 1:
        y[i, label_to_index[emotion_1]] = salience_1 / 100
        y[i, label_to_index[emotion_2]] = salience_2 / 100
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
np.savez_compressed(VECTOR_TRAINING_SET_PATH,
                    X=X,
                    y=y,
                    indices=original_indices,
                    folds=folds)
