import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

from config import ROOT_DIR

from utils.standardization import standardize_by_group

# --- Settings ---
data_folder = "/home/tim/Work/quantum/data/blemore/"

train_metadata_path = os.path.join(data_folder, "train_metadata.csv")

encoding_paths = {
    "openface": os.path.join(data_folder, "encoded_videos/static_data/openface_static_features.npz"),
    "imagebind": os.path.join(data_folder, "encoded_videos/static_data/imagebind_static_features.npz"),
    "clip": os.path.join(data_folder, "encoded_videos/static_data/clip_static_features.npz"),
    "videoswintransformer": os.path.join(data_folder, "encoded_videos/static_data/videoswintransformer_static_features.npz"),
    "videomae": os.path.join(data_folder, "encoded_videos/static_data/videomae_static_features.npz"),
    "wavlm": os.path.join(data_folder, "encoded_videos/static_data/wavlm_static_features.npz"),
    "hubert": os.path.join(data_folder, "encoded_videos/static_data/hubert_static_features.npz"),
}

# encoding = "hubert"
# encoding = "wavlm"  # <-- Choose your encoder here

# encoding = "imagebind"  # <-- Choose your encoder 3here
encoding = "videomae"  # <-- Choose your encoder here

save_path = os.path.join(ROOT_DIR, 'data/plots/pca/pca_{}_hap_sad.png'.format(encoding))

all_emotions = {'ang', 'disg', 'fea', 'hap', 'sad'}

# Focus emotions
focus_emotions = {'sad', 'hap'}  # Only keep these emotions

# --- Load data ---
imagebind_data = np.load(encoding_paths[encoding])
X = imagebind_data['X']
filenames = imagebind_data['filenames']

train_metadata = pd.read_csv(train_metadata_path)

# Match filenames
metadata_filenames = train_metadata['filename'].values
feature_filenames = np.array([fname.replace(".npy", "") for fname in filenames])

filename_to_idx = {fname: idx for idx, fname in enumerate(feature_filenames)}
indices = [filename_to_idx[fname] for fname in metadata_filenames if fname in filename_to_idx]

# Align features and metadata
X_aligned = X[indices]
metadata_aligned = train_metadata.iloc[[i for i, fname in enumerate(metadata_filenames) if fname in filename_to_idx]]

# --- New column: emotion_label ---
def create_emotion_label(row):
    emotions = []
    if row['mix'] == 0:
        emotions.append(row['emotion_1'])
    else:
        emotions.append(row['emotion_1'])
        emotions.append(row['emotion_2'])
    return '-'.join(sorted(set(emotions)))

metadata_aligned['emotion_label'] = metadata_aligned.apply(create_emotion_label, axis=1)

# --- Filter rows to focus only on happy/sad ---
def emotion_is_only_focus(row):
    if row['mix'] == 0:
        return row['emotion_1'] in focus_emotions
    else:
        return (row['emotion_1'] in focus_emotions and row['emotion_2'] in focus_emotions)

mask = metadata_aligned.apply(emotion_is_only_focus, axis=1)
metadata_aligned = metadata_aligned[mask]
X_aligned = X_aligned[mask.values]

# --- Normalize features ---
video_ids = metadata_aligned['video_id'].values
X_aligned = standardize_by_group(X_aligned, video_ids)

# Assume X_aligned and metadata_aligned are already prepared
# X_aligned: (n_samples, n_features)
# metadata_aligned: DataFrame with 'emotion_label' column

# Run PCA
pca = PCA(n_components=2, whiten=True)
X_pca = pca.fit_transform(X_aligned)

# Unique emotion labels
emotion_labels = metadata_aligned['emotion_label'].unique()

# Define colors and markers for each label
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blue, orange, green
markers = ['o', 's', '^']  # circle, square, triangle

label_to_color = {label: colors[i % len(colors)] for i, label in enumerate(emotion_labels)}
label_to_marker = {label: markers[i % len(markers)] for i, label in enumerate(emotion_labels)}

# Plot
fig, ax = plt.subplots(figsize=(10, 8))

for label in emotion_labels:
    idx = metadata_aligned['emotion_label'] == label
    ax.scatter(
        X_pca[idx, 0],
        X_pca[idx, 1],
        c=label_to_color[label],
        marker=label_to_marker[label],
        label=label,
        edgecolors='k',
        s=80,  # marker size
        alpha=1.0
    )

ax.legend(fontsize=20, loc='best')
ax.grid(True)
ax.tick_params(labelsize=18)
plt.tight_layout()


# Save figure
plt.savefig(save_path, dpi=300)
plt.show()
