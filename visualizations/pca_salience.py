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
    if row['mix'] == 0:  # single emotion (100%)
        return f"{row['emotion_1']}_100"
    elif row['mix'] == 1:  # blended emotions (e.g. 70-30, 50-50)
        sal1 = int(row['emotion_1_salience'])
        sal2 = int(row['emotion_2_salience'])
        return f"{row['emotion_1']}_{sal1}_{row['emotion_2']}_{sal2}"
    else:
        return "unknown"

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

import matplotlib.colors as mcolors

import matplotlib.colors as mcolors

# base colors
hap_color = "#1f77b4"   # blue
sad_color = "#ff7f0e"   # orange
equal_color = "#2ca02c" # green for 50/50

hap_rgb = np.array(mcolors.to_rgb(hap_color))
sad_rgb = np.array(mcolors.to_rgb(sad_color))
equal_rgb = np.array(mcolors.to_rgb(equal_color))

# assign marker shapes by base emotion
emotion_markers = {
    "hap": "o",   # circle
    "sad": "s",   # square
}

# special shape for 50/50 mix
equal_marker = "D"  # diamond

def parse_label_style(label):
    parts = label.split("_")
    if len(parts) == 2:
        # e.g. hap_100
        emotion, sal = parts
        color = hap_rgb if emotion == "hap" else sad_rgb
        marker = emotion_markers.get(emotion, "x")
        return color, marker

    elif len(parts) == 4:
        # e.g. hap_70_sad_30
        emotion1, sal1, emotion2, sal2 = parts
        sal1, sal2 = int(sal1), int(sal2)

        if sal1 == sal2:   # 50/50 blend
            return equal_rgb, equal_marker

        # align hap vs sad saliences
        if emotion1 == "hap":
            hap_sal, sad_sal = sal1, sal2
        else:
            hap_sal, sad_sal = sal2, sal1

        total = hap_sal + sad_sal
        hap_ratio = hap_sal / total if total > 0 else 0.5

        # interpolate color
        color = hap_rgb * hap_ratio + sad_rgb * (1 - hap_ratio)

        # marker = dominant emotion
        dominant_emotion = emotion1 if sal1 > sal2 else emotion2
        marker = emotion_markers.get(dominant_emotion, "x")

        return color, marker

    else:
        return "gray", "x"

# map label → color/marker
label_to_color = {label: parse_label_style(label)[0] for label in emotion_labels}
label_to_marker = {label: parse_label_style(label)[1] for label in emotion_labels}





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
