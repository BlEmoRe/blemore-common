import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import itertools

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

all_emotions = {'ang', 'disg', 'fea', 'hap', 'sad'}

# Generate all 2-emotion combos (change r=2 if you want 3-emotion combos too)
emotion_combos = list(itertools.combinations(all_emotions, 2))


# --- Function to create PCA plot for one encoder + combo ---
def plot_pca_for_combo(encoding, focus_emotions, X, metadata, save_path):
    # Create new column for emotion label
    def create_emotion_label(row):
        emotions = [row['emotion_1']]
        if row['mix'] != 0:
            emotions.append(row['emotion_2'])
        return '-'.join(sorted(set(emotions)))

    metadata['emotion_label'] = metadata.apply(create_emotion_label, axis=1)

    # Keep only samples with the focus emotions
    def emotion_is_only_focus(row):
        if row['mix'] == 0:
            return row['emotion_1'] in focus_emotions
        else:
            return (row['emotion_1'] in focus_emotions and
                    row['emotion_2'] in focus_emotions)

    mask = metadata.apply(emotion_is_only_focus, axis=1)
    metadata_focused = metadata[mask]
    X_focused = X[mask.values]

    if len(metadata_focused) == 0:
        print(f"⚠️ No data for combo {focus_emotions} in {encoding}")
        return

    # Normalize by group
    video_ids = metadata_focused['video_id'].values
    X_focused = standardize_by_group(X_focused, video_ids)

    # PCA
    X_pca = PCA(n_components=2, whiten=True).fit_transform(X_focused)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = metadata_focused['emotion_label'].unique()

    # Define colors and markers for each label
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blue, orange, green
    markers = ['o', 's', '^']  # circle, square, triangle

    label_to_color = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
    label_to_marker = {label: markers[i % len(markers)] for i, label in enumerate(unique_labels)}

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    for label in unique_labels:
        idx = metadata_focused['emotion_label'] == label
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

    # cmap = plt.cm.get_cmap('tab10', len(unique_labels))
    #
    # for i, label in enumerate(unique_labels):
    #     idx = metadata_focused['emotion_label'] == label
    #     ax.scatter(X_pca[idx, 0], X_pca[idx, 1],
    #                c=[cmap(i)], label=label, s=80, edgecolors='k')
    #
    # ax.legend(fontsize=14, loc='best')
    # ax.set_title(f"{encoding.upper()} – {', '.join(focus_emotions)}")
    # ax.grid(True)
    # plt.tight_layout()
    # plt.savefig(save_path, dpi=300)
    # plt.close()


# --- Main loop: iterate encoders × emotion combos ---
train_metadata = pd.read_csv(train_metadata_path)

for encoding, path in encoding_paths.items():
    print(f"Processing encoder: {encoding}")
    enc_data = np.load(path)
    X = enc_data['X']
    filenames = enc_data['filenames']

    # Align features and metadata
    metadata_filenames = train_metadata['filename'].values
    feature_filenames = np.array([fname.replace(".npy", "") for fname in filenames])
    filename_to_idx = {fname: idx for idx, fname in enumerate(feature_filenames)}
    indices = [filename_to_idx[fname] for fname in metadata_filenames if fname in filename_to_idx]

    X_aligned = X[indices]
    metadata_aligned = train_metadata.iloc[[i for i, fname in enumerate(metadata_filenames) if fname in filename_to_idx]]

    for combo in emotion_combos:
        focus_emotions = set(combo)
        save_dir = os.path.join(ROOT_DIR, "data/plots/pca", encoding)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(
            save_dir,
            f"pca_{encoding}_{'_'.join(sorted(focus_emotions))}.png"
        )
        plot_pca_for_combo(encoding, focus_emotions, X_aligned, metadata_aligned.copy(), save_path)
