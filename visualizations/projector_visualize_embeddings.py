import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorboard.plugins import projector

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
}

encoding = "videomae"  # <-- Choose your encoder here

# Focus emotions
focus_emotions = {'hap', 'sad'}  # Only keep these emotions

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

print(f"Filtered and aligned features shape: {X_aligned.shape}")
print(f"Filtered and aligned metadata shape: {metadata_aligned.shape}")

# --- Prepare logs for TensorBoard Projector ---
log_dir = "projector_logs_{}_hap_sad".format(encoding)
os.makedirs(log_dir, exist_ok=True)

metadata_filename = "metadata.tsv"
tensor_filename = "tensor.tsv"

# Save metadata
metadata_df = pd.DataFrame({
    'emotion_label': metadata_aligned['emotion_label'],
    'filename': metadata_aligned['filename'],
    'emotion_1': metadata_aligned['emotion_1'],
    'emotion_2': metadata_aligned['emotion_2'],
    'emotion_1_salience': metadata_aligned['emotion_1_salience'],
    'emotion_2_salience': metadata_aligned['emotion_2_salience'],
    'mix': metadata_aligned['mix'],
    'video_id': metadata_aligned['video_id'],
    'fold': metadata_aligned['fold'],
    'intensity_level': metadata_aligned['intensity_level']
})

metadata_df.to_csv(os.path.join(log_dir, metadata_filename), sep='\t', index=False)

# Save embeddings
pd.DataFrame(X_aligned).to_csv(os.path.join(log_dir, tensor_filename), sep='\t', index=False, header=False)

# --- TensorBoard Projector Config ---
writer = tf.summary.create_file_writer(log_dir)
writer.close()

config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.metadata_path = metadata_filename
embedding.tensor_path = tensor_filename
projector.visualize_embeddings(log_dir, config)

# Write config to event file
with tf.summary.create_file_writer(log_dir).as_default():
    projector.visualize_embeddings(log_dir, config)

print(f"✅ Projector setup complete at {log_dir}")
