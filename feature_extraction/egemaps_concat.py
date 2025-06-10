import os
import glob
import pandas as pd
import numpy as np

output_path = "/home/tim/Work/quantum/data/blemore/encoded_videos/static_data/egemaps_static_features.npz"

opensmile_train_path = "/media/tim/Seagate Hub/mixed_emotion_challenge/opensmile_files/train"
opensmile_test_path = "/media/tim/Seagate Hub/mixed_emotion_challenge/opensmile_files/test"

all_paths = [opensmile_train_path, opensmile_test_path]
csv_files = []

for path in all_paths:
    csv_files.extend(glob.glob(os.path.join(path, '**', '*.csv'), recursive=True))

X_list = []
filenames = []

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    df = df.drop(columns=['file', 'start', 'end'], errors='ignore')
    X_list.append(df.values[0])
    filenames.append(os.path.splitext(os.path.basename(csv_file))[0])

X = np.stack(X_list)

np.savez(output_path, X=X, filenames=filenames)