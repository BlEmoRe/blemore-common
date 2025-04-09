import os
from config import ROOT_DIR

# Input files (modify with your owns paths to replicate the results)
RAW_OPENFACE_FILES_PATH = "/media/tim/Seagate Hub/mixed_emotion_challenge/openface_files/train"
METADATA_PATH = os.path.join(ROOT_DIR, "data/train_metadata.csv")

# Internal paths (used within the baseline code to save/load files)
AGGREGATED_OPENFACE_PATH = os.path.join(ROOT_DIR, "data/baselines/simple/agg_openface_data.csv")
LABEL_MAPPING_PATH = os.path.join(ROOT_DIR, "data/baselines/simple/emotion_label_mapping_probabilistic.json")
VECTOR_TRAINING_SET_PATH = os.path.join(ROOT_DIR, "data/baselines/simple/train_data_probabilistic.npz")


AU_INTENSITY_COLS = ['AU01_r',
                     'AU02_r',
                     'AU04_r',
                     'AU05_r',
                     'AU06_r',
                     'AU07_r',
                     'AU09_r',
                     'AU10_r',
                     'AU12_r',
                     'AU14_r',
                     'AU15_r',
                     'AU17_r',
                     'AU20_r',
                     'AU23_r',
                     'AU25_r',
                     'AU26_r',
                     'AU45_r']

POSE_COLS = [
    "pose_Rx",
    "pose_Ry",
    "pose_Rz",
    "pose_Tx",
    "pose_Ty",
    "pose_Tz"
]

GAZE_COLS = [
    'gaze_0_x',
    'gaze_0_y',
    'gaze_0_z',
    'gaze_1_x',
    'gaze_1_y',
    'gaze_1_z',
    'gaze_angle_x',
    'gaze_angle_y'
]

SUCCESS_COLS = ["success", "confidence"]

feature_columns = AU_INTENSITY_COLS + POSE_COLS + GAZE_COLS

openface_columns = SUCCESS_COLS + feature_columns