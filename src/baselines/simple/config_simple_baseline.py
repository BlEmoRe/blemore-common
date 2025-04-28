
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