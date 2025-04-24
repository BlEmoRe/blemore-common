import pandas as pd
import numpy as np

from src.baselines.simple.config_simple_baseline import AGGREGATED_OPENFACE_PATH
from src.baselines.simple.train_eval.blend_operations.blend_utils import convert_probs_to_salience
from src.tools.generic_accuracy.accuracy_funcs import acc_presence_total, acc_salience_total


def get_filename_from_indices(indices):
    df = pd.read_csv(AGGREGATED_OPENFACE_PATH)
    df_val = df.iloc[indices]
    return df_val["filename"].values


def generic_accuracy(preds, salience_threshold, indices, index2emotion):
    filenames = get_filename_from_indices(indices)

    preds_mapped = convert_probs_to_salience(preds, salience_threshold)
    pred_dict = label_vector2dict(filenames, preds_mapped, index2emotion)

    presence = acc_presence_total(pred_dict, AGGREGATED_OPENFACE_PATH)
    salience = acc_salience_total(pred_dict, AGGREGATED_OPENFACE_PATH)

    print(f"Presence Accuracy: {presence:.2f}")
    print(f"Salience Accuracy: {salience:.2f}")


def label_vector2dict(filenames, y_pred, index2emotion):
    """
    Converts a 2D numpy array of predictions to a list of dictionaries.

    :param filenames: List of filenames
    :param y_pred: 2D numpy array of shape (n_samples, n_labels).
    :param index2emotion: Mapping from indices to emotion labels.
    :return: List of dictionaries with 'emotion' and 'salience' keys.
    """
    ret = {}

    for i in range(y_pred.shape[0]):
        filename = filenames[i]

        prediction = []
        for j in range(y_pred.shape[1]):
            if y_pred[i][j] > 0:
                prediction.append({"emotion": index2emotion[j], "salience": np.round(y_pred[i][j], 1).item() * 100})

        if len(prediction) == 0:
            prediction.append({"emotion": "neu", "salience": 100})

        ret[filename] = prediction
    return ret


def main():
    # Example usage
    index2emotion = {
        0: "ang",
        1: "disg",
        2: "fea",
        3: "hap",
        4: "sad"    }

    y_pred = np.array([
        [0.5, 0., 0., 0., 0.5],
        [0., 0., 0., 0.3, 0.7],
        [1, 0., 0., 0., 0.],
        [0, 0., 0., 0., 0.]
    ])

    filenames = [
        "A102_ang_int1_ver1",
        "A102_ang_int2_ver1",
        "A102_disg_int1_ver1",
        "A102_neu_sit2_ver1"
    ]

    y_pred_dict = label_vector2dict(filenames, y_pred, index2emotion)
    print(y_pred_dict)


if __name__ == "__main__":
    main()