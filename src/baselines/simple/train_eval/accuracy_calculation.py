import pandas as pd
import numpy as np

from src.baselines.simple.train_eval.blend_operations.blend_utils import convert_probs_to_salience
from src.tools.generic_accuracy.accuracy_funcs import acc_presence_total, acc_salience_total


def get_filename_from_indices(indices, df):
    df_val = df.iloc[indices]
    return df_val["filename"].values


def generic_accuracy(salience_predictions, prediction_as_filenames, index2emotion):
    pred_dict = convert_labels_to_dictionary(prediction_as_filenames, salience_predictions, index2emotion)

    presence = acc_presence_total(pred_dict)
    salience = acc_salience_total(pred_dict)

    print(f"Presence Accuracy: {presence:.2f}")
    print(f"Salience Accuracy: {salience:.2f}")

    return presence, salience


def convert_labels_to_dictionary(filenames, salience_predictions, index2emotion):
    """
    Converts a 2D numpy array of predictions to a list of dictionaries.

    :param filenames: List of filenames
    :param y_pred: 2D numpy array of shape (n_samples, n_labels).
    :param index2emotion: Mapping from indices to emotion labels.
    :return: List of dictionaries with 'emotion' and 'salience' keys.
    """
    ret = {}

    for i in range(salience_predictions.shape[0]):
        filename = filenames[i]

        prediction = []
        for j in range(salience_predictions.shape[1]):
            if salience_predictions[i][j] > 0:
                prediction.append({"emotion": index2emotion[j], "salience": np.round(salience_predictions[i][j], 1).item() * 100})

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

    y_pred_dict = convert_labels_to_dictionary(filenames, y_pred, index2emotion)
    print(y_pred_dict)


if __name__ == "__main__":
    main()