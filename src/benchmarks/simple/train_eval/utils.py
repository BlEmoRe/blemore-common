import numpy as np


def get_top_k_predictions(y_pred_matrix, k=2):
    """
    Applies a probability threshold and ensures at most `max_positive_labels` predictions per row.

    :param k: maximum number of positive labels per row.
    :param y_pred_matrix: np.array of shape (n_samples, n_labels) with predicted probabilities.
    :return: np.array of shape (n_samples, n_labels) with binary predictions.
    """
    # Step 2: Get sorting indices (descending order)
    sorted_indices = np.argsort(-y_pred_matrix, axis=1)  # Sort each row in descending order

    # Step 3: Select only the top `max_positive_labels` indices
    top_k_indices = sorted_indices[:, :k]  # Keep only the top `max_positive_labels` per row

    # Step 4: Create a mask to enforce at most `max_positive_labels` per row
    mask = np.zeros_like(y_pred_matrix, dtype=bool)  # Initialize mask with all False
    np.put_along_axis(mask, top_k_indices, True, axis=1)  # Set True only for top values

    # Step 5: Apply mask
    ret = y_pred_matrix * mask

    return ret


def get_blend_indices(y: np.ndarray) -> np.ndarray:
    """Returns indices of samples with more than one non-zero label."""
    return np.where((y > 0).sum(axis=1) > 1)[0]


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
                prediction.append({"emotion": index2emotion[j], "salience": y_pred[i][j].item()})

        if len(prediction) == 0:
            prediction.append({"emotion": "neu", "salience": 1.0})

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