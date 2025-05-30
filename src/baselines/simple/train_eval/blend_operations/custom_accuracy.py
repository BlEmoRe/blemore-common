import numpy as np

from src.baselines.simple.train_eval.blend_operations.blend_utils import convert_probs_to_salience

"""
Accuracy functions adapted for salience vector labels.
"""


def custom_acc_presence(y_true, y_pred):
    """
    Computes presence accuracy by checking if the set of active labels (non-zero values)
    in prediction matches the ground truth exactly.

    Parameters:
    - y_true: np.ndarray (shape: [n_samples, n_classes]), ground truth labels (non-zero = present).
    - y_pred: np.ndarray (shape: [n_samples, n_classes]), predicted labels (non-zero = predicted).

    Returns:
    - acc_presence: float, proportion of exact matches.
    - true_preds: np.ndarray, boolean array of per-sample correctness.
    """
    # Binarize: convert non-zero values to 1
    y_true_bin = (y_true != 0).astype(int)
    y_pred_bin = (y_pred != 0).astype(int)

    # Compare row-wise
    true_preds = np.all(y_true_bin == y_pred_bin, axis=1)
    acc_presence = np.mean(true_preds)

    return acc_presence, true_preds


def custom_acc_salience(y_true, y_pred, threshold=0.1):
    """
    Compute salience accuracy by comparing canonical predictions to true salience vectors.

    Parameters:
    - y_true (np.ndarray): Ground truth salience vectors.
    - y_pred (np.ndarray): Predicted salience vectors.
    - threshold (float): Tolerance for 0.5/0.5 mapping.

    Returns:
    - accuracy (float), correct (list of bools)
    """
    y_pred_mapped = convert_probs_to_salience(y_pred, threshold)

    correct = []
    for t, p in zip(y_true, y_pred_mapped):
        pred_nz = np.count_nonzero(p)
        if pred_nz != 2:
            correct.append(False)
        else:
            correct.append(np.array_equal(t, p))

    return np.mean(correct), correct


def main():
    labels = np.array([
        [0.5, 0., 0., 0., 0.5],
        [0., 0., 0., 0.3, 0.7],
        [0., 0.7, 0.3, 0., 0.],
        [0., 0., 0., 0.3, 0.7],
        [0., 0.5, 0., 0., 0.5],
        [0.0, 0.5, 0.0, 0.0, 0.5]
    ])

    preds = np.array([
        [0.4, 0., 0., 0., 0.4],
        [0., 0., 0., .2, 0.2],
        [0., 0.51, 0.4, 0., 0.],
        [0., 0., 0., 0.1, 0.9],
        [0., 0.4, 0., 0., 0.6],
        [0.0, 0.41860163, 0.0, 0.0, 0.448838]
    ])

    # preds_mapped = map_vector_pairwise(preds, threshold=0.1)
    # print(preds_mapped)

    # acc, idx = compute_acc_salience(labels, preds, distance_tolerance=0.0001)
    # print(acc)
    # print(idx)
    #

if __name__ == "__main__":
    main()