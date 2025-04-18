import numpy as np

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
    y_pred_mapped = map_vector_pairwise(y_pred, threshold)

    correct = []
    for t, p in zip(y_true, y_pred_mapped):
        pred_nz = np.count_nonzero(p)
        if pred_nz != 2:
            correct.append(False)
        else:
            correct.append(np.array_equal(t, p))

    return np.mean(correct), correct


def map_vector_pairwise(y_pred, threshold=0.1):
    """
    Map predicted salience vectors to canonical forms (0.5/0.5 or 0.7/0.3), in NumPy format.

    Parameters:
    - y_pred (np.ndarray): Array of shape (n_samples, n_classes)
    - threshold (float): Max allowed difference for neutral (0.5/0.5) mapping.

    Returns:
    - np.ndarray: Canonical salience predictions, same shape as y_pred.
    """
    y_pred = np.copy(y_pred)  # avoid modifying in-place
    mapped = []

    for vec in y_pred:
        non_zero_indices = np.where(vec > 0)[0]

        if len(non_zero_indices) != 2:
            mapped.append(vec)
            continue

        i, j = non_zero_indices
        v1, v2 = vec[i], vec[j]

        new_vec = np.zeros_like(vec)

        if abs(v1 - v2) <= threshold:
            new_vec[i], new_vec[j] = 0.5, 0.5
        elif v1 > v2:
            new_vec[i], new_vec[j] = 0.7, 0.3
        else:
            new_vec[i], new_vec[j] = 0.3, 0.7

        mapped.append(new_vec)

    return np.stack(mapped)


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

    preds_mapped = map_vector_pairwise(preds, threshold=0.1)
    print(preds_mapped)

    # acc, idx = compute_acc_salience(labels, preds, distance_tolerance=0.0001)
    # print(acc)
    # print(idx)
    #

if __name__ == "__main__":
    main()