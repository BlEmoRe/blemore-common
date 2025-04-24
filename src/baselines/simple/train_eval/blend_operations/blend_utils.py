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


def convert_probs_to_salience(y_pred, threshold=0.1):
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