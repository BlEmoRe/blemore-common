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


# def convert_preds_to_standard(y_pred, index2emotion):
#     n_samples = y_pred.shape[0]
#
#     for i in range(n_samples):
#         pred_indices = np.nonzero(y_pred[i])[0]
#         emotion_1 = index2emotion[pred_indices[0]]
#         emotion_2 = index2emotion[pred_indices[1]]
#
#         emotion_1_raw_salience = y_pred[i][pred_indices[0]]
#         emotion_2_raw_salience = y_pred[i][pred_indices[1]]
#
#         if emotion_1_raw_salience - emotion_2_raw_salience:
#
#
#         elif emotion_1_salience - emotion_2_salience:
#             y_pred[i] = np.zeros_like(y_pred[i])
#             y_pred[i][pred_indices[1]] = 1
#         else:
#             y_pred[i] = np.zeros_like(y_pred[i])
#             y_pred[i][pred_indices[0]] = 0.5
#             y_pred[i][pred_indices[1]] = 0.5
#
