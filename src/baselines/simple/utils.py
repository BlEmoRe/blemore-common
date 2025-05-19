import numpy as np

LABEL_TO_INDEX = {
    "ang": 0,
    "disg": 1,
    "fea": 2,
    "hap": 3,
    "sad": 4
}
INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}


def get_top_2_predictions(y_pred):
    # Step 2: Get sorting indices (descending order)
    sorted_indices = np.argsort(-y_pred, axis=1)  # Sort each row in descending order

    # Step 3: Select only the top `max_positive_labels` indices
    top_k_indices = sorted_indices[:, :2]  # Keep only the top `max_positive_labels` per row

    # Step 4: Create a mask to enforce at most `max_positive_labels` per row
    mask = np.zeros_like(y_pred, dtype=bool)  # Initialize mask with all False
    np.put_along_axis(mask, top_k_indices, True, axis=1)  # Set True only for top values

    # Step 5: Apply mask
    ret = y_pred * mask

    return ret

def probs2dict(y_pred,
               filenames,
               salience_threshold=0.1,
               presence_threshold=0.1):
    """
    Convert predicted probability vectors to a filename → prediction dictionary
    with canonical salience values and thresholding.

    Args:
        y_pred (np.ndarray): [N, C] array of predicted class probabilities
        filenames (List[str]): List of corresponding filenames
        index2emotion (Dict[int, str]): Mapping from class index to emotion label
        salience_threshold (float): Max difference to consider equal salience (50/50)
        presence_threshold (float): Min prob to include a class at all

    Returns:
        Dict[str, List[Dict[str, float]]]: Formatted predictions per file
    """
    y_pred = np.copy(y_pred)
    y_pred[y_pred < presence_threshold] = 0  # mask low confidence

    result = {}

    for fname, vec in zip(filenames, y_pred):
        nonzero = np.where(vec > 0)[0]

        if len(nonzero) != 2:
            # Return original (unnormalized) values if not a blend
            preds = [
                {"emotion": INDEX_TO_LABEL[i], "salience": round(100 * vec[i], 1)}
                for i in nonzero
            ]
            result[fname] = preds
            continue

        i, j = nonzero
        p1, p2 = vec[i], vec[j]

        if abs(p1 - p2) <= salience_threshold:
            sal1, sal2 = 0.5, 0.5
        elif p1 > p2:
            sal1, sal2 = 0.7, 0.3
        else:
            sal1, sal2 = 0.3, 0.7

        result[fname] = [
            {"emotion": INDEX_TO_LABEL[i], "salience": round(100 * sal1, 1)},
            {"emotion": INDEX_TO_LABEL[j], "salience": round(100 * sal2, 1)}
        ]

    return result