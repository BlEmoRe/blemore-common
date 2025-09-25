import numpy as np
from collections import Counter

import torch

from config import INDEX_TO_LABEL, NEUTRAL_INDEX
from utils.generic_accuracy.accuracy_funcs import acc_presence_total, acc_salience_total
from visualizations.epoch_results import plot_grid_heatmap, summarize_prediction_distribution


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
               presence_threshold=0.1,
               salience_threshold=0.1):
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
    result = {}

    for fname, vec in zip(filenames, y_pred):
        pred_top_index = np.argmax(vec)  # Get the index of the highest probability

        if vec[pred_top_index] < presence_threshold:
            preds = [{"emotion": INDEX_TO_LABEL[pred_top_index], "salience": 100.0}]
            result[fname] = preds
            continue

        vec[vec < presence_threshold] = 0  # mask low confidence
        nonzero = np.where(vec > 0)[0]

        if len(nonzero) == 1:
            preds = [{"emotion": INDEX_TO_LABEL[nonzero[0]], "salience": 100.0}]
            result[fname] = preds
            continue

        if NEUTRAL_INDEX in nonzero:
            if vec[nonzero[0]] >= vec[NEUTRAL_INDEX]:
                # If neutral is present but has lower salience than the first emotion
                preds = [{"emotion": INDEX_TO_LABEL[nonzero[0]], "salience": 100.0}]
            else:
                # If neutral is present and has higher salience than the first emotion
                preds = [{"emotion": INDEX_TO_LABEL[NEUTRAL_INDEX], "salience": 100.0}]
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


def grid_search_thresholds(filenames, preds, presence_weight=0.5, debug_plots=False):
    """
    Perform grid search over presence and salience thresholds to maximize weighted accuracy.
    :param filenames: list of filenames
    :param preds: predicted probabilities with rows corresponding to filenames
    :param presence_weight: optimization weight for presence accuracy vs salience accuracy
    :param debug_plots: plot heatmaps and prediction distributions if True
    :return: statistics dictionary with the best thresholds and accuracies

    alpha: best presence threshold (include class if prob >= alpha)
    beta: best salience threshold (difference to consider equal salience)

    """
    preds = get_top_2_predictions(preds)
    grid = []

    alpha = np.linspace(0.05, 0.95, 20)
    beta = np.linspace(0.05, 0.95, 20)

    for a in alpha:
        for b in beta:
            label_dict = probs2dict(preds, filenames, a, b)
            acc_presence = acc_presence_total(label_dict)
            acc_salience = acc_salience_total(label_dict)
            grid.append((a.item(), b.item(), acc_presence, acc_salience))

    if debug_plots:
        plot_grid_heatmap(grid, metric_index=2, title="Presence", cmap="viridis")
        plot_grid_heatmap(grid, metric_index=3, title="Salience", cmap="viridis")

    best_presence_only = max(grid, key=lambda x: x[2])[2]
    best_salience_only = max(grid, key=lambda x: x[3])[3]

    sorted_grid = sorted(
        grid,
        key=lambda x: presence_weight * x[2] + (1 - presence_weight) * x[3],
        reverse=True
    )

    best_alpha = sorted_grid[0][0]
    best_beta = sorted_grid[0][1]
    best_acc_presence = sorted_grid[0][2]
    best_acc_salience = sorted_grid[0][3]

    # After best_alpha, best_beta have been found
    final_preds = probs2dict(preds, filenames, best_alpha, best_beta)

    if debug_plots:
        summarize_prediction_distribution(final_preds)

    return {
        "alpha": best_alpha,
        "beta": best_beta,
        "acc_presence": best_acc_presence,
        "acc_salience": best_acc_salience,
        "presence_only": best_presence_only,
        "salience_only": best_salience_only
    }


def main():
    # Example usage
    # Simulated predicted probabilities for 5 samples and 6 classes
    # (In practice, replace this with your model's output)

    # assuming the following class order:
    # LABEL_TO_INDEX = {
    #     "ang": 0,
    #     "disg": 1,
    #     "fea": 2,
    #     "hap": 3,
    #     "sad": 4,
    #     "neu": 5
    # }

    y_pred = np.array([[0.7, 0.1, 0.05, 0.03, 0.02, 0.1],
                       [0.2, 0.6, 0.05, 0.03, 0.02, 0.1],
                       [0.1, 0.1, 0.6, 0.05, 0.05, 0.1],
                       [0.0, 0.0, 0.0, 0.0, 0.4, 0.6],
                       [0.9, 0.0, 0.0, 0.0, 0.0, 0.1],
                       ])  # Example predictions

    filenames = ['A405_mix_ang_disg_70_30_ver1',
                 'A334_mix_disg_fea_50_50_ver1',
                 'A427_mix_fea_sad_50_50_ver1',
                 'A303_neu_sit2_ver1',
                 'A411_ang_int1_ver1']

    stats = grid_search_thresholds(filenames, y_pred)

    print(stats)
    # Example output (actual values will depend on y_pred):
    # {'alpha': 0.05, 'beta': 0.05, 'acc_presence': 0.6, 'acc_salience': 0.3333333333333333, 'presence_only': 0.6, 'salience_only': 0.3333333333333333}
    # acc_presence and acc_salience represent the best accuracies found
    # at the optimal thresholds alpha and beta
    # presence_only and salience_only represent the best possible accuracies
    # if we only optimized for presence or salience respectively



    # Apply the best thresholds to get final predictions
    y_pred = get_top_2_predictions(y_pred)

    preds = probs2dict(y_pred, filenames, presence_threshold=stats["alpha"], salience_threshold=stats["beta"])
    for i in preds.items():
        print(i)


if __name__ == "__main__":
    main()
