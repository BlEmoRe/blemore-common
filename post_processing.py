import numpy as np
from collections import Counter

from config import INDEX_TO_LABEL
from utils.generic_accuracy.accuracy_funcs import acc_presence_total, acc_salience_total
from visualizations import plot_grid_heatmap, summarize_prediction_distribution


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
    y_pred[y_pred < presence_threshold] = 0  # mask low confidence

    result = {}

    for fname, vec in zip(filenames, y_pred):
        nonzero = np.where(vec > 0)[0]

        if len(nonzero) == 0:
            preds = [{"emotion": "neu", "salience": 100.0}]
            result[fname] = preds
            continue

        if len(nonzero) == 1:
            preds = [{"emotion": INDEX_TO_LABEL[nonzero[0]], "salience": 100.0}]
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


def post_process(filenames, preds, presence_weight=0.5):
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

    plot_grid_heatmap(grid, metric_index=2, title="Presence", cmap="viridis")
    plot_grid_heatmap(grid, metric_index=3, title="Salience", cmap="viridis")

    print("Best presence ", max(grid, key=lambda x: x[2]))
    print("Best salience ", max(grid, key=lambda x: x[3]))

    # df = pd.DataFrame(grid, columns=["alpha", "beta", "Presence", "Salience"])

    sorted_grid = sorted(
        grid,
        key=lambda x: presence_weight * x[2] + (1 - presence_weight) * x[3],
        reverse=True
    )

    print(f"With score Best alpha: {sorted_grid[0][0]}, Best beta: {sorted_grid[0][1]}, Presence: {sorted_grid[0][2]}, Salience: {sorted_grid[0][3]}")

    best_alpha = sorted_grid[0][0]
    best_beta = sorted_grid[0][1]
    best_acc_presence = sorted_grid[0][2]
    best_acc_salience = sorted_grid[0][3]

    # After best_alpha, best_beta have been found
    final_preds = probs2dict(preds, filenames, best_alpha, best_beta)
    summarize_prediction_distribution(final_preds)

    return best_alpha, best_beta, best_acc_presence, best_acc_salience




def main():

    # Example usage
    y_pred = np.array([[0.8, 0.1, 0.05, 0.03, 0.02],
                       [0.2, 0.7, 0.05, 0.03, 0.02],
                       [0.1, 0.1, 0.7, 0.05, 0.05],
                      [0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.9, 0.0, 0.0, 0.0, 0.0],
                       ])  # Example predictions

    y_pred = get_top_2_predictions(y_pred)

    filenames = ["file1", "file2", "file3", "file4", "file5"]

    preds = probs2dict(y_pred, filenames, presence_threshold=0.1, salience_threshold=0.9)
    for i in preds.items():
        print(i)


if __name__ == "__main__":
    main()
