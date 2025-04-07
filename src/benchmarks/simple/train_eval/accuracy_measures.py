import numpy as np

def compute_acc_presence(y_true, y_pred):
    """
    Computes strict accuracy by checking if the set of active labels (non-zero values)
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


def compute_distance(salience_scalars):
    """Computes the distance between two scalar values."""
    return salience_scalars[0] - salience_scalars[1]

def is_prediction_correct(true_proportions, pred_proportions, distance_tolerance=0.1):
    true_distance = compute_distance(true_proportions)
    pred_distance = compute_distance(pred_proportions)

    if true_distance > 0:
        return pred_distance > distance_tolerance
    elif true_distance == 0:
        return abs(pred_distance) < distance_tolerance
    elif true_distance < 0:
        return pred_distance < -distance_tolerance
    else:
        raise ValueError("Invalid true distance")


def compute_acc_salience(y_true, y_pred, distance_tolerance=0.1):
    num_samples = y_true.shape[0]

    correct = []

    for i in range(num_samples):
        true_indices = np.nonzero(y_true[i])[0]
        pred_indices = np.nonzero(y_pred[i])[0]

        if not np.array_equal(true_indices, pred_indices):
            correct.append(False)
            continue

        true_salience = y_true[i][true_indices]
        pred_salience = y_pred[i][true_indices]

        ret = is_prediction_correct(true_salience, pred_salience, distance_tolerance=distance_tolerance)

        correct.append(ret)

    accuracy = sum(correct) / len(correct) if len(correct) > 0 else 0
    return accuracy, correct



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

    # acc, idx = compute_acc_salience(labels, preds, distance_tolerance=0.0001)
    # print(acc)
    # print(idx)
    #

if __name__ == "__main__":
    main()