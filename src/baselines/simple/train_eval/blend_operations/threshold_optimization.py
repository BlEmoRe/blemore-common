import numpy as np
import matplotlib.pyplot as plt

from src.baselines.simple.train_eval.blend_operations.custom_accuracy import custom_acc_salience, custom_acc_presence


def find_optimal_salience_threshold(y_true, y_pred, thresholds=np.linspace(0.00001, 0.4, 10), plot=True):
    accuracies = []

    for threshold in thresholds:
        acc, _ = custom_acc_salience(y_true, y_pred, threshold)
        accuracies.append(acc)

    best_idx = np.argmax(accuracies)
    best_tol = thresholds[best_idx]

    if plot:
        plot_salience_accuracy_vs_threshold(thresholds, accuracies, best_tol)

    return best_tol


def find_optimal_presence_threshold(y_true,
                                    y_pred,
                                    positive_thresholds=np.linspace(0.05, 0.95, 19),
                                    plot=True,
                                    presence_weight=0.3):
    """
    presence_weight: float between 0 and 1
        - 1.0 = only presence matters
        - 0.0 = only salience matters
        - 0.5 = equal weighting
    """

    accuracies_presence = []
    accuracies_salience = []
    combined_scores = []

    for threshold in positive_thresholds:
        binary_preds = (y_pred >= threshold).astype(int)

        # Compute accuracies
        acc_presence, _ = custom_acc_presence(y_true, binary_preds)

        y_pred_copy = y_pred.copy()
        y_pred_copy[y_pred_copy < threshold] = 0
        salience_threshold = find_optimal_salience_threshold(y_true, y_pred_copy, plot=False)
        acc_salience, _ = custom_acc_salience(y_true, y_pred_copy, salience_threshold)

        print(f"Presence Threshold: {threshold:.2f}, Presence Accuracy: {acc_presence:.2f}, Salience Accuracy: {acc_salience:.2f}")

        accuracies_presence.append(acc_presence)
        accuracies_salience.append(acc_salience)

        # Combine accuracies
        combined_score = presence_weight * acc_presence + (1 - presence_weight) * acc_salience
        combined_scores.append(combined_score)

    # Find best threshold based on combined score
    best_threshold_idx = np.argmax(combined_scores)
    best_threshold = positive_thresholds[best_threshold_idx]

    if plot:
        plot_accuracy_vs_threshold(positive_thresholds, accuracies_presence, accuracies_salience, combined_scores, best_threshold)

    return best_threshold


def plot_salience_accuracy_vs_threshold(tolerances, accuracies, best_tol):
    plt.figure(figsize=(8, 6))
    plt.plot(tolerances, accuracies, label="Acc Salience", marker="o")
    plt.axvline(best_tol, color="red", linestyle="dashed", label=f"Optimal Tolerance: {best_tol:.5f}")
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.title("Salience Accuracy vs. Salience Diff Threshold")
    plt.legend()
    plt.grid(True)
    plt.show()


# def plot_accuracy_vs_threshold(positive_thresholds, accuracies, best_threshold):
#     plt.figure(figsize=(8, 6))
#     plt.plot(positive_thresholds, accuracies, label="Acc Presence", marker="o")
#
#     plt.axvline(best_threshold, color="red", linestyle="dashed", label=f"Optimal Threshold: {best_threshold:.2f}")
#     plt.xlabel("Threshold")
#     plt.ylabel("Score")
#     plt.title("Accuracy vs. Probability Threshold")
#     plt.legend()
#     plt.grid(True)
#     plt.show()

def plot_accuracy_vs_threshold(thresholds, acc_presence, acc_salience, combined_scores, best_threshold):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, acc_presence, label="Presence Accuracy", marker='o')
    plt.plot(thresholds, acc_salience, label="Salience Accuracy", marker='s')
    plt.plot(thresholds, combined_scores, label="Combined Score", marker='x', linestyle='--')
    plt.axvline(best_threshold, color='red', linestyle='--', label=f"Best Threshold: {best_threshold:.2f}")

    plt.xlabel("Presence Threshold")
    plt.ylabel("Accuracy / Combined Score")
    plt.title("Accuracy vs Threshold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()