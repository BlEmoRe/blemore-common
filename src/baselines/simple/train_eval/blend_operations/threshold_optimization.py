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
                                    plot=True):

    # Store accuracy scores
    accuracies = []

    for threshold in positive_thresholds:
        binary_preds = (y_pred >= threshold).astype(int)

        # Compute accuracy
        acc_presence, _ = custom_acc_presence(y_true, binary_preds)

        salience_threshold = find_optimal_salience_threshold(y_true, binary_preds)
        # Compute accuracy
        acc_salience, _ = custom_acc_salience(y_true, binary_preds, salience_threshold)


        # Store results
        accuracies.append(acc_presence)

    # Find optimal threshold (based on highest AUC score)
    best_threshold_idx = np.argmax(accuracies)
    best_threshold = positive_thresholds[best_threshold_idx]

    if plot:
        plot_accuracy_vs_threshold(positive_thresholds, accuracies, best_threshold)

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


def plot_accuracy_vs_threshold(positive_thresholds, accuracies, best_threshold):
    plt.figure(figsize=(8, 6))
    plt.plot(positive_thresholds, accuracies, label="Acc Presence", marker="o")

    plt.axvline(best_threshold, color="red", linestyle="dashed", label=f"Optimal Threshold: {best_threshold:.2f}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Accuracy vs. Probability Threshold")
    plt.legend()
    plt.grid(True)
    plt.show()