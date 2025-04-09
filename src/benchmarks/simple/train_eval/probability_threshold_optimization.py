import numpy as np
import matplotlib.pyplot as plt

from src.benchmarks.simple.train_eval.custom_accuracy import custom_acc_presence, custom_acc_salience


def find_optimal_salience_discriminator(y_true, y_pred, tolerances=np.linspace(0.00001, 0.4, 100), plot=True):
    accuracies = []

    for tol in tolerances:
        acc, _ = custom_acc_salience(y_true, y_pred, distance_tolerance=tol)
        accuracies.append(acc)

    best_idx = np.argmax(accuracies)
    best_tol = tolerances[best_idx]

    if plot:
        plot_tolerance_vs_accuracy(tolerances, accuracies, best_tol)

    return best_tol


def find_optimal_positive_threshold(y_true,
                                    y_pred,
                                    positive_thresholds=np.linspace(0.05, 0.95, 19),
                                    plot=True):

    # Store accuracy scores
    accuracies = []

    for threshold in positive_thresholds:
        binary_preds = (y_pred >= threshold).astype(int)

        # Compute accuracy
        acc_presence, _ = custom_acc_presence(y_true, binary_preds)

        # Store results
        accuracies.append(acc_presence)

    # Find optimal threshold (based on highest AUC score)
    best_threshold_idx = np.argmax(accuracies)
    best_threshold = positive_thresholds[best_threshold_idx]

    if plot:
        plot_accuracy_vs_threshold(positive_thresholds, accuracies, best_threshold)

    return best_threshold


def plot_tolerance_vs_accuracy(tolerances, accuracies, best_tol):
    plt.figure(figsize=(8, 6))
    plt.plot(tolerances, accuracies, label="Very Strict Accuracy", marker="o")
    plt.axvline(best_tol, color="red", linestyle="dashed", label=f"Optimal Tolerance: {best_tol:.5f}")
    plt.xlabel("Tolerance")
    plt.ylabel("Accuracy")
    plt.title("Very Strict Accuracy vs. Tolerance")
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