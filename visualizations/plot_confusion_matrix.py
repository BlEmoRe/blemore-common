from utils.filename_parser import parse_filename
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import json
from config import ROOT_DIR

import os

def plot_confusion_matrix(final_preds, save_path=None):

    def build_label(parsed):
        if parsed["mix"] == 1:
            emotions = sorted([parsed["emotion_1"], parsed["emotion_2"]])
            label = "-".join(emotions)
        else:
            label = parsed["emotion_1"]
        return label

    def build_pred_label(pred_list):
        emotions = sorted([pred["emotion"] for pred in pred_list])
        label = "-".join(emotions)
        return label

    # Process
    true_labels = []
    pred_labels = []

    for filename, preds in final_preds.items():
        parsed = parse_filename(filename)

        true_label = build_label(parsed)
        pred_label = build_pred_label(preds)

        true_labels.append(true_label)
        pred_labels.append(pred_label)

    # Get all unique labels
    unique_labels = set(true_labels) | set(pred_labels)

    # Sort: singles first, blends after
    singles = sorted([l for l in unique_labels if '-' not in l])
    blends = sorted([l for l in unique_labels if '-' in l])
    all_labels = singles + blends  # new order!

    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=all_labels)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=all_labels, yticklabels=all_labels,
                cbar=True, square=True, linewidths=0.5,
                annot_kws={"size": 20})

    # Change colorbar font size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)

    ax.set_xlabel('Predicted label', fontsize=20)
    ax.set_ylabel('True label', fontsize=20)
    # ax.set_title("Confusion Matrix (Emotions and Blends as Unique Classes)", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved confusion matrix to {save_path}")

    plt.show()

# encoder = "wavlm"  # Choose your encoder here
encoder = "videomae_hubert"

with open(os.path.join(ROOT_DIR, "data/{}_test_predictions.json".format(encoder)), "r") as f:
    p = json.load(f)

plot_confusion_matrix(p, os.path.join(ROOT_DIR, "data/plots/confusion_matrix_{}_test.png".format(encoder)))