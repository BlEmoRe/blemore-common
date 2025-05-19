import numpy as np

from src.tools.filename_parser import parse_filename
from src.tools.generic_accuracy.metadata2labels import metadata2label


def acc_presence_single(label, pred):
    """
    Check if predicted emotions match ground truth emotions, ignoring order and salience.

    Parameters:
        label (list of dict): Ground truth for one item, e.g., [{'emotion': 'ang', 'salience': 70.0}, ...]
        pred (list of dict): Predictions, same format as label.

    Returns:
        bool: True if all predicted emotions match the ground truth emotions (order and salience ignored).
    """
    label_emotions = set([l['emotion'] for l in label])
    pred_emotions = set([p['emotion'] for p in pred])
    return label_emotions == pred_emotions


def acc_salience_single(label, pred):
    """
    Check if both emotions and their salience values match exactly between prediction and ground truth.

    Parameters:
        label (list of dict): Ground truth, must contain exactly two items with keys 'emotion' and 'salience'.
        pred (list of dict): Predictions, same format as label.

    Returns:
        bool: True if both emotion sets and their salience values match exactly.

    Raises:
        ValueError: If label or pred does not contain exactly two emotions.
    """
    if len(label) != 2 or len(pred) != 2:
        raise ValueError("Both label and prediction must contain exactly two emotions to calculate salience accuracy.")
    label_dict = {l["emotion"]: round(float(l["salience"])) for l in label}
    pred_dict = {p["emotion"]: round(float(p["salience"])) for p in pred}
    return label_dict == pred_dict


def acc_presence_total(preds):
    """
    Compute average presence accuracy across all samples.

    Parameters:
        preds (dict): Mapping from filename to list of predictions (each a dict with 'emotion' and 'salience').
                        e.g. {'A102_ang_int1_ver1': [{'emotion': 'neu', 'salience': 100.0}], ...}
                        note that the salience values are completely ignored in this function.
                        salience is only used in acc_salience_total, and only applicable for samples with exactly two emotions.
    Returns:
        float: Mean presence accuracy across all matched files.
    """

    res = []
    for filename, predictions in preds.items():
        metadata = parse_filename(filename)
        label = metadata2label(metadata)[filename]
        presence = acc_presence_single(label, predictions)
        res.append(presence)
    return np.mean(res).item()


def acc_salience_total(preds):
    """
    Compute average salience accuracy for samples with exactly two emotions.

    Parameters:
        preds (dict): Mapping from filename to list of predictions (each a dict with 'emotion' and 'salience').
                        e.g. {'A411_mix_ang_hap_30_70_ver1': [
                                {'emotion': 'hap', 'salience': 70.0},
                                {'emotion': 'ang', 'salience': 30.0}
                            ], ...}
                            note that this function will only consider samples with exactly two emotions.
    Returns:
        float: Mean salience accuracy for samples with exactly two ground truth emotions.
    """
    res = []
    for filename, predictions in preds.items():
        metadata = parse_filename(filename)
        label = metadata2label(metadata)[filename]
        if len(label) != 2:
            continue
        if len(predictions) == 2:
            salience = acc_salience_single(label, predictions)
        else:
            salience = False
        res.append(salience)

    if len(res) == 0:
        return 0.0
    else:
        return np.mean(res).item()



def main():
    # Provide labels in the following format
    y_pred = {'A102_ang_int1_ver1': [{'emotion': 'neu', 'salience': 100.0}],
              'A102_ang_int2_ver1': [{'emotion': 'ang', 'salience': 100.0}],
              'A102_disg_int1_ver1': [{'emotion': 'fea', 'salience': 100.0}],
              'A102_disg_int2_ver1': [{'emotion': 'disg', 'salience': 100.0}],
              'A55_fea_int1_ver2': [{'emotion': 'hap', 'salience': 100.0}],
              'A55_fea_int3_ver1': [{'emotion': 'fea', 'salience': 100.0}],
              'A407_mix_disg_hap_50_50_ver1': [{'emotion': 'disg', 'salience': 50},
                                               {'emotion': 'hap', 'salience': 50}],
              'A407_mix_disg_hap_70_30_ver1': [{'emotion': 'disg', 'salience': 30.0},
                                               {'emotion': 'hap', 'salience': 70.0}],
              'A411_mix_ang_hap_30_70_ver1': [{'emotion': 'hap', 'salience': 70.0},
                                              {'emotion': 'ang', 'salience': 30.0}],
              'A411_mix_ang_hap_50_50_ver1': [{'emotion': 'ang', 'salience': 50.0},
                                              {'emotion': 'hap', 'salience': 50.0}],
              'A427_mix_ang_hap_30_70_ver1': [{'emotion': 'hap', 'salience': 70.0},
                                              {'emotion': 'ang', 'salience': 30.0}],
              'A427_mix_ang_hap_50_50_ver1': [{'emotion': 'ang', 'salience': 30.0},
                                              {'emotion': 'hap', 'salience': 70.0}]
              }
    # Calculate accuracy for presence
    presence_accuracy = acc_presence_total(y_pred)
    print(f"Presence Accuracy: {presence_accuracy:.2f}")

    # Calculate accuracy for salience
    salience_accuracy = acc_salience_total(y_pred)
    print(f"Salience Accuracy: {salience_accuracy:.2f}")

if __name__ == "__main__":
    main()