import os
import pandas as pd
from config import ROOT_DIR
from src.tools.generic_accuracy.accuracy_funcs import acc_presence_total, acc_salience_total


def metadata2labels(df):
    """
    Convert the metadata DataFrame to a dictionary of labels.
    """
    ret = {}

    for _, row in df.iterrows():
        filename = row["filename"]
        emotion_1 = row["emotion_1"]
        emotion_2 = row["emotion_2"]

        if pd.notna(row["emotion_2"]):
            emotion_1_salience = row["emotion_1_salience"]
            emotion_2_salience = row["emotion_2_salience"]

            ret[filename] = [
                {"emotion": emotion_1, "salience": emotion_1_salience},
                {"emotion": emotion_2, "salience": emotion_2_salience}
            ]
        else:
            ret[filename] = [
                {"emotion": emotion_1, "salience": 1.0}
            ]

    return ret


def main():
    path = os.path.join(ROOT_DIR, "data/test_metadata.csv")
    df_metadata = pd.read_csv(path)
    labels = metadata2labels(df_metadata)

    from collections import Counter
    all_single_emotions = []
    all_blends = []

    for preds in labels.values():
        if len(preds) == 1:
            all_single_emotions.append(preds[0]["emotion"])
        elif len(preds) == 2:
            blend = tuple(sorted([preds[0]["emotion"], preds[1]["emotion"]]))
            all_blends.append(blend)

    # Find most common single emotion (for presence)
    single_counts = Counter(all_single_emotions)
    most_common_single, _ = single_counts.most_common(1)[0]
    print(f"Most common single emotion: {most_common_single}")
    # Build predictions
    y_pred = {}
    for filename in labels.keys():
        # Always predict the most common blend
        y_pred[filename] = [
            {"emotion": most_common_single, "salience": 1.0}
        ]

    # Evaluate
    presence_acc = acc_presence_total(y_pred)

    # Find most common blend (for salience)
    blend_counts = Counter(all_blends)
    most_common_blend, _ = blend_counts.most_common(1)[0]
    print(f"Most common blend: {most_common_blend}")

    # Build predictions
    y_pred = {}
    for filename in labels.keys():
        # Always predict the most common blend
        y_pred[filename] = [
            {"emotion": most_common_blend[0], "salience": 70.0},
            {"emotion": most_common_blend[1], "salience": 30.0}
        ]

    # Evaluate
    salience_acc = acc_salience_total(y_pred)

    print(f"Presence Accuracy: {presence_acc:.3f}")
    print(f"Salience Accuracy: {salience_acc:.3f}")


if __name__ == "__main__":
    main()