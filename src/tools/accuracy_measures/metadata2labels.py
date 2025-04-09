import os
import pandas as pd
from config import ROOT_DIR


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
    path = os.path.join(ROOT_DIR, "data/train_metadata.csv")
    df_metadata = pd.read_csv(path)
    labels = metadata2labels(df_metadata)
    print(labels)
    # for i in labels.items():
    #     print(i)

if __name__ == "__main__":
    main()