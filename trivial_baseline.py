import pandas as pd

from utils.generic_accuracy.accuracy_funcs import acc_presence_total, acc_salience_total


# Helper to create blend labels
def make_blend_label(row):
    emotions = sorted([row['emotion_1'], row['emotion_2']])
    return '-'.join(emotions)

# Helper to create blend+salience labels
def make_blend_salience_label(row):
    if row['mix'] == 1:
        e1 = row['emotion_1']
        e2 = row['emotion_2']
        s1 = int(row['emotion_1_salience'])  # integer
        s2 = int(row['emotion_2_salience'])
        if e1 < e2:
            return f"{e1}-{e2} ({s1}/{s2})"
        else:
            return f"{e2}-{e1} ({s2}/{s1})"
    else:
        return None

######################
# Test the generic accuracy functions
def test_single(filenames, single_emotion):
    preds = {}
    for f in filenames:
        preds[f] = [{'emotion': single_emotion, 'salience': 1.0}]
    acc_pres = acc_presence_total(preds)
    print(f"Presence accuracy as determined by generic accuracy function '{single_emotion}': {acc_pres:.4f}")
    return acc_pres

def test_blend(filenames, blend_label):
    preds = {}
    for f in filenames:
        preds[f] = [{'emotion': e, 'salience': s} for e, s in zip(blend_label.split('-'), [0.5, 0.5])]
    acc_pres = acc_presence_total(preds)
    print(f"Presence accuracy as determined by generic accuracy function '{blend_label}': {acc_pres:.4f}")
    return acc_pres

def test_blend_salience(filenames, blend_salience_label):
    preds = {}
    for f in filenames:
        e1, e2 = blend_salience_label.split(' ')[0].split('-')
        s1, s2 = map(int, blend_salience_label.split(' ')[1][1:-1].split('/'))
        preds[f] = [{'emotion': e1, 'salience': s1}, {'emotion': e2, 'salience': s2}]
    acc_sal = acc_salience_total(preds)
    print(f"Salience accuracy as determined by generic accuracy function '{blend_salience_label}': {acc_sal:.4f}")
    return acc_sal


def do_accuracy_calcs(df_train, df_test):
    for df in [df_train, df_test]:
        # Create blend and blend+salience labels
        df['blend_label'] = df.apply(lambda row: make_blend_label(row) if row['mix'] == 1 else None, axis=1)
        df['blend_salience_label'] = df.apply(make_blend_salience_label, axis=1)

    # -- Trivial Single Emotion Baseline --

    # Most common single emotion
    single_counts = df_train[df_train['mix'] == 0]['emotion_1'].value_counts()
    most_common_single = single_counts.idxmax()
    print(f"Most common single emotion: {most_common_single}")
    presence_accuracy_single = test_single(df_test['filename'].values, most_common_single)

    # Salience accuracy will be 0 because only one emotion is predicted
    salience_accuracy_single = 0.0

    # -- Trivial Blend Baseline --

    # Most common blend (regardless of salience)
    blend_counts = df_train[df_train['mix'] == 1]['blend_label'].value_counts()
    most_common_blend = blend_counts.idxmax()
    presence_accuracy_blend = test_blend(df_test['filename'].values, most_common_blend)

    print(f"Most common blend: {most_common_blend}")

    # Salience Accuracy: best we can do is the most common salience configuration
    salience_counts = df_train[df_train['mix'] == 1]['blend_salience_label'].value_counts()
    most_common_blend_salience = salience_counts.idxmax()
    salience_accuracy_blend = test_blend_salience(df_test['filename'].values, most_common_blend_salience)

    print(f"Most common blend with salience: {most_common_blend_salience}")

    return (presence_accuracy_single, salience_accuracy_single), (presence_accuracy_blend, salience_accuracy_blend)




metadata_path_train = "/home/tim/Work/quantum/data/blemore/train_metadata.csv"
df_full_train = pd.read_csv(metadata_path_train)
folds = [0, 1, 2, 3, 4]
results = []

for fold_id in folds:
    train_mask = df_full_train["fold"] != fold_id
    val_mask = ~train_mask
    df_train_ = df_full_train.loc[train_mask].copy()
    df_val = df_full_train.loc[val_mask].copy()
    print(f"\nFold {fold_id} Accuracy Calculations:")
    (pres_single, sal_single), (pres_blend, sal_blend) = do_accuracy_calcs(df_train_, df_val)
    results.append((pres_single, sal_single, pres_blend, sal_blend))

    # Convert to DataFrame for convenience
results_df = pd.DataFrame(
    results,
    columns=["presence_single", "salience_single", "presence_blend", "salience_blend"]
)

# Calculate mean and std
means = results_df.mean()
stds = results_df.std()

print("\nMean accuracies:")
print(means)
print("\nStd of accuracies:")
print(stds)




# print("Training Data Accuracy Calculations:")
# do_accuracy_calcs(df_full_train)


# metadata_path_test = "/home/tim/Work/quantum/data/blemore/test_metadata.csv"
# df_full_test = pd.read_csv(metadata_path_test)
# print("\nTest Data Accuracy Calculations:")
# do_accuracy_calcs(df_full_train, df_full_test)




