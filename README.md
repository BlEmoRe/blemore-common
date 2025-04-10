# BlEmoRe-common

## Tools

### Accuracy metrics

Generic functions to calculate the accuracy metrics are available in: `src/tools/generic_accuracy/accuracy_funcs.py`.
These functions rely on predictions provided in the following dictionary format:

```python
{   
    # The key is the filename of the video
    'A411_mix_ang_hap_30_70_ver1':
        # The value is a list of dictionaries, 
        # each containing the predicted emotion and its salience
        [
            {'emotion': 'hap', 'salience': 70.0},
            {'emotion': 'ang', 'salience': 30.0}
        ],
    'A102_ang_int1_ver1':
        [
            {'emotion': 'neu', 'salience': 100.0}
        ]
        ...
}
```

We employ two main evaluation metrics: `ACC_presence` and `ACC_salience`.

- `ACC_presence` measures whether the correct label(s) are predicted without errors.
  A correct prediction must include all present emotions while avoiding false negatives
  (e.g., predicting only one emotion in a blend) and false positives
  (e.g., predicting emotions that are not part of the label).

- `ACC_salience` extends `ACC_presence` by considering the relative prominence of each emotion.
  It evaluates whether the predicted proportions reflect the correct ranking — whether the emotions
  are equally present or one is more dominant than the other. This metric applies only to blended emotions.

### Filename parser

Simple filename parser, provided as a convenience. Note that it is not necessary to use the filename parser, since the
filenames and their
corresponding labels and other metadata is already available in
the train_metadata.csv on [Zenodo](https://zenodo.org/records/15096942).

## Baselines

### Simple Baseline (OpenFace + MLP)

Using a feature-based model (`MultiLabelSoftmaxNN`) trained on OpenFace features with a
softmax + KL-divergence objective, evaluated using 5-fold cross-validation.

| Fold | Accuracy (Presence) | Accuracy (Salience) |
|------|---------------------|---------------------|
| 0    | 0.2372              | 0.1037              |
| 1    | 0.2407              | 0.0824              |
| 2    | 0.2456              | 0.0917              |
| 3    | 0.2358              | 0.0730              |
| 4    | 0.2396              | 0.0845              |

* Loss: `KLDivLoss` vs. probabilistic multi-label targets
* Optimizer: `Adam (lr=0.001, weight_decay=1e-4)`
* Epochs: `100`

**Running the baseline code:**

1. Download the data set from [Zenodo](https://zenodo.org/records/15096942)
2. Extract OpenFace features from the videos and provide the path to the files, along with the train_metadata.csv in
   `src/baselines/simple/config_simple_baseline.py`
3. Aggregate statistical features, and merge with metadata using the script `src/baselines/simple/aggregate_data.py`
4. Create the dataset with label vectors using the script `src/baselines/simple/create_train_set.py`
5. Train and evaluate the model using the script `src/baselines/simple/evaluate.py`