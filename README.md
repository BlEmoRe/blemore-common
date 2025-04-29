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

Simple filename parser, provided as a convenience. The filenames and their
corresponding labels and other metadata is already available in
the train_metadata.csv on [Zenodo](https://zenodo.org/records/15096942).

## Baselines

### Simple Baseline (OpenFace + MLP)

Using a feature-based model (`MultiLabelSoftmaxNN`) trained on [OpenFace 2.2.0](https://github.com/TadasBaltrusaitis/OpenFace) features with a
softmax + KL-divergence objective. 

* Loss: `KLDivLoss` vs. probabilistic multi-label targets
* Optimizer: `Adam (lr=0.001, weight_decay=1e-4)`
* Epochs: `100`

We compare this to a trivial baseline that predicts the most frequent label in the training set.

#### Cross Validation

We provide pre-defined folds in the dataset, the baseline results with the OpenFace + MLP classifier for each fold are as follows:

| Fold | Accuracy (Presence) | Accuracy (Salience) |
|------|---------------------|---------------------|
| 0    | 0.24                | 0.14                |
| 1    | 0.19                | 0.11                |
| 2    | 0.20                | 0.15                |
| 3    | 0.22                | 0.10                |
| 4    | 0.22                | 0.10                |

#### Results on the test set

The results on the test set are as follows:

| Method         | Accuracy (Presence) | Accuracy (Salience) |
|----------------|---------------------|---------------------|
| OpenFace + MLP | 0.21                | 0.10                |
| Trivial        | 0.074               | 0.033               |


**Running the baseline code:**

Simple Openface + MLP baseline:

1. Download the data set from [Zenodo](https://zenodo.org/records/15096942)
2. Extract [OpenFace 2.2.0](https://github.com/TadasBaltrusaitis/OpenFace) features from the videos.
3. Run the baseline code in `src/baselines/simple/pipeline.py`. Adjust the paths as necessary.

Trivial baseline: `src/baselines/trivial/most_frequent_classifier.py`.
