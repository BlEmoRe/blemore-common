# BlEmoRe-common

## Tools 

### Filename parser

Simple filename parser. Note that it is not necessary to use the filename parser at all, since the filenames and their 
corresponding labels and other metadata is already available in the train_metadata.csv 
on [Zenodo](https://zenodo.org/records/15096942).

## Benchmarks 

### Simple Baseline Benchmarks (OpenFace + MLP)

Using a feature-based model (`MultiLabelSoftmaxNN`) trained on OpenFace features with a 
softmax + KL-divergence objective, evaluated using 5-fold cross-validation.

| Fold | Accuracy (Presence) | Accuracy (Salience) |
|------|---------------------|----------------------|
| 0    | 0.2372              | 0.1037               |
| 1    | 0.2407              | 0.0824               |
| 2    | 0.2456              | 0.0917               |
| 3    | 0.2358              | 0.0730               |
| 4    | 0.2396              | 0.0845               |

* Loss: `KLDivLoss` vs. probabilistic multi-label targets
* Optimizer: `Adam (lr=0.001, weight_decay=1e-4)`
* Epochs: `100`

Running these baselines require:

1. Extracting OpenFace features from the videos.
2. Aggregating statistical features, and merging with metadata using the script `src/benchmarks/simple/aggregate_data.py` (replace paths appropriately)
3. Creating the dataset with labels using the script `src/benchmarks/simple/create_train_set.py`.
4. Training and evaluating the model using the script `src/benchmarks/simple/evaluate.py`.