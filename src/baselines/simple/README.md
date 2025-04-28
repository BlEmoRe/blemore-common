# Simple Baseline Benchmarks (OpenFace + MLP)
Using a feature-based model (`MultiLabelSoftmaxNN`) trained on OpenFace features with a 
softmax + KL-divergence objective, evaluated using 5-fold cross-validation.

| Fold | Accuracy (Presence) | Accuracy (Salience) |
|------|---------------------|---------------------|
| 0    | 0.24                | 0.14                |
| 1    | 0.19                | 0.11                |
| 2    | 0.20                | 0.15                |
| 3    | 0.22                | 0.10                |
| 4    | 0.22                | 0.10                |

* Loss: `KLDivLoss` vs. probabilistic multi-label targets
* Optimizer: `Adam (lr=0.001, weight_decay=1e-4)`
* Epochs: `100`
