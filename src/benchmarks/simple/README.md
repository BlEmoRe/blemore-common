# Simple Baseline Benchmarks (OpenFace + MLP)
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
