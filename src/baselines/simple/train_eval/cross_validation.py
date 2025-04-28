import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd

from src.baselines.simple.train_eval.accuracy_calculation import generic_accuracy, get_filename_from_indices
from src.baselines.simple.train_eval.blend_operations.blend_utils import get_top_k_predictions, get_blend_indices, \
    convert_probs_to_salience
from src.baselines.simple.train_eval.blend_operations.threshold_optimization import find_optimal_presence_threshold, \
    find_optimal_salience_threshold
from src.baselines.simple.train_eval.model.load_data import get_index2emotion, load_data
from src.baselines.simple.train_eval.model.model import MultiLabelSoftmaxNN

from src.baselines.simple.train_eval.model.train import Trainer


def run_validation(data, fold, df, index2emotion):
    train_dataset, val_dataset = load_data(data, fold_id=fold)  # Assuming fold_id=0 for validation

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = MultiLabelSoftmaxNN(train_dataset.input_dim, train_dataset.output_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Predict
    trainer = Trainer(model, optimizer)
    trainer.train(train_loader, epochs=100)

    y_pred = trainer.predict(val_loader)
    y_pred_top_2 = get_top_k_predictions(y_pred, k=2)  # use top 2 predictions

    labels = val_dataset.labels.numpy()

    # Find the best positive thresholds
    best_presence_threshold = find_optimal_presence_threshold(labels, y_pred_top_2)

    # Zero out predictions below positive threshold
    y_pred_top_2[y_pred_top_2 < best_presence_threshold] = 0

    # Find the best salience threshold (when to consider salience equal or one dominant)
    mixed_indices = get_blend_indices(labels)
    best_salience_threshold = find_optimal_salience_threshold(labels[mixed_indices], y_pred_top_2[mixed_indices])

    # map probabilities to salience
    salience_predictions = convert_probs_to_salience(y_pred_top_2, best_salience_threshold)

    predicted_filenames = get_filename_from_indices(val_dataset.indices, df)

    presence, salience = generic_accuracy(salience_predictions, predicted_filenames, index2emotion)

    ret = {
        "acc_pres": presence,
        "acc_sal": salience,
        "pres_threshold": best_presence_threshold,
        "sal_threshold": best_salience_threshold
    }
    return ret


def cross_validate(dataset, df, index2emotion):
    folds = [0, 1, 2, 3, 4]

    results = []

    for fold in folds:
        print(f"Fold {fold}")
        # Load data
        data = dataset
        res = run_validation(data, fold, df, index2emotion)

        results.append(res)

    # Calculate average results
    avg_results = {
        "acc_pres": np.mean([res["acc_pres"] for res in results]).item(),
        "acc_sal": np.mean([res["acc_sal"] for res in results]).item(),
        "pres_threshold": np.mean([res["pres_threshold"] for res in results]).item(),
        "sal_threshold": np.mean([res["sal_threshold"] for res in results]).item()
    }
    print(f"Average Results: {avg_results}")

    df_results = pd.DataFrame(results)
    print(df_results)





