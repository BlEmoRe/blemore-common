import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.baselines.simple.train_eval.custom_accuracy import custom_acc_presence, custom_acc_salience, \
    map_vector_pairwise

from src.baselines.simple.train_eval.load_data import get_index2emotion, load_data
from src.baselines.simple.train_eval.model import MultiLabelSoftmaxNN
from src.baselines.simple.train_eval.probability_threshold_optimization import find_optimal_positive_threshold, \
    find_optimal_salience_threshold

from src.baselines.simple.train_eval.train import Trainer
from src.baselines.simple.train_eval.utils import get_top_k_predictions, get_blend_indices, label_vector2dict
from src.tools.generic_accuracy.accuracy_funcs import acc_presence_total, acc_salience_total

from src.baselines.simple.config_simple_baseline import AGGREGATED_OPENFACE_PATH


def standard_accuracy_calculation(preds, salience_threshold, indices):
    df = pd.read_csv(AGGREGATED_OPENFACE_PATH)
    index2emotion = get_index2emotion()

    df_val = df.iloc[indices]
    filenames = df_val["filename"].values

    preds_mapped = map_vector_pairwise(preds, salience_threshold)
    pred_dict = label_vector2dict(filenames, preds_mapped, index2emotion)

    presence = acc_presence_total(pred_dict, AGGREGATED_OPENFACE_PATH)
    salience = acc_salience_total(pred_dict, AGGREGATED_OPENFACE_PATH)

    print(f"Presence Accuracy: {presence:.2f}")
    print(f"Salience Accuracy: {salience:.2f}")



folds = [0, 1, 2, 3, 4]


for fold in folds:
    print(f"Fold {fold}")
    train_dataset, val_dataset = load_data(fold_id=fold)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = MultiLabelSoftmaxNN(train_dataset.input_dim, train_dataset.output_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Predict
    trainer = Trainer(model, optimizer)
    trainer.train(train_loader, epochs=100)

    y_pred = trainer.predict(val_loader)
    y_pred_top_k = get_top_k_predictions(y_pred, k=2) # use top 2 predictions

    labels = val_dataset.labels.numpy()

    # Find the best positive thresholds
    best_positive_threshold = find_optimal_positive_threshold(labels, y_pred_top_k)

    # Zero out predictions below positive threshold
    y_pred_top_k[y_pred_top_k < best_positive_threshold] = 0

    # Find the best salience threshold (when to consider salience equal or one dominant)
    mixed_indices = get_blend_indices(labels)
    best_salience_threshold = find_optimal_salience_threshold(labels[mixed_indices], y_pred_top_k[mixed_indices])

    standard_accuracy_calculation(y_pred_top_k, best_salience_threshold, val_dataset.indices)







