import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import ROOT_DIR
from src.benchmarks.simple.train_eval.custom_accuracy import custom_acc_presence, custom_acc_salience

from src.benchmarks.simple.train_eval.load_data import get_index2emotion, load_data
from src.benchmarks.simple.train_eval.model import MultiLabelSoftmaxNN
from src.benchmarks.simple.train_eval.probability_threshold_optimization import find_optimal_positive_threshold, \
    find_optimal_salience_threshold

from src.benchmarks.simple.train_eval.train import Trainer
from src.benchmarks.simple.train_eval.utils import get_top_k_predictions, get_blend_indices, label_vector2dict

index2emotion = get_index2emotion()

folds = [0, 1, 2, 3, 4]

metadata_path = ROOT_DIR + "/data/train_metadata.csv"

df_metadata = pd.read_csv(metadata_path)


for fold in folds:
    print(f"Fold {fold}")
    train_dataset, val_dataset = load_data(fold_id=fold)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = MultiLabelSoftmaxNN(train_dataset.input_dim, train_dataset.output_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # predict
    trainer = Trainer(model, optimizer)
    trainer.train(train_loader, epochs=100)

    y_pred = trainer.predict(val_loader)
    y_pred_top_k = get_top_k_predictions(y_pred, k=2)

    labels = val_dataset.labels.numpy()
    indices = val_dataset.indices

    # Find best thresholds
    best_positive_threshold = find_optimal_positive_threshold(labels, y_pred_top_k)

    # zero out predictions below threshold
    y_pred_top_k[y_pred_top_k < best_positive_threshold] = 0
    acc_presence, _ = custom_acc_presence(labels, y_pred_top_k)
    print("acc presence: ", acc_presence)

    mixed_indices = get_blend_indices(labels)
    best_salience_diff_threshold = find_optimal_salience_threshold(labels[mixed_indices], y_pred_top_k[mixed_indices])

    acc_salience, salience_preds = custom_acc_salience(labels[mixed_indices],
                                          y_pred_top_k[mixed_indices],
                                          best_salience_diff_threshold)

    print("acc salience: ", acc_salience)

    # convert predictions to standard format
    df_metadata_val = df_metadata.iloc[indices]
    filenames = df_metadata_val["filename"].values

    label_dict = label_vector2dict(filenames, y_pred_top_k, index2emotion)




