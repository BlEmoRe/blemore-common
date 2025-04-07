import torch
from torch.utils.data import DataLoader

from src.benchmarks.simple.train_eval.accuracy_measures import compute_acc_presence, compute_acc_salience
from src.benchmarks.simple.train_eval.load_data import get_index2emotion, load_data
from src.benchmarks.simple.train_eval.model import MultiLabelSoftmaxNN
from src.benchmarks.simple.train_eval.probability_threshold_optimization import find_optimal_positive_threshold, \
    find_optimal_salience_discriminator

from src.benchmarks.simple.train_eval.train import Trainer
from src.benchmarks.simple.train_eval.utils import get_top_k_predictions, get_blend_indices

index2emotion = get_index2emotion()

folds = [0, 1, 2, 3, 4]

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

    # Find best thresholds
    best_threshold = find_optimal_positive_threshold(labels, y_pred_top_k)

    # zero out predictions below threshold
    y_pred_top_k[y_pred_top_k < best_threshold] = 0
    acc_presence, _ = compute_acc_presence(labels, y_pred_top_k)
    print("acc presence: ", acc_presence)

    mixed_indices = get_blend_indices(labels)
    best_disc = find_optimal_salience_discriminator(labels[mixed_indices], y_pred_top_k[mixed_indices])

    acc_salience, _ = compute_acc_salience(labels[mixed_indices],
                                           y_pred_top_k[mixed_indices],
                                           distance_tolerance=best_disc)

    print("acc salience: ", acc_salience)
