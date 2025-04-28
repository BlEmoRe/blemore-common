import torch
from torch.utils.data import DataLoader
from src.baselines.simple.train_eval.model.model import MultiLabelSoftmaxNN
from src.baselines.simple.train_eval.model.train import Trainer
from src.baselines.simple.train_eval.blend_operations.blend_utils import get_top_k_predictions, convert_probs_to_salience
from src.baselines.simple.train_eval.accuracy_calculation import generic_accuracy, get_filename_from_indices
from src.baselines.simple.train_eval.blend_operations.threshold_optimization import find_optimal_presence_threshold, find_optimal_salience_threshold
from src.baselines.simple.train_eval.model.load_data import get_index2emotion, load_data


def train_and_evaluate(train_data, test_data, test_df, index2label):
    train_dataset = load_data(train_data, split=False)
    test_dataset = load_data(test_data, split=False)

    # Load datasets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Define model
    model = MultiLabelSoftmaxNN(train_dataset.input_dim, test_dataset.output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    trainer = Trainer(model, optimizer)

    # Train
    trainer.train(train_loader, epochs=100)

    # Determined in the validation phase
    best_presence_threshold = 0.05
    best_salience_threshold = 0.000010

    # Predict on test set
    y_test_pred = trainer.predict(test_loader)
    y_test_pred_top_2 = get_top_k_predictions(y_test_pred, k=2)

    y_test_pred_top_2[y_test_pred_top_2 < best_presence_threshold] = 0
    salience_predictions = convert_probs_to_salience(y_test_pred_top_2, best_salience_threshold)

    predicted_filenames = get_filename_from_indices(test_dataset.indices, test_df)
    generic_accuracy(salience_predictions, predicted_filenames, index2label)






