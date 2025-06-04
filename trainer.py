import torch
import numpy as np
import os

from post_processing import grid_search_thresholds
from utils.subsample_utils import aggregate_subsamples

class Trainer(object):

    def __init__(self, model, optimizer, data_loader, epochs, valid_data_loader=None, subsample_aggregation=False, save_dir="checkpoints"):
        self.model = model
        self.optimizer = optimizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model.to(self.device)

        self.data_loader = data_loader
        self.epochs = epochs
        self.valid_data_loader = valid_data_loader
        self.subsample_aggregation = subsample_aggregation

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_val_score = -np.inf  # Higher is better

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            probs, logits, loss = self.model(data, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(self.data_loader)
        return avg_loss

    def validate(self):
        print("\nValidating model...")

        self.model.eval()
        total_loss = 0
        all_probs = []
        all_logits = []

        with torch.no_grad():
            for data, target in self.valid_data_loader:
                data, target = data.to(self.device), target.to(self.device)
                probs, logits, loss = self.model(data, target)
                total_loss += loss.item()
                all_probs.append(probs.cpu().numpy())
                all_logits.append(logits.cpu().numpy())

        all_probs = np.concatenate(all_probs, axis=0)
        all_logits = np.concatenate(all_logits, axis=0)
        val_filenames = self.valid_data_loader.dataset.filenames

        if self.subsample_aggregation:
            val_filenames, all_probs = aggregate_subsamples(val_filenames, all_logits)

        ret = grid_search_thresholds(val_filenames, all_probs)
        avg_loss = total_loss / len(self.valid_data_loader)
        ret["val_loss"] = avg_loss
        return ret

    def train(self, writer=None, save_prefix="model"):
        best_epoch_stats = None
        best_model_path = None

        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            print(f"Epoch [{epoch + 1}/{self.epochs}], Train Loss: {train_loss:.4f}")

            if writer:
                writer.add_scalar("Loss/train", train_loss, epoch)

            if self.valid_data_loader is not None:
                stats = self.validate()
                val_score = 0.5 * stats['acc_presence'] + 0.5 * stats['acc_salience']  # scoring metric

                print(f"Epoch [{epoch + 1}/{self.epochs}], "
                      f"Validation Loss: {stats['val_loss']:.4f}, "
                      f"Best Alpha: {stats['alpha']:.4f}, "
                      f"Best Beta: {stats['beta']:.4f}, "
                      f"Best Acc Presence: {stats['acc_presence']:.4f}, "
                      f"Best Acc Salience: {stats['acc_salience']:.4f}, "
                      f"Best possible Acc Presence : {stats['presence_only']:.4f}, "
                      f"Best possible Acc Salience : {stats['salience_only']:.4f}")

                if writer:
                    writer.add_scalar("Loss/val", stats['val_loss'], epoch)
                    writer.add_scalar("Accuracy/presence", stats['acc_presence'], epoch)
                    writer.add_scalar("Accuracy/salience", stats['acc_salience'], epoch)
                    writer.add_scalar("Alpha", stats['alpha'], epoch)
                    writer.add_scalar("Beta", stats['beta'], epoch)
                    writer.add_scalar("Best possible Accuracy/presence", stats['presence_only'], epoch)
                    writer.add_scalar("Best possible Accuracy/salience", stats['salience_only'], epoch)

                if val_score > self.best_val_score:
                    # Save best model
                    self.best_val_score = val_score
                    best_model_path = os.path.join(self.save_dir, f"{save_prefix}_best.pth")
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'input_dim': self.model.input_dim,
                        'output_dim': self.model.output_dim,
                        'model_type': self.model.model_type,
                    }, best_model_path)

                    best_epoch_stats = {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                        "val_loss": stats['val_loss'],
                        "best_alpha": stats['alpha'],
                        "best_beta": stats['beta'],
                        "best_acc_presence": stats['acc_presence'],
                        "best_acc_salience": stats['acc_salience'],
                    }

        return best_epoch_stats, best_model_path
