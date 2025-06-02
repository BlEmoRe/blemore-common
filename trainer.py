import torch
import numpy as np

from post_processing import grid_search_thresholds


class Trainer(object):

    def __init__(self, model, optimizer, data_loader, epochs, valid_data_loader=None):
        self.model = model
        self.optimizer = optimizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model.to(self.device)

        self.data_loader = data_loader
        self.epochs = epochs
        self.valid_data_loader = valid_data_loader

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output, loss = self.model(data, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(self.data_loader)
        return avg_loss

    def validate(self):

        print("\nValidating model...")

        self.model.eval()
        total_loss = 0
        all_preds = []

        with torch.no_grad():
            for data, target in self.valid_data_loader:

                data, target = data.to(self.device), target.to(self.device)

                output, loss = self.model(data, target)

                total_loss += loss.item()

                all_preds.append(output.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        val_filenames = self.valid_data_loader.dataset.filenames

        ret = grid_search_thresholds(val_filenames, all_preds)
        avg_loss = total_loss / len(self.valid_data_loader)
        ret["val_loss"] = avg_loss
        return ret

    def train(self, writer=None):
        results = []

        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            print(f"Epoch [{epoch + 1}/{self.epochs}], Train Loss: {train_loss:.4f}")

            # Log training loss
            if writer:
                writer.add_scalar("Loss/train", train_loss, epoch)

            stats = self.validate()
            print(f"Epoch [{epoch + 1}/{self.epochs}], "
                  f"Validation Loss: {stats['val_loss']:.4f}, "
                  f"Best Alpha: {stats['alpha']:.4f}, "
                  f"Best Beta: {stats['beta']:.4f}, "
                  f"Best Acc Presence: {stats['acc_presence']:.4f}, "
                  f"Best Acc Salience: {stats['acc_salience']:.4f},"
                  f"Best possible Acc Presence : {stats['presence_only']:.4f}, "
                  f"Best possible Acc Salience : {stats['salience_only']:.4f}")

            # Log validation metrics
            if writer:
                writer.add_scalar("Loss/val", stats['val_loss'], epoch)
                writer.add_scalar("Accuracy/presence", stats['acc_presence'], epoch)
                writer.add_scalar("Accuracy/salience", stats['acc_salience'], epoch)
                writer.add_scalar("Alpha", stats['alpha'], epoch)
                writer.add_scalar("Beta", stats['beta'], epoch)
                writer.add_scalar("Best possible Accuracy/presence", stats['presence_only'], epoch)
                writer.add_scalar("Best possible Accuracy/salience", stats['salience_only'], epoch)

            results.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": stats['val_loss'] if self.valid_data_loader else None,
                "best_alpha": stats['alpha'] if self.valid_data_loader else None,
                "best_beta": stats['beta'] if self.valid_data_loader else None,
                "best_acc_presence": stats['acc_presence'] if self.valid_data_loader else None,
                "best_acc_salience": stats['acc_salience'] if self.valid_data_loader else None
            })

        return results


