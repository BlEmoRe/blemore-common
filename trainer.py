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
        self.model.eval()
        total_loss = 0
        all_preds = []

        with torch.no_grad():
            for data, target in self.valid_data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output, loss = self.model(data, target)
                total_loss += loss.item()

                preds = torch.topk(output, 1)

                all_top2 = {
                    "values": preds[0].cpu().numpy(),
                    "indices": preds[1].cpu().numpy(),
                }

                all_preds.append(all_top2)

        all_preds = np.concatenate(all_preds, axis=0)
        val_filenames = self.valid_data_loader.dataset.filenames

        best_alpha, best_beta, best_acc_presence, best_acc_salience = grid_search_thresholds(val_filenames, all_preds)
        avg_loss = total_loss / len(self.valid_data_loader)
        return avg_loss, best_alpha, best_beta, best_acc_presence, best_acc_salience

    def train(self):
        results = []

        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            print(f"Epoch [{epoch + 1}/{self.epochs}], Train Loss: {train_loss:.4f}")

            if self.valid_data_loader and epoch % 10 == 0:
                val_loss, best_alpha, best_beta, best_acc_presence, best_acc_salience = self.validate()
                print(f"Epoch [{epoch + 1}/{self.epochs}], "
                      f"Validation Loss: {val_loss:.4f}, "
                      f"Best Alpha: {best_alpha:.4f}, "
                      f"Best Beta: {best_beta:.4f}, "
                      f"Best Acc Presence: {best_acc_presence:.4f}, "
                      f"Best Acc Salience: {best_acc_salience:.4f}")

            results.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss if self.valid_data_loader else None,
                "best_alpha": best_alpha if self.valid_data_loader else None,
                "best_beta": best_beta if self.valid_data_loader else None,
                "best_acc_presence": best_acc_presence if self.valid_data_loader else None,
                "best_acc_salience": best_acc_salience if self.valid_data_loader else None
            })

        return results


