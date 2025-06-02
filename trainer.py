import torch
import numpy as np

from post_processing import grid_search_thresholds


class Trainer(object):

    def __init__(self, model, optimizer, data_loader, epochs, valid_data_loader=None, subsample_aggregation=True):
        self.model = model
        self.optimizer = optimizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model.to(self.device)

        self.data_loader = data_loader
        self.epochs = epochs
        self.valid_data_loader = valid_data_loader
        self.subsample_aggregation = subsample_aggregation

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

                if loss is None:
                    print("Warning: Loss is None")
                    print("data", data.shape)
                    print("target", target.shape)
                    print("output", output.shape)

                total_loss += loss.item()

                all_preds.append(output.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        val_filenames = self.valid_data_loader.dataset.filenames

        if self.subsample_aggregation:
            # Aggregate predictions by video_id
            val_filenames, all_preds = self.aggregate_subsamples(val_filenames, all_preds)

        best_alpha, best_beta, best_acc_presence, best_acc_salience = grid_search_thresholds(val_filenames, all_preds)
        avg_loss = total_loss / len(self.valid_data_loader)
        return avg_loss, best_alpha, best_beta, best_acc_presence, best_acc_salience

    def aggregate_subsamples(self, all_video_ids, all_preds):
        # New aggregation by video_id
        video_pred_dict = {}
        for video_id, pred in zip(all_video_ids, all_preds):
            if video_id not in video_pred_dict:
                video_pred_dict[video_id] = []
            video_pred_dict[video_id].append(pred)

        aggregated_preds = []
        aggregated_video_ids = []
        for video_id, preds in video_pred_dict.items():
            preds = np.stack(preds, axis=0)  # (num_subsamples, num_classes)
            avg_preds = np.mean(preds, axis=0)  # Average over subsamples
            aggregated_preds.append(avg_preds)
            aggregated_video_ids.append(video_id)

        aggregated_preds = np.stack(aggregated_preds, axis=0)
        aggregated_video_ids = np.array(aggregated_video_ids)
        return aggregated_video_ids, aggregated_preds

    def train(self, writer=None):
        results = []

        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            print(f"Epoch [{epoch + 1}/{self.epochs}], Train Loss: {train_loss:.4f}")

            # Log training loss
            if writer:
                writer.add_scalar("Loss/train", train_loss, epoch)

            val_loss, best_alpha, best_beta, best_acc_presence, best_acc_salience = self.validate()
            print(f"Epoch [{epoch + 1}/{self.epochs}], "
                  f"Validation Loss: {val_loss:.4f}, "
                  f"Best Alpha: {best_alpha:.4f}, "
                  f"Best Beta: {best_beta:.4f}, "
                  f"Best Acc Presence: {best_acc_presence:.4f}, "
                  f"Best Acc Salience: {best_acc_salience:.4f}")

            # Log validation metrics
            if writer:
                writer.add_scalar("Loss/val", val_loss, epoch)
                writer.add_scalar("Accuracy/presence", best_acc_presence, epoch)
                writer.add_scalar("Accuracy/salience", best_acc_salience, epoch)
                writer.add_scalar("Alpha", best_alpha, epoch)
                writer.add_scalar("Beta", best_beta, epoch)

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
