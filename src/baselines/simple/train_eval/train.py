import torch
import os

from config import ROOT_DIR


class Trainer:

    default_path = os.path.join(ROOT_DIR, "data/baselines/simple/multi_label_nn.pth")

    def __init__(self, model, optimizer, path=default_path):
        self.model = model
        self.optimizer = optimizer
        self.path = path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model.to(self.device)


    def train(self, train_loader, epochs=100):
        for epoch in range(epochs):

            self.model.train()
            total_loss = 0

            for X_batch, y_batch in train_loader:

                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs, loss = self.model(X_batch, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")

    def predict(self, test_loader):
        self.model.eval()
        y_pred_list = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_pred, _ = self.model(X_batch)
                y_pred_list.append(y_pred.cpu())

        y_pred_tensor = torch.cat(y_pred_list, dim=0)
        return y_pred_tensor.numpy()

    def save_model(self):
        torch.save(self.model.state_dict(), self.path)
        print(f"Model saved at {self.path}")

    def load_model(self):
        self.model.load_state_dict(torch.load(self.path))
        self.model.to(self.device)
        print("Model loaded successfully!")

