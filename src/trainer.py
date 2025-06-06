from typing import List
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch
import os


class LSTM(nn.Module):
    def __init__(
            self,
            input_size: int = 5,
            hidden_size: int = 128,
            num_layers: int = 2
    ) -> None:
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def train(
        X: List[List[List[float]]],
        y: List[List[float]],
        checkpoint_dir: str,
        epochs: int = 10,
        batch_size: int = 32
) -> LSTM:
    os.makedirs(checkpoint_dir, exist_ok=True)

    model = LSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss = torch.tensor(0.0)
    best_loss = float("inf")
    for epoch in range(epochs):
        current_loss = 0.0
        batch_count = 0
        for batch_x, batch_y in loader:
            output = model(batch_x)
            loss = criterion(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_loss += loss.item()
            batch_count += 1
        epoch_loss = current_loss / \
            batch_count if batch_count > 0 else float("inf")
        print(f"Epochs: {epoch+1}/{epochs} | Loss: {loss.item():.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = os.path.join(checkpoint_dir, "best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"Saved new model at epoch {
                  epoch + 1}. Loss: {best_loss:.6f}")

        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"checkpoint_epoch_{epoch}+1.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint at epoch {epoch + 1}")

    return model
