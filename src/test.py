from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from typing import List

import torch.nn as nn
import torch
import os


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class LSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        return self.fc(output)


def train(
    X_sequence: List[List[List[float]]],
    checkpoint_dir: str,
    epochs: int = 8,
    batch_size: int = 8,
    device: torch.device = torch.device("cpu")
) -> LSTM:
    os.makedirs(checkpoint_dir, exist_ok=True)
    model = LSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        count = 0

        torch.random.manual_seed(epoch)
        batches = [X_sequence[i:i+batch_size]
                   for i in range(0, len(X_sequence), batch_size)]

        for batch in batches:
            sequences = [torch.tensor(seq, dtype=torch.float32)
                         for seq in batch if len(seq) > 2]
            lengths = [len(seq) - 1 for seq in sequences]
            if not sequences:
                continue

            inputs = [seq[:-1] for seq in sequences]
            targets = [seq[1:] for seq in sequences]

            padded_inputs = pad_sequence(inputs, batch_first=True).to(device)
            padded_targets = pad_sequence(targets, batch_first=True).to(device)

            output = model(padded_inputs, lengths)
            loss = criterion(output, padded_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg_loss = total_loss / count if count else float("inf")
        print(f"Epoch {epoch + 1} / {epochs} | Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(
                checkpoint_dir, "best.pt"))
            print("\u2705 Saved new best model")

    return model


def load_model(
    checkpoint_dir: str,
    input_size: int = 6,
    hidden_size: int = 64,
    num_layers: int = 2,
    device: torch.device = torch.device("cpu")
) -> LSTM:
    model = LSTM(input_size, hidden_size, num_layers).to(device)
    model.load_state_dict(torch.load(checkpoint_dir, map_location=device))
    model.eval()
    return model


def genrate_sequence(
    model: LSTM,
    seed: List[List[float]],
    max_len: int = 100,
    device: torch.device = torch.device("cpu")
) -> List[List[float]]:
    model.eval()
    generated = torch.tensor([seed], dtype=torch.float32).to(device)

    with torch.no_grad():
        for _ in range(max_len - len(seed)):
            length = [generated.shape[1]]
            out = model(generated, length)
            next_step = out[:, -1:, :]
            generated = torch.cat([generated, next_step], dim=1)

    return generated.squeeze(0).cpu().tolist()
