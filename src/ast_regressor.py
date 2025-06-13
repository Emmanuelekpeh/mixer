import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from sklearn.metrics import mean_absolute_error

FEATURES_FOLDER = Path(__file__).resolve().parent.parent / "data" / "ast_features"
TARGETS_FILE = Path(__file__).resolve().parent.parent / "data" / "targets_generated.json"
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_OUTPUTS = 10

class ASTFeatureDataset(Dataset):
    def __init__(self, features_dir, targets_file, n_outputs=N_OUTPUTS):
        self.samples = []
        self.targets = json.load(open(targets_file))
        for track_dir in Path(features_dir).rglob("*_ast_cls.npy"):
            self.samples.append(track_dir)
        self.n_outputs = n_outputs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feat_path = self.samples[idx]
        feature = np.load(feat_path)
        # Extract track name for target lookup
        track_name = feat_path.parent.name
        target = self.targets.get(track_name, [0.0]*self.n_outputs)
        target = np.array(target, dtype=np.float32)
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

class ASTRegressor(nn.Module):
    def __init__(self, input_dim, n_outputs, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_outputs)
        )
    def forward(self, x):
        return self.net(x)

def evaluate(model, dataloader):
    model.eval()
    mse_losses = []
    mae_losses = []
    criterion = nn.MSELoss()
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for features, targets in dataloader:
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            outputs = model(features)
            mse_loss = criterion(outputs, targets).item()
            mae_loss = torch.mean(torch.abs(outputs - targets)).item()
            mse_losses.append(mse_loss)
            mae_losses.append(mae_loss)
            all_targets.append(targets.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())
    all_targets = np.concatenate(all_targets, axis=0)
    all_outputs = np.concatenate(all_outputs, axis=0)
    mae = mean_absolute_error(all_targets, all_outputs)
    return np.mean(mse_losses), np.mean(mae_losses), mae

def train():
    # Infer input dimension from one feature file
    sample_file = next((FEATURES_FOLDER / "train").rglob("*_ast_cls.npy"))
    input_dim = np.load(sample_file).shape[0]
    train_dataset = ASTFeatureDataset(FEATURES_FOLDER / "train", TARGETS_FILE)
    val_dataset = ASTFeatureDataset(FEATURES_FOLDER / "val", TARGETS_FILE)
    test_dataset = ASTFeatureDataset(FEATURES_FOLDER / "test", TARGETS_FILE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    model = ASTRegressor(input_dim, N_OUTPUTS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for features, targets in train_loader:
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        val_mse, val_mae, val_mae_skl = evaluate(model, val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {running_loss/len(train_loader):.4f}, Val MSE: {val_mse:.4f}, Val MAE: {val_mae:.4f}, Val MAE (sklearn): {val_mae_skl:.4f}")
    print("Training complete.")
    test_mse, test_mae, test_mae_skl = evaluate(model, test_loader)
    print(f"Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}, Test MAE (sklearn): {test_mae_skl:.4f}")

if __name__ == "__main__":
    train() 