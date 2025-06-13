import numpy as np
import torch
from pathlib import Path
from baseline_cnn import (
    SpectrogramDataset, BaselineCNN, EnhancedCNN, N_OUTPUTS, N_CONV_LAYERS, DROPOUT, SPECTROGRAMS_FOLDER, TARGETS_FILE, DEVICE, BATCH_SIZE
)
from ast_regressor import ASTFeatureDataset, ASTRegressor, FEATURES_FOLDER as AST_FEATURES_FOLDER

from torch.utils.data import DataLoader
import json

# Train and evaluate all models, generate predictions for the same test item

def train_and_predict():
    # --- Train Baseline CNN ---
    print("Training Baseline CNN...")
    train_dataset = SpectrogramDataset(SPECTROGRAMS_FOLDER / "train", TARGETS_FILE, augment=True)
    val_dataset = SpectrogramDataset(SPECTROGRAMS_FOLDER / "val", TARGETS_FILE)
    test_dataset = SpectrogramDataset(SPECTROGRAMS_FOLDER / "test", TARGETS_FILE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=1)
    baseline_model = BaselineCNN(N_OUTPUTS, n_conv_layers=N_CONV_LAYERS, dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(baseline_model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    baseline_model.train()
    for epoch in range(10):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = baseline_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    print("Baseline CNN training complete.")

    # --- Train Enhanced CNN ---
    print("Training Enhanced CNN...")
    enhanced_model = EnhancedCNN(N_OUTPUTS, dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(enhanced_model.parameters(), lr=1e-3)
    enhanced_model.train()
    for epoch in range(10):
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = enhanced_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    print("Enhanced CNN training complete.")

    # --- Train AST Regressor ---
    print("Training AST Regressor...")
    ast_train_dataset = ASTFeatureDataset(AST_FEATURES_FOLDER / "train", TARGETS_FILE)
    ast_val_dataset = ASTFeatureDataset(AST_FEATURES_FOLDER / "val", TARGETS_FILE)
    ast_test_dataset = ASTFeatureDataset(AST_FEATURES_FOLDER / "test", TARGETS_FILE)
    ast_train_loader = DataLoader(ast_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    ast_test_loader = DataLoader(ast_test_dataset, batch_size=1)
    sample_file = next((AST_FEATURES_FOLDER / "train").rglob("*_ast_cls.npy"))
    input_dim = np.load(sample_file).shape[0]
    ast_model = ASTRegressor(input_dim, N_OUTPUTS).to(DEVICE)
    optimizer = torch.optim.Adam(ast_model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    ast_model.train()
    for epoch in range(10):
        for features, targets in ast_train_loader:
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = ast_model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    print("AST Regressor training complete.")

    # --- Generate predictions for the same test item ---
    print("Generating predictions for the same test item...")
    # Use the first item from the test set
    item_spec, _ = next(iter(test_loader))
    item_spec = item_spec.to(DEVICE)
    baseline_model.eval()
    enhanced_model.eval()
    with torch.no_grad():
        baseline_pred = baseline_model(item_spec).cpu().numpy()[0]
        enhanced_pred = enhanced_model(item_spec).cpu().numpy()[0]
    # For AST, get the corresponding feature
    # Find the corresponding track name
    test_sample_path = test_loader.dataset.samples[0]
    track_name = Path(test_sample_path).parent.name
    ast_feat_path = AST_FEATURES_FOLDER / "test" / track_name / (Path(test_sample_path).stem + "_ast_cls.npy")
    ast_feat = np.load(ast_feat_path)
    ast_feat_tensor = torch.tensor(ast_feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    ast_model.eval()
    with torch.no_grad():
        ast_pred = ast_model(ast_feat_tensor).cpu().numpy()[0]
    # Save predictions
    np.save(SPECTROGRAMS_FOLDER.parent / "baseline_cnn_pred.npy", baseline_pred)
    np.save(SPECTROGRAMS_FOLDER.parent / "enhanced_cnn_pred.npy", enhanced_pred)
    np.save(SPECTROGRAMS_FOLDER.parent / "ast_pred.npy", ast_pred)
    print("Saved predictions for baseline CNN, enhanced CNN, and AST regressor for the same test item.")

if __name__ == "__main__":
    train_and_predict() 