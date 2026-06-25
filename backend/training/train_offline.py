import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from backend.models.cnn_bilstm import CNNBiLSTM

def load_data(processed_dir):
    """
    Load combined windows from all four client partitions across all subsets.
    Excludes fault windows, as they are not standard RUL regression labels.
    """
    windows_files = glob.glob(os.path.join(processed_dir, '*_windows.npy'))
    labels_files = [f.replace('_windows.npy', '_labels.npy') for f in windows_files]
    
    all_windows = []
    all_labels = []
    
    for w_f, l_f in zip(windows_files, labels_files):
        if 'fault' in w_f:
            continue
        all_windows.append(np.load(w_f))
        all_labels.append(np.load(l_f))
        
    if not all_windows:
        raise ValueError(f"No valid numpy window files found in {processed_dir}")
        
    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    return X, y

def pinball_loss(y_true, y_pred, under_weight, over_weight):
    """
    Custom pinball/quantile loss.
    Err = y_true - y_pred.
    If Err > 0 (underestimation), penalty is weighted by under_weight.
    If Err < 0 (overestimation), penalty is weighted by over_weight.
    """
    err = y_true - y_pred
    loss = torch.where(err >= 0, under_weight * err, over_weight * -err)
    return loss.mean()

def train():
    processed_dir = 'backend/data/processed'
    saved_model_path = 'backend/models/saved/global_model.pth'
    os.makedirs(os.path.dirname(saved_model_path), exist_ok=True)
    
    print("Loading data...")
    X, y = load_data(processed_dir)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # Split 80% Train, 20% Validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = CNNBiLSTM(num_features=12).to(device)
    
    # Adam optimizer (lr=0.001, weight_decay=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # ReduceLROnPlateau (factor=0.5, patience=10)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    num_epochs = 100
    early_stopping_patience = 15
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    mae_criterion = nn.L1Loss()
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            out_10, out_50, out_90 = model(batch_X)
            
            # 10th Percentile Head: Penalize underestimation 9x more than overestimation.
            loss_10 = pinball_loss(batch_y, out_10, under_weight=0.9, over_weight=0.1)
            # 50th Percentile Head: Standard MAE.
            loss_50 = mae_criterion(out_50, batch_y)
            # 90th Percentile Head: Penalize overestimation 9x more than underestimation.
            loss_90 = pinball_loss(batch_y, out_90, under_weight=0.1, over_weight=0.9)
            
            # Total Loss = Sum of all 3 head losses.
            loss = loss_10 + loss_50 + loss_90
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                out_10, out_50, out_90 = model(batch_X)
                
                loss_10 = pinball_loss(batch_y, out_10, under_weight=0.9, over_weight=0.1)
                loss_50 = mae_criterion(out_50, batch_y)
                loss_90 = pinball_loss(batch_y, out_90, under_weight=0.1, over_weight=0.9)
                
                loss = loss_10 + loss_50 + loss_90
                val_loss += loss.item() * batch_X.size(0)
                
        val_loss /= len(val_loader.dataset)
        
        # Step the learning rate scheduler
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1:03d}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
        
        # Early Stopping Logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), saved_model_path)
            print(f" => Best model saved to {saved_model_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

if __name__ == "__main__":
    train()
