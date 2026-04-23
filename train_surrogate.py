import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import mlflow
import matplotlib.pyplot as plt
import shap

# 1. Define Model Architecture
class BeamSurrogateMLP(nn.Module):
    def __init__(self):
        super(BeamSurrogateMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(6, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.model(x)

def calc_mape(y_true, y_pred):
    return torch.mean(torch.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

def train_model():
    print("Loading data...")
    train_df = pd.read_parquet("train_dataset.parquet")
    val_df = pd.read_parquet("val_dataset.parquet")
    test_df = pd.read_parquet("test_dataset.parquet")
    
    input_cols = ['L', 'b', 'h', 'F', 'E', 'rho']
    output_cols = ['max_deflection_m', 'max_stress_Pa', 'natural_freq_Hz']
    
    X_train, y_train = train_df[input_cols].values, train_df[output_cols].values
    X_val, y_val = val_df[input_cols].values, val_df[output_cols].values
    X_test, y_test = test_df[input_cols].values, test_df[output_cols].values
    
    # 2. Scale Data
    print("Scaling inputs and outputs...")
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_train_scaled = X_scaler.fit_transform(X_train)
    y_train_scaled = y_scaler.fit_transform(y_train)
    
    X_val_scaled = X_scaler.transform(X_val)
    y_val_scaled = y_scaler.transform(y_val)
    
    X_test_scaled = X_scaler.transform(X_test)
    y_test_scaled = y_scaler.transform(y_test)
    
    # Save scalers
    os.makedirs("models", exist_ok=True)
    joblib.dump(X_scaler, "models/X_scaler.pkl")
    joblib.dump(y_scaler, "models/y_scaler.pkl")
    
    # Create DataLoaders
    batch_size = 256
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), 
                                            torch.tensor(y_train_scaled, dtype=torch.float32)), 
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32), 
                                          torch.tensor(y_val_scaled, dtype=torch.float32)), 
                            batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), 
                                           torch.tensor(y_test_scaled, dtype=torch.float32)), 
                             batch_size=batch_size)
    
    # Setup MLflow
    mlflow.set_experiment("beam_surrogate")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = BeamSurrogateMLP().to(device)
    epochs = 300
    patience = 20
    learning_rate = 1e-3
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    train_losses = []
    val_losses = []
    
    with mlflow.start_run():
        mlflow.log_params({
            "model": "MLP_4_layer_BatchNorm_GELU",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "optimizer": "Adam",
            "scheduler": "CosineAnnealingLR"
        })
        
        print("Starting training...")
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
            
            train_loss /= len(train_loader.dataset)
            train_losses.append(train_loss)
            
            model.eval()
            val_loss = 0.0
            val_mape_agg = torch.zeros(3).to(device)
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    y_pred = model(X_batch)
                    val_loss += criterion(y_pred, y_batch).item() * X_batch.size(0)
                    
                    # Compute unscaled metrics
                    y_pred_unscaled = torch.tensor(y_scaler.inverse_transform(y_pred.cpu().numpy())).to(device)
                    y_batch_unscaled = torch.tensor(y_scaler.inverse_transform(y_batch.cpu().numpy())).to(device)
                    
                    val_mape_agg += torch.sum(torch.abs((y_batch_unscaled - y_pred_unscaled) / (y_batch_unscaled + 1e-8)), dim=0)
            
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)
            
            val_mape_agg = (val_mape_agg / len(val_loader.dataset)) * 100
            scheduler.step()
            
            # Log metrics
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_mape_deflection", val_mape_agg[0].item(), step=epoch)
            mlflow.log_metric("val_mape_stress", val_mape_agg[1].item(), step=epoch)
            mlflow.log_metric("val_mape_freq", val_mape_agg[2].item(), step=epoch)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAPEs: {val_mape_agg.cpu().numpy().round(2)}")
                
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                torch.save(model.state_dict(), "models/surrogate_model.pt")
            else:
                early_stop_counter += 1
                
            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
                
        # Run evaluations on test set
        model.load_state_dict(torch.load("models/surrogate_model.pt"))
        model.eval()
        
        test_preds_scaled = []
        test_true_scaled = []
        start_time = time.time()
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                preds = model(X_batch)
                test_preds_scaled.append(preds.cpu().numpy())
                test_true_scaled.append(y_batch.numpy())
        
        infer_time = time.time() - start_time
        print(f"Inference time for {len(test_loader.dataset)} samples: {infer_time:.4f}s")
                
        test_preds_scaled = np.vstack(test_preds_scaled)
        test_true_scaled = np.vstack(test_true_scaled)
        
        test_preds = y_scaler.inverse_transform(test_preds_scaled)
        test_true = y_scaler.inverse_transform(test_true_scaled)
        
        test_mape = np.mean(np.abs((test_true - test_preds) / (test_true + 1e-8)), axis=0) * 100
        print(f"Test MAPE: Deflection: {test_mape[0]:.2f}%, Stress: {test_mape[1]:.2f}%, Freq: {test_mape[2]:.2f}%")
        
        mlflow.log_metric("test_mape_deflection", test_mape[0])
        mlflow.log_metric("test_mape_stress", test_mape[1])
        mlflow.log_metric("test_mape_freq", test_mape[2])
        mlflow.log_artifact("models/surrogate_model.pt")
        mlflow.log_artifact("models/X_scaler.pkl")
        mlflow.log_artifact("models/y_scaler.pkl")
        
    # Plot losses
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.yscale('log')
    plt.legend()
    plt.title("Learning Curves")
    plt.savefig("figures/learning_curves.png")
    
    # Parity Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ['Max Deflection (m)', 'Max Stress (Pa)', 'Natural Frequency (Hz)']
    
    for i in range(3):
        axes[i].scatter(test_true[:, i], test_preds[:, i], alpha=0.3, s=5)
        # diagonal line
        min_v = min(np.min(test_true[:, i]), np.min(test_preds[:, i]))
        max_v = max(np.max(test_true[:, i]), np.max(test_preds[:, i]))
        axes[i].plot([min_v, max_v], [min_v, max_v], 'r--')
        
        if i < 2:  # log scale for deflection and stress parity plots
            axes[i].set_xscale('log')
            axes[i].set_yscale('log')
            
        axes[i].set_title(titles[i])
        axes[i].set_xlabel("Ground Truth")
        axes[i].set_ylabel("Predicted")
        
    plt.tight_layout()
    plt.savefig("figures/parity_plots.png")
    
    # SHAP Analysis
    print("Running SHAP feature importance...")
    # Use a background set of 100 samples
    background = torch.tensor(X_train_scaled[:100], dtype=torch.float32).to(device)
    explainer = shap.DeepExplainer(model, background)
    
    # Calculate SHAP values for 500 test samples
    test_samples = torch.tensor(X_test_scaled[:500], dtype=torch.float32).to(device)
    shap_values = explainer.shap_values(test_samples)
    
    print(f"SHAP values type: {type(shap_values)}")
    # SHAP values returns a list of arrays (one for each output) or a 3D array
    for i, title in enumerate(['Deflection', 'Stress', 'Frequency']):
        plt.figure()
        if isinstance(shap_values, list):
            sv = shap_values[i]
        else: # it might be 3D array (samples, outputs, features) or (samples, features, outputs)
            if shap_values.ndim == 3:
                if shap_values.shape[-1] == 3:
                    sv = shap_values[:, :, i]
                else:
                    sv = shap_values[:, i, :]
            else:
                sv = shap_values
                
        shap.summary_plot(sv, test_samples.cpu().numpy(), feature_names=input_cols, show=False)
        plt.title(f"SHAP Summary - {title}")
        plt.tight_layout()
        plt.savefig(f"figures/shap_summary_{title.lower()}.png")
        plt.close()
        
    print("Done! All figures saved to figures/")

if __name__ == "__main__":
    train_model()
