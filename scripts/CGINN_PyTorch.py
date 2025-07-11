# ----------------------------------------------------------------------------
# SOHAR PORT CAUSAL ML PROJECT
#
# SCRIPT: 3b_train_evaluate_cginn_pytorch.py
#
# DESCRIPTION:
# This script is a PyTorch implementation of the Causal Graph-Informed
# Neural Network (CGINN). It replicates the exact same rigorous workflow as the
# TensorFlow/Keras version for direct comparison.
#
# METHODOLOGY:
# 1.  Uses PyTorch to define the neural network architecture.
# 2.  Uses Optuna and TimeSeriesSplit for hyperparameter tuning.
# 3.  Implements a custom training loop with manual early stopping and dropout
#     layers to prevent overfitting.
# 4.  Trains a final model on the full training set using the best parameters.
# 5.  Evaluates the final model on both train and test sets.
# 6.  Performs a SHAP analysis for model explainability.
# 7.  Saves all artifacts systematically.
#
# USAGE:
# Ensure required libraries (torch, optuna, shap, graphviz) are installed.
# Run from terminal: `python 3b_train_evaluate_cginn_pytorch.py`
# ----------------------------------------------------------------------------

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import optuna
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from graphviz import Digraph

# --- 1. CONFIGURATION AND PATHS ---
PROCESSED_DATA_DIR = '/home/artorias/Desktop/Sohar_Causal_Modeling/Data/Processed/Features_CGINN'
OUTPUT_DIR = '/home/artorias/Desktop/Sohar_Causal_Modeling/Output/CGIN_PyTorch'
MODEL_NAME = 'cgin_pytorch'

N_TRIALS = 100
N_SPLITS_CV = 5
DEBUG_MODE = False
DEBUG_DATA_FRACTION = 0.05

TARGET_VARIABLE = 'Hm0_buoy'
CAUSAL_PARENTS = [
    'WindSpeed_buoy', 'Effective_VHM0', 'Effective_VHM0_lag3h', 'Effective_VTPK',
    'Effective_VTPK_lag3h', 'Effective_VTPK_lag6h', 'Path_Mean_Slope_m_per_km',
    'Effective_WindDirection', 'Path_Rugosity_m', 'Path_Rugosity_m_lag3h',
    'Shoaling_Factor_Proxy_lag3h', 'Tp_buoy', 'Tp_buoy_lag6h', 'Tp_buoy_lag24h'
]

# --- 2. HELPER FUNCTIONS (METRICS, PLOTTING, ETC.) ---
# These are identical to previous scripts. Omitted for brevity, but should be included.
def evaluate_performance(y_true, y_pred, set_name):
    y_true_np, y_pred_np = np.array(y_true).flatten(), np.array(y_pred).flatten()
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true_np, y_pred_np)),
        'MAE': mean_absolute_error(y_true_np, y_pred_np),
        'R_squared': r2_score(y_true_np, y_pred_np)
    }
    print(f"\n--- {set_name} Set Performance ---")
    for metric, value in metrics.items(): print(f"{metric}: {value:.4f}")
    return metrics

def plot_causal_graph(parents, target, output_path):
    dot = Digraph(comment='Causal Graph', engine='dot')
    dot.attr('node', shape='ellipse', style='filled', color='skyblue')
    dot.attr(rankdir='LR', size='8,5')
    dot.node(target, shape='doubleoctagon', color='lightcoral')
    for parent in parents:
        dot.node(parent)
        dot.edge(parent, target)
    dot.render(output_path, format='png', view=False, cleanup=True)
    print(f"Saved causal graph visualization to: {output_path}.png")

def plot_training_history(train_loss, val_loss, output_path):
    """Plots the training and validation loss over epochs."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_loss, label='Training Loss', color='dodgerblue', lw=2)
    ax.plot(val_loss, label='Validation Loss', color='orangered', lw=2)
    ax.set_title('CGINN Model Training History', fontsize=16, weight='bold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved training history plot to: {output_path}")

def plot_predictions_vs_actuals(y_true, y_pred, set_name, output_path):
    """Generates and saves a scatter plot of predicted vs. actual values."""
    y_true_flat = np.array(y_true).flatten()
    y_pred_flat = np.array(y_pred).flatten()

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 8))
    max_val = max(y_true_flat.max(), y_pred_flat.max()) * 1.05
    min_val = min(y_true_flat.min(), y_pred_flat.min()) * 0.95
    ax.scatter(y_true_flat, y_pred_flat, alpha=0.5, s=20, label='Predictions', color='darkcyan')
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Line')
    ax.set_xlabel('Actual Significant Wave Height (m)', fontsize=12)
    ax.set_ylabel('Predicted Significant Wave Height (m)', fontsize=12)
    ax.set_title(f'CGINN: Predicted vs. Actual Values ({set_name} Set)', fontsize=14, weight='bold')
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved {set_name} scatter plot to: {output_path}")

def plot_timeseries_forecast(y_true_series, y_pred_series, set_name, output_path):
    """Generates and saves a time-series plot of predictions vs. actuals."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(y_true_series.index, y_true_series, label='Actual Values', color='dodgerblue', lw=2)
    ax.plot(y_pred_series.index, y_pred_series, label='Predicted Values', color='orangered', linestyle='--', lw=2)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Significant Wave Height (m)', fontsize=12)
    ax.set_title(f'CGINN: Time-Series Forecast ({set_name} Set)', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved {set_name} time-series plot to: {output_path}")

# --- 3. PYTORCH MODEL AND EARLY STOPPING ---

class CGINN_PyTorch(nn.Module):
    def __init__(self, trial, input_dim):
        super(CGINN_PyTorch, self).__init__()
        layers = []
        in_features = input_dim
        n_layers = trial.suggest_int('n_layers', 1, 3)
        for i in range(n_layers):
            out_features = trial.suggest_int(f'n_units_l{i}', 16, 128, log=True)
            layers.append(nn.Linear(in_features, out_features))
            layers.append(nn.ReLU())
            dropout_rate = trial.suggest_float(f'dropout_l{i}', 0.1, 0.5)
            layers.append(nn.Dropout(dropout_rate))
            in_features = out_features
        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class EarlyStopper:
    def __init__(self, patience=1, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# --- 4. OPTUNA OBJECTIVE FUNCTION ---

def objective(trial, X, y) -> float:
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
    val_scores = []

    for train_index, val_index in tscv.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Convert to PyTorch Tensors
        X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
        X_val_t = torch.tensor(X_val.values, dtype=torch.float32)
        y_val_t = torch.tensor(y_val.values, dtype=torch.float32)
        
        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=False)

        model = CGINN_PyTorch(trial, X_train.shape[1])
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        early_stopper = EarlyStopper(patience=15, min_delta=0.0001)
        
        # Initialize val_loss to handle linter warning
        val_loss = torch.tensor(float('inf'))

        for epoch in range(200):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t)
            
            if early_stopper.early_stop(val_loss):
                break
        
        val_scores.append(val_loss.item())

    return float(np.mean(val_scores))
# --- 5. SHAP ANALYSIS FUNCTION ---

def perform_shap_analysis(model, X_train, X_test, output_dir):
    print("\nPerforming SHAP analysis...")
    model.eval() # Set model to evaluation mode

    # SHAP requires a function that takes a numpy array and returns a numpy array
    def predict_fn(x):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            return model(x_tensor).numpy()

    background_data = shap.sample(X_train, 100)
    explainer = shap.KernelExplainer(predict_fn, background_data)
    
    test_sample = shap.sample(X_test, 200)
    shap_values = explainer.shap_values(test_sample)

    plt.figure()
    shap.summary_plot(shap_values, test_sample, plot_type="bar", show=False)
    shap_summary_path = os.path.join(output_dir, f'{MODEL_NAME}_shap_summary.png')
    plt.savefig(shap_summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SHAP summary plot saved to: {shap_summary_path}")


# --- 6. MAIN EXECUTION BLOCK ---

def main():
    print("--- Starting CGINN (PyTorch) Pipeline ---")
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    plot_causal_graph(CAUSAL_PARENTS, TARGET_VARIABLE, os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_causal_graph'))

    # Load data
    try:
        X_train_full = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'cgin_X_train.csv'), index_col='time', parse_dates=True)
        y_train_full = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'cgin_y_train.csv'), index_col='time', parse_dates=True)
        X_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'cgin_X_test.csv'), index_col='time', parse_dates=True)
        y_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'cgin_y_test.csv'), index_col='time', parse_dates=True)
    except FileNotFoundError as e:
        print(f"FATAL ERROR: Data file not found. {e}"); return

    if DEBUG_MODE:
        train_size = int(len(X_train_full) * DEBUG_DATA_FRACTION)
        X_train_full, y_train_full = X_train_full.iloc[:train_size], y_train_full.iloc[:train_size]
    
    print(f"Training set size: {len(X_train_full)}, Test set size: {len(X_test)}")

    # Hyperparameter Tuning
    print(f"\nStarting hyperparameter tuning with Optuna ({N_TRIALS} trials)...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train_full, y_train_full), n_trials=N_TRIALS)

    print("\nTuning complete.")
    print(f"Best trial validation loss (MSE): {study.best_value:.4f}")
    
    best_params_path = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_best_hyperparameters.json')
    with open(best_params_path, 'w') as f: json.dump(study.best_params, f, indent=4)
    print(f"Best hyperparameters saved to: {best_params_path}")

# --- Train Final Model ---
    print("\nTraining final model on the full training set with best hyperparameters...")
    final_model = CGINN_PyTorch(study.best_trial, X_train_full.shape[1])
    
    best_lr = study.best_params['learning_rate']
    optimizer = optim.Adam(final_model.parameters(), lr=best_lr)
    criterion = nn.MSELoss()

    X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
        X_train_full, y_train_full, test_size=0.2, shuffle=False
    )

    X_train_sub_t = torch.tensor(X_train_sub.values, dtype=torch.float32)
    y_train_sub_t = torch.tensor(y_train_sub.values, dtype=torch.float32)
    X_val_sub_t = torch.tensor(X_val_sub.values, dtype=torch.float32)
    y_val_sub_t = torch.tensor(y_val_sub.values, dtype=torch.float32)
    
    train_loader = DataLoader(TensorDataset(X_train_sub_t, y_train_sub_t), batch_size=32, shuffle=False)
    early_stopper = EarlyStopper(patience=20, min_delta=0.0001)
    
    # Lists to store loss history for plotting
    train_loss_history = []
    val_loss_history = []

    for epoch in range(500):
        final_model.train()
        epoch_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = final_model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        train_loss_history.append(epoch_train_loss / len(train_loader))
        
        final_model.eval()
        with torch.no_grad():
            val_outputs = final_model(X_val_sub_t)
            val_loss = criterion(val_outputs, y_val_sub_t)
            val_loss_history.append(val_loss.item())
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Validation Loss: {val_loss.item():.6f}")

        if early_stopper.early_stop(val_loss):
            print(f"Early stopping triggered at epoch {epoch}.")
            break
    
    print("Final model training complete.")
    
    model_path = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_final_model.pth')
    torch.save(final_model.state_dict(), model_path)
    print(f"Final model saved to: {model_path}")

    # --- Final Evaluation ---
    print("\nEvaluating final model...")
    final_model.eval() # Ensure model is in evaluation mode
    
    # Convert full datasets to tensors for prediction
    X_train_full_t = torch.tensor(X_train_full.values, dtype=torch.float32)
    X_test_t = torch.tensor(X_test.values, dtype=torch.float32)

    with torch.no_grad():
        y_pred_train = final_model(X_train_full_t).numpy()
        y_pred_test = final_model(X_test_t).numpy()

    train_metrics = evaluate_performance(y_train_full, y_pred_train, "Final Train")
    test_metrics = evaluate_performance(y_test, y_pred_test, "Final Test")

    # Save metrics and predictions
    all_metrics = {'train_metrics': train_metrics, 'test_metrics': test_metrics}
    metrics_path = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_performance_metrics.json')
    with open(metrics_path, 'w') as f: json.dump(all_metrics, f, indent=4)
    
    predictions_df = pd.DataFrame({
        'actual_Hm0_buoy': y_test.values.flatten(), 
        'predicted_Hm0_buoy': y_pred_test.flatten()
    }, index=y_test.index)
    predictions_path = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_predictions_vs_actuals.csv')
    predictions_df.to_csv(predictions_path)

# --- Generate Plots ---
    print("\nGenerating final plots...")
    # Training History
    plot_training_history(train_loss_history, val_loss_history, os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_training_history.png'))

    # Scatter Plots
    plot_predictions_vs_actuals(y_train_full, y_pred_train, "Train", os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_scatter_train.png'))
    plot_predictions_vs_actuals(y_test, y_pred_test, "Test", os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_scatter_test.png'))

    # Time-Series Plot
    y_pred_test_series = pd.Series(y_pred_test.flatten(), index=y_test.index)
    plot_timeseries_forecast(y_test, y_pred_test_series, "Test", os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_timeseries_test.png'))

    # --- SHAP Analysis ---
    perform_shap_analysis(final_model, X_train_full, X_test, OUTPUT_DIR)
    
    print("\n--- CGINN (PyTorch) Pipeline Finished ---")
    print("NOTE: Final training loop and evaluation are simplified for this example.")

if __name__ == "__main__":
    main()
