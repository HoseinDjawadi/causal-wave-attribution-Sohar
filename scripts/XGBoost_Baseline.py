# ----------------------------------------------------------------------------
# SOHAR PORT CAUSAL ML PROJECT
#
# SCRIPT: 2b_train_evaluate_xgb_rigorous.py
#
# DESCRIPTION:
# This script implements a scientifically rigorous pipeline for the XGBoost
# baseline model. It incorporates automated hyperparameter tuning with Optuna,
# time-series cross-validation to prevent data leakage, and early stopping
# to mitigate overfitting.
#
# METHODOLOGY:
# 1.  Implements a DEBUG_MODE flag to run on a small data subset for speed.
# 2.  Loads the pre-processed data for the XGBoost model.
# 3.  Defines an Optuna objective function to search for the best
#     hyperparameters.
# 4.  Inside the objective function, TimeSeriesSplit is used to create
#     chronologically-correct validation folds from the training data.
# 5.  For each trial, XGBoost is trained with early stopping on each fold to
#     find the optimal number of trees and prevent overfitting.
# 6.  The average validation RMSE across all folds is returned as the
#     objective to be minimized.
# 7.  After the best parameters are found, a final XGBoost model is trained
#     on the ENTIRE training set using these parameters.
# 8.  The final model is evaluated on both the train and the held-out test
#     sets to assess performance and diagnose overfitting.
# 9.  All outputs (best parameters, metrics, predictions, plots) are saved.
#
# USAGE:
# Ensure the required libraries (optuna, xgboost) are installed.
# Run the script from your terminal: `python 2b_train_evaluate_xgb_rigorous.py`
# ----------------------------------------------------------------------------

import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- 1. CONFIGURATION AND PATHS ---
# Use placeholder paths. Update these to your actual file locations.
PROCESSED_DATA_DIR = '/home/artorias/Desktop/Sohar_Causal_Modeling/Data/Processed/Features_XGB'
OUTPUT_DIR = '/home/artorias/Desktop/Sohar_Causal_Modeling/Output/XGBoost'
MODEL_NAME = 'xgb'

# Optuna Configuration
N_TRIALS = 75  # Number of hyperparameter combinations to test. Increase for more thorough search.
N_SPLITS_CV = 5 # Number of folds for time-series cross-validation.

# Development flag for rapid testing
DEBUG_MODE = False  # Set to True to run on a small subset of data
DEBUG_DATA_FRACTION = 0.05

# --- 2. HELPER FUNCTIONS (METRICS AND PLOTTING) ---

def calculate_mape(y_true, y_pred):
    """Calculates the Mean Absolute Percentage Error (MAPE)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0
    if not np.any(non_zero_mask): return np.inf
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

def calculate_pearson_r(y_true, y_pred):
    """Calculates the Pearson correlation coefficient (r)."""
    y_true_flat = np.array(y_true).flatten()
    y_pred_flat = np.array(y_pred).flatten()
    return np.corrcoef(y_true_flat, y_pred_flat)[0, 1]

def evaluate_performance(y_true, y_pred, set_name):
    """Calculates and prints a dictionary of performance metrics."""
    y_true_flat = np.array(y_true).flatten()
    y_pred_flat = np.array(y_pred).flatten()
    
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    mape = calculate_mape(y_true_flat, y_pred_flat)
    r2 = r2_score(y_true_flat, y_pred_flat)
    pearson_r = calculate_pearson_r(y_true_flat, y_pred_flat)
    
    metrics = {'RMSE': rmse, 'MAE': mae, 'MAPE (%)': mape, 'R_squared': r2, 'Pearson_r': pearson_r}
    print(f"\n--- {set_name} Set Performance ---")
    for metric, value in metrics.items(): print(f"{metric}: {value:.4f}")
    print("---------------------------\n")
    return metrics

def plot_predictions_vs_actuals(y_true, y_pred, set_name, output_path):
    """Generates and saves a scatter plot of predicted vs. actual values."""
    y_true_flat = np.array(y_true).flatten()
    y_pred_flat = np.array(y_pred).flatten()

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 8))
    max_val, min_val = max(y_true_flat.max(), y_pred_flat.max()) * 1.05, min(y_true_flat.min(), y_pred_flat.min()) * 0.95
    ax.scatter(y_true_flat, y_pred_flat, alpha=0.5, s=20, label='Predictions')
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Line')
    ax.set_xlabel('Actual Significant Wave Height (m)', fontsize=12)
    ax.set_ylabel('Predicted Significant Wave Height (m)', fontsize=12)
    ax.set_title(f'XGBoost: Predicted vs. Actual Values ({set_name} Set)', fontsize=14, weight='bold')
    ax.set_xlim(min_val, max_val); ax.set_ylim(min_val, max_val)
    ax.legend(); ax.grid(True)
    plt.tight_layout(); plt.savefig(output_path, dpi=300); plt.close()
    print(f"Saved {set_name} scatter plot to: {output_path}")

def plot_timeseries_forecast(y_true_series, y_pred_series, set_name, output_path):
    """Generates and saves a time-series plot of predictions vs. actuals."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(y_true_series.index, y_true_series, label='Actual Values', color='dodgerblue', lw=2)
    ax.plot(y_pred_series.index, y_pred_series, label='Predicted Values', color='orangered', linestyle='--', lw=2)
    ax.set_xlabel('Date', fontsize=12); ax.set_ylabel('Significant Wave Height (m)', fontsize=12)
    ax.set_title(f'XGBoost: Time-Series Forecast ({set_name} Set)', fontsize=14, weight='bold')
    ax.legend(); ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout(); plt.savefig(output_path, dpi=300); plt.close()
    print(f"Saved {set_name} time-series plot to: {output_path}")


# --- 3. OPTUNA OBJECTIVE FUNCTION ---

def objective(trial, X, y) -> float:
    """
    Optuna objective function for hyperparameter tuning.
    This function defines the search space and evaluates a set of
    hyperparameters using time-series cross-validation.
    """
    param = {
        'objective': 'reg:squarederror',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'random_state': 42,
        'n_jobs': -1
    }

    tscv = TimeSeriesSplit(n_splits=N_SPLITS_CV)
    val_scores = []

    for train_index, val_index in tscv.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50, **param)
        
        # CORRECTED FIT METHOD: Pass early_stopping_rounds directly.
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)

        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        val_scores.append(rmse)

    # Corrected return type for Pylance
    return float(np.mean(val_scores))


# --- 4. MAIN EXECUTION BLOCK ---

def main():
    """Main function to run the full XGBoost pipeline."""
    print("--- Starting Rigorous XGBoost Baseline Model Pipeline ---")
    print(f"DEBUG MODE: {'ON' if DEBUG_MODE else 'OFF'}")

    # Load Data
    print("Loading pre-processed XGBoost data...")
    try:
        X_train_full = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{MODEL_NAME}_X_train.csv'), index_col='time', parse_dates=True)
        y_train_full = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{MODEL_NAME}_y_train.csv'), index_col='time', parse_dates=True)
        X_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{MODEL_NAME}_X_test.csv'), index_col='time', parse_dates=True)
        y_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{MODEL_NAME}_y_test.csv'), index_col='time', parse_dates=True)
    except FileNotFoundError as e:
        print(f"FATAL ERROR: Data file not found. {e}")
        return

    # --- DEBUG MODE LOGIC ---
    if DEBUG_MODE:
        print(f"Debug mode is ON. Using {DEBUG_DATA_FRACTION*100}% of the data.")
        train_size = int(len(X_train_full) * DEBUG_DATA_FRACTION)
        test_size = int(len(X_test) * DEBUG_DATA_FRACTION)
        X_train_full, y_train_full = X_train_full.iloc[:train_size], y_train_full.iloc[:train_size]
        X_test, y_test = X_test.iloc[:test_size], y_test.iloc[:test_size]
    
    print(f"Training set size: {len(X_train_full)}, Test set size: {len(X_test)}")

    # Hyperparameter Tuning with Optuna
    print(f"\nStarting hyperparameter tuning with Optuna ({N_TRIALS} trials)...")
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train_full, y_train_full), n_trials=N_TRIALS)

    print("\nTuning complete.")
    print(f"Best trial validation RMSE: {study.best_value:.4f}")
    print("Best hyperparameters found:")
    print(study.best_params)

    # Save best hyperparameters
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    best_params_path = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_best_hyperparameters.json')
    with open(best_params_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    print(f"Best hyperparameters saved to: {best_params_path}")

    # Train Final Model with Best Parameters
    print("\nTraining final model on the full training set with best hyperparameters...")
    final_params = study.best_params
    
    final_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=2000, early_stopping_rounds=50, random_state=42, n_jobs=-1, **final_params)

    X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
        X_train_full, y_train_full, test_size=0.2, shuffle=False
    )
    
    # CORRECTED FIT METHOD: Pass early_stopping_rounds directly.
    final_model.fit(X_train_sub, y_train_sub,
                    eval_set=[(X_val_sub, y_val_sub)],
                    verbose=False)
    print("Final model training complete.")

    # Final Evaluation
    print("\nEvaluating final model...")
    y_pred_train = final_model.predict(X_train_full)
    y_pred_test = final_model.predict(X_test)

    train_metrics = evaluate_performance(y_train_full, y_pred_train, "Final Train")
    test_metrics = evaluate_performance(y_test, y_pred_test, "Final Test")

    # Save all outputs
    all_metrics = {'train_metrics': train_metrics, 'test_metrics': test_metrics}
    metrics_path = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_performance_metrics.json')
    with open(metrics_path, 'w') as f: json.dump(all_metrics, f, indent=4)
    print(f"Final performance metrics saved to: {metrics_path}")

    predictions_df = pd.DataFrame({'actual_Hm0_buoy': y_test.values.flatten(), 'predicted_Hm0_buoy': y_pred_test}, index=y_test.index)
    predictions_path = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_predictions_vs_actuals.csv')
    predictions_df.to_csv(predictions_path)
    print(f"Test set predictions saved to: {predictions_path}")

    # Generate and Save Plots
    plot_predictions_vs_actuals(y_train_full, y_pred_train, "Train", os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_scatter_train.png'))
    plot_predictions_vs_actuals(y_test, y_pred_test, "Test", os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_scatter_test.png'))
    plot_timeseries_forecast(y_test, pd.Series(y_pred_test, index=y_test.index), "Test", os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_timeseries_test.png'))

    print("\n--- Rigorous XGBoost Baseline Model Pipeline Finished Successfully ---")

if __name__ == "__main__":
    main()
