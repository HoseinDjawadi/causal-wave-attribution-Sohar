# ----------------------------------------------------------------------------
# SOHAR PORT CAUSAL ML PROJECT
#
# SCRIPT: 2a_train_evaluate_mlr.py
#
# DESCRIPTION:
# This script trains and evaluates the Multiple Linear Regression (MLR)
# baseline model. This model serves as the primary performance benchmark,
# representing a traditional, physics-inspired statistical approach.
#
# METHODOLOGY:
# 1.  Loads the pre-processed and scaled training and testing data specific
#     to the MLR model.
# 2.  Instantiates and trains a LinearRegression model from scikit-learn.
# 3.  Generates predictions on the unseen test set.
# 4.  Calculates a comprehensive set of performance metrics:
#     - Root Mean Squared Error (RMSE)
#     - Mean Absolute Error (MAE)
#     - Mean Absolute Percentage Error (MAPE)
#     - R-squared (R²)
#     - Pearson's correlation coefficient (r)
# 5.  Saves the following outputs to the `3_outputs/mlr_model/` directory:
#     - A CSV file comparing actual vs. predicted values for later analysis
#       and plotting.
#     - A JSON file containing the calculated performance metrics.
#
# USAGE:
# Ensure the input paths point to the correct processed data files.
# Run the script from your terminal: `python 2a_train_evaluate_mlr.py`
# ----------------------------------------------------------------------------

import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- 1. CONFIGURATION AND PATHS ---
# Use placeholder paths. Update these to your actual file locations.
PROCESSED_DATA_DIR = '/home/artorias/Desktop/Sohar_Causal_Modeling/Data/Processed/Features_MLR'
OUTPUT_DIR = '/home/artorias/Desktop/Sohar_Causal_Modeling/Output/MLR'
MODEL_NAME = 'mlr'

# --- 2. HELPER FUNCTIONS ---

def calculate_mape(y_true, y_pred):
    """
    Calculates the Mean Absolute Percentage Error (MAPE).
    Handles cases where y_true is zero to avoid division by zero errors.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Filter out zero values in y_true to prevent division by zero
    non_zero_mask = y_true != 0
    if not np.any(non_zero_mask):
        return np.inf  # Or 0, depending on convention when all true values are zero
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100

def calculate_pearson_r(y_true, y_pred):
    """
    Calculates the Pearson correlation coefficient (r).
    """
    return np.corrcoef(y_true, y_pred)[0, 1]


# --- 3. MAIN EXECUTION BLOCK ---

def main():
    """Main function to run the MLR model training and evaluation pipeline."""
    print(f"--- Starting MLR Baseline Model Pipeline ---")

    # --- 3.1. Load Data ---
    print("Loading pre-processed MLR data...")
    try:
        X_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{MODEL_NAME}_X_train.csv'), index_col='time', parse_dates=True)
        y_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{MODEL_NAME}_y_train.csv'), index_col='time', parse_dates=True)
        X_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{MODEL_NAME}_X_test.csv'), index_col='time', parse_dates=True)
        y_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, f'{MODEL_NAME}_y_test.csv'), index_col='time', parse_dates=True)
        print("Data loaded successfully.")
    except FileNotFoundError as e:
        print(f"FATAL ERROR: Data file not found. {e}")
        print("Please ensure you have run the '1_data_preparation.py' script first.")
        return

    # --- 3.2. Model Training ---
    print("Training the Linear Regression model...")
    mlr_model = LinearRegression()
    mlr_model.fit(X_train, y_train)
    print("Model training complete.")

    # --- 3.3. Prediction ---
    print("Generating predictions on the test set...")
    y_pred = mlr_model.predict(X_test)
    # The output of predict is a numpy array, flatten it for metrics calculation
    y_pred_flat = y_pred.flatten()
    y_test_flat = y_test.values.flatten()
    print("Prediction complete.")

    # --- 3.4. Performance Evaluation ---
    print("Calculating performance metrics...")
    rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_flat))
    mae = mean_absolute_error(y_test_flat, y_pred_flat)
    mape = calculate_mape(y_test_flat, y_pred_flat)
    r2 = r2_score(y_test_flat, y_pred_flat)
    pearson_r = calculate_pearson_r(y_test_flat, y_pred_flat)

    performance_metrics = {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE (%)': mape,
        'R_squared': r2,
        'Pearson_r': pearson_r
    }

    print("\n--- MLR Model Performance ---")
    for metric, value in performance_metrics.items():
        print(f"{metric}: {value:.4f}")
    print("---------------------------\n")

    # --- 3.5. Save Outputs ---
    print("Saving outputs...")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Save performance metrics to a JSON file
    metrics_path = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_performance_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(performance_metrics, f, indent=4)
    print(f"Performance metrics saved to: {metrics_path}")

    # Save predictions vs. actuals to a CSV file
    predictions_df = pd.DataFrame({
        'actual_Hm0_buoy': y_test_flat,
        'predicted_Hm0_buoy': y_pred_flat
    }, index=y_test.index)
    predictions_path = os.path.join(OUTPUT_DIR, f'{MODEL_NAME}_predictions_vs_actuals.csv')
    predictions_df.to_csv(predictions_path)
    print(f"Predictions saved to: {predictions_path}")

    print("\n--- MLR Baseline Model Pipeline Finished Successfully ---")


if __name__ == "__main__":
    main()
