# ----------------------------------------------------------------------------
# SOHAR PORT CAUSAL ML PROJECT
#
# SCRIPT: 1_data_preparation.py
#
# DESCRIPTION:
# This script performs the crucial data preparation steps for the causal
# machine learning modeling pipeline. It is designed to be executed once to
# create the necessary training and testing datasets for all models, ensuring
# a consistent and leak-proof foundation.
#
# METHODOLOGY:
# 1.  Loads the final, time-harmonized master dataset.
# 2.  Defines the specific feature sets for each of the three models:
#     - Causal Graph-Informed Neural Network (CGINN)
#     - XGBoost Baseline ("Black-Box")
#     - Multiple Linear Regression Baseline ("Physics-Inspired")
# 3.  Performs a strict 70/30 temporal split to create training and test sets.
#     This prevents data leakage by ensuring the model is tested on data that
#     occurs chronologically after the training data.
# 4.  Applies StandardScaler to the features. The scaler is fit *only* on the
#     training data and then applied to both training and test sets.
# 5.  Saves the processed data (X_train, y_train, X_test, y_test) for each
#     model into a dedicated output directory for easy access in subsequent
#     modeling scripts.
#
# USAGE:
# Ensure the MASTER_DATA_PATH points to your final, unified dataset.
# Run the script from your terminal: `python 1_data_preparation.py`
# ----------------------------------------------------------------------------

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 1. CONFIGURATION AND PATHS ---
# Use placeholder paths. Update these to your actual file locations.
MASTER_DATA_PATH = '/home/artorias/Desktop/Sohar_Causal_Modeling/Data/Raw/master_analysis_dataset.csv'
PROCESSED_DATA_DIR = '/home/artorias/Desktop/Sohar_Causal_Modeling/Data/Processed'
SPLIT_RATIO = 0.3 # 30% of data for testing
TARGET_VARIABLE = 'Hm0_buoy'

# --- 2. DEFINE MODEL-SPECIFIC FEATURE SETS ---
# This section translates our research plan into concrete feature lists.

# The target variable is the same for all models
TARGET = [TARGET_VARIABLE]

# 2.1. Causal Graph-Informed Neural Network (CGINN)
# Features are strictly the direct causal parents of Hm0_buoy identified by PCMCI+.
# IMPORTANT: Update this list based on your final `significant_causal_links.csv`.
# The list below is a placeholder based on our previous discussion.
FEATURES_CGINN = [
    'WindSpeed_buoy',
    'Effective_VHM0',
    'Effective_VHM0_lag3h',
    'Effective_VTPK',
    'Effective_VTPK_lag3h',
    'Effective_VTPK_lag6h',
    'Path_Mean_Slope_m_per_km',
    'Effective_WindDirection',
    'Path_Rugosity_m',
    'Path_Rugosity_m_lag3h',
    'Shoaling_Factor_Proxy_lag3h',
    'Tp_buoy',
    'Tp_buoy_lag6h',
    'Tp_buoy_lag24h'
    # Add any other direct parent variables identified by the causal analysis
    # e.g., 'Some_Other_Parent_at_Lag_X'
]

# 2.2. XGBoost Baseline ("Black-Box")
# This model gets all potentially relevant variables to test predictive power
# without causal constraints.
# IMPORTANT: This list should be comprehensive. Exclude only the target variable
# and identifiers or variables that would cause perfect multicollinearity.
FEATURES_XGB = [
    'Effective_VHM0', 'Effective_VTPK', 'Effective_VMDR', # Offshore forcing
#    'Tp_buoy', 'Mdir_buoy', 'WindSpeed_buoy','WindDirection_buoy',
#    'WaterTemp_buoy', 'Salinity_buoy', 'CurrSpd_buoy', 'CurrDir_buoy',  # Local forcing
    'Path_Mean_Slope_m_per_km', 'Path_Rugosity_m', 'Shoaling_Factor_Proxy', 'Path_Slope_StdDev', # Bathymetry
    'Wind_Wave_Angle_Diff', 'Shore_Normal_Angle', # Interaction features
    'Effective_ssh', 'Effective_msl', 'Effective_sst',
    'Effective_vtide', 'Effective_utide', 'Effective_vo', 'Effective_uo', 'Effective_northward_wind',
    'Effective_eastward_wind'
      # Other environmental factors
    # Add all other relevant columns from your master dataset
]

# 2.3. Multiple Linear Regression Baseline ("Physics-Inspired")
# A simple, traditional model based on general physical intuition.
FEATURES_MLR = [
    'Effective_VHM0',
    'WindSpeed_buoy'
]

# Dictionary to iterate through models
MODEL_FEATURES = {
    'cgin': FEATURES_CGINN,
    'xgb': FEATURES_XGB,
    'mlr': FEATURES_MLR
}

# --- 3. DATA PROCESSING FUNCTIONS ---

def load_and_prepare_data(filepath):
    """
    Loads the master dataset and prepares it for processing.
    """
    print(f"Loading data from: {filepath}")
    try:
        df = pd.read_csv(filepath, index_col='time', parse_dates=True)
        # Ensure data is sorted chronologically for the temporal split
        df.sort_index(inplace=True)
        print("Data loaded successfully.")

        # --- INSERT SNIPPET START ---
        # Create lagged features required for the CGINN model.
        # The data has a 3-hour frequency, so we shift by periods.
        # 3h lag = 1 period, 6h lag = 2 periods, 24h lag = 8 periods.
        print("Creating additional lagged features for CGINN...")
        df['Path_Rugosity_m_lag3h'] = df['Path_Rugosity_m'].shift(1)
        df['Shoaling_Factor_Proxy_lag3h'] = df['Shoaling_Factor_Proxy'].shift(1)
        df['Tp_buoy_lag6h'] = df['Tp_buoy'].shift(2)
        df['Tp_buoy_lag24h'] = df['Tp_buoy'].shift(8)
        # --- INSERT SNIPPET END ---

        return df
    except FileNotFoundError:
        print(f"FATAL ERROR: The file was not found at {filepath}")
        print("Please update the MASTER_DATA_PATH variable in the script.")
        return None

def split_scale_and_save(df, features, target, model_name, output_dir):
    """
    Performs the temporal split, scaling, and saving for a given model's dataset.

    Args:
        df (pandas.DataFrame): The full master dataframe.
        features (list): The list of feature columns for the model.
        target (list): The list containing the target variable name.
        model_name (str): The short name of the model (e.g., 'cgin').
        output_dir (str): The directory to save the processed files.
    """
    print(f"\n--- Processing data for model: {model_name.upper()} ---")

    # Select the relevant columns for this model
    model_df = df[features + target].copy()
    model_df.dropna(inplace=True) # Drop rows with any missing values in the selected columns

    X = model_df[features]
    y = model_df[target]

    # --- Temporal Split ---
    # We split based on the index, not randomly, to preserve time order.
    # `shuffle=False` is the key to preventing data leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=SPLIT_RATIO, shuffle=False
    )
    print(f"Temporal split complete. Training set size: {len(X_train)}, Test set size: {len(X_test)}")

    # --- Feature Scaling ---
    # Fit the scaler ONLY on the training data.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # Apply the same fitted scaler to the test data.
    X_test_scaled = scaler.transform(X_test)

    # Convert scaled arrays back to dataframes to retain column names and index
    X_train_scaled_df = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
    print("Feature scaling complete.")

    # --- Save Processed Data ---
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define file paths
    train_features_path = os.path.join(output_dir, f"{model_name}_X_train.csv")
    test_features_path = os.path.join(output_dir, f"{model_name}_X_test.csv")
    train_target_path = os.path.join(output_dir, f"{model_name}_y_train.csv")
    test_target_path = os.path.join(output_dir, f"{model_name}_y_test.csv")

    # Save to CSV
    X_train_scaled_df.to_csv(train_features_path)
    X_test_scaled_df.to_csv(test_features_path)
    y_train.to_csv(train_target_path)
    y_test.to_csv(test_target_path)
    print(f"Processed data for {model_name.upper()} saved successfully to {output_dir}")


# --- 4. MAIN EXECUTION BLOCK ---

def main():
    """Main function to run the data preparation pipeline."""
    print("Starting Data Preparation Pipeline...")
    master_df = load_and_prepare_data(MASTER_DATA_PATH)

    if master_df is not None:
        for model_name, feature_list in MODEL_FEATURES.items():
            split_scale_and_save(
                df=master_df,
                features=feature_list,
                target=TARGET,
                model_name=model_name,
                output_dir=PROCESSED_DATA_DIR
            )
        print("\nData preparation pipeline finished successfully.")

if __name__ == "__main__":
    main()
