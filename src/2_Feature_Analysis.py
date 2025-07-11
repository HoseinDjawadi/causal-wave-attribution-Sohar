# ----------------------------------------------------------------------------
# SOHAR PORT CAUSAL ML PROJECT
#
# SCRIPT: 1b_feature_and_sensitivity_analysis.py
#
# DESCRIPTION:
# This script performs a comprehensive, model-aware feature analysis to
# generate deep insights into the dataset and model behaviors. It is
# structured into three parts, each tailored to a specific model's objective.
#
# METHODOLOGY:
# 1.  MLR Analysis: Visually proves the non-linearity of the system using
#     scatter plots with linear and LOESS regression lines.
# 2.  XGBoost Analysis: Diagnoses overfitting by performing a robust
#     Permutation Feature Importance analysis on the held-out test set.
# 3.  CGINN Analysis: Validates the discovered causal dynamics using
#     Time-Lagged Cross-Correlation plots and a focused Correlation Matrix
#     of the causal features.
#
# USAGE:
# Ensure all required libraries are installed.
# Update the placeholder paths for data and model outputs.
# Run the script from your terminal: `python 1b_feature_and_sensitivity_analysis.py`
# ----------------------------------------------------------------------------

import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

# --- 1. CONFIGURATION AND PATHS ---
# Use placeholder paths. Update these to your actual file locations.
MASTER_DATA_PATH = '/home/artorias/Desktop/Sohar_Causal_Modeling/Data/Raw/master_analysis_dataset.csv'
PROCESSED_DATA_DIR = '/home/artorias/Desktop/Sohar_Causal_Modeling/Data/Processed'
XGB_HYPERPARAMS_PATH = '/home/artorias/Desktop/Sohar_Causal_Modeling/Output/XGBoost/xgb_best_hyperparameters.json'
ANALYSIS_OUTPUT_DIR = '/home/artorias/Desktop/Sohar_Causal_Modeling/Output/Feature Analysis'

TARGET_VARIABLE = 'Hm0_buoy'

# --- Feature sets defined in the data preparation script ---
FEATURES_MLR = ['Effective_VHM0', 'WindSpeed_buoy']
FEATURES_CGINN = [
    'WindSpeed_buoy', 'Effective_VHM0', 'Effective_VHM0_lag3h', 'Effective_VTPK',
    'Effective_VTPK_lag3h', 'Effective_VTPK_lag6h', 'Path_Mean_Slope_m_per_km',
    'Effective_WindDirection', 'Path_Rugosity_m', 'Path_Rugosity_m_lag3h',
    'Shoaling_Factor_Proxy_lag3h', 'Tp_buoy', 'Tp_buoy_lag6h', 'Tp_buoy_lag24h'
]

# --- 2. ANALYSIS I: MLR - PROVING NON-LINEARITY ---

def analyze_mlr_features(df, output_dir):
    """
    Generates scatter plots with linear and LOESS fits to visualize
    the non-linearity of predictor-target relationships.
    """
    print("\n--- Starting Analysis for MLR: Proving Non-Linearity ---")
    output_path = os.path.join(output_dir, 'mlr_analysis')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for feature in FEATURES_MLR:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 7))

        # Use seaborn's regplot, which can show both linear and LOESS fits
        # Plot a sample of the data to avoid overplotting
        sample_df = df.sample(n=min(5000, len(df)), random_state=42)
        
        # Linear fit
        sns.regplot(x=feature, y=TARGET_VARIABLE, data=sample_df, ax=ax,
                    scatter_kws={'alpha': 0.2, 's': 15},
                    line_kws={'color': 'dodgerblue', 'linewidth': 3, 'label': 'Linear Fit'},
                    ci=None)
        
        # LOESS fit
        sns.regplot(x=feature, y=TARGET_VARIABLE, data=sample_df, ax=ax,
                    lowess=True,
                    scatter=False, # Don't plot the scatter points again
                    line_kws={'color': 'orangered', 'linewidth': 3, 'linestyle': '--', 'label': 'LOESS Fit'})

        ax.set_title(f'Relationship between {feature} and {TARGET_VARIABLE}', fontsize=16, weight='bold')
        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel(TARGET_VARIABLE, fontsize=12)
        ax.legend()

        plot_filename = os.path.join(output_path, f'{feature}_vs_{TARGET_VARIABLE}_linearity.png')
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved non-linearity plot to: {plot_filename}")
    print("--- MLR Analysis Complete ---")


# --- 3. ANALYSIS II: XGBOOST - DIAGNOSING OVERFITTING ---

def analyze_xgb_features(output_dir):
    """
    Trains the tuned XGBoost model and performs permutation importance
    on the test set to identify key drivers and diagnose overfitting.
    """
    print("\n--- Starting Analysis for XGBoost: Diagnosing Overfitting ---")
    output_path = os.path.join(output_dir, 'xgb_analysis')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load data and hyperparameters
    try:
        X_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'xgb_X_train.csv'), index_col='time', parse_dates=True)
        y_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'xgb_y_train.csv'), index_col='time', parse_dates=True)
        X_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'xgb_X_test.csv'), index_col='time', parse_dates=True)
        y_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'xgb_y_test.csv'), index_col='time', parse_dates=True)
        with open(XGB_HYPERPARAMS_PATH, 'r') as f:
            best_params = json.load(f)
    except FileNotFoundError as e:
        print(f"FATAL ERROR: Could not load data or hyperparameters for XGBoost. {e}")
        print("Please ensure the XGBoost model has been trained first.")
        return

    # Train the final model
    print("Training XGBoost model with best hyperparameters...")
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, random_state=42, early_stopping_rounds=50, n_jobs=-1, **best_params)
    # Use the compatible early stopping mechanism
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Calculate Permutation Importance on the TEST set
    print("Calculating Permutation Importance on the test set...")
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    
    # --- SNIPPET CORRECTION START ---
    # Process and save results using dictionary-style access
    sorted_idx = perm_importance['importances_mean'].argsort()
    importance_df = pd.DataFrame(
        data={'feature': X_test.columns[sorted_idx], 'importance_mean': perm_importance['importances_mean'][sorted_idx]}
    )
    # --- SNIPPET CORRECTION END ---
    
    csv_path = os.path.join(output_path, 'permutation_importance.csv')
    importance_df.to_csv(csv_path, index=False)
    print(f"Saved permutation importance data to: {csv_path}")

    # Plot top 20 features
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    top_20 = importance_df.tail(20)
    ax.barh(top_20['feature'], top_20['importance_mean'], color='mediumseagreen')
    ax.set_xlabel("Permutation Importance (Drop in R²)", fontsize=12)
    ax.set_title("XGBoost: Top 20 Most Important Features on Test Set", fontsize=16, weight='bold')
    plt.tight_layout()
    
    plot_filename = os.path.join(output_path, 'permutation_importance_plot.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print(f"Saved permutation importance plot to: {plot_filename}")
    print("--- XGBoost Analysis Complete ---")


# --- 4. ANALYSIS III: CGINN - VALIDATING CAUSAL DYNAMICS ---

def plot_cross_correlation(df, cause, effect, max_lags, output_dir):
    """Calculates and plots the time-lagged cross-correlation between two series."""
    lags = range(-max_lags, max_lags + 1)
    corrs = [df[cause].shift(lag).corr(df[effect]) for lag in lags]
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    markerline, stemlines, baseline = ax.stem(lags, corrs, linefmt='grey', markerfmt='o', basefmt='r-')
    plt.setp(markerline, 'color', 'dodgerblue', 'markersize', 6)
    plt.setp(stemlines, 'color', 'dodgerblue', 'linewidth', 2)
    
    # Highlight the most significant positive and negative correlation
    max_corr_lag = lags[np.argmax(corrs)]
    min_corr_lag = lags[np.argmin(corrs)]
    ax.plot(max_corr_lag, max(corrs), 'go', markersize=10, label=f'Max Corr at lag {max_corr_lag} ({max(corrs):.2f})')
    ax.plot(min_corr_lag, min(corrs), 'ro', markersize=10, label=f'Min Corr at lag {min_corr_lag} ({min(corrs):.2f})')

    ax.set_title(f'Cross-Correlation: {cause} vs. {effect}', fontsize=16, weight='bold')
    ax.set_xlabel('Lag (3-hour steps)', fontsize=12)
    ax.set_ylabel('Pearson Correlation', fontsize=12)
    ax.set_xticks(lags)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()
    
    plot_filename = os.path.join(output_dir, f'xcorr_{cause}_vs_{effect}.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved cross-correlation plot to: {plot_filename}")

def analyze_cginn_features(df, output_dir):
    """
    Analyzes the features selected for the CGINN model, focusing on
    temporal dynamics and inter-feature correlations.
    """
    print("\n--- Starting Analysis for CGINN: Validating Causal Dynamics ---")
    output_path = os.path.join(output_dir, 'cginn_analysis')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Part A: Time-Lagged Cross-Correlation
    xcorr_path = os.path.join(output_path, 'cross_correlations')
    if not os.path.exists(xcorr_path):
        os.makedirs(xcorr_path)
    
    print("Generating cross-correlation plots...")
    plot_cross_correlation(df, 'Effective_VHM0', TARGET_VARIABLE, max_lags=8, output_dir=xcorr_path)
    plot_cross_correlation(df, 'WindSpeed_buoy', TARGET_VARIABLE, max_lags=8, output_dir=xcorr_path)
    plot_cross_correlation(df, 'Effective_VTPK', TARGET_VARIABLE, max_lags=8, output_dir=xcorr_path)

    # Part B: Causal Feature Correlation Matrix
    print("Generating correlation matrix for causal features...")
    causal_df = df[FEATURES_CGINN].copy()
    corr_matrix = causal_df.corr()
    
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                linewidths=.5, ax=ax, annot_kws={"size": 8})
    ax.set_title('Correlation Matrix of CGINN Input Features', fontsize=16, weight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plot_filename = os.path.join(output_path, 'causal_features_correlation_matrix.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved causal feature correlation matrix to: {plot_filename}")
    print("--- CGINN Analysis Complete ---")


# --- 5. MAIN EXECUTION BLOCK ---

# --- 5. MAIN EXECUTION BLOCK ---

def main():
    """Main function to run the full feature analysis pipeline."""
    print("="*60)
    print("Starting Comprehensive Feature and Sensitivity Analysis")
    print("="*60)

    try:
        master_df = pd.read_csv(MASTER_DATA_PATH, index_col='time', parse_dates=True)
        master_df.sort_index(inplace=True)
    except FileNotFoundError as e:
        print(f"FATAL ERROR: Master data file not found. {e}")
        return

    # --- SNIPPET START: CREATE LAGGED FEATURES ---
    # This step is crucial as the master dataset does not contain these columns by default.
    # The data has a 3-hour frequency, so we shift by periods.
    # 3h lag = 1 period, 6h lag = 2 periods, 24h lag = 8 periods.
    print("Creating required lagged features for CGINN analysis...")
    master_df['Effective_VHM0_lag3h'] = master_df['Effective_VHM0'].shift(1)
    master_df['Effective_VTPK_lag3h'] = master_df['Effective_VTPK'].shift(1)
    master_df['Effective_VTPK_lag6h'] = master_df['Effective_VTPK'].shift(2)
    master_df['Path_Rugosity_m_lag3h'] = master_df['Path_Rugosity_m'].shift(1)
    master_df['Shoaling_Factor_Proxy_lag3h'] = master_df['Shoaling_Factor_Proxy'].shift(1)
    master_df['Tp_buoy_lag6h'] = master_df['Tp_buoy'].shift(2)
    master_df['Tp_buoy_lag24h'] = master_df['Tp_buoy'].shift(8)
    # --- SNIPPET END ---

    # Run the three analysis modules
    analyze_mlr_features(master_df, ANALYSIS_OUTPUT_DIR)
    analyze_xgb_features(ANALYSIS_OUTPUT_DIR)
    analyze_cginn_features(master_df, ANALYSIS_OUTPUT_DIR)

    print("\n--- Feature Analysis Pipeline Finished Successfully ---")

if __name__ == "__main__":
    main()
