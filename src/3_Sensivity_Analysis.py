# ----------------------------------------------------------------------------
# SOHAR PORT CAUSAL ML PROJECT
#
# SCRIPT: 4_sensitivity_analysis.py
#
# DESCRIPTION:
# This script performs a formal sensitivity analysis to quantify the robustness
# of a key causal finding to potential unobserved confounding variables. It uses
# the methodology of Cinelli & Hazlett (2020) via the `sensemakr` library.
#
# METHODOLOGY:
# 1.  Loads the master dataset and creates necessary lagged features.
# 2.  Defines a core causal relationship to test: the effect of local
#     wind speed on nearshore wave height.
# 3.  Constructs a linear regression model to estimate this effect while
#     controlling for all other known direct causal parents.
# 4.  Uses `sensemakr` to calculate the Robustness Value (RV) and other
#     sensitivity statistics.
# 5.  Generates and saves a detailed text report and a publication-quality
#     sensitivity contour plot.
#
# USAGE:
# Ensure the `sensemakr` library is installed (`pip install sensemakr`).
# Update placeholder paths for input and output.
# Run from terminal: `python 4_sensitivity_analysis.py`
# ----------------------------------------------------------------------------

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sensemakr import Sensemakr

# --- 1. CONFIGURATION AND PATHS ---
# Use placeholder paths. Update these to your actual file locations.
MASTER_DATA_PATH = '/home/artorias/Desktop/Sohar_Causal_Modeling/Data/Raw/master_analysis_dataset.csv'
ANALYSIS_OUTPUT_DIR = '/home/artorias/Desktop/Sohar_Causal_Modeling/Output/Sensivity_Analysis'

# --- 2. DEFINE THE CAUSAL MODEL FOR SENSITIVITY TESTING ---
# We will test the robustness of the WindSpeed_buoy -> Hm0_buoy link.

TREATMENT = 'WindSpeed_buoy'
OUTCOME = 'Hm0_buoy'

# The controls are all other direct causal parents of the outcome.
# This list should match the CGINN features, excluding the treatment itself.
CONTROLS = [
    'Effective_VHM0', 'Effective_VHM0_lag3h', 'Effective_VTPK',
    'Effective_VTPK_lag3h', 'Effective_VTPK_lag6h', 'Path_Mean_Slope_m_per_km',
    'Effective_WindDirection', 'Path_Rugosity_m', 'Path_Rugosity_m_lag3h',
    'Shoaling_Factor_Proxy_lag3h', 'Tp_buoy', 'Tp_buoy_lag6h', 'Tp_buoy_lag24h'
]

# --- 3. MAIN EXECUTION BLOCK ---

def main():
    """Main function to run the sensitivity analysis pipeline."""
    print("="*60)
    print("Starting Sensitivity Analysis for Unobserved Confounding")
    print("="*60)

    # --- 3.1. Load and Prepare Data ---
    print("Loading master dataset...")
    try:
        df = pd.read_csv(MASTER_DATA_PATH, index_col='time', parse_dates=True)
        df.sort_index(inplace=True)
    except FileNotFoundError as e:
        print(f"FATAL ERROR: Master data file not found. {e}"); return

    # Create lagged features required for the control set
    print("Creating required lagged features...")
    df['Effective_VHM0_lag3h'] = df['Effective_VHM0'].shift(1)
    df['Effective_VTPK_lag3h'] = df['Effective_VTPK'].shift(1)
    df['Effective_VTPK_lag6h'] = df['Effective_VTPK'].shift(2)
    df['Path_Rugosity_m_lag3h'] = df['Path_Rugosity_m'].shift(1)
    df['Shoaling_Factor_Proxy_lag3h'] = df['Shoaling_Factor_Proxy'].shift(1)
    df['Tp_buoy_lag6h'] = df['Tp_buoy'].shift(2)
    df['Tp_buoy_lag24h'] = df['Tp_buoy'].shift(8)

    model_vars = [OUTCOME, TREATMENT] + CONTROLS
    model_df = df[model_vars].dropna().copy()
    print(f"Prepared analysis dataset with {len(model_df)} observations.")

    # --- 3.2. Fit the Benchmark Regression Model ---
    print("\nFitting the benchmark linear model...")
    formula = f"{OUTCOME} ~ {TREATMENT} + " + " + ".join(CONTROLS)
    model = smf.ols(formula=formula, data=model_df).fit()
    print("Model fitting complete.")
    print("\n--- Benchmark Model Summary ---")
    print(model.summary())

    # --- 3.3. Perform Sensitivity Analysis ---
    print("\nPerforming sensitivity analysis using sensemakr...")
    sensitivity = Sensemakr(model=model, treatment=TREATMENT, benchmark_covariates=['Effective_VHM0', 'Effective_VTPK'])
    summary_report = sensitivity.summary()
    print("\n--- Sensemakr Sensitivity Analysis Summary ---")
    print(summary_report)
    print("------------------------------------------")

    # --- 3.4. Save Outputs ---
    print("\nSaving analysis outputs...")
    if not os.path.exists(ANALYSIS_OUTPUT_DIR):
        os.makedirs(ANALYSIS_OUTPUT_DIR)

    report_path = os.path.join(ANALYSIS_OUTPUT_DIR, 'sensitivity_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*80 + "\nSENSITIVITY ANALYSIS REPORT\n" + "="*80 + "\n\n")
        f.write(f"This report assesses the robustness of the estimated causal effect\n")
        f.write(f"of '{TREATMENT}' on '{OUTCOME}' to potential unobserved confounding.\n\n")
        f.write("--- Benchmark OLS Model ---\n")
        f.write(str(model.summary()))
        f.write("\n\n--- Sensemakr Sensitivity Analysis ---\n")
        f.write(str(summary_report))
    print(f"Detailed sensitivity report saved to: {report_path}")

    # --- FINAL ROBUST PLOTTING SNIPPET ---
    print("Generating and saving sensitivity contour plot manually...")
    plot_path = os.path.join(ANALYSIS_OUTPUT_DIR, 'sensitivity_contour_plot.png')
    
    # Get necessary stats from the sensitivity object
    stats = sensitivity.sensitivity_stats
    estimate = stats['estimate']
    se = stats['se']
    dof = stats['dof']

    # Create grid for contour plot
    r2_d = np.linspace(1e-4, 0.4, 100)
    r2_y = np.linspace(1e-4, 0.4, 100)
    X_grid, Y_grid = np.meshgrid(r2_d, r2_y)
    
    # Manually calculate the adjusted estimate grid using the vectorized numpy formula
    # This bypasses the buggy, non-vectorized adjusted_estimate function in the library.
    bias = se * np.sqrt(dof) * np.sqrt( (Y_grid * X_grid) / (1 - X_grid) )
    Z_grid = estimate - bias # Assuming reduce=True

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    contour = ax.contourf(X_grid, Y_grid, Z_grid, levels=15, cmap='coolwarm_r', extend='both')
    fig.colorbar(contour, ax=ax, label='Adjusted Estimate')

    CS = ax.contour(X_grid, Y_grid, Z_grid, levels=np.arange(0, round(estimate, 2), 0.01), colors='black', linewidths=0.5)
    ax.clabel(CS, inline=True, fontsize=8)
    
    ax.contour(X_grid, Y_grid, Z_grid, levels=[0], colors='red', linewidths=3, linestyles='--')

    bounds = sensitivity.bounds
    if bounds is not None and not bounds.empty:
        ax.scatter(bounds['r2dz_x'], bounds['r2yz_dx'], c='black', marker='D', s=80, label='Benchmarks')
        for i, row in bounds.iterrows():
            ax.text(row['r2dz_x'] + 0.005, row['r2yz_dx'], row['bound_label'], fontsize=9, verticalalignment='center')

    ax.set_xlabel(r'Partial $R^2$ of Confounder with Treatment ($R^2_{D \sim Z|X}$)', fontsize=12)
    ax.set_ylabel(r'Partial $R^2$ of Confounder with Outcome ($R^2_{Y \sim Z|X,D}$)', fontsize=12)
    ax.set_title(f'Sensitivity of the Effect of {TREATMENT} on {OUTCOME}', fontsize=16, weight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(0, 0.4)
    ax.set_ylim(0, 0.4)
    ax.legend()
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Sensitivity contour plot saved to: {plot_path}")
    # --- END SNIPPET ---

    print("\n--- Sensitivity Analysis Pipeline Finished Successfully ---")

if __name__ == "__main__":
    main()
