import pandas as pd
import numpy as np
import os
import joblib # For saving Python objects like lists and dictionaries

# --- Configuration ---
# 1. Define file paths
BASE_PATH = "/home/artorias/Desktop/Causal_Modeling/Data"
ANALYSIS_READY_PATH = os.path.join(BASE_PATH, "Analysis_Ready")

# Input file from the previous step
MASTER_DATASET_FILE = os.path.join(ANALYSIS_READY_PATH, "master_analysis_dataset.csv")

# Output files for the prepared data
OUTPUT_DATA_FILE = os.path.join(ANALYSIS_READY_PATH, "Casual_Discovery_Ready/tigramite_data.npy")
OUTPUT_VARS_FILE = os.path.join(ANALYSIS_READY_PATH, "Casual_Discovery_Ready/tigramite_variable_names.pkl")
OUTPUT_PRIORS_FILE = os.path.join(ANALYSIS_READY_PATH, "Casual_Discovery_Ready/tigramite_priors.pkl")

# --- Main Script ---
if __name__ == '__main__':
    # --- 1. Load the Master Dataset ---
    print("Loading the master analysis dataset...")
    try:
        df = pd.read_csv(MASTER_DATASET_FILE, index_col='time', parse_dates=True)
    except FileNotFoundError as e:
        raise SystemExit(f"FATAL: Master dataset not found. Please run the previous script. Error: {e}")

    # --- 2. Select Variables for the Causal Model ---
    # We select key outcomes, contemporaneous drivers, and mediators.
    # The lagged variables will be handled automatically by tigramite.
    print("Selecting variables for the causal model...")
    
    # Rename buoy columns for clarity in the graph
    df.rename(columns={'Hm0': 'Hm0_buoy', 'Tp': 'Tp_buoy', 'WindSpeed': 'WindSpeed_buoy'}, inplace=True)

    selected_variables = [
        # Outcomes (Buoy)
        'Hm0_buoy',
        'Tp_buoy',
        'WindSpeed_buoy',
        # Contemporaneous Drivers (Offshore)
        'Effective_VHM0',
        'Effective_VTPK',
        'Effective_WindDirection', # We calculated this from components
        # Mediators (Static Bathymetry)
        'Path_Mean_Slope_m_per_km',
        'Path_Rugosity_m',
        'Shoaling_Factor_Proxy'
    ]
    
    # Select only the chosen contemporaneous columns
    df_model = df[selected_variables]
    
    # --- 3. Define Causal Graph Priors ---
    # This is where we inject our domain knowledge.
    print("Defining causal graph priors (forbidden links)...")

    # Initialize list for forbidden links: (cause, effect, lag)
    # The 'lag' here is in time steps. 'lag=0' means contemporaneous.
    # Using a wildcard 'None' for lag forbids links at all time lags.
    forbidden_links = []

    # Rule 1: Buoy measurements cannot cause offshore conditions.
    for buoy_var in ['Hm0_buoy', 'Tp_buoy', 'WindSpeed_buoy']:
        for offshore_var in ['Effective_VHM0', 'Effective_VTPK', 'Effective_WindDirection']:
            forbidden_links.append((buoy_var, offshore_var, None))

    # Rule 2: Nothing can cause static bathymetry features.
    for bathy_var in ['Path_Mean_Slope_m_per_km', 'Path_Rugosity_m', 'Shoaling_Factor_Proxy']:
        for other_var in selected_variables:
            if bathy_var != other_var:
                forbidden_links.append((other_var, bathy_var, None))
    
    # Rule 3: Buoy wind cannot cause offshore wave properties.
    for offshore_wave_var in ['Effective_VHM0', 'Effective_VTPK']:
        forbidden_links.append(('WindSpeed_buoy', offshore_wave_var, None))

    priors = {'forbidden_links': forbidden_links}
    print(f"Defined {len(forbidden_links)} forbidden link rules.")

    # --- 4. Prepare Data for Tigramite ---
    print("Converting data to Tigramite format...")
    # Tigramite works with numpy arrays and a list of variable names
    
    # The data array
    data_array = df_model.values
    
    # The list of variable names
    variable_names = df_model.columns.tolist()

    # --- 5. Save Prepared Objects for Next Step ---
    print("Saving prepared data, variable names, and priors...")
    np.save(OUTPUT_DATA_FILE, data_array)
    joblib.dump(variable_names, OUTPUT_VARS_FILE)
    joblib.dump(priors, OUTPUT_PRIORS_FILE)

    print("\n--- Causal Model Preparation Complete ---")
    print("The following files have been created:")
    print(f" - Data Array: {OUTPUT_DATA_FILE}")
    print(f" - Variable Names: {OUTPUT_VARS_FILE}")
    print(f" - Causal Priors: {OUTPUT_PRIORS_FILE}")
    print("\nNext step: Run the PCMCI+ algorithm using these files.")