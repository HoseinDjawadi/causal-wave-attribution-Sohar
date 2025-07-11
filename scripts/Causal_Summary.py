import pandas as pd
import numpy as np
import os
import joblib

# --- Configuration ---
BASE_PATH = "/home/artorias/Desktop/Causal_Modeling/Data"
RESULTS_PATH = os.path.join(BASE_PATH, "Causal_Discovery") # Corrected path
VARS_PATH = os.path.join(BASE_PATH, "Analysis_Ready/Casual_Discovery_Ready") # Corrected path

# Input files
RESULTS_FILE = os.path.join(RESULTS_PATH, "V4/causal_graph_results.pkl")
VARS_FILE = os.path.join(VARS_PATH, "tigramite_variable_names.pkl")

# Output file
SUMMARY_FILE = os.path.join(RESULTS_PATH, "V4/significant_causal_links.csv")

# The alpha level used in the last run
ALPHA_LEVEL = 0.001 # IMPORTANT: Use the same alpha as your final graph run

# --- Main Script ---
if __name__ == '__main__':
    print("Loading results and variable names...")
    try:
        results = joblib.load(RESULTS_FILE)
        variable_names = joblib.load(VARS_FILE)
    except FileNotFoundError as e:
        raise SystemExit(f"FATAL: Could not find input file. Error: {e}")

    # Extract the relevant matrices from the results object
    graph = results['graph']
    p_matrix = results['p_matrix']
    val_matrix = results['val_matrix']
    
    significant_links = []

    # Iterate through the graph to find significant links
    # The graph shape is (target, source, lag)
    for effect_idx, cause_idx, lag in np.argwhere(graph != ''):
        p_value = p_matrix[effect_idx, cause_idx, lag]

        # Check if the link is statistically significant
        if p_value < ALPHA_LEVEL:
            link_data = {
                "Effect": variable_names[effect_idx],
                "Cause": variable_names[cause_idx],
                "Lag (3h steps)": lag,
                "Lag (hours)": lag * 3,
                "Strength (MCI)": val_matrix[effect_idx, cause_idx, lag],
                "P-Value": p_value
            }
            significant_links.append(link_data)

    if not significant_links:
        print("No significant links found at the specified alpha level. Consider using a less strict alpha.")
    else:
        # Create and sort the summary DataFrame
        df_summary = pd.DataFrame(significant_links)
        df_summary = df_summary.sort_values(by=['Effect', 'P-Value'], ascending=[True, True])
        
        # Save the summary to a CSV file
        df_summary.to_csv(SUMMARY_FILE, index=False, float_format='%.4f')

        print(f"\n--- Significant Causal Links (alpha < {ALPHA_LEVEL}) ---")
        print(df_summary.to_string())
        print(f"\nSummary table saved to: {SUMMARY_FILE}")