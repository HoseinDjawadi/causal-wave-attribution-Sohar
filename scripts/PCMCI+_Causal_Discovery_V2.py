import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from tigramite.pcmci import PCMCI
from tigramite.data_processing import DataFrame
from tigramite.plotting import plot_graph
from tigramite.independence_tests.parcorr import ParCorr

# --- Configuration ---
# 1. Define file paths
BASE_PATH = "/home/artorias/Desktop/Causal_Modeling/Data"
ANALYSIS_READY_PATH = os.path.join(BASE_PATH, "Analysis_Ready/Casual_Discovery_Ready")
RESULTS_PATH = os.path.join(BASE_PATH, "Causal_Discovery")

# Input files from the previous step
DATA_FILE = os.path.join(ANALYSIS_READY_PATH, "tigramite_data.npy")
VARS_FILE = os.path.join(ANALYSIS_READY_PATH, "tigramite_variable_names.pkl")
PRIORS_FILE = os.path.join(ANALYSIS_READY_PATH, "tigramite_priors.pkl")

# Output files for the results
RESULTS_FILE = os.path.join(RESULTS_PATH, "causal_graph_results.pkl")
PLOT_FILE = os.path.join(RESULTS_PATH, "causal_graph_plot.png")

# 2. Define PCMCI+ Algorithm Parameters
TAU_MAX = 8  # Maximum time lag (8 steps = 24 hours for 3-hour intervals)
PC_ALPHA = 0.05  # Significance level for parent discovery stage
ALPHA_LEVEL = 0.05  # Significance level for MCI test

# --- Main Script ---
if __name__ == '__main__':
    try:
        # --- 1. Load Prepared Data and Priors ---
        print("Loading prepared data, variable names, and priors...")
        
        # Check if files exist before loading
        for file_path, file_name in [(DATA_FILE, "data file"), (VARS_FILE, "variables file"), (PRIORS_FILE, "priors file")]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required {file_name} not found at: {file_path}")
        
        data_array = np.load(DATA_FILE)
        variable_names = joblib.load(VARS_FILE)
        priors = joblib.load(PRIORS_FILE)
        forbidden_links_list = priors['forbidden_links']

        # --- 2. Initialize Tigramite Objects ---
        print("Initializing Tigramite DataFrame and PCMCI objects...")
        tigramite_dataframe = DataFrame(data_array, var_names=variable_names)
        pcmci = PCMCI(dataframe=tigramite_dataframe, cond_ind_test=ParCorr())

        # --- 3. Build the 'link_assumptions' Dictionary ---
        print("Building 'link_assumptions' dictionary from forbidden link rules...")
        
        var_to_idx = {name: i for i, name in enumerate(variable_names)}
        num_vars = len(variable_names)
        link_assumptions = {j: {} for j in range(num_vars)}

        for cause_name, effect_name, _ in forbidden_links_list:
            if cause_name in var_to_idx and effect_name in var_to_idx:
                cause_idx = var_to_idx[cause_name]
                effect_idx = var_to_idx[effect_name]
                
                if '->'' not in link_assumptions[effect_idx]:
                    link_assumptions[effect_idx]['->'] = []
                
                if cause_idx not in link_assumptions[effect_idx]['->']:
                     link_assumptions[effect_idx]['->'].append(cause_idx)
                    
        print(f"Built link assumptions dictionary with {len(forbidden_links_list)} forbidden rules.")

        # --- 4. Run PCMCI+ Analysis ---
        print(f"Running PCMCI+ with tau_max = {TAU_MAX} and significance level = {ALPHA_LEVEL}...")
        
        results = pcmci.run_pcmci(
            tau_max=TAU_MAX,
            pc_alpha=PC_ALPHA,
            alpha_level=ALPHA_LEVEL,
            link_assumptions=link_assumptions
        )
        print("PCMCI+ analysis complete.")
        
        # --- 5. Process, Save, and Plot Results ---
        print("Processing and saving results...")
        os.makedirs(RESULTS_PATH, exist_ok=True)
        joblib.dump(results, RESULTS_FILE)
        print(f"Full results object saved to: {RESULTS_FILE}")

        print("Generating and saving the causal graph plot...")
        plt.style.use('default')
        
        plot_graph(
            val_matrix=results['val_matrix'],
            graph=results['graph'],
            var_names=variable_names,
            link_colorbar_label='cross-MCI (linear)',
            node_colorbar_label='auto-MCI (linear)',
            figsize=(14, 10),
            node_size=12,
            arrowhead_size=1.5,
            node_label_size=12,
            link_label_fontsize=10,
            link_sig_matrix=results['p_matrix'],
            sig_alpha=ALPHA_LEVEL
        )
        
        plt.savefig(PLOT_FILE, dpi=300, bbox_inches='tight')
        print(f"Causal graph plot saved to: {PLOT_FILE}")

        print("\n--- Causal Discovery Complete ---")
        print("The final analysis is finished. Please inspect the output files.")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure all required input files exist.")
    except KeyError as e:
        print(f"Missing key in data: {e}")
        print("Please check the structure of your priors file.")
    except Exception as e:
        print(f"Unexpected error during causal discovery: {e}")
        print("Please check the error details and your data format.")
        import traceback
        traceback.print_exc()