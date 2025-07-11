import pandas as pd
import numpy as np
import os

# --- Configuration ---
# Please ensure these paths are correct for your system
BASE_PATH = "/home/artorias/Desktop/Causal_Modeling/Data"
HYCOM_PATH = os.path.join(BASE_PATH, "Reanalysis/HYCOM")
OUTPUT_PATH = os.path.join(BASE_PATH, "Analysis_Ready")
HYCOM_FILENAME = "HYCOM_SSH_Sohar_2010_2024.csv"
OUTPUT_FILENAME = "effective_ssh_3hourly_final.csv"

def process_hycom_standalone():
    """
    Loads daily HYCOM data, computes a daily spatial average, then upsamples
    and interpolates it to a clean 3-hourly time series.
    """
    print(f"--- Processing Standalone: {HYCOM_FILENAME} ---")

    # 1. Load the data
    try:
        file_path = os.path.join(HYCOM_PATH, HYCOM_FILENAME)
        df = pd.read_csv(file_path)
        # Rename columns for clarity
        df.rename(columns={'date': 'time', 'surface_elevation': 'ssh'}, inplace=True)
    except Exception as e:
        print(f"FATAL: Could not load {file_path}. Error: {e}")
        return

    # 2. Convert to datetime and calculate daily spatial average
    df['time'] = pd.to_datetime(df['time'])
    print("Calculating daily spatial average...")
    daily_ssh = df.groupby(df['time'].dt.date)['ssh'].mean()
    daily_ssh.index = pd.to_datetime(daily_ssh.index)
    
    print("\n--- Intermediate Daily Data ---")
    print("Original daily data shape:", daily_ssh.shape)
    print(daily_ssh.head())
    print("-----------------------------\n")

    # 3. Upsample to 3-hourly frequency and interpolate
    print("Upsampling to 3-hourly frequency...")
    # Create a new, complete 3-hourly index from the start to the end of your data
    full_3h_index = pd.date_range(start=daily_ssh.index.min(), end=daily_ssh.index.max(), freq='3h')

    # Reindex the daily data to this new 3-hourly index. This creates NaNs.
    ssh_3hourly = daily_ssh.reindex(full_3h_index)

    print(f"Shape after reindexing to 3-hourly: {ssh_3hourly.shape}")
    print("Number of non-NaN values before interpolation:", ssh_3hourly.notna().sum())

    print("Interpolating to fill gaps...")
    # Use time-based linear interpolation to fill the NaNs
    ssh_3hourly_interpolated = ssh_3hourly.interpolate(method='time')

    print("Number of non-NaN values after interpolation:", ssh_3hourly_interpolated.notna().sum())
    
    # 4. Finalize and save the data
    final_df = ssh_3hourly_interpolated.to_frame(name="Effective_ssh")
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    output_path = os.path.join(OUTPUT_PATH, OUTPUT_FILENAME)
    final_df.to_csv(output_path)

    print("\n--- HYCOM Processing Complete ---")
    print(f"Successfully created 3-hourly SSH data.")
    print(f"Output saved to: {output_path}")
    print("\nFirst 10 rows of the final output file:")
    print(final_df.head(10))

if __name__ == '__main__':
    process_hycom_standalone()