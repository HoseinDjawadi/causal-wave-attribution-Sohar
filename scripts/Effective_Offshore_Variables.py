import pandas as pd
import numpy as np
import os
import glob

# --- Configuration ---
BASE_PATH = "/home/artorias/Desktop/Causal_Modeling/Data"
CMEMS_PATH = os.path.join(BASE_PATH, "Reanalysis/CMEMS/Processed/CSVs_by_Variable/Selected")
ERA5_PATH = os.path.join(BASE_PATH, "Reanalysis/ERA5/Cleaned_Analyzed/CSVs")
HYCOM_PATH = os.path.join(BASE_PATH, "Reanalysis/HYCOM")
OUTPUT_PATH = os.path.join(BASE_PATH, "Analysis_Ready")

causal_forcing_sectors = {
    'SE_Swell_Corridor': [(24.6, 57.2), (24.6, 57.4), (24.6, 57.6), (24.6, 57.8), (24.6, 58.0), (24.8, 57.6), (24.8, 57.8), (24.8, 58.0)],
    'Local_ENE_Sea': [(24.6, 56.6), (24.6, 56.8), (24.6, 57.0), (24.8, 56.8), (24.8, 57.0), (24.8, 57.2), (24.8, 57.4), (25.0, 57.0), (25.0, 57.2), (25.0, 57.4), (25.0, 57.6), (25.0, 57.8), (25.0, 58.0)],
    'NW_Shamal_Sector': [(24.8, 56.6), (25.0, 56.4), (25.0, 56.6), (25.0, 56.8), (25.2, 56.6), (25.2, 56.8), (25.2, 57.0), (25.2, 57.2), (25.2, 57.4), (25.2, 57.6), (25.2, 57.8), (25.2, 58.0)]
}
CMEMS_VARS = ['VHM0', 'VMDR', 'VTPK', 'eastward_wind', 'northward_wind', 'uo', 'vo', 'utide', 'vtide']

SECTOR_CENTERS = {'SE_Swell_Corridor': 135, 'Local_ENE_Sea': 67.5, 'NW_Shamal_Sector': 315}
WEIGHTING_SPREAD = 45

# --- Helper Functions ---

def calculate_dynamic_weights(wave_direction: float) -> dict:
    if pd.isna(wave_direction):
        return {sector: 1/len(SECTOR_CENTERS) for sector in SECTOR_CENTERS}
    weights = {}
    total_weight = 0
    for sector, center_deg in SECTOR_CENTERS.items():
        diff = 180 - abs(abs(wave_direction - center_deg) - 180)
        weight = np.exp(-diff**2 / (2 * WEIGHTING_SPREAD**2))
        weights[sector] = weight
        total_weight += weight
    return {sector: w / total_weight for sector, w in weights.items()} if total_weight > 0 else calculate_dynamic_weights(np.nan)

def process_cmems_data():
    """Loads and processes only the CMEMS data using dynamic weighting."""
    print("--- Processing CMEMS Data ---")
    cmems_files = glob.glob(os.path.join(CMEMS_PATH, "*.csv"))
    cmems_dfs = []
    for f in cmems_files:
        var_name = os.path.basename(f).replace('.csv', '')
        df = pd.read_csv(f)
        df_long = df.melt(id_vars=['time'], var_name='location', value_name=var_name)
        locs = df_long['location'].str.extract(r'lat(\d+\.\d+)_lon(\d+\.\d+)')
        df_long['lat'] = pd.to_numeric(locs[0])
        df_long['lon'] = pd.to_numeric(locs[1])
        df_long.set_index(['time', 'lat', 'lon'], inplace=True)
        cmems_dfs.append(df_long.drop(columns=['location']))

    if not cmems_dfs: return pd.DataFrame()
    
    df_cmems = pd.concat(cmems_dfs, axis=1).reset_index()
    df_cmems['time'] = pd.to_datetime(df_cmems['time'], format='mixed')
    df_cmems['sector'] = df_cmems.apply(
        lambda row: next((s for s, p in causal_forcing_sectors.items() if (row['lat'], row['lon']) in p), None),
        axis=1
    )
    
    effective_rows = []
    for timestamp, group_df in df_cmems.groupby('time'):
        row_data = {'time': timestamp}
        mean_vmdr = group_df['VMDR'].mean()
        weights = calculate_dynamic_weights(mean_vmdr)
        for var in CMEMS_VARS:
            if var in group_df.columns:
                vals = [weights[s] * group_df[group_df['sector'] == s][var].mean() for s in causal_forcing_sectors if not group_df[group_df['sector'] == s][var].empty]
                row_data[f'Effective_{var}'] = sum(v for v in vals if not pd.isna(v)) or np.nan
        effective_rows.append(row_data)
        
    return pd.DataFrame(effective_rows).set_index('time')

def process_external_data(path, filename, var_map, resample=False):
    """Loads and processes ERA5 or HYCOM data with simple spatial average."""
    print(f"--- Processing {filename} ---")
    try:
        df = pd.read_csv(os.path.join(path, filename))
        df = df[list(var_map.keys())]
        df.rename(columns=var_map, inplace=True)
        df['time'] = pd.to_datetime(df['time'], format='mixed')
        df.set_index('time', inplace=True)
        
        # FIX: Resample HERE before doing anything else
        if resample:
            df = df.resample('3h').mean()

        # Calculate simple spatial average for each timestamp
        df_effective = df.groupby('time').mean()
        # Rename column to be 'Effective_...'
        df_effective.rename(columns={list(var_map.values())[-1]: f"Effective_{list(var_map.values())[-1]}"}, inplace=True)
        return df_effective.drop(columns=['lat', 'lon'], errors='ignore')

    except Exception as e:
        print(f"Warning: Could not process {filename}. Error: {e}")
        return pd.DataFrame()


# --- Main Execution ---
if __name__ == '__main__':
    # 1. Process each data source independently
    df_eff_cmems = process_cmems_data()
    
    # Process ERA5 files (assuming they are already 3-hourly)
    df_eff_sst = process_external_data(ERA5_PATH, 'sst.csv', {'time':'time', 'latitude':'lat', 'longitude':'lon', 'sst':'sst'})
    df_eff_msl = process_external_data(ERA5_PATH, 'msl.csv', {'time':'time', 'latitude':'lat', 'longitude':'lon', 'msl':'msl'})

    # Process HYCOM file and RESAMPLE it to 3-hourly grid
    df_eff_ssh = process_external_data(HYCOM_PATH, 'HYCOM_SSH_Sohar_2010_2024.csv', 
                                     {'date':'time', 'latitude':'lat', 'longitude':'lon', 'surface_elevation':'ssh'},
                                     resample=True) # <-- Key change is here

    # 2. Combine the final time series
    print("\n--- Merging all effective time series ---")
    final_df = pd.concat([df_eff_cmems, df_eff_sst, df_eff_msl, df_eff_ssh], axis=1)
    
    # 3. Save the result
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    output_filename = os.path.join(OUTPUT_PATH, "effective_offshore_variables_v4_final.csv")
    final_df.to_csv(output_filename)

    print(f"\n--- Task 1 Complete (v4) ---")
    print(f"Output saved to: {output_filename}")
    print("\nFinal Data Info:")
    final_df.info()
    print("\nFirst 5 rows of the new output file:")
    print(final_df.head())