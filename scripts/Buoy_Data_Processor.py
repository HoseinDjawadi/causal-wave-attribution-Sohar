import pandas as pd
import numpy as np

def harmonize_buoy_data(file_path: str, output_path: str) -> pd.DataFrame:
    """
    Loads hourly buoy data, resamples it to a 3-hourly interval using a
    hybrid aggregation strategy, and saves the result to a new CSV file.

    The strategy is:
    - .max() for Hm0, Tp, WindSpeed, CurrSpd.
    - .mean() for WaterTemp, Salinity.
    - Vector averaging for Mdir, WindDirection, CurrDir.

    Args:
        file_path (str): The full path to the input buoy CSV file.
        output_path (str): The full path where the processed 3-hourly
                           CSV file will be saved.

    Returns:
        pd.DataFrame: The processed and resampled DataFrame.
    """
    print(f"--- Processing file: {file_path} ---")

    # 1. Load data and set a proper DatetimeIndex
    try:
        df = pd.read_csv(file_path)
        # Assuming the time column is named 'time'
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
    except FileNotFoundError:
        print(f"Error: Input file not found at {file_path}")
        return None
    except KeyError:
        print("Error: 'time' column not found. Please check your CSV file.")
        return None


    # 2. Define aggregation for scalar variables
    # This dictionary defines which function to apply to each column.
    scalar_agg_logic = {
        'Hm0': 'max',
        'Tp': 'max',
        'WindSpeed': 'max',
        'CurrSpd': 'max',
        'WaterTemp': 'mean',
        'Salinity': 'mean'
    }

    # Resample the scalar columns according to the logic
    df_scalar_resampled = df[scalar_agg_logic.keys()].resample('3H').agg(scalar_agg_logic)
    print("Resampled scalar variables.")

    # 3. Handle directional variables using vector averaging
    dir_cols = ['Mdir', 'WindDirection', 'CurrDir']
    
    # Create a new DataFrame to hold the resampled directional data
    df_dir_resampled = pd.DataFrame(index=df_scalar_resampled.index)

    for col in dir_cols:
        if col in df.columns:
            # Convert degrees to radians for trigonometric functions
            radians = np.deg2rad(df[col])

            # Calculate U (East-West) and V (North-South) components
            df[f'{col}_u'] = np.sin(radians)
            df[f'{col}_v'] = np.cos(radians)

            # Resample U and V components by taking their mean
            u_mean = df[f'{col}_u'].resample('3h').mean()
            v_mean = df[f'{col}_v'].resample('3h').mean()

            # Convert averaged U, V back to degrees
            avg_rad = np.arctan2(u_mean, v_mean)
            avg_deg = np.rad2deg(avg_rad)

            # Normalize to 0-360 degree range
            df_dir_resampled[col] = (avg_deg + 360) % 360
            print(f"Performed vector averaging on '{col}'.")

    # 4. Combine the processed scalar and directional data
    df_processed = pd.concat([df_scalar_resampled, df_dir_resampled], axis=1)
    
    # Reorder columns to match original where possible
    original_order = [col for col in df.columns if col in df_processed.columns]
    df_processed = df_processed[original_order]

    # 5. Save the final DataFrame to a new CSV file
    df_processed.to_csv(output_path)
    print(f"Successfully saved processed data to: {output_path}\n")

    return df_processed

if __name__ == '__main__':
    # --- Configuration ---
    # Please update these paths to match your file locations.
    # I'm assuming you have a directory for your original buoy data and one for output.
    
    # Input files
    buoy_file_072 = "/home/artorias/Desktop/Causal_Modeling/Data/Buoys/Processed/SWMidi073_Final_Merged_Imputed_Hourly.csv"  # Replace with your actual path
    buoy_file_103 = "/home/artorias/Desktop/Causal_Modeling/Data/Buoys/Processed/SWMidi103_Final_Merged_Imputed_Hourly.csv"  # Replace with your actual path

    # Output files
    output_file_072 = "/home/artorias/Desktop/Causal_Modeling/Data/Buoys/Processed/buoy_072_3hourly.csv"
    output_file_103 = "/home/artorias/Desktop/Causal_Modeling/Data/Buoys/Processed/buoy_103_3hourly.csv"

    # --- Execution ---
    processed_df_072 = harmonize_buoy_data(buoy_file_072, output_file_072)
    processed_df_103 = harmonize_buoy_data(buoy_file_103, output_file_103)

    # --- Verification ---
    if processed_df_072 is not None:
        print("--- Verification for Buoy SWMidi-072 ---")
        print("Data Info:")
        processed_df_072.info()
        print("\nFirst 5 rows of processed data:")
        print(processed_df_072.head())