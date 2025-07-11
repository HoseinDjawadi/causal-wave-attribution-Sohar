import pandas as pd
import numpy as np
import xarray as xr
import os
from geopy.distance import great_circle
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# --- Configuration ---
BASE_PATH = "/home/artorias/Desktop/Causal_Modeling/Data"
BATHYMETRY_PATH = os.path.join(BASE_PATH, "Bathymetry")
GEBCO_FILENAME = "gebco_2024_n26.0_s23.5_w56.2_e58.8.nc"
OUTPUT_PATH = os.path.join(BASE_PATH, "Analysis_Ready")
OUTPUT_CSV_FILENAME = "bathymetric_path_features.csv"
OUTPUT_MAP_FILENAME = "bathymetric_paths_map.png"
OUTPUT_PLOT_FILENAME = "depth_profiles.png"

# --- NEW: Manually set the known ground-truth depth of the buoy ---
MANUAL_BUOY_DEPTH = 18.0  # Depth in meters

causal_forcing_sectors = {
    'SE_Swell_Corridor': [(24.6, 57.2), (24.6, 57.4), (24.6, 57.6), (24.6, 57.8), (24.6, 58.0), (24.8, 57.6), (24.8, 57.8), (24.8, 58.0)],
    'Local_ENE_Sea': [(24.6, 56.6), (24.6, 56.8), (24.6, 57.0), (24.8, 56.8), (24.8, 57.0), (24.8, 57.2), (24.8, 57.4), (25.0, 57.0), (25.0, 57.2), (25.0, 57.4), (25.0, 57.6), (25.0, 57.8), (25.0, 58.0)],
    'NW_Shamal_Sector': [(24.8, 56.6), (25.0, 56.4), (25.0, 56.6), (25.0, 56.8), (25.2, 56.6), (25.2, 56.8), (25.2, 57.0), (25.2, 57.2), (25.2, 57.4), (25.2, 57.6), (25.2, 57.8), (25.2, 58.0)]
}
primary_buoy_location = {'name': 'SWMidi-072', 'lat': 24.469299, 'lon': 56.629028}

# --- Main Script ---
if __name__ == '__main__':
    print("Loading GEBCO bathymetry data...")
    ds = xr.open_dataset(os.path.join(BATHYMETRY_PATH, GEBCO_FILENAME))
    elevation_var = next(v for v in ds.data_vars if 'mean' not in v)
    print(f"Using elevation variable: '{elevation_var}'")

    sector_centroids = {sector: {'lat': np.mean([p[0] for p in points]), 'lon': np.mean([p[1] for p in points])} 
                        for sector, points in causal_forcing_sectors.items()}

    path_features = []
    fig_profile, ax_profile = plt.subplots(figsize=(12, 7))

    for sector, start_point in sector_centroids.items():
        print(f"\n--- Processing path for: {sector} ---")
        end_point = primary_buoy_location
        path_lats = np.linspace(start_point['lat'], end_point['lat'], 100)
        path_lons = np.linspace(start_point['lon'], end_point['lon'], 100)
        depth_profile_ds = ds[elevation_var].interp(lat=xr.DataArray(path_lats, dims="points"), lon=xr.DataArray(path_lons, dims="points"), method="linear")
        depths = -depth_profile_ds.values
        
        total_distance_km = great_circle((start_point['lat'], start_point['lon']), (end_point['lat'], end_point['lon'])).kilometers
        
        # Use GEBCO depth for offshore, but manual depth for nearshore
        start_depth = depths[0]
        end_depth = MANUAL_BUOY_DEPTH # <-- Using the manual depth here
        print(f"Using manual buoy depth of {end_depth}m.")
        
        mean_slope = (start_depth - end_depth) / total_distance_km
        rugosity = np.std(depths)
        
        # The last segment of the depth profile is now from the last GEBCO point to the manual depth
        profile_with_manual_end = np.append(depths[:-1], end_depth)
        segment_slopes = -np.diff(profile_with_manual_end) / (total_distance_km / (len(depths)-1))
        slope_std_dev = np.std(segment_slopes)
        
        shoaling_proxy = start_depth / end_depth

        path_features.append({
            'Sector': sector, 'Path_Mean_Slope_m_per_km': mean_slope, 'Path_Rugosity_m': rugosity,
            'Path_Slope_StdDev': slope_std_dev, 'Shoaling_Factor_Proxy': shoaling_proxy
        })
        print(f"Calculated features for {sector}.")
        ax_profile.plot(np.linspace(0, total_distance_km, 100), depths, label=sector)

    # Finalize and save plots
    ax_profile.set_title('Depth Profiles from Sector Centroids to Buoy', fontsize=16)
    ax_profile.set_xlabel('Distance from Offshore (km)'); ax_profile.set_ylabel('Depth (m)')
    ax_profile.grid(True, linestyle='--'); ax_profile.legend(); ax_profile.invert_yaxis()
    fig_profile.savefig(os.path.join(OUTPUT_PATH, OUTPUT_PLOT_FILENAME), bbox_inches='tight')

    fig_map = plt.figure(figsize=(12, 10))
    ax_map = fig_map.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax_map.set_extent([56.2, 58.2, 24.2, 25.4], crs=ccrs.PlateCarree())
    ax_map.add_feature(cfeature.LAND, edgecolor='black', facecolor='#d9d9d9'); ax_map.add_feature(cfeature.OCEAN); ax_map.coastlines()
    for sector, start_point in sector_centroids.items():
        ax_map.plot(start_point['lon'], start_point['lat'], 'o', markersize=10, label=f'Centroid: {sector}', zorder=6)
        ax_map.plot([start_point['lon'], primary_buoy_location['lon']], [start_point['lat'], primary_buoy_location['lat']], '--', transform=ccrs.Geodetic(), zorder=5)
    ax_map.plot(primary_buoy_location['lon'], primary_buoy_location['lat'], 'X', color='black', markersize=14, label='Buoy (18m depth)', zorder=10)
    ax_map.set_title('Wave Propagation Paths', fontsize=16); ax_map.legend()
    fig_map.savefig(os.path.join(OUTPUT_PATH, OUTPUT_MAP_FILENAME), bbox_inches='tight')
    
    # Create and save the final DataFrame
    df_features = pd.DataFrame(path_features).set_index('Sector')
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    df_features.to_csv(os.path.join(OUTPUT_PATH, OUTPUT_CSV_FILENAME))
    
    print("\n--- Task 2 Complete ---")
    print(f"Saved plots to: {OUTPUT_PATH}")
    print("\nFinal Path Features (with manual depth):")
    print(df_features)