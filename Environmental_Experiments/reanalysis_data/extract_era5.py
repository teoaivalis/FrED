import os
import pandas as pd
import xarray as xr
from datasets import load_dataset


# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("teoaivalis/extreme-floods-kg")

# Print dataset info
print(ds)

# Inspect one record
print(ds["train"][0])

# Convert to pandas DataFrame
import pandas as pd
df = pd.DataFrame(ds["train"])
print(df.head())

#################################################################

# Path to the full ERA5 historical dataset (1959 - Jan 2023)
era5_path = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr'

# Open the Zarr store lazily
ds_era5 = xr.open_zarr(era5_path)

print("ERA5 Dataset Connected.")
# Note: ERA5 often uses 'latitude' and 'longitude' (full names)


#################################################################

# --- 1. SETTINGS & PATHS ---
base_output_dir = "era5_ground_truth"
os.makedirs(base_output_dir, exist_ok=True)

# ERA5 variable names in WeatherBench 2 (WB13)
surface_vars = ['10m_u_component_of_wind', '10m_v_component_of_wind', '10m_wind_speed', '2m_temperature', 'mean_sea_level_pressure', 'surface_pressure', 'total_precipitation_24hr']
level_vars = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity', 'wind_speed']
target_levels = [1000, 850, 500, 250] # Key levels for flood analysis

# Coordinate names in ERA5 WB2
lat_n, lon_n = 'latitude', 'longitude'
is_lat_descending = bool((ds_era5[lat_n][0] > ds_era5[lat_n][-1]).item())
era5_limit = pd.Timestamp('2023-01-10')

def safe_name(text):
    return "".join([c if c.isalnum() else "_" for c in str(text)])

# --- 2. EXTRACTION LOOP ---
print(f"Starting ERA5 Extraction for {len(df)} potential events...")

for _, row in df.iterrows():
    event_id = row['event_id']
    event_date = pd.to_datetime(row['date'])
    
    # Check archive limit
    if event_date > era5_limit:
        continue

    # Create event-specific subfolder
    event_folder = os.path.join(base_output_dir, safe_name(event_id))
    os.makedirs(event_folder, exist_ok=True)
    
    # Build list of all points to extract for this event
    # Format: (FileSuffix, Lat, Lon)
    points = [("center", row['location']['lat'], row['location']['lon'])]
    
    regions_list = row.get('affected_regions', [])
    if isinstance(regions_list, list):
        for reg in regions_list:
            points.append((reg.get('region', 'Unnamed_Region'), reg.get('lat'), reg.get('lon')))

    # Pre-Processing Validation Print
    print(f"\n>>> Processing {event_id} | Date: {event_date.date()} | Points: {len(points)}")

    for name, lat, lon in points:
        if lat is None or lon is None: continue
        
        s_region = safe_name(name)
        lon_360 = lon % 360 # Convert to 0-360 for ERA5
        filepath = os.path.join(event_folder, f"{s_region}_ERA5_truth.csv")
        
        # Resume Capability: Skip if already done
        if os.path.exists(filepath):
            print(f"    [EXISTS] {name}")
            continue

        try:
            # Step A: Spatial Slice (8x8 grid / 2.0 degrees)
            lat_slice = slice(lat + 1.0, lat - 1.0) if is_lat_descending else slice(lat - 1.0, lat + 1.0)
            lon_slice = slice(lon_360 - 1.0, lon_360 + 1.0)
            
            subset = ds_era5.sel({lat_n: lat_slice, lon_n: lon_slice})
            
            # Step B: Time Nearest Selection
            patch = subset.sel(time=event_date, method='nearest').compute()
            
            # Step C: Merge Surface and Atmospheric Levels
            df_s = patch[surface_vars].to_dataframe().reset_index()
            df_l = patch[level_vars].sel(level=target_levels).to_dataframe().reset_index()
            df_final = pd.merge(df_s, df_l, on=[lat_n, lon_n, 'time'])
            
            # Step D: Add metadata and convert units
            df_final['precipitation_mm_24h'] = df_final['total_precipitation_24hr'] * 1000
            df_final['event_id'] = event_id
            df_final['region_label'] = name
            
            df_final.to_csv(filepath, index=False)
            print(f"    [SAVED]  {name}")

        except Exception as e:
            print(f"    [ERROR]  {name}: {e}")
            continue

print("\n--- ERA5 EXTRACTION FINISHED ---")