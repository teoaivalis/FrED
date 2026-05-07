import os
import pandas as pd
import xarray as xr
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("yourname/extreme-floods-kg")

# Print dataset info
print(ds)

# Inspect one record
print(ds["train"][0])

# Convert to pandas DataFrame
import pandas as pd
df = pd.DataFrame(ds["train"])
print(df.head())

#################################################################

# 1. SETUP DATA
# Note: Using the WB2 2020 GraphCast set (Covers late 2019 to Feb 2021)
gc_path = 'gs://weatherbench2/datasets/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr'
ds_gc = xr.open_zarr(gc_path)

base_output_dir = "graphcast_predictions"
os.makedirs(base_output_dir, exist_ok=True)

# 2. CONFIGURATION (Matches your ERA5 setup exactly)
target_levels = [1000, 850, 500, 250]

# Surface variables (GraphCast uses 'total_precipitation' for accumulated rain)
surface_vars = [
    '10m_u_component_of_wind', 
    '10m_v_component_of_wind', 
    '10m_wind_speed', 
    '2m_temperature', 
    'mean_sea_level_pressure', 
    'total_precipitation_24hr', 
    'total_precipitation_6hr'
]

# level_vars remain the same as they are all present in your list
level_vars = [
    'geopotential', 
    'specific_humidity', 
    'temperature', 
    'u_component_of_wind', 
    'v_component_of_wind', 
    'vertical_velocity', 
    'wind_speed'
]

# GraphCast uses 'lat' and 'lon' (short names)
lat_n, lon_n = 'lat', 'lon'
is_lat_descending = False # GraphCast is usually ascending [-90 to 90]

def safe_name(text):
    return "".join([c if c.isalnum() else "_" for c in str(text)])

# 3. EXTRACTION LOOP
print(f"Starting GraphCast Extraction (24h Lead)...")

for _, row in df.iterrows():
    event_id = row['event_id']
    event_date = pd.to_datetime(row['date'])
    
    # GraphCast 2020 Availability Filter
    if not (pd.Timestamp('2019-11-16') <= event_date <= pd.Timestamp('2021-02-01')):
        continue

    # Create subfolder
    event_folder = os.path.join(base_output_dir, safe_name(event_id))
    os.makedirs(event_folder, exist_ok=True)
    
    # Define Time Windows
    # We want the forecast that is VALID at event_date, which was INITIALIZED 24h prior.
    init_time = event_date - pd.Timedelta(hours=24)
    lead_time = pd.Timedelta(hours=24)

    # Build points list (Center + Regions)
    points = [("center", row['location']['lat'], row['location']['lon'])]
    regions_list = row.get('affected_regions', [])
    if isinstance(regions_list, list):
        for reg in regions_list:
            points.append((reg.get('region', 'Unnamed_Region'), reg.get('lat'), reg.get('lon')))

    print(f"\n>>> GraphCast Forecast for {event_id} | Init: {init_time.date()} | Valid: {event_date.date()}")

    for name, lat, lon in points:
        if lat is None or lon is None: continue
        
        s_region = safe_name(name)
        lon_360 = lon % 360
        filepath = os.path.join(event_folder, f"{s_region}_GC_pred.csv")
        
        if os.path.exists(filepath):
            print(f"    [EXISTS] {name}")
            continue

        try:
            # Step A: Spatial & Lead Time Slice
            # We select the 24h lead time immediately to reduce data overhead
            subset = ds_gc.sel({
                lat_n: slice(lat - 1.0, lat + 1.0), 
                lon_n: slice(lon_360 - 1.0, lon_360 + 1.0),
                'prediction_timedelta': lead_time
            })
            
            # Step B: Select Initialization Time (The "Input" moment)
            patch = subset.sel(time=init_time, method='nearest').compute()
            
            # Step C: Extract Surface and Levels
            df_s = patch[surface_vars].to_dataframe().reset_index()
            df_l = patch[level_vars].sel(level=target_levels).to_dataframe().reset_index()
            
            # Step D: Merge on spatial and temporal coords
            # GraphCast adds 'prediction_timedelta' to the index
            df_final = pd.merge(df_s, df_l, on=[lat_n, lon_n, 'time', 'prediction_timedelta'])
            
            # Standardize column for KG (matching ERA5 notebook)
            df_final['precipitation_mm_24h'] = df_final['total_precipitation_24hr'] * 1000
            df_final['event_id'] = event_id
            df_final['region_label'] = name
            
            df_final.to_csv(filepath, index=False)
            print(f"    [SAVED]  {name}")

        except Exception as e:
            print(f"    [ERROR]  {name}: {e}")
            continue

print("\n--- GRAPHCAST EXTRACTION FINISHED ---")