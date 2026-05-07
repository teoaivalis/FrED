import os
import pandas as pd
import xarray as xr
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

ERA5_START = pd.Timestamp('1959-01-01')
ERA5_LIMIT = pd.Timestamp('2023-01-10') 

dataset_name = "yourname/extreme-floods-kg"
era5_path = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr'

base_output_dir = "era5_baselines_clean" 
os.makedirs(base_output_dir, exist_ok=True)

surface_vars = ['10m_u_component_of_wind', '10m_v_component_of_wind', '10m_wind_speed', 
                '2m_temperature', 'mean_sea_level_pressure', 'surface_pressure', 'total_precipitation_24hr']
level_vars = ['geopotential', 'specific_humidity', 'temperature', 'u_component_of_wind', 
              'v_component_of_wind', 'vertical_velocity', 'wind_speed']
target_levels = [1000, 850, 500, 250]
deltas_months = [6, 12]


def safe_name(text):
    """Clean strings for filenames."""
    return "".join([c if c.isalnum() else "_" for c in str(text)])

def get_safe_historical_date(event_date, months_back):
    """
    Calculates a date X months before the event.
    If that date is outside the 1959-2023 ERA5 window, it jumps back 
    by years to maintain seasonality until it finds a valid date.
    """
    target = event_date - pd.DateOffset(months=months_back)
    
    while target > ERA5_LIMIT:
        target -= pd.DateOffset(years=1)
        
    while target < ERA5_START:
        target += pd.DateOffset(years=1)
        
    return target

try:
    ds = load_dataset(dataset_name, split="train", trust_remote_code=True)
    df = pd.DataFrame(ds)
except Exception as e:
    df = pd.read_csv(f"https://huggingface.co/datasets/{dataset_name}/resolve/main/extreme_floods.csv")

ds_era5 = xr.open_zarr(era5_path, storage_options={'token': 'anon'}, consolidated=True)
lat_n, lon_n = 'latitude', 'longitude'
is_lat_descending = bool((ds_era5[lat_n][0] > ds_era5[lat_n][-1]).item())

print(f"Starting Extraction for {len(df)} events...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    event_id = row['event_id']
    event_date = pd.to_datetime(row['date'])
    
    event_folder = os.path.join(base_output_dir, safe_name(event_id))
    os.makedirs(event_folder, exist_ok=True)
    
    lat, lon = row['location']['lat'], row['location']['lon']
    if lat is None or lon is None: continue
    lon_360 = lon % 360
    
    lat_slice = slice(lat + 1.0, lat - 1.0) if is_lat_descending else slice(lat - 1.0, lat + 1.0)
    lon_slice = slice(lon_360 - 1.0, lon_360 + 1.0)
    spatial_subset = ds_era5.sel({lat_n: lat_slice, lon_n: lon_slice})

    for months_back in deltas_months:
        clean_target_date = get_safe_historical_date(event_date, months_back)
        
        filepath = os.path.join(event_folder, f"delta_{months_back}m_normal.csv")
        if os.path.exists(filepath):
            continue

        try:
            patch = spatial_subset.sel(time=clean_target_date, method='nearest').compute()
            
            df_s = patch[surface_vars].to_dataframe().reset_index()
            df_l = patch[level_vars].sel(level=target_levels).to_dataframe().reset_index()
            df_final = pd.merge(df_s, df_l, on=[lat_n, lon_n, 'time'])
            
            df_final['precipitation_mm_24h'] = df_final['total_precipitation_24hr'] * 1000
            df_final['event_id'] = event_id
            df_final['delta_months'] = months_back
            df_final['is_flood'] = 0 
            df_final['actual_era5_time'] = clean_target_date # Store for verification
            
            df_final.to_csv(filepath, index=False)
            
        except Exception as e:
            print(f"Skipping {event_id} delta {months_back}m: {e}")
