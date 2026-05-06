import os
import pandas as pd
import xarray as xr
from datasets import load_dataset


ds = load_dataset("teoaivalis/extreme-floods-kg")

print(ds)

print(ds["train"][0])

import pandas as pd
df = pd.DataFrame(ds["train"])
print(df.head())

era5_path = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr'

ds_era5 = xr.open_zarr(era5_path)

era5_path = 'gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr'
ds_era5 = xr.open_zarr(era5_path, consolidated=True)

base_output_dir = "era5_ground_truth_timedelta_previous_day"
os.makedirs(base_output_dir, exist_ok=True)

surface_vars = [
    '10m_u_component_of_wind', '10m_v_component_of_wind', '10m_wind_speed', 
    '2m_temperature', 'mean_sea_level_pressure', 'surface_pressure', 
    'total_precipitation_24hr'
]
level_vars = [
    'geopotential', 'specific_humidity', 'temperature', 
    'u_component_of_wind', 'v_component_of_wind', 
    'vertical_velocity', 'wind_speed'
]
target_levels = [1000, 850, 500, 250]

lat_n, lon_n = 'latitude', 'longitude'
is_lat_descending = True 
era5_limit = pd.Timestamp('2023-01-10')

def safe_name(text):
    return "".join([c if c.isalnum() else "_" for c in str(text)])


for _, row in df.iterrows():
    event_id = row['event_id']
    event_date = pd.to_datetime(row['date'])
    
    if event_date > era5_limit:
        continue

    event_folder = os.path.join(base_output_dir, safe_name(event_id))
    os.makedirs(event_folder, exist_ok=True)
    
    #target_time = event_date
    target_time = event_date - pd.Timedelta(days=1)

    points = [("center", row['location']['lat'], row['location']['lon'])]
    for reg in row.get('affected_regions', []):
        points.append((reg['region'], reg['lat'], reg['lon']))

    print(f"\n>>> ERA5 Truth for {event_id} | Target Date: {target_time.date()}")

    for name, lat, lon in points:
        s_region = safe_name(name)
        lon_360 = lon % 360
        filepath = os.path.join(event_folder, f"{s_region}_ERA5_truth.csv")
        
        if os.path.exists(filepath):
            print(f"    [EXISTS] {name}")
            continue

        try:
            lat_slice = slice(lat + 1.0, lat - 1.0) # Higher to Lower
            lon_slice = slice(lon_360 - 1.0, lon_360 + 1.0)
            
            subset = ds_era5.sel({lat_n: lat_slice, lon_n: lon_slice})
            
            patch = subset.sel(time=target_time, method='nearest').compute()
            
            df_s = patch[surface_vars].to_dataframe().reset_index()
            df_l = patch[level_vars].sel(level=target_levels).to_dataframe().reset_index()
            df_final = pd.merge(df_s, df_l, on=[lat_n, lon_n, 'time'])
            
            df_final['precipitation_mm_24h'] = df_final['total_precipitation_24hr'] * 1000
            df_final['event_id'] = event_id
            df_final['region_label'] = name
            
            df_final.to_csv(filepath, index=False)
            print(f"    [SAVED]  {name}")

        except Exception as e:
            print(f"    [ERROR]  {name}: {e}")
            continue

print("\n--- ERA5 EXTRACTION COMPLETE ---")
