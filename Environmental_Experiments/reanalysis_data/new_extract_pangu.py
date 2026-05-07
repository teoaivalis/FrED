import os
import pandas as pd
import xarray as xr
import numpy as np
from datasets import load_dataset

pangu_path = 'gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr'
base_output_dir = "new_pangu_predictions"
os.makedirs(base_output_dir, exist_ok=True)

target_levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

surface_vars = [
    'mean_sea_level_pressure', 
    '10m_u_component_of_wind', 
    '10m_v_component_of_wind', 
    '2m_temperature'
]
upper_vars = [
    'geopotential', 
    'specific_humidity', 
    'temperature', 
    'u_component_of_wind', 
    'v_component_of_wind'
]
all_vars = surface_vars + upper_vars

def safe_name(text):
    return "".join([c if c.isalnum() else "_" for c in str(text)])

ds_pangu = xr.open_zarr(pangu_path, consolidated=True)

ds = load_dataset("yourname/extreme-floods-kg")
df = pd.DataFrame(ds["train"])

processed_count = 0

for _, row in df.iterrows():
    event_id = row['event_id']
    event_date = pd.to_datetime(row['date']).floor('6h') # Align to 6h steps
    
    if not (pd.Timestamp('2018-01-01') <= event_date <= pd.Timestamp('2022-12-31')):
        continue

    init_time = event_date - pd.Timedelta(hours=24)
    lead_time = pd.Timedelta(hours=24)

    event_folder = os.path.join(base_output_dir, safe_name(event_id))
    os.makedirs(event_folder, exist_ok=True)
    
    points = [("center", row['location']['lat'], row['location']['lon'])]
    for reg in row.get('affected_regions', []):
        points.append((reg['region'], reg['lat'], reg['lon']))

    print(f"\n>>> EVENT: {event_id} | Time: {event_date}")

    for name, lat, lon in points:
        s_region = safe_name(name)
        lon_360 = lon % 360
        filepath = os.path.join(event_folder, f"{s_region}_Pangu_data.nc")
        
        if os.path.exists(filepath):
            continue

        try:
            subset = ds_pangu[all_vars].sel({
                'latitude': slice(lat + 1.0, lat - 1.0), 
                'longitude': slice(lon_360 - 1.0, lon_360 + 1.0),
                'prediction_timedelta': lead_time
            })

            patch = subset.sel(time=init_time, level=target_levels, method='nearest').compute()
            
            if patch.latitude.size == 0:
                print(f"    [EMPTY] {name} at {lat}, {lon_360}")
                continue

            patch.attrs['event_id'] = event_id
            patch.attrs['region_label'] = name
            
            patch.to_netcdf(filepath)
            processed_count += 1
            print(f"    [SAVED NC] {name}")

        except Exception as e:
            print(f"    [ERROR] {name}: {e}")
            continue

print(f"\nFinished! Total 3D Patches extracted: {processed_count}")
