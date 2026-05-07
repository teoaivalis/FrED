import os
import pandas as pd
import xarray as xr
from datasets import load_dataset

ds = load_dataset("yourname/extreme-floods-kg")

print(ds)

print(ds["train"][0])

import pandas as pd
df = pd.DataFrame(ds["train"])
print(df.head())

pangu_path = 'gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr'
ds_pangu = xr.open_zarr(pangu_path, consolidated=True)

base_output_dir = "pangu_predictions"
os.makedirs(base_output_dir, exist_ok=True)

target_levels = [1000, 850, 500, 250]
pangu_vars = [
    '10m_u_component_of_wind', '10m_v_component_of_wind', '10m_wind_speed', 
    '2m_temperature', 'mean_sea_level_pressure', 'geopotential', 
    'specific_humidity', 'temperature', 'u_component_of_wind', 
    'v_component_of_wind', 'wind_speed'
]

def safe_name(text):
    return "".join([c if c.isalnum() else "_" for c in str(text)])

processed_count = 0
print("Starting Pangu Extraction (Descending Latitude fix)...")

for _, row in df.iterrows():
    event_id = row['event_id']
    event_date = pd.to_datetime(row['date'])
    
    if not (pd.Timestamp('2018-01-01') <= event_date <= pd.Timestamp('2022-12-31')):
        continue

    init_time = event_date - pd.Timedelta(hours=24)
    lead_time = pd.Timedelta(hours=24)

    event_folder = os.path.join(base_output_dir, safe_name(event_id))
    os.makedirs(event_folder, exist_ok=True)
    
    points = [("center", row['location']['lat'], row['location']['lon'])]
    for reg in row.get('affected_regions', []):
        points.append((reg['region'], reg['lat'], reg['lon']))

    print(f"\n>>> EVENT: {event_id} | Points: {len(points)}")

    for name, lat, lon in points:
        s_region = safe_name(name)
        lon_360 = lon % 360
        filepath = os.path.join(event_folder, f"{s_region}_Pangu_pred.csv")
        
        if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
            continue

        try:
            subset = ds_pangu.sel({
                'latitude': slice(lat + 1.0, lat - 1.0), 
                'longitude': slice(lon_360 - 1.0, lon_360 + 1.0),
                'prediction_timedelta': lead_time
            })
            
            patch = subset.sel(time=init_time, level=target_levels, method='nearest').compute()
            
            df_final = patch[pangu_vars].to_dataframe().reset_index()
            
            if df_final.empty:
                print(f"    [EMPTY]  {name} - check coords: {lat}, {lon_360}")
                continue
            
            df_final['event_id'] = event_id
            df_final['region_label'] = name
            df_final['valid_time'] = df_final['time'] + df_final['prediction_timedelta']
            
            df_final.to_csv(filepath, index=False)
            processed_count += 1
            print(f"    [SAVED]  {name}")

        except Exception as e:
            print(f"    [ERROR]  {name}: {e}")
            continue

print(f"\nFinished! Total Pangu files: {processed_count}")
