import os
import json
import requests
import time
import pandas as pd
from datasets import load_dataset

# --- CONFIGURATION ---
OUTPUT_BASE_DIR = "flood_event_species"
RADIUS = 10  
INAT_URL = "https://api.inaturalist.org/v1/observations/species_counts"
HEADERS = {'User-Agent': 'FloodBiodiversityResearchBot/1.0 (contact: your@email.com)'}

ds = load_dataset("yourname/extreme-floods-kg")
df = pd.DataFrame(ds["train"])

def fetch_species_for_event(event_id, lat, lon):
    """Fetches full iNaturalist JSON and saves it in a structured folder."""
    
    event_folder = os.path.join(OUTPUT_BASE_DIR, str(event_id))
    os.makedirs(event_folder, exist_ok=True)

    params = {
        'lat': lat,
        'lng': lon,
        'radius': RADIUS,
        'quality_grade': 'research'
    }

    try:
        response = requests.get(INAT_URL, params=params, headers=HEADERS)
        response.raise_for_status()
        data = response.json()

        file_path = os.path.join(event_folder, "species_info.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        
        count = len(data.get('results', []))
        print(f" {event_id}: Saved {count} species.")

    except Exception as e:
        print(f" {event_id}: Failed to fetch. Error: {e}")


print(f"Starting processing for {len(df)} events...")

for index, row in df.iterrows():
    event_id = row['event_id']
    
    location = row['location']
    lat = location.get('lat')
    lon = location.get('lon')

    if lat is not None and lon is not None:
        fetch_species_for_event(event_id, lat, lon)
        time.sleep(1)
    else:
        print(f"{event_id}: Missing coordinates. Skipping.")

print("\n--- All events processed ---")
