import os
import re

INPUT_DIR = "./disaster_region_by_id"
CLEAN_DIR = "./cleaned_regions"
os.makedirs(CLEAN_DIR, exist_ok=True)

REGION_PATTERN = r",\s*['\"]?([^'\")]+)['\"]?\)"

print(f"Cleaning files from {INPUT_DIR}...")

files_processed = 0

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".txt"):
        continue
        
    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(CLEAN_DIR, filename)

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned_names = []
    for line in lines:
        match = re.search(REGION_PATTERN, line)
        if match:
            name = match.group(1).strip()
            if name and name not in cleaned_names:
                cleaned_names.append(name)

    if cleaned_names:
        with open(output_path, "w", encoding="utf-8") as f_out:
            f_out.write("\n".join(cleaned_names))
        files_processed += 1

print(f"Cleaned {files_processed} files. Check the '{CLEAN_DIR}' folder.")
