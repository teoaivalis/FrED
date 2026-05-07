import os
import shutil
from pathlib import Path

base_path = Path("/home/yourname/neurips/artbench-10-imagefolder")
merged_path = Path("/home/yourname/neurips/artbench-merged")

os.makedirs(merged_path, exist_ok=True)

for folder in ["train", "test"]:
    current_dir = base_path / folder
    
    if not current_dir.is_dir():
        continue
        
    for movement in os.listdir(current_dir):
        if movement.startswith('.'):
            continue
            
        source_folder = current_dir / movement
        dest_folder = merged_path / movement
        
        if source_folder.is_dir():
            os.makedirs(dest_folder, exist_ok=True)
            
            for img in os.listdir(source_folder):
                if img.startswith('.'):
                    continue
                shutil.copy(source_folder / img, dest_folder / img)
print(f"Dataset merged into {merged_path}")
