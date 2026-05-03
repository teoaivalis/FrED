import os
import shutil
from pathlib import Path

# Use absolute paths to be safe
base_path = Path("/home/teoaivalis/neurips/artbench-10-imagefolder")
merged_path = Path("/home/teoaivalis/neurips/artbench-merged")

# Create the new root directory
os.makedirs(merged_path, exist_ok=True)

for folder in ["train", "test"]:
    current_dir = base_path / folder
    
    # Skip if it's not a directory (like .DS_Store at the train/test level)
    if not current_dir.is_dir():
        continue
        
    for movement in os.listdir(current_dir):
        # Ignore hidden files like .DS_Store
        if movement.startswith('.'):
            continue
            
        source_folder = current_dir / movement
        dest_folder = merged_path / movement
        
        if source_folder.is_dir():
            os.makedirs(dest_folder, exist_ok=True)
            
            for img in os.listdir(source_folder):
                # Ignore hidden files inside the movement folders
                if img.startswith('.'):
                    continue
                
                # Copying is safer than moving (preserves your original data)
                shutil.copy(source_folder / img, dest_folder / img)

print(f"Success! Dataset merged into {merged_path}")
