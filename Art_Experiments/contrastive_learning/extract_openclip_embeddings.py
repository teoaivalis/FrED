#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

import open_clip

ImageFile.LOAD_TRUNCATED_IMAGES = True

def main():
    print("=== Starting OpenCLIP Embedding Extraction ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_name = 'ViT-g-14'
    pretrained_data = 'laion2b_s34b_b88k'
    
    print(f"Loading {model_name} model ({pretrained_data})...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, 
        pretrained=pretrained_data
    )
    model.to(device)
    model.eval()

    embed_dim = model.visual.output_dim
    print(f"Model loaded. Embedding dimension: {embed_dim}\n")

    print("Loading mapping CSVs...")
    try:
        aligned_df = pd.read_csv('aligned_paintings.csv')
        mapping_df = pd.read_csv('wikiart_art_pieces.csv')
    except FileNotFoundError as e:
        print(f"ERROR: Could not find CSV file. {e}")
        sys.exit(1)

    print("Cleaning URLs")
    aligned_df['clean_URL'] = aligned_df['URL'].astype(str).apply(lambda x: x.split('!')[0])
    mapping_df['clean_img'] = mapping_df['img'].astype(str).apply(lambda x: x.split('!')[0])

    mapping_df = mapping_df.drop_duplicates(subset=['clean_img'], keep='first')

    merged_df = pd.merge(aligned_df, mapping_df[['clean_img', 'file_name']], 
                         left_on='clean_URL', right_on='clean_img', how='left')
    
    missing_mappings = merged_df['file_name'].isna().sum()
    print(f"Total images to process: {len(merged_df)}")
    if missing_mappings > 0:
        print(f"WARNING: {missing_mappings} URLs did not match a local 'file_name'.")

    image_embeddings = []
    missing_files = []
    
    image_folder = "/home/teoaivalis/contrastive_learning/worldkg_images"

    print("\nfeature extraction...")
    
    with torch.no_grad():
        for idx, row in tqdm(merged_df.iterrows(), total=len(merged_df), desc="Processing Images"):
            
            if pd.isna(row['file_name']):
                missing_files.append((idx, row['URL'], "Missing from CSV mapping"))
                image_embeddings.append(np.zeros(embed_dim, dtype=np.float32))
                continue
                
            local_path = os.path.join(image_folder, str(row['file_name']))
            
            try:
                image = Image.open(local_path).convert("RGB")
                image_input = preprocess(image).unsqueeze(0).to(device)
                image_features = model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                vector = image_features.cpu().numpy().flatten().astype(np.float32)
                image_embeddings.append(vector)
                
            except Exception as e:
                error_msg = str(e)
                missing_files.append((idx, local_path, error_msg))
                image_embeddings.append(np.zeros(embed_dim, dtype=np.float32))
                
                if "out of memory" in error_msg.lower():
                    torch.cuda.empty_cache()
                
                if len(missing_files) == 50:
                    sys.exit(1)

    image_embeddings_matrix = np.stack(image_embeddings)
    
    output_filename = "image_embeddings_vit_g_14.npy"
    np.save(output_filename, image_embeddings_matrix)
    print(f"Successfully saved {output_filename} with shape: {image_embeddings_matrix.shape}")

    if missing_files:
        for err in missing_files[:5]:
            print(f" - Row {err[0]}: {err[1]} -> {err[2]}")
    else:
        print("\nimages processed!")

if __name__ == "__main__":
    main()
