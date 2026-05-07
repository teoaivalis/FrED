import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Using device: {device}")

train_roots = [
    "/home/yourname/floods_kg/reanalysis_data/era5_ground_truth",
    "/home/yourname/floods_kg/reanalysis_data/era5_baselines_clean"
    #"/home/yourname/floods_kg/reanalysis_data/era5_ground_truth_timedelta_next_day",
    #"/home/yourname/floods_kg/reanalysis_data/era5_ground_truth_timedelta_previous_day"
]


target_day_label = "era5_ground_truth"

save_emb_path = "era5_cnn_512_event_day_only.npy"
save_model_path = "weather_autoencoder_baseline.pth"

surface_vars = ['2m_temperature', 'mean_sea_level_pressure']
level_vars = ['temperature', 'specific_humidity', 'u_component_of_wind', 'v_component_of_wind', 'geopotential']
levels = [1000, 850, 500, 250]


def physics_aware_loss(pred, target):
    l1_loss = F.l1_loss(pred, target)
    pred_dx = pred[..., 1:] - pred[..., :-1]
    target_dx = target[..., 1:] - target[..., :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    grad_loss = F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)
    return l1_loss + (0.5 * grad_loss)


class ResBlock3D(nn.Module):
    def __init__(self, channels):
        super(ResBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class WeatherEncoder3D(nn.Module):
    def __init__(self, in_channels=7, out_dim=512):
        super(WeatherEncoder3D, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.res1 = ResBlock3D(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2)) 
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.res2 = ResBlock3D(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2)) 
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 2 * 2 * 2, out_dim)
        )
    def forward(self, x):
        x = self.stem(x)
        x = self.pool1(self.res1(x))
        x = self.pool2(self.res2(self.conv2(x)))
        return self.fc(x)

class WeatherDecoder3D(nn.Module):
    def __init__(self, in_dim=512, out_channels=7):
        super(WeatherDecoder3D, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, 512), nn.ReLU())
        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.res3 = ResBlock3D(32)
        self.up2 = nn.ConvTranspose3d(32, 16, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.res4 = ResBlock3D(16)
        self.out_conv = nn.Conv3d(16, out_channels, kernel_size=3, padding=1)
    def forward(self, x):
        x = self.fc(x).view(-1, 64, 2, 2, 2)
        x = self.res3(self.up1(x))
        x = self.res4(self.up2(x))
        return self.out_conv(x)

class WeatherAutoencoder3D(nn.Module):
    def __init__(self):
        super(WeatherAutoencoder3D, self).__init__()
        self.encoder = WeatherEncoder3D()
        self.decoder = WeatherDecoder3D()
    def forward(self, x):
        emb = self.encoder(x)
        rec = self.decoder(emb)
        return rec, emb


def process_to_3d_cube(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={'lat': 'latitude', 'lon': 'longitude'})
    
    level_tensors = []
    for v in level_vars:
        var_levels = []
        for lvl in levels: 
            lvl_df = df[df['level'] == lvl].sort_values(['latitude', 'longitude'], ascending=[False, True])
            grid = lvl_df[v].values[:64].reshape(8, 8)
            grid = (grid - grid.mean()) / (grid.std() + 1e-6)
            var_levels.append(grid)
        level_tensors.append(np.stack(var_levels))
    
    surf_df = df[df['level'] == 1000].sort_values(['latitude', 'longitude'], ascending=[False, True])
    surf_tensors = []
    for v in surface_vars:
        grid = surf_df[v].values[:64].reshape(8, 8)
        grid = (grid - grid.mean()) / (grid.std() + 1e-6)
        grid_3d = np.repeat(grid[np.newaxis, :, :], 4, axis=0) 
        surf_tensors.append(grid_3d)
    
    cube = np.concatenate([np.stack(level_tensors), np.stack(surf_tensors)], axis=0)
    return torch.from_numpy(cube).float().to(device)


all_tensors = []
valid_metadata = [] 

print(f"\n Scanning {len(train_roots)} directories...")
for root in train_roots:
    day_type = os.path.basename(root)
    folders = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]
    for folder_name in tqdm(folders, desc=f"Loading {day_type}"):
        csv_path = os.path.join(root, folder_name, "center_ERA5_truth.csv")
        if os.path.exists(csv_path):
            try:
                x = process_to_3d_cube(csv_path)
                all_tensors.append(x)
                valid_metadata.append((folder_name, day_type))
            except Exception: continue

dataset_tensor = torch.stack(all_tensors) 
dataloader = DataLoader(TensorDataset(dataset_tensor), batch_size=32, shuffle=True)
print(f" Total Cubes Loaded for Training: {len(all_tensors)}")


autoencoder = WeatherAutoencoder3D().to(device)
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
epochs = 100 

print(f"\n Training Denoising Autoencoder...")
for epoch in range(epochs):
    total_loss = 0
    autoencoder.train()
    for batch in dataloader:
        clean_data = batch[0]
        noise = torch.randn_like(clean_data) * 0.1 
        optimizer.zero_grad()
        reconstruction, _ = autoencoder(clean_data + noise)
        loss = physics_aware_loss(reconstruction, clean_data)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:02d}/{epochs}], Loss: {total_loss/len(dataloader):.4f}")

torch.save(autoencoder.state_dict(), save_model_path)


print(f"\n Extracting embeddings for '{target_day_label}' only...")
autoencoder.eval()
all_embeddings = {}

with torch.inference_mode():
    for i, (folder_name, day_type) in enumerate(tqdm(valid_metadata)):
        # ONLY extract if it matches the main event day directory
        if day_type == target_day_label:
            x = dataset_tensor[i].unsqueeze(0)
            embedding = autoencoder.encoder(x).cpu().numpy().flatten()
            all_embeddings[folder_name] = embedding

np.save(save_emb_path, all_embeddings)
print(f"Extracted {len(all_embeddings)} embeddings to: {save_emb_path}")

pangu_root = "/home/yourname/floods_kg/reanalysis_data/pangu_predictions"
save_pangu_emb_path = "pangu_cnn_512_embeddings.npy"

print(f"\n Scanning Pangu directory: {pangu_root}")
pangu_embeddings = {}

autoencoder.eval()

pangu_folders = [f for f in os.listdir(pangu_root) if os.path.isdir(os.path.join(pangu_root, f))]

with torch.inference_mode():
    for folder_name in tqdm(pangu_folders, desc="Processing Pangu"):
        csv_path = os.path.join(pangu_root, folder_name, "center_ERA5_truth.csv")
        
        if os.path.exists(csv_path):
            try:
                x_pangu = process_to_3d_cube(csv_path) 
                x_pangu = x_pangu.unsqueeze(0) 
                embedding = autoencoder.encoder(x_pangu).cpu().numpy().flatten()
                pangu_embeddings[folder_name] = embedding
            except Exception as e:
                print(f" Error processing {folder_name}: {e}")
                continue


np.save(save_pangu_emb_path, pangu_embeddings)
print(f" Extracted {len(pangu_embeddings)} Pangu embeddings to: {save_pangu_emb_path}")
