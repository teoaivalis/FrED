import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

class ArtEmbeddingDataset(Dataset):
    def __init__(self, image_npy_path, graph_npy_path):
        # Load pre-computed embeddings
        self.image_embeds = torch.tensor(np.load(image_npy_path), dtype=torch.float32)
        self.graph_embeds = torch.tensor(np.load(graph_npy_path), dtype=torch.float32)
        
    def __len__(self):
        return len(self.image_embeds)

    def __getitem__(self, idx):
        return self.image_embeds[idx], self.graph_embeds[idx]

# Hyperparameters
BATCH_SIZE = 512 
EPOCHS = 100
LEARNING_RATE = 1e-3
TEMPERATURE = 0.07 
SHARED_DIM = 512 

# Setup DataLoader
#dataset = ArtEmbeddingDataset('image_embeddings_vitb32.npy', 'node2vec_embeddings.npy')
#dataset = ArtEmbeddingDataset('image_embeddings_vit336.npy', 'node2vec_embeddings.npy')
#dataset = ArtEmbeddingDataset('image_embeddings_rn5064.npy', 'node2vec_embeddings.npy'))
dataset = ArtEmbeddingDataset('image_embeddings_vit_g_14.npy', 'node2vec_embeddings.npy')
#dataset = ArtEmbeddingDataset('image_embeddings_dinov2.npy', 'node2vec_embeddings.npy')

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

print(f"Dataset loaded: {len(dataset)} paintings. Batches per epoch: {len(dataloader)}")

#MODEL ARCHITECTURE
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.mlp(x)
        return F.normalize(x, p=2, dim=1)

class MultiModalAlignmentModel(nn.Module):
    def __init__(self, img_in_dim=1024, graph_in_dim=512, shared_dim=512):
        super().__init__()
        self.image_proj = ProjectionHead(input_dim=img_in_dim, hidden_dim=512, output_dim=shared_dim)
        self.graph_proj = ProjectionHead(input_dim=graph_in_dim, hidden_dim=512, output_dim=shared_dim)

    def forward(self, img_embeds, graph_embeds):
        z_img = self.image_proj(img_embeds)
        z_graph = self.graph_proj(graph_embeds)
        return z_img, z_graph


#INFONCE LOSS FUNCTION
def info_nce_loss(z_img, z_graph, temperature=0.07):
    logits = torch.matmul(z_img, z_graph.T) / temperature
    
    N = logits.shape[0]
    labels = torch.arange(N).to(logits.device)
    
    loss_i2g = F.cross_entropy(logits, labels)
    loss_g2i = F.cross_entropy(logits.T, labels)
    
    return (loss_i2g + loss_g2i) / 2

# THE TRAINING LOOP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

model = MultiModalAlignmentModel(img_in_dim=1024, graph_in_dim=512, shared_dim=SHARED_DIM).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

print("\nStarting Training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
    
    for batch_img, batch_graph in progress_bar:
        batch_img = batch_img.to(device)
        batch_graph = batch_graph.to(device)
        
        optimizer.zero_grad()
        
        z_img, z_graph = model(batch_img, batch_graph)
        
        loss = info_nce_loss(z_img, z_graph, temperature=TEMPERATURE)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Average Loss: {avg_loss:.4f}")

# SAVE THE MODEL
save_path = "new_multimodal_projection_heads_vitg14_node2vec.pth"

torch.save(model.state_dict(), save_path)
print(f"\n Training Complete! Model weights saved to '{save_path}'")
