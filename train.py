import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split # NEW: Imported random_split
import torch.optim as optim
import os

# --- Import our custom modules ---
from nuscenes.nuscenes import NuScenes
from src.datasets.nuscenes_dataset import NuScenesTrajectoryDataset
from src.datasets.transformer import ComposeTransforms, RandomRotate
from src.models.encoder import TrajectoryLSTMEncoder, TrajectoryTransformerEncoder, MapCNNEncoder
from src.models.social_layers import SocialPooling
from src.models.decoder import MultiModalDecoder
from src.utils.metrics import WTALoss, compute_distances, compute_min_ade, compute_min_fde
from src.utils.visualization import plot_multimodal_predictions

# ==========================================
# 1. The Master Model Wrapper
# ==========================================
class TrajectoryPredictor(nn.Module):
    def __init__(self, hidden_dim=64, num_modes=2, future_frames=3, use_transformer=True):
        super(TrajectoryPredictor, self).__init__()
        
        # 1. The Momentum Stream
        if use_transformer:
            self.encoder = TrajectoryTransformerEncoder(input_dim=2, d_model=hidden_dim, nhead=4, num_layers=2)
            print("Initialized with: TRANSFORMER Encoder")
        else:
            self.encoder = TrajectoryLSTMEncoder(input_dim=2, hidden_dim=hidden_dim)
            print("Initialized with: LSTM Encoder")
            
        # 2. The Visual Stream
        self.map_encoder = MapCNNEncoder(in_channels=3, hidden_dim=hidden_dim)
        
        # 3. The Fusion Layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
        # 4. Neighbors & Decoder
        self.social_pooling = SocialPooling(hidden_dim=hidden_dim)
        self.decoder = MultiModalDecoder(hidden_dim=hidden_dim, future_frames=future_frames, num_modes=num_modes)

    def forward(self, past_coords, map_images):
        # A. Process Momentum
        traj_features = self.encoder(past_coords) # -> [Batch, 64]
        
        # B. Process Visual Map
        map_features = self.map_encoder(map_images) # -> [Batch, 64]
        
        # C. Late Fusion: Glue them together and project back to 64
        fused_state = torch.cat([traj_features, map_features], dim=1) # -> [Batch, 128]
        fused_state = self.fusion_layer(fused_state) # -> [Batch, 64]
        
        # D. Social Context 
        current_positions = past_coords[:, -1, :] 
        social_context = self.social_pooling(fused_state, current_positions)
        
        # E. Final combination and Decode
        combined_state = fused_state + social_context
        trajectories, confidences = self.decoder(combined_state)
        
        return trajectories, confidences

# ==========================================
# 2. The Main Training Loop
# ==========================================
def main():
    # --- Configuration ---
    DATAROOT = r'data' 
    BATCH_SIZE = 32 
    EPOCHS = 70
    LEARNING_RATE = 1e-3
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    print(f"--- Starting Training on {DEVICE} ---")
    
    os.makedirs('outputs', exist_ok=True)

    # --- Data Loading & Splitting ---
    print("Loading nuScenes database...")
    nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=False)
    
    train_transform = ComposeTransforms([
        RandomRotate(max_angle_degrees=180)
    ])
    
    # 1. Create the full dataset
    import copy

    # 1. Create the dataset WITHOUT transforms first for validation
    val_full_dataset = NuScenesTrajectoryDataset(nusc, past_frames=2, future_frames=3, transform=None)

    # 2. Create the dataset WITH transforms for training
    train_full_dataset = NuScenesTrajectoryDataset(nusc, past_frames=2, future_frames=3, transform=train_transform)

    # 3. Lock seed and generate indices
    torch.manual_seed(42)
    dataset_size = len(val_full_dataset)
    indices = torch.randperm(dataset_size).tolist()
    train_size = int(0.8 * dataset_size)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # 4. Use PyTorch's Subset
    from torch.utils.data import Subset
    train_dataset = Subset(train_full_dataset, train_indices)
    val_dataset = Subset(val_full_dataset, val_indices)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    # --- Initialization ---
    model = TrajectoryPredictor(hidden_dim=64, num_modes=3, use_transformer=True).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = WTALoss(alpha=0.05, temperature=0.5) 

    # --- The Epoch Loop ---
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        
        # --- Train on TRAIN_DATALOADER ---
        for batch_idx, (past_coords, gt_coords, map_images) in enumerate(train_dataloader):
            
            past_coords = past_coords.to(DEVICE).float()
            gt_coords = gt_coords.to(DEVICE).float()
            map_images = map_images.to(DEVICE)
            
            # Note: Checking model.training is redundant here since we set model.train() above,
            # but leaving it as it works fine.
            if model.training: 
                 noise = torch.randn_like(past_coords) * 0.02
                 past_coords_noisy = past_coords + noise
            else:
                 past_coords_noisy = past_coords

            optimizer.zero_grad()
            
            pred_traj, pred_conf = model(past_coords_noisy, map_images)
            loss = criterion(pred_traj, pred_conf, gt_coords)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

            

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {avg_loss:.4f}")
        
        # --- Validation & Visualization ---
        model.eval()
        val_ade_total = 0.0
        val_fde_total = 0.0
        
        with torch.no_grad():
            # --- Evaluate on the ENTIRE VAL_DATALOADER ---
            for val_past, val_gt, val_map in val_dataloader:
                val_past = val_past.to(DEVICE).float()
                val_gt = val_gt.to(DEVICE).float()
                val_map = val_map.to(DEVICE)
                
                val_pred, val_conf = model(val_past, val_map)
                
                distances = compute_distances(val_pred, val_gt)
                # .item() extracts the float from the tensor so we can sum it
                val_ade_total += compute_min_ade(distances).item()
                val_fde_total += compute_min_fde(distances).item()
            
            # Calculate averages across all validation batches
            avg_val_ade = val_ade_total / len(val_dataloader)
            avg_val_fde = val_fde_total / len(val_dataloader)
            
            print(f"   -> Val minADE: {avg_val_ade:.3f}m | Val minFDE: {avg_val_fde:.3f}m")
            
            # Visualization: This just grabs the final batch from the loop above
            plot_path = f"outputs/epoch_{epoch+1}_vis.png"
            plot_multimodal_predictions(val_past, val_gt, val_pred, val_conf, sample_idx=0, save_path=plot_path)

    print("Training Complete! Check the 'outputs/' folder for trajectory progression images.")
    torch.save(model.state_dict(), "outputs/final_model.pth")

if __name__ == '__main__':
    main()