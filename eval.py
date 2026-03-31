import torch
from torch.utils.data import DataLoader, random_split # NEW: Imported random_split
from nuscenes.nuscenes import NuScenes

# Import your custom modules
from src.datasets.nuscenes_dataset import NuScenesTrajectoryDataset
from src.utils.metrics import compute_distances
from src.utils.visualization import save_animated_radar, plot_multimodal_predictions
from train import TrajectoryPredictor  # Import the Master Model wrapper

def evaluate_model():
    # --- Configuration ---
    DATAROOT = r'data' 
    MODEL_PATH = 'outputs/final_model.pth'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading Dataset...")
    nusc = NuScenes(version='v1.0-mini', dataroot=DATAROOT, verbose=False)
    
    # 1. Load the ENTIRE dataset (No transforms needed for eval)
    full_dataset = NuScenesTrajectoryDataset(nusc, past_frames=2, future_frames=3, transform=None)
    
    # 2. SYNC THE SPLIT: Lock the random seed to match train.py exactly!
    torch.manual_seed(42)
    
    # 3. Calculate identical sizes and split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, eval_dataset = random_split(full_dataset, [train_size, val_size]) # We ignore the train part here
    
    # 4. Create loader ONLY from the eval split (shuffle=False for consistent debugging)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)

    print(f"Loading Model from {MODEL_PATH}...")
    model = TrajectoryPredictor(hidden_dim=64, num_modes=3, use_transformer=True)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval() 

    # --- Enhanced Debugging Metrics ---
    total_min_ade = 0.0
    total_min_fde = 0.0
    total_top1_ade = 0.0 # Error of the model's MOST confident prediction
    total_top1_fde = 0.0
    misses_2m = 0 # Count how many times the best prediction was off by > 2 meters
    num_samples = 0
    
    print("\n--- Running Evaluation ---")
    
    with torch.no_grad():
        for i, (past_coords, gt_coords, map_images) in enumerate(eval_loader):
            # Move to GPU and strictly enforce float32 to prevent NumPy double errors
            past_coords = past_coords.to(DEVICE).float()
            gt_coords = gt_coords.to(DEVICE).float()
            map_images = map_images.to(DEVICE)
            
            pred_traj, pred_conf = model(past_coords, map_images)
            
            # --- Extract Detailed Mode Statistics ---
            distances = compute_distances(pred_traj, gt_coords) # Shape: [1, 3, 6]
            ade_per_mode = distances.mean(dim=-1)[0]            # Shape: [3]
            fde_per_mode = distances[:, :, -1][0]               # Shape: [3]
            confs = pred_conf[0]                                # Shape: [3]
            
            # Find the "Geometric Best" (The line closest to reality)
            best_geometric_idx = torch.argmin(ade_per_mode).item()
            min_ade = ade_per_mode[best_geometric_idx].item()
            min_fde = fde_per_mode[best_geometric_idx].item()
            
            # Find the "Confidence Best" (The line the model THOUGHT was best)
            best_conf_idx = torch.argmax(confs).item()
            top1_ade = ade_per_mode[best_conf_idx].item()
            top1_fde = fde_per_mode[best_conf_idx].item()
            
            # Aggregate totals
            total_min_ade += min_ade
            total_min_fde += min_fde
            total_top1_ade += top1_ade
            total_top1_fde += top1_fde
            
            if min_fde > 2.0:
                misses_2m += 1
                
            num_samples += 1
            
            # --- Detailed Console Logging for the first 20 samples ---
            if i < 20:
                print(f"\n[Sample {i+1} Debug Breakdown]")
                for m in range(model.decoder.num_modes):
                    marker = " (<- Model Choice)" if m == best_conf_idx else ""
                    marker += " (<- Actual Best)" if m == best_geometric_idx else ""
                    print(f"  Mode {m+1}: Conf = {confs[m]*100:04.1f}% | ADE = {ade_per_mode[m]:.3f}m | FDE = {fde_per_mode[m]:.3f}m{marker}")
                
                # Save both static PNG and animated GIF for comparison
                png_path = f"outputs/eval_sample_{i+1}.png"
                gif_path = f"outputs/radar_sample_{i+1}.gif"
                plot_multimodal_predictions(past_coords, gt_coords, pred_traj, pred_conf, sample_idx=0, save_path=png_path)
                save_animated_radar(past_coords, gt_coords, pred_traj, pred_conf, sample_idx=0, save_path=gif_path)
                
    # --- Final Aggregated Metrics ---
    print("\n========================================")
    print("          FINAL BENCHMARK SCORES          ")
    print("========================================")
    print(f"Total Trajectories Evaluated: {num_samples}")
    print("\n--- Geometric Limits (Trajectory Head) ---")
    print("If these are low, the network is successfully drawing valid paths.")
    print(f"  Overall minADE: {(total_min_ade/num_samples):.3f} meters")
    print(f"  Overall minFDE: {(total_min_fde/num_samples):.3f} meters")
    
    print("\n--- Model Accuracy (Confidence Head) ---")
    print("If these are close to the Geometric Limits, the confidence head is working perfectly.")
    print(f"  Overall Top-1 ADE: {(total_top1_ade/num_samples):.3f} meters")
    print(f"  Overall Top-1 FDE: {(total_top1_fde/num_samples):.3f} meters")
    
    miss_rate = (misses_2m / num_samples) * 100
    print(f"\n  Miss Rate (>2.0m Error): {miss_rate:.1f}%")
    print("========================================\n")

if __name__ == '__main__':
    evaluate_model() 