import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalDecoder(nn.Module):
    def __init__(self, hidden_dim=64, future_frames=6, num_modes=3):
        super(MultiModalDecoder, self).__init__()
        self.future_frames = future_frames
        self.num_modes = num_modes
        
        # Branch 1: The Trajectory Predictor
        self.traj_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_modes * future_frames * 2)
        )
        
        # --- NEW: Trajectory-Aware Confidence Predictor ---
        # Notice the input size: hidden_dim + the specific coordinates of ONE trajectory
        self.conf_mlp = nn.Sequential(
            nn.Linear(hidden_dim + (future_frames * 2), 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Outputs a single raw score per mode
        )

    def forward(self, final_state):
        batch_size = final_state.size(0)
        
        # 1. Generate Coordinates (Exactly as before)
        raw_trajectories = self.traj_mlp(final_state)
        trajectories = raw_trajectories.view(batch_size, self.num_modes, self.future_frames, 2)
        
        # 2. Generate Confidence (The Upgrade)
        conf_scores = []
        for m in range(self.num_modes):
            # Extract the shape of this specific predicted path
            mode_traj = trajectories[:, m, :, :].view(batch_size, -1) 
            
            # Glue the map/momentum state together with the drawn path.
            # We use .detach() on the trajectory so the confidence gradients 
            # don't accidentally mess up the coordinate weights!
            conf_input = torch.cat([final_state, mode_traj.detach()], dim=1)
            
            # Grade this specific line
            score = self.conf_mlp(conf_input)
            conf_scores.append(score)
            
        # 3. Combine the 3 scores and convert to percentages
        raw_confidences = torch.cat(conf_scores, dim=1) 
        confidences = F.softmax(raw_confidences, dim=1)
        
        return trajectories, confidences