import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_distances(pred_trajectories, gt_trajectories):
    """
    Calculates the Euclidean distance between predicted and ground truth points.
    
    Args:
        pred_trajectories: Shape [Batch, Num_Modes, Future_Frames, 2]
        gt_trajectories: Shape [Batch, Future_Frames, 2]
    Returns:
        distances: Shape [Batch, Num_Modes, Future_Frames]
    """
    # Expand ground truth to broadcast across the 'Num_Modes' dimension
    # Shape becomes: [Batch, 1, Future_Frames, 2]
    gt_expanded = gt_trajectories.unsqueeze(1)
    
    # Calculate Euclidean distance (L2 Norm) along the coordinate dimension (dim=-1)
    distances = torch.norm(pred_trajectories - gt_expanded, p=2, dim=-1)
    
    return distances

def compute_min_ade(distances):
    """
    Calculates the minimum Average Displacement Error (minADE) over all predicted modes.
    ADE is the mean Euclidean distance across all future time steps.
    """
    # Average the distance across the time dimension (Future_Frames)
    # Shape: [Batch, Num_Modes]
    ade_per_mode = distances.mean(dim=-1)
    
    # For each batch item, find the mode with the lowest ADE
    # Shape: [Batch]
    min_ade, _ = torch.min(ade_per_mode, dim=-1)
    
    # Return the mean across the entire batch
    return min_ade.mean()

def compute_min_fde(distances):
    """
    Calculates the minimum Final Displacement Error (minFDE) over all predicted modes.
    FDE is the Euclidean distance at the very last predicted time step.
    """
    # Grab the distances at the final timestep (-1)
    # Shape: [Batch, Num_Modes]
    fde_per_mode = distances[:, :, -1]
    
    # For each batch item, find the mode with the lowest FDE
    # Shape: [Batch]
    min_fde, _ = torch.min(fde_per_mode, dim=-1)
    
    return min_fde.mean()



class WTALoss(nn.Module):
    def __init__(self, alpha=1.0, temperature=0.5):
        super(WTALoss, self).__init__()
        self.regression_loss = nn.SmoothL1Loss(reduction='none') 
        self.alpha = alpha 
        self.temperature = temperature # Controls how strictly we penalize bad lines

    def forward(self, pred_trajectories, pred_confidences, gt_trajectories):
        batch_size = pred_trajectories.size(0)
        
        # Calculate errors for all 3 modes
        with torch.no_grad():
            distances = compute_distances(pred_trajectories, gt_trajectories)
            ade_per_mode = distances.mean(dim=-1) 
            best_mode_indices = torch.argmin(ade_per_mode, dim=-1)
            
            # --- NEW: Soft Targets ---
            # Convert physical error distances into target percentages
            # A lower ADE gets a higher percentage.
            target_confidences = F.softmax(-ade_per_mode / self.temperature, dim=-1)
        
        # 1. Geometry Loss (Still pure WTA! Only train the winner so lines stay spread out)
        batch_indices = torch.arange(batch_size, device=pred_trajectories.device)
        best_trajectories = pred_trajectories[batch_indices, best_mode_indices]
        reg_loss = self.regression_loss(best_trajectories, gt_trajectories).mean()
        
        # 2. Confidence Loss (Train against the physical distances)
        # We use CrossEntropy to teach the network to match its percentages to the actual physics
        cls_loss = torch.sum(-target_confidences * torch.log(pred_confidences + 1e-9), dim=-1).mean()
        
        return reg_loss + (self.alpha * cls_loss)