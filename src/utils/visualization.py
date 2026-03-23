import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_multimodal_predictions(past_coords, gt_coords, pred_coords, confidences, sample_idx=0, save_path=None):
    """
    Plots the past trajectory, the ground truth future, and the K predicted futures.
    
    Args:
        past_coords: Tensor/Array of shape [Batch, Past_Frames, 2]
        gt_coords: Tensor/Array of shape [Batch, Future_Frames, 2]
        pred_coords: Tensor/Array of shape [Batch, Num_Modes, Future_Frames, 2]
        confidences: Tensor/Array of shape [Batch, Num_Modes]
        sample_idx: Which item in the batch to visualize.
        save_path: Optional string path to save the figure instead of showing it.
    """
    # 1. Convert PyTorch tensors to NumPy arrays for Matplotlib
    if isinstance(past_coords, torch.Tensor):
        past = past_coords[sample_idx].detach().cpu().numpy()
        truth = gt_coords[sample_idx].detach().cpu().numpy()
        preds = pred_coords[sample_idx].detach().cpu().numpy()
        confs = confidences[sample_idx].detach().cpu().numpy()
    else:
        past = past_coords[sample_idx]
        truth = gt_coords[sample_idx]
        preds = pred_coords[sample_idx]
        confs = confidences[sample_idx]

    plt.figure(figsize=(8, 8))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title("Multi-Modal Trajectory Prediction", fontsize=14, fontweight='bold')
    plt.xlabel("X Coordinate (meters)", fontsize=12)
    plt.ylabel("Y Coordinate (meters)", fontsize=12)

    # 2. Plot the Past Trajectory (Blue)
    # We plot the lines and the individual coordinate nodes
    plt.plot(past[:, 0], past[:, 1], color='blue', linewidth=2, label='Past Trajectory', zorder=3)
    plt.scatter(past[:, 0], past[:, 1], color='blue', s=40, zorder=4)
    
    # Mark the exact origin point (the last known frame, ideally at 0,0)
    plt.scatter(past[-1, 0], past[-1, 1], color='black', s=80, marker='X', label='Current Position', zorder=5)

    # 3. Plot the Ground Truth Future (Green)
    # We connect the last past frame to the first future frame so the line is continuous
    truth_full = np.vstack([past[-1:], truth])
    plt.plot(truth_full[:, 0], truth_full[:, 1], color='green', linewidth=2.5, linestyle='--', label='Ground Truth', zorder=3)
    plt.scatter(truth[:, 0], truth[:, 1], color='green', s=40, marker='*', zorder=4)

    # 4. Plot the Multi-Modal Predictions (Warm Colors)
    # We use a color map and map the confidence score to the line's opacity (alpha)
    colors = ['orange', 'purple', 'red']
    num_modes = preds.shape[0]
    
    for mode in range(num_modes):
        pred_path = preds[mode]
        confidence = confs[mode]
        
        # Connect to origin for a continuous line
        pred_full = np.vstack([past[-1:], pred_path])
        
        # Higher confidence = thicker, more opaque line
        alpha_val = max(0.3, confidence) # Ensure even low-confidence paths are slightly visible
        line_weight = 1.5 + (confidence * 2) 
        
        plt.plot(pred_full[:, 0], pred_full[:, 1], color=colors[mode % len(colors)], 
                 linewidth=line_weight, alpha=alpha_val, linestyle='-', 
                 label=f'Prediction {mode+1} (Conf: {confidence*100:.1f}%)', zorder=2)
        plt.scatter(pred_path[:, 0], pred_path[:, 1], color=colors[mode % len(colors)], s=20, alpha=alpha_val, zorder=2)

    # 5. Final formatting
    plt.legend(loc='best')
    plt.axis('equal') # Ensures 1 meter X looks the same as 1 meter Y (no stretching)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

import matplotlib.animation as animation

def save_animated_radar(past_coords, gt_coords, pred_coords, confidences, sample_idx=0, save_path="outputs/radar_anim.gif"):
    """
    Creates an animated Bird's Eye View (Radar) GIF of the predictions unfolding over time.
    """
    # 1. Convert to NumPy
    if isinstance(past_coords, torch.Tensor):
        past = past_coords[sample_idx].detach().cpu().numpy()
        truth = gt_coords[sample_idx].detach().cpu().numpy()
        preds = pred_coords[sample_idx].detach().cpu().numpy()
        confs = confidences[sample_idx].detach().cpu().numpy()
    else:
        past = past_coords[sample_idx]; truth = gt_coords[sample_idx]
        preds = pred_coords[sample_idx]; confs = confidences[sample_idx]

    # 2. Setup the "Radar" Figure
    fig, ax = plt.subplots(figsize=(8, 8))
    # Dark background for a "radar" aesthetic
    ax.set_facecolor('#111111')
    fig.patch.set_facecolor('#111111')
    
    # We need fixed axis limits so the camera doesn't jump around during the video
    # We'll set a 10x10 meter grid around the pedestrian
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.grid(True, color='#333333', linestyle='--', alpha=0.7)
    
    title = ax.set_title("BEV Radar: Multi-Modal Trajectory Sweep", color='white', fontsize=14)
    ax.tick_params(colors='white')

    # 3. Draw the static Past Trajectory (Bright Blue)
    ax.plot(past[:, 0], past[:, 1], color='#00aaff', linewidth=2, marker='o', markersize=4, label='Past')
    ax.scatter(past[-1, 0], past[-1, 1], color='white', s=100, marker='X', zorder=5, label='Current Pos')

    # 4. Initialize the dynamic lines (empty for now)
    truth_line, = ax.plot([], [], color='#00ff00', linewidth=2.5, linestyle='--', label='Ground Truth')
    
    colors = ['#ffaa00', '#aa00ff', '#ff0055'] # Orange, Purple, Pink
    pred_lines = []
    for mode in range(preds.shape[0]):
        alpha_val = max(0.3, confs[mode])
        line_weight = 1.5 + (confs[mode] * 2)
        line, = ax.plot([], [], color=colors[mode], linewidth=line_weight, alpha=alpha_val, 
                        label=f'Mode {mode+1} ({confs[mode]*100:.1f}%)')
        pred_lines.append(line)

    ax.legend(loc='upper right', facecolor='#222222', edgecolor='none', labelcolor='white')

    # 5. The Animation Function (called once per future frame)
    future_frames = truth.shape[0]
    
    def update(frame):
        # We start drawing from the 'current' position (the last past frame)
        current_pos = past[-1:]
        
        # Slice the arrays up to the current animation frame
        truth_slice = np.vstack([current_pos, truth[:frame+1]])
        truth_line.set_data(truth_slice[:, 0], truth_slice[:, 1])
        
        for mode in range(preds.shape[0]):
            pred_slice = np.vstack([current_pos, preds[mode, :frame+1]])
            pred_lines[mode].set_data(pred_slice[:, 0], pred_slice[:, 1])
            
        title.set_text(f"BEV Radar: Sweeping Future T+{((frame+1) * 0.5):.1f}s")
        return [truth_line] + pred_lines + [title]

    # 6. Render and Save the GIF
    print(f"Rendering radar animation to {save_path}...")
    # interval=300 means 300 milliseconds per frame
    ani = animation.FuncAnimation(fig, update, frames=future_frames, interval=300, blit=True)
    
    # Save as GIF (Requires Pillow, which you already have installed)
    ani.save(save_path, writer='pillow')
    plt.close(fig)