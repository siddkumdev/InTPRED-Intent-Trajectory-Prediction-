import torch
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

class NuScenesTrajectoryDataset(Dataset):
    def __init__(self, nusc, category_filters=None, past_frames=2, future_frames=3, transform=None):
        if category_filters is None:
            category_filters = ['human.pedestrian', 'vehicle.bicycle']
            
        self.nusc = nusc
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.seq_length = past_frames + future_frames
        self.transform = transform
        
        # --- HACKATHON MODE: Load Raw PNGs into Memory ---
        self.raw_maps = {}
        for map_record in self.nusc.map:
            # map_record['filename'] looks like 'maps/53992ee3023e549...png'
            map_path = os.path.join(self.nusc.dataroot, map_record['filename'])
            if os.path.exists(map_path):
                # Load as a grayscale image and convert to PyTorch Tensor
                img = Image.open(map_path).convert('L')
                self.raw_maps[map_record['token']] = TF.to_tensor(img)
            else:
                print(f"Warning: Could not find map image at {map_path}")
        
        self.valid_sequences = []
        self.sequence_metadata = [] 
        
        print("Pre-processing trajectory windows. This might take a moment...")
        self._extract_all_sequences(category_filters)
        print(f"Extraction complete! Found {len(self.valid_sequences)} valid trajectory windows.")

    def _extract_all_sequences(self, category_filters):
        """Iterates through all instances and chunks their paths into valid time windows."""
        for instance in self.nusc.instance:
            category = self.nusc.get('category', instance['category_token'])
            
            # Check if this instance matches our targeted categories
            is_valid_category = any(cat in category['name'] for cat in category_filters)
            if not is_valid_category:
                continue
                
            # Extract the full continuous path for this instance
            full_path = self._get_instance_trajectory(instance)
            
            # If it's too short, skip it
            if len(full_path) < self.seq_length:
                continue
                
            # --- FIND THE MAP TOKEN FOR THIS SPECIFIC INSTANCE ---
            first_ann = self.nusc.get('sample_annotation', instance['first_annotation_token'])
            sample = self.nusc.get('sample', first_ann['sample_token'])
            scene = self.nusc.get('scene', sample['scene_token'])
            log = self.nusc.get('log', scene['log_token'])
            map_token = log['map_token'] # We use the token instead of the name now!
            
            # Slice into sliding windows
            for i in range(len(full_path) - self.seq_length + 1):
                window = full_path[i : i + self.seq_length]
                self.valid_sequences.append(window)
                
                # Grab the global X and Y at the exact moment of prediction (T=0)
                current_global_x = window[self.past_frames - 1, 0]
                current_global_y = window[self.past_frames - 1, 1]
                
                self.sequence_metadata.append((map_token, current_global_x, current_global_y))
                
    def _get_instance_trajectory(self, instance):
        """Follows an instance's annotations to build its full (x, y) coordinate history."""
        trajectory = []
        first_token = instance['first_annotation_token']
        current_ann = self.nusc.get('sample_annotation', first_token)
        
        while current_ann is not None:
            x, y, _ = current_ann['translation']
            trajectory.append([x, y])
            
            next_token = current_ann['next']
            if next_token == "":
                break
            current_ann = self.nusc.get('sample_annotation', next_token)
            
        return np.array(trajectory)

    def get_map_crop(self, map_token, global_x, global_y, patch_size_meters=50.0):
        """Mathematically slices a bounding box out of the raw PNG map."""
        map_tensor = self.raw_maps.get(map_token)
        
        if map_tensor is None:
            # Fallback to a blank image if something goes wrong
            return torch.zeros((3, 224, 224))
            
        # The nuScenes standard map resolution
        PIXELS_PER_METER = 10.0
        
        # Convert global coordinates to pixel locations on the PNG
        center_x_px = int(global_x * PIXELS_PER_METER)
        center_y_px = int(global_y * PIXELS_PER_METER)
        patch_size_px = int(patch_size_meters * PIXELS_PER_METER)
        
        # Calculate the top-left pixel for the crop
        top = center_y_px - (patch_size_px // 2)
        left = center_x_px - (patch_size_px // 2)
        
        # Torchvision's crop function is brilliant: if our bounding box goes off 
        # the edge of the map, it automatically pads it with zeros (black pixels).
        cropped = TF.crop(map_tensor, top, left, patch_size_px, patch_size_px)
        
        # Squeeze it down to 224x224 so our CNN can read it
        resized = TF.resize(cropped, [224, 224], antialias=True)
        
        # Our CNN expects 3 color channels, so we just duplicate the grayscale mask 3 times
        return resized.repeat(3, 1, 1)
    
    def __len__(self):
        return len(self.valid_sequences)

    def __getitem__(self, idx):
        # 1. Get the coordinates
        sequence = self.valid_sequences[idx]
        past_coords = sequence[:self.past_frames]
        future_coords = sequence[self.past_frames:]
        
        # Normalization (Agent-Centric Coordinates)
        origin = past_coords[-1].copy() 
        past_normalized = past_coords - origin
        future_normalized = future_coords - origin
        
        # Apply Transforms
        if self.transform:
            x, y = self.transform(past_normalized, future_normalized)
        else:
            x = torch.tensor(past_normalized, dtype=torch.float32)
            y = torch.tensor(future_normalized, dtype=torch.float32)
        
        # 2. Get the Map Image!
        map_token, global_x, global_y = self.sequence_metadata[idx]
        map_tensor = self.get_map_crop(map_token, global_x, global_y)
        
        return x, y, map_tensor