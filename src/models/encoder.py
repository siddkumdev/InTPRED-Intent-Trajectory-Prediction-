import torch
import torch.nn as nn

class TrajectoryLSTMEncoder(nn.Module):
    """
    Processes the past (x,y) coordinates using an LSTM network.
    Excellent for capturing sequential momentum over short time frames.
    """
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=1):
        super(TrajectoryLSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # The LSTM processes the sequence of (x, y) coordinates
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True  # Expects input shape: [Batch, Time, Features]
        )

    def forward(self, x):
        # x shape: [Batch_Size, Past_Frames (4), Features (2)]
        
        # h0 and c0 are the initial hidden and cell states (defaulted to zero)
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)

        # Pass the sequence through the LSTM
        # out: The hidden states for every single frame
        # (h_n, c_n): The final states after seeing the very last frame
        out, (h_n, c_n) = self.lstm(x, (h0, c0))

        # We only care about the network's "final thought" after observing 
        # the entire 2-second history. We extract the last hidden state.
        final_hidden_state = h_n[-1] # Shape: [Batch, Hidden_dim]

        return final_hidden_state


class TrajectoryTransformerEncoder(nn.Module):
    """
    Processes the past coordinates using a Transformer architecture.
    Better at handling long-range dependencies and avoiding vanishing gradients.
    """
    def __init__(self, input_dim=2, d_model=64, nhead=4, num_layers=2):
        super(TrajectoryTransformerEncoder, self).__init__()
        
        # Transformers perform poorly on raw 2D inputs. We must project the 
        # (x, y) coordinates into a richer, higher-dimensional embedding space first.
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # The core Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
    def forward(self, x):
        # x shape: [Batch, Past_Frames (4), Features (2)]
        
        # 1. Project 2D coordinates into d_model space
        x_emb = self.input_projection(x) # Shape: [Batch, 4, 64]
        
        # 2. Pass through the Transformer
        encoded_seq = self.transformer_encoder(x_emb) # Shape: [Batch, 4, 64]
        
        # 3. Aggregate the sequence into a single state vector.
        # Taking the mean across the time dimension is a standard approach 
        # to summarize the entire sequence's context.
        final_feature = torch.mean(encoded_seq, dim=1) # Shape: [Batch, 64]
        
        return final_feature
    

import torch
import torch.nn as nn

class MapCNNEncoder(nn.Module):
    """
    A lightweight CNN to extract semantic map features.
    Compresses a [Batch, 3, 224, 224] image into a [Batch, hidden_dim] vector.
    """
    def __init__(self, in_channels=3, hidden_dim=64):
        super(MapCNNEncoder, self).__init__()
        
        # Convolutional feature extractor
        self.features = nn.Sequential(
            # Block 1: 224x224 -> 112x112
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Block 2: 112x112 -> 56x56
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 3: 56x56 -> 28x28
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 4: 28x28 -> 14x14
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Adaptive pooling ensures the output is ALWAYS 1x1 spatially, 
        # protecting the network from crashing if the map image size ever changes.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final projection to match the Transformer's hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, map_images):
        # Input: [Batch, 3, 224, 224]
        x = self.features(map_images)  # [Batch, 128, 14, 14]
        x = self.avgpool(x)            # [Batch, 128, 1, 1]
        x = torch.flatten(x, 1)        # [Batch, 128]
        
        # Output: [Batch, 64]
        out = self.projection(x)
        return out