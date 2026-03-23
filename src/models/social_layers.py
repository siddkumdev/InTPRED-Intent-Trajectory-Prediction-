import torch
import torch.nn as nn

class SocialPooling(nn.Module):
    """
    Computes a social context vector for each pedestrian by observing the 
    relative positions and hidden states of all other pedestrians in the scene.
    """
    def __init__(self, hidden_dim=64, embedding_dim=64):
        super(SocialPooling, self).__init__()
        
        # We embed the spatial distance between two pedestrians into a higher dimension
        self.spatial_embedding = nn.Sequential(
            nn.Linear(2, embedding_dim),
            nn.ReLU()
        )
        
        # We combine the spatial embedding with the neighbor's hidden momentum state
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim + hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, hidden_dim)
        )

    def forward(self, hidden_states, current_positions):
        """
        Args:
            hidden_states: Tensor of shape [Num_Agents, Hidden_Dim]. 
                           The outputs from your TrajectoryLSTMEncoder.
            current_positions: Tensor of shape [Num_Agents, 2]. 
                               The exact (x, y) coordinates of each agent at the current timestep.
        Returns:
            pooled_context: Tensor of shape [Num_Agents, Hidden_Dim].
                            The aggregated social awareness vector for each agent.
        """
        num_agents = hidden_states.size(0)
        
        # If there is only one pedestrian in the scene, there is no one to avoid.
        # Return a zero tensor for the social context.
        if num_agents == 1:
            return torch.zeros_like(hidden_states)

        # 1. Compute Pairwise Relative Positions
        # We expand the positions to create a matrix of every agent's distance to every other agent.
        # pos_expanded_1 shape: [Num_Agents, 1, 2]
        # pos_expanded_2 shape: [1, Num_Agents, 2]
        pos_expanded_1 = current_positions.unsqueeze(1)
        pos_expanded_2 = current_positions.unsqueeze(0)
        
        # relative_pos shape: [Num_Agents, Num_Agents, 2]
        # relative_pos[i, j] gives the (x, y) vector pointing from Agent i to Agent j.
        relative_pos = pos_expanded_2 - pos_expanded_1
        
        # 2. Embed the Relative Spatial Distances
        # spatial_emb shape: [Num_Agents, Num_Agents, Embedding_Dim]
        spatial_emb = self.spatial_embedding(relative_pos)
        
        # 3. Combine with Neighbor's Hidden States
        # We expand the hidden states so Agent i can look at Agent j's hidden state.
        # hidden_expanded shape: [1, Num_Agents, Hidden_Dim] -> [Num_Agents, Num_Agents, Hidden_Dim]
        hidden_expanded = hidden_states.unsqueeze(0).expand(num_agents, -1, -1)
        
        # Concatenate spatial distance and hidden states
        # interaction_features shape: [Num_Agents, Num_Agents, Embedding_Dim + Hidden_Dim]
        interaction_features = torch.cat([spatial_emb, hidden_expanded], dim=2)
        
        # Pass through the MLP
        # interaction_features shape: [Num_Agents, Num_Agents, Hidden_Dim]
        interaction_features = self.mlp(interaction_features)
        
        # 4. Mask out self-interactions (an agent doesn't need to avoid itself)
        mask = torch.eye(num_agents, device=hidden_states.device).bool()
        interaction_features[mask] = float('-inf') # Set self-interaction to negative infinity
        
        # 5. Max Pooling across the "Neighbor" dimension (dim=1)
        # This isolates the most dominant/threatening feature from all neighbors for each agent.
        # pooled_context shape: [Num_Agents, Hidden_Dim]
        pooled_context, _ = torch.max(interaction_features, dim=1)
        
        # Replace the -inf back to 0 (in case there were no valid neighbors)
        pooled_context[pooled_context == float('-inf')] = 0.0
        
        return pooled_context