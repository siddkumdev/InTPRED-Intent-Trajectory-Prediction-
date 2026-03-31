"""
model.py
--------
Transformer Encoder-Decoder for pedestrian/cyclist trajectory prediction.
Outputs 3 multi-modal paths and their confidence scores.

Architecture:
  1. Linear input projection       (2 → d_model)
  2. Sinusoidal positional encoding
  3. Social Attention               (agents attend to each other)
  4. Transformer Encoder            (encodes 4-step history)
  5. Learnable future queries       (one embedding per future step)
  6. Transformer Decoder            (cross-attends over encoded history)
  7. Multi-Modal heads:
       traj_proj  → (B, K=3, T_future=6, 2)
       conf_proj  → (B, K=3)
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


class TrajectoryTransformer(nn.Module):
    def __init__(
        self,
        past_steps:   int   = 2,
        future_steps: int   = 3,
        d_model:      int   = 64,
        nhead:        int   = 4,
        num_layers:   int   = 2,
        dim_ff:       int   = 128,
        dropout:      float = 0.1,
        num_modes:    int   = 3,    # number of predicted paths
    ):
        super().__init__()
        self.future_steps = future_steps
        self.num_modes    = num_modes

        # ── Input projection ──────────────────────────────────────────────
        self.input_proj = nn.Linear(2, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)

        # ── Social Attention (agents attend to each other in the batch) ───
        self.social_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True
        )

        # ── Temporal Encoder (processes each agent's own history) ─────────
        enc_layer    = nn.TransformerEncoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)

        # ── Learnable future queries (one per future timestep) ────────────
        self.future_queries = nn.Embedding(future_steps, d_model)

        # ── Decoder (queries attend over encoded past) ────────────────────
        dec_layer    = nn.TransformerDecoderLayer(
            d_model, nhead, dim_ff, dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)

        # ── Multi-Modal output heads ───────────────────────────────────────
        self.traj_proj = nn.Linear(d_model, num_modes * 2)   # K trajectories
        self.conf_proj = nn.Linear(d_model, num_modes)        # K confidences

        # Weight initialisation
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, past_xy: torch.Tensor):
        """
        Args:
            past_xy : (B, PAST_STEPS, 2)
        Returns:
            trajs       : (B, K, FUTURE_STEPS, 2)
            conf_logits : (B, K)
        """
        B = past_xy.size(0)

        # 1. Embed + positional encode
        x = self.pos_enc(self.input_proj(past_xy))   # (B, T_past, D)

        # 2. Social attention — each agent sees all others in the batch
        x_t = x.transpose(0, 1)                       # (T_past, B, D)
        social_x, _ = self.social_attn(x_t, x_t, x_t)
        x = social_x.transpose(0, 1) + x              # residual (B, T_past, D)

        # 3. Temporal encoding
        memory = self.encoder(x)                       # (B, T_past, D)

        # 4. Future queries → decode
        idx = torch.arange(self.future_steps, device=past_xy.device)
        tgt = self.future_queries(idx).unsqueeze(0).expand(B, -1, -1)
        out = self.decoder(tgt, memory)                # (B, T_future, D)

        # 5. Multi-modal projection
        trajs = self.traj_proj(out)                    # (B, T_future, K*2)
        trajs = trajs.view(B, self.future_steps, self.num_modes, 2)
        trajs = trajs.permute(0, 2, 1, 3)             # (B, K, T_future, 2)

        conf_logits = self.conf_proj(out.mean(dim=1))  # (B, K)

        return trajs, conf_logits


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = TrajectoryTransformer()
    dummy = torch.randn(8, 4, 2)
    trajs, conf = model(dummy)
    print("Input :", dummy.shape)   # (8, 4, 2)
    print("Trajs :", trajs.shape)   # (8, 3, 6, 2)
    print("Conf  :", conf.shape)    # (8, 3)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n:,}")
