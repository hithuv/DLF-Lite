import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int = 50):
        super().__init__()
        pe = torch.zeros(max_len, hidden_dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        # x: (B, T, H)
        return x + self.pe[: x.size(1)]


class ModalityTransformer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        max_len: int = 50,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor):
        # x: (B, T, in_dim)
        h = self.input_proj(x)    # (B, T, H)
        h = self.pos_enc(h)       # Add positional encoding (B, T, H)
        h = self.encoder(h)       # (B, T, H)
        return h.mean(dim=1)      # (B, H)


class CrossModalAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor):
        # q:  (B, 1, H)
        # kv: (B, M, H)
        attn_out, _ = self.attn(q, kv, kv)   # (B, 1, H)
        out = q + self.dropout(attn_out)
        return self.norm(out).squeeze(1)     # (B, H)


class LateFusionWithCrossModalOrtho(nn.Module):
    """
    Baseline2 + Orthogonality:
     - Per-modality transformers
     - Cross-modal attention
     - Returns (logits, text_feat, av_feat) for orthogonality loss
    """
    def __init__(
        self,
        D_text: int,
        D_audio: int,
        D_vision: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Per-modality encoders
        self.text_enc = ModalityTransformer(D_text, hidden_dim, n_heads, n_layers, dropout)
        self.audio_enc = ModalityTransformer(D_audio, hidden_dim, n_heads, n_layers, dropout)
        self.vision_enc = ModalityTransformer(D_vision, hidden_dim, n_heads, n_layers, dropout)

        # Cross-modal attention
        self.cross_attn = CrossModalAttentionBlock(hidden_dim, n_heads, dropout)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, text, audio, vision):
        """
        Returns:
          logits:   (B, 3)
          t_feat:   (B, H) — text representation
          av_feat:  (B, H) — combined audio+vision representation
        """
        # Encode modalities
        t = self.text_enc(text)      # (B, H)
        a = self.audio_enc(audio)    # (B, H)
        v = self.vision_enc(vision)  # (B, H)

        # Cross-attend
        t_q = t.unsqueeze(1)                         # (B, 1, H)
        kv_t = torch.stack([a, v], dim=1)            # (B, 2, H)
        t2 = self.cross_attn(t_q, kv_t)              # (B, H)

        a_q = a.unsqueeze(1)                         # (B, 1, H)
        kv_a = torch.stack([t, v], dim=1)            # (B, 2, H)
        a2 = self.cross_attn(a_q, kv_a)              # (B, H)

        v_q = v.unsqueeze(1)                         # (B, 1, H)
        kv_v = torch.stack([t, a], dim=1)            # (B, 2, H)
        v2 = self.cross_attn(v_q, kv_v)              # (B, H)

        # Fuse enhanced features
        x = torch.cat([t2, a2, v2], dim=1)           # (B, 3H)
        logits = self.classifier(x)                  # (B, 3)

        return logits, t, a, v