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
        # x: (B, T, in_dim), where T is fixed (e.g., 50) and no padding
        h = self.input_proj(x)    # (B, T, H)
        h = self.pos_enc(h)       # add positional encoding (B, T, H)
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
        attn_out, _ = self.attn(q, kv, kv)    # (B, 1, H)
        out = q + self.dropout(attn_out)
        return self.norm(out).squeeze(1)      # (B, H)


class LateFusionWithCrossModalAuxHeads(nn.Module):
    """
    Baseline2 + Auxiliary Heads:
     - per-modality Transformers
     - cross-modal attention
     - main classifier
     - auxiliary classifiers for each unimodal representation (t, a, v)
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
        num_classes: int = 3,
    ):
        super().__init__()
        # Per-modality sequence encoders
        self.text_enc = ModalityTransformer(D_text, hidden_dim, n_heads, n_layers, dropout)
        self.audio_enc = ModalityTransformer(D_audio, hidden_dim, n_heads, n_layers, dropout)
        self.vision_enc = ModalityTransformer(D_vision, hidden_dim, n_heads, n_layers, dropout)

        # Cross-modal attention blocks
        self.cross_attn = CrossModalAttentionBlock(hidden_dim, n_heads, dropout)

        # Main classification head
        self.main_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Auxiliary classification heads for each modality
        aux_head_hidden_dim = max(hidden_dim // 2, 1)  # Ensure positive hidden dim
        self.aux_classifier_text = nn.Sequential(
            nn.Linear(hidden_dim, aux_head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(aux_head_hidden_dim, num_classes),
        )
        self.aux_classifier_audio = nn.Sequential(
            nn.Linear(hidden_dim, aux_head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(aux_head_hidden_dim, num_classes),
        )
        self.aux_classifier_vision = nn.Sequential(
            nn.Linear(hidden_dim, aux_head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(aux_head_hidden_dim, num_classes),
        )

    def forward(self, text_data, audio_data, vision_data):
        # text_data, audio_data, vision_data: (B, 50, D_*)
        t = self.text_enc(text_data)     # (B, H)
        a = self.audio_enc(audio_data)   # (B, H)
        v = self.vision_enc(vision_data) # (B, H)

        # Auxiliary predictions
        logits_text_aux = self.aux_classifier_text(t)     # (B, num_classes)
        logits_audio_aux = self.aux_classifier_audio(a)   # (B, num_classes)
        logits_vision_aux = self.aux_classifier_vision(v) # (B, num_classes)

        # Cross-modal attention
        t_q = t.unsqueeze(1)                                # (B, 1, H)
        a_q = a.unsqueeze(1)                                # (B, 1, H)
        v_q = v.unsqueeze(1)                                # (B, 1, H)

        kv_t = torch.stack([a, v], dim=1)                   # (B, 2, H)
        kv_a = torch.stack([t, v], dim=1)                   # (B, 2, H)
        kv_v = torch.stack([t, a], dim=1)                   # (B, 2, H)

        t2 = self.cross_attn(t_q, kv_t)                     # (B, H)
        a2 = self.cross_attn(a_q, kv_a)                     # (B, H)
        v2 = self.cross_attn(v_q, kv_v)                     # (B, H)

        # Main prediction
        x_fused = torch.cat([t2, a2, v2], dim=1)            # (B, 3H)
        main_logits = self.main_classifier(x_fused)         # (B, num_classes)

        return main_logits, logits_text_aux, logits_audio_aux, logits_vision_aux