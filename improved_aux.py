import torch
import torch.nn as nn
import math

# Your original PositionalEncoding - this is fine.
class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int = 50):
        super().__init__()
        pe = torch.zeros(max_len, hidden_dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, hidden_dim, 2).float() *
                        (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        # x: (B, T, H)
        return x + self.pe[: x.size(1)]


# Your original ModalityTransformer - restored as per no padding.
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

    def forward(self, x: torch.Tensor): # No lengths argument needed if no padding
        # x: (B, T, in_dim), where T is fixed (e.g., 50) and no padding
        h = self.input_proj(x)    # (B, T, H)
        h = self.pos_enc(h)       # add positional encoding (B, T, H)
        # No src_key_padding_mask needed if all tokens are real
        h = self.encoder(h)       # (B, T, H) 
        # Simple mean pooling is fine if no padding
        return h.mean(dim=1)      # (B, H)


# Your original CrossModalAttentionBlock - this is fine.
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
        # Since t,a,v are already pooled, no key_padding_mask needed for kv's M dimension.
        attn_out, _ = self.attn(q, kv, kv)    # (B,1,H)
        out = q + self.dropout(attn_out)
        return self.norm(out).squeeze(1)      # (B, H)


class LateFusionWithCrossModalAuxHeads(nn.Module):
    """
    Baseline2 + Auxiliary Heads:
     - per-modality Transformers (original version, no explicit padding logic)
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
        n_layers: int = 1, # n_layers for ModalityTransformer
        dropout: float = 0.1,
        num_classes: int = 7, # Number of output classes for sentiment
    ):
        super().__init__()
        # 1) per-modality sequence encoders
        self.text_enc   = ModalityTransformer(D_text,   hidden_dim, n_heads, n_layers, dropout)
        self.audio_enc  = ModalityTransformer(D_audio,  hidden_dim, n_heads, n_layers, dropout)
        self.vision_enc = ModalityTransformer(D_vision, hidden_dim, n_heads, n_layers, dropout)

        # 2) cross-modal attention blocks
        self.cross_attn = CrossModalAttentionBlock(hidden_dim, n_heads, dropout)

        # 3) Main classification head (same as Baseline 2)
        self.main_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), # Takes concatenated t2, a2, v2
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # 4) Auxiliary classification heads for each modality (t, a, v)
        # These take the output of ModalityTransformer directly (B, H)
        # Using a simpler structure for auxiliary heads: Linear -> ReLU -> Linear
        aux_head_hidden_dim = hidden_dim // 2 if hidden_dim // 2 > 0 else hidden_dim # Ensure positive
        if aux_head_hidden_dim == 0 and hidden_dim > 0 : aux_head_hidden_dim = 1 # safety for tiny hidden_dim
        
        self.aux_classifier_text = nn.Sequential(
            nn.Linear(hidden_dim, aux_head_hidden_dim), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(aux_head_hidden_dim, num_classes)
        )
        self.aux_classifier_audio = nn.Sequential(
            nn.Linear(hidden_dim, aux_head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(aux_head_hidden_dim, num_classes)
        )
        self.aux_classifier_vision = nn.Sequential(
            nn.Linear(hidden_dim, aux_head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(aux_head_hidden_dim, num_classes)
        )

    def forward(self, text_data, audio_data, vision_data): # No lengths needed if no padding
        # text_data, audio_data, vision_data: (B, 50, D_*)
        # All sequences are assumed to be of length 50 with real data.

        # 1) Per-modality sequence encoding
        t = self.text_enc(text_data)     # (B, H)
        a = self.audio_enc(audio_data)   # (B, H)
        v = self.vision_enc(vision_data) # (B, H)

        # 2) Generate Auxiliary Predictions from unimodal features (t, a, v)
        logits_text_aux = self.aux_classifier_text(t)     # (B, num_classes)
        logits_audio_aux = self.aux_classifier_audio(a)   # (B, num_classes)
        logits_vision_aux = self.aux_classifier_vision(v) # (B, num_classes)

        # 3) Cross-attend: each modality attends to the other two (same as Baseline 2)
        t_q = t.unsqueeze(1)                                # (B,1,H)
        a_q = a.unsqueeze(1)                                # (B,1,H)
        v_q = v.unsqueeze(1)                                # (B,1,H)

        kv_t = torch.stack([a, v], dim=1)                   # (B,2,H)
        kv_a = torch.stack([t, v], dim=1)                   # (B,2,H)
        kv_v = torch.stack([t, a], dim=1)                   # (B,2,H)
        
        t2 = self.cross_attn(t_q, kv_t)                     # (B, H)
        a2 = self.cross_attn(a_q, kv_a)                     # (B, H)
        v2 = self.cross_attn(v_q, kv_v)                     # (B, H)

        # 4) Fuse enhanced features for main prediction
        x_fused = torch.cat([t2, a2, v2], dim=1)            # (B, 3H)
        main_logits = self.main_classifier(x_fused)         # (B, num_classes)
        
        # Return main logits and auxiliary logits
        return main_logits, logits_text_aux, logits_audio_aux, logits_vision_aux