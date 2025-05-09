import torch
import torch.nn as nn

class LateFusionTransformer7(nn.Module):
    def __init__(
        self,
        D_text: int,
        D_audio: int,
        D_vision: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        # Project each modality
        self.text_enc = nn.Sequential(
            nn.Linear(D_text, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.audio_enc = nn.Sequential(
            nn.Linear(D_audio, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.vision_enc = nn.Sequential(
            nn.Linear(D_vision, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Norm + dropout per token
        self.text_norm   = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.audio_norm  = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.vision_norm = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout))

        # Transformer fusion (batch_first, smaller FFN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 7-way classification head
        self.class7 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 7)
        )

    def forward(self, x_text, x_audio, x_vision):
        # Mean-pool time dimension
        t = x_text.mean(dim=1)
        a = x_audio.mean(dim=1)
        v = x_vision.mean(dim=1)

        # Encode each
        t = self.text_norm(self.text_enc(t))
        a = self.audio_norm(self.audio_enc(a))
        v = self.vision_norm(self.vision_enc(v))

        # Stack tokens and fuse
        x = torch.stack([t, a, v], dim=1)  # (B,3,H)
        fused = self.fusion(x)             # (B,3,H)

        # Pool and classify
        pooled = fused.mean(dim=1)         # (B,H)
        return self.class7(pooled)         # (B,7)
