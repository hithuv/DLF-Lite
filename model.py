import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int = 50):
        super().__init__()
        pe = torch.zeros(max_len, hidden_dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, hidden_dim, 2).float() *
                        (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # (max_len, hidden_dim)

    def forward(self, x: torch.Tensor):
        # x: (B, T, hidden_dim)
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
        # 1) token embedding
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        # 2) positional encoding
        self.pos_enc = PositionalEncoding(hidden_dim, max_len)
        # 3) transformer encoder
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor):
        # x: (B, T, in_dim)
        h = self.input_proj(x)     # (B, T, H)
        h = self.pos_enc(h)        # add positional
        h = self.encoder(h)        # (B, T, H)
        return h.mean(dim=1)       # (B, H)





class LateFusionTransformer(nn.Module):
    def __init__(
        self,
        D_text: int,
        D_audio: int,
        D_vision: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        # per-modality transformers
        self.text_enc = ModalityTransformer(
            D_text, hidden_dim, n_heads, n_layers, dropout, max_len=50
        )
        self.audio_enc = ModalityTransformer(
            D_audio, hidden_dim, n_heads, n_layers, dropout, max_len=50
        )
        self.vision_enc = ModalityTransformer(
            D_vision, hidden_dim, n_heads, n_layers, dropout, max_len=50
        )

        # learned modality embeddings for late fusion
        self.modality_tokens = nn.Parameter(torch.randn(3, hidden_dim))

        # late-fusion transformer (3 tokens)
        fuse_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.fusion = nn.TransformerEncoder(fuse_layer, num_layers=n_layers)

        # classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 7),
        )

    def forward(self, text, audio, vision):
        """
        Args:
          text:  (B, 50, D_text)
          audio: (B, 50, D_audio)
          vision:(B, 50, D_vision)
        Returns:
          logits: (B, 7)
        """
        # 1) per-modality encoding
        t = self.text_enc(text)       # (B, H)
        a = self.audio_enc(audio)     # (B, H)
        v = self.vision_enc(vision)   # (B, H)

        # 2) add modality token embeddings
        t = t + self.modality_tokens[0]
        a = a + self.modality_tokens[1]
        v = v + self.modality_tokens[2]

        # 3) stack and fuse
        x = torch.stack([t, a, v], dim=1)  # (B, 3, H)
        fused = self.fusion(x)             # (B, 3, H)

        # 4) pool and classify
        pooled = fused.mean(dim=1)         # (B, H)
        return self.classifier(pooled)     # (B, 7)