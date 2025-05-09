import torch
import torch.nn as nn
from transformers import BertModel

class LateFusionTransformer(nn.Module):
    def __init__(self, audio_dim, vision_dim, hidden_dim=128, n_heads=4, n_layers=6, n_classes=7):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.text_proj = nn.Linear(self.bert.config.hidden_size, hidden_dim)
        self.audio_enc = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.vision_enc = nn.Sequential(
            nn.Linear(vision_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads)
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, input_ids, attention_mask, audio, vision):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        txt = self.text_proj(bert_out.pooler_output)
        aud = self.audio_enc(audio)
        vid = self.vision_enc(vision)
        x = torch.stack([txt, aud, vid], dim=0)
        fused = self.fusion(x)
        pooled = fused.mean(dim=0)
        return self.classifier(pooled)
