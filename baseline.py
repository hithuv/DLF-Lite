# baseline.py

# BASIC LATE FUSION MODEL

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 1) Dataset
class MoseiDataset(Dataset):
    def __init__(self, npy_path, split='train'):
        data = np.load(npy_path, allow_pickle=True).item()[split]
        self.text   = data['text']    # (N,50,300)
        self.audio  = data['audio']   # (N,50,74)
        self.vision = data['vision']  # (N,50,35)
        self.labels = data['labels'].astype(int)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        t = torch.from_numpy(self.text[idx]).float()
        a = torch.from_numpy(self.audio[idx]).float()
        v = torch.from_numpy(self.vision[idx]).float()
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return t, a, v, y

# 2) Model
class LateFusionModel(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.text_fc   = nn.Linear(300, hidden_dim)
        self.audio_fc  = nn.Linear(74,  hidden_dim)
        self.vision_fc = nn.Linear(35,  hidden_dim)
        self.classifier = nn.Linear(hidden_dim * 3, 7)
    def forward(self, t, a, v):
        # t, a, v: (B, 50, D)
        t = t.mean(dim=1)             # (B,300)
        a = a.mean(dim=1)             # (B,74)
        v = v.mean(dim=1)             # (B,35)
        t = torch.relu(self.text_fc(t))
        a = torch.relu(self.audio_fc(a))
        v = torch.relu(self.vision_fc(v))
        x = torch.cat([t, a, v], dim=1)  # (B, 3*H)
        return self.classifier(x)        # (B,7)

# 3) Train / Eval
def train_epoch(model, loader, opt, crit, device):
    model.train()
    total_loss, total_acc, n = 0, 0, 0
    for t,a,v,y in loader:
        t,a,v,y = t.to(device), a.to(device), v.to(device), y.to(device)
        logits = model(t,a,v)
        loss = crit(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        preds = logits.argmax(dim=1)
        total_loss += loss.item() * y.size(0)
        total_acc  += (preds==y).sum().item()
        n += y.size(0)
    return total_loss/n, total_acc/n

def eval_epoch(model, loader, crit, device):
    model.eval()
    total_loss, total_acc, n = 0, 0, 0
    with torch.no_grad():
        for t,a,v,y in loader:
            t,a,v,y = t.to(device), a.to(device), v.to(device), y.to(device)
            logits = model(t,a,v)
            loss = crit(logits, y)
            preds = logits.argmax(dim=1)
            total_loss += loss.item() * y.size(0)
            total_acc  += (preds==y).sum().item()
            n += y.size(0)
    return total_loss/n, total_acc/n

# 4) Main
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data',       type=str, default='aligned_mosei_dataset.npy')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--epochs',     type=int, default=10)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds = MoseiDataset(args.data, split='train')
    val_ds   = MoseiDataset(args.data, split='valid')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size)

    model = LateFusionModel().to(device)
    crit  = nn.CrossEntropyLoss()
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_epoch(model, train_loader, opt, crit, device)
        val_loss, val_acc = eval_epoch(model, val_loader, crit, device)
        print(f"Epoch {epoch:02d}  "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

if __name__ == '__main__':
    main()
