import numpy as np
import torch
from torch.utils.data import Dataset

class MoseiDataset(Dataset):
    def __init__(self, npy_path, split="train"):
        data = np.load(npy_path, allow_pickle=True).item()
        sd = data.get(split)
        if sd is None:
            raise ValueError(f"Split '{split}' not found in {npy_path}")

        self.X_text   = sd["text"]    # (N,50,300)
        self.X_audio  = sd["audio"]   # (N,50,74)
        self.X_vision = sd["vision"]  # (N,50,35)
        self.y7       = sd["labels"].astype(int)  # 0..6

    def __len__(self):
        return len(self.y7)

    def __getitem__(self, idx):
        return {
            "text":   torch.from_numpy(self.X_text[idx]).float(),
            "audio":  torch.from_numpy(self.X_audio[idx]).float(),
            "vision": torch.from_numpy(self.X_vision[idx]).float(),
            "label7": torch.tensor(self.y7[idx], dtype=torch.long),
        }
