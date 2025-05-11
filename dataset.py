import numpy as np
import torch
from torch.utils.data import Dataset

class MoseiDataset(Dataset):
    """
    Loads aligned_mosei_dataset.npy with keys 'train','valid','test'.
    Cleans ±inf in audio and z-scores using train-split stats.
    """
    def __init__(self, npy_path, split="train", stats=None):
        data = np.load(npy_path, allow_pickle=True).item()
        sd = data.get(split)
        if sd is None:
            raise ValueError(f"Split '{split}' not found in {npy_path}")

        text_np = sd["text"]    # (N,50,300)
        audio_np = sd["audio"]   # (N,50,74)  may contain ±inf
        vision_np = sd["vision"]  # (N,50,35)
        labels_np = sd["labels"].astype(int)

        # 1) clean ±inf in audio
        audio_np = np.nan_to_num(audio_np, nan=0.0, posinf=0.0, neginf=0.0)

        # 2) compute train‐split stats
        if split == "train":
            stats = {
                "t_mean": text_np.mean((0,1)),
                "t_std": text_np.std((0,1)) + 1e-6,
                "a_mean": audio_np.mean((0,1)),
                "a_std": audio_np.std((0,1)) + 1e-6,
                "v_mean": vision_np.mean((0,1)),
                "v_std": vision_np.std((0,1)) + 1e-6,
            }
        elif stats is None:
            raise RuntimeError("Need train-split stats for valid/test")

        # 3) z-score each modality
        self.X_text = (text_np - stats["t_mean"]) / stats["t_std"]
        self.X_audio = (audio_np - stats["a_mean"]) / stats["a_std"]
        self.X_vision = (vision_np - stats["v_mean"]) / stats["v_std"]
        self.y7 = labels_np
        self.stats = stats

    def __len__(self):
        return len(self.y7)

    def __getitem__(self, idx):
        return {
            "text": torch.from_numpy(self.X_text[idx]).float(),
            "audio": torch.from_numpy(self.X_audio[idx]).float(),
            "vision": torch.from_numpy(self.X_vision[idx]).float(),
            "label7": torch.tensor(self.y7[idx], dtype=torch.long),
        }
