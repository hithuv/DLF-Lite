from torch.utils.data import DataLoader
from dataset import MoseiDataset

def get_data_loaders(npy_path, batch_size=32):
    train_ds = MoseiDataset(npy_path, split="train")
    val_ds   = MoseiDataset(npy_path, split="valid")
    test_ds  = MoseiDataset(npy_path, split="test")

    return (
        DataLoader(train_ds, batch_size, shuffle=True),
        DataLoader(val_ds,   batch_size),
        DataLoader(test_ds,  batch_size)
    )
