from torch.utils.data import DataLoader
from dataset import MoseiDataset

def get_data_loaders(npy_path, batch_size):
    # build train to get stats
    train_ds = MoseiDataset(npy_path, split="train", stats=None)
    stats    = train_ds.stats
    val_ds   = MoseiDataset(npy_path, split="valid", stats=stats)
    test_ds  = MoseiDataset(npy_path, split="test",  stats=stats)

    return (
        DataLoader(train_ds, batch_size, shuffle=True),
        DataLoader(val_ds,   batch_size),
        DataLoader(test_ds,  batch_size)
    )
