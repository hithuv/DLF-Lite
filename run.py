import argparse
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from config import Config
from utils import get_data_loaders
from model import LateFusionTransformer7
from train import train_epoch
from eval import eval_epoch

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",   default="config.json")
    p.add_argument("--npy_path", default="aligned_mosei_dataset.npy")
    args = p.parse_args()
    cfg = Config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_data_loaders(
        args.npy_path, cfg.train["batch_size"]
    )

    # infer dims
    sample = next(iter(train_loader))
    D_text, D_audio, D_vision = (
        sample["text"].shape[-1],
        sample["audio"].shape[-1],
        sample["vision"].shape[-1]
    )

    model = LateFusionTransformer7(
        D_text, D_audio, D_vision,
        hidden_dim=cfg.model["hidden_dim"],
        n_heads=cfg.model["n_heads"],
        n_layers=cfg.model["n_layers"],
        dropout=cfg.model["dropout"]
    ).to(device)

    total_steps = len(train_loader) * cfg.train["epochs"]
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.train["lr"],
        weight_decay=cfg.train["weight_decay"]
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.train["warmup_ratio"] * total_steps),
        num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()

    best_val = float("inf")
    for epoch in range(1, cfg.train["epochs"] + 1):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, cfg.train["max_grad_norm"], device
        )
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch:02d}  "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_model.pt")

    # final test
    model.load_state_dict(torch.load("best_model.pt"))
    te_loss, te_acc = eval_epoch(model, test_loader, criterion, device)
    print(f"\nTest ▶ loss={te_loss:.4f} acc={te_acc:.4f}")

if __name__ == "__main__":
    main()
