import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from utils import get_data_loaders
from model_baseline1 import LateFusionTransformer
from model_baseline2 import LateFusionWithCrossModal
from improved_ortho import LateFusionWithCrossModalOrtho
from improved_aux import LateFusionWithCrossModalAuxHeads
from train import train_epoch, train_epoch_ortho, train_epoch_aux
from eval import eval_epoch, eval_epoch_ortho, eval_epoch_aux
import pandas as pd

# Function to save metrics to CSV
def dump_csv(metrics, fname):
    pd.DataFrame(metrics).to_csv(fname, index=False)

# Run the Late Fusion Transformer model (Baseline 1).
def main1(cfg, data):
    train_loader, val_loader, test_loader = get_data_loaders(data, cfg.train["batch_size"])

    # Infer input dimensions from one batch
    sample = next(iter(train_loader))
    D_text, D_audio, D_vision = (
        sample["text"].shape[-1],
        sample["audio"].shape[-1],
        sample["vision"].shape[-1]
    )

    # Initialize the model
    model = LateFusionTransformer(
        D_text, D_audio, D_vision,
        hidden_dim=cfg.model["hidden_dim"],
        n_heads=cfg.model["n_heads"],
        n_layers=cfg.model["n_layers"],
        dropout=cfg.model["dropout"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.train["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    out = []
    for epoch in range(1, cfg.train["epochs"] + 1):
        # Train and evaluate
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device, cfg.train["max_grad_norm"])
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        te_loss, te_acc = eval_epoch(model, test_loader, criterion, device)

        # Log metrics
        out.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_loss": te_loss,
            "test_acc": te_acc
        })

        scheduler.step(val_loss)
        print(f"Epoch {epoch:02d}  train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    # Final evaluation and save results
    te_loss, te_acc = eval_epoch(model, test_loader, criterion, device)
    print(f"\nTest ▶ loss={te_loss:.4f} acc={te_acc:.4f}")
    torch.save(model.state_dict(), "models/baseline1.pth")
    dump_csv(out, "csv/baseline1_metrics.csv")

# Run the Late Fusion with Cross-Modal Attention model (Baseline 2).
def main2(cfg, data):
    train_loader, val_loader, test_loader = get_data_loaders(data, cfg.train["batch_size"])

    # Infer input dimensions from one batch
    sample = next(iter(train_loader))
    D_text, D_audio, D_vision = (
        sample["text"].shape[-1],
        sample["audio"].shape[-1],
        sample["vision"].shape[-1]
    )

    # Initialize the model
    model = LateFusionWithCrossModal(
        D_text, D_audio, D_vision,
        hidden_dim=cfg.model["hidden_dim"],
        n_heads=cfg.model["n_heads"],
        n_layers=cfg.model["n_layers"],
        dropout=cfg.model["dropout"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.train["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    out = []
    for epoch in range(1, cfg.train["epochs"] + 1):
        # Train and evaluate
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device, cfg.train["max_grad_norm"])
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        te_loss, te_acc = eval_epoch(model, test_loader, criterion, device)

        # Log metrics
        out.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_loss": te_loss,
            "test_acc": te_acc
        })

        scheduler.step(val_loss)
        print(f"Epoch {epoch:02d}  train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    # Final evaluation and save results
    te_loss, te_acc = eval_epoch(model, test_loader, criterion, device)
    print(f"\nTest ▶ loss={te_loss:.4f} acc={te_acc:.4f}")
    torch.save(model.state_dict(), "models/baseline2.pth")
    dump_csv(out, "csv/baseline2_metrics.csv")

# Run the Late Fusion with Orthogonality model (Improved 1).
def main3(cfg, data):
    train_loader, val_loader, test_loader = get_data_loaders(data, cfg.train["batch_size"])

    # Infer input dimensions
    sample = next(iter(train_loader))
    D_text, D_audio, D_vision = (
        sample["text"].shape[-1],
        sample["audio"].shape[-1],
        sample["vision"].shape[-1]
    )

    model = LateFusionWithCrossModalOrtho(
        D_text, D_audio, D_vision,
        hidden_dim=cfg.model["hidden_dim"],
        n_heads=cfg.model["n_heads"],
        n_layers=cfg.model["n_layers"],
        dropout=cfg.model["dropout"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.train["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    ortho_weight = cfg.train.get("ortho_weight", 0.01)
    max_grad_norm = cfg.train.get("max_grad_norm", 1.0)

    print(f"Ortho Weight = {ortho_weight}")

    out = []
    for epoch in range(1, cfg.train["epochs"] + 1):
        # Train and evaluate
        tr_loss, tr_acc, ortho = train_epoch_ortho(
            model, train_loader, optimizer, criterion, device, ortho_weight, max_grad_norm
        )
        val_loss, val_acc = eval_epoch_ortho(model, val_loader, criterion, device)
        te_loss, te_acc = eval_epoch_ortho(model, test_loader, criterion, device)

        # Log metrics
        out.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "ortho_loss": ortho,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_loss": te_loss,
            "test_acc": te_acc
        })

        scheduler.step(val_loss)
        print(f"Epoch {epoch:02d}  train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    # Final evaluation and save results
    te_loss, te_acc = eval_epoch_ortho(model, test_loader, criterion, device)
    print(f"\nTest ▶ loss={te_loss:.4f} acc={te_acc:.4f}")
    torch.save(model.state_dict(), "models/improved_ortho.pth")
    dump_csv(out, "csv/ortho_metrics.csv")

# Run the Late Fusion with Auxiliary Heads model (Improved 2).
def main4(cfg, data):
    train_loader, val_loader, test_loader = get_data_loaders(data, cfg.train["batch_size"])

    # Infer input dimensions
    sample = next(iter(train_loader))
    D_text, D_audio, D_vision = (
        sample["text"].shape[-1],
        sample["audio"].shape[-1],
        sample["vision"].shape[-1]
    )

    model = LateFusionWithCrossModalAuxHeads(
        D_text, D_audio, D_vision,
        hidden_dim=cfg.model["hidden_dim"],
        n_heads=cfg.model["n_heads"],
        n_layers=cfg.model["n_layers"],
        dropout=cfg.model["dropout"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.train["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    out = []
    for epoch in range(1, cfg.train["epochs"] + 1):
        # Train and evaluate
        tr_loss, tr_acc, text_l, audio_l, video_l = train_epoch_aux(
            model, train_loader, optimizer, criterion, device, cfg.train["aux_weight"], cfg.train["max_grad_norm"]
        )
        val_loss, val_acc = eval_epoch_aux(model, val_loader, criterion, device)
        te_loss, te_acc = eval_epoch_aux(model, test_loader, criterion, device)

        # Log metrics
        out.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "aux_text_loss": text_l,
            "aux_audio_loss": audio_l,
            "aux_video_loss": video_l,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_loss": te_loss,
            "test_acc": te_acc
        })

        scheduler.step(val_loss)
        print(f"Epoch {epoch:02d}  train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    # Final evaluation and save results
    te_loss, te_acc = eval_epoch_aux(model, test_loader, criterion, device)
    print(f"\nTest ▶ loss={te_loss:.4f} acc={te_acc:.4f}")
    torch.save(model.state_dict(), "models/improved_aux.pth")
    dump_csv(out, "csv/aux_metrics.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--data", default="data/aligned_mosei_dataset.pkl")
    parser.add_argument('--run1', action='store_true', help='Run the Late Fusion model (baseline1)')
    parser.add_argument('--run2', action='store_true', help='Run the Cross Attention model (baseline2)')
    parser.add_argument('--run3', action='store_true', help='Run the Orthogonality model (improved3)')
    parser.add_argument('--run4', action='store_true', help='Run the Aux Heads model (improved4)')
    args = parser.parse_args()

    cfg = Config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.run1:
        print("Running Late Fusion Transformer (Baseline 1)")
        main1(cfg, args.data)
    if args.run2:
        print("Running Cross Attention Model (Baseline 2)")
        main2(cfg, args.data)
    if args.run3:
        print("Running Orthogonality Model (Improved 3)")
        main3(cfg, args.data)
    if args.run4:
        print("Running Auxiliary Heads Model (Improved 4)")
        main4(cfg, args.data)