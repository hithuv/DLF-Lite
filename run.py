import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from config import Config
from utils import get_data_loaders
from model import LateFusionTransformer
from model_baseline2 import LateFusionWithCrossModal
from improved_ortho import LateFusionWithCrossModalOrtho
from improved_aux import LateFusionWithCrossModalAuxHeads
from train import train_epoch, train_epoch_ortho, train_epoch_aux
from eval import eval_epoch, eval_epoch_ortho, eval_epoch_aux
import torch.nn.functional as F
import pandas as pd

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps: float = 0.1, reduction='mean'):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, preds, target):
        # preds: (B, C), target: (B,)
        log_preds = F.log_softmax(preds, dim=-1)
        nll = -log_preds.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth = -log_preds.mean(dim=-1)
        loss = (1 - self.eps) * nll + self.eps * smooth
        return loss.mean() if self.reduction=='mean' else loss.sum()

def dump_csv(metrics, fname):
    pd.DataFrame(metrics).to_csv(fname, index=False)

def main1(cfg, data):
    train_loader, val_loader, test_loader = get_data_loaders(
        data, cfg.train["batch_size"]
    )

    # infer dims from one batch
    sample = next(iter(train_loader))
    D_text, D_audio, D_vision = (
        sample["text"].shape[-1],
        sample["audio"].shape[-1],
        sample["vision"].shape[-1]
    )

    # model = BasicLateFusionModel(
    #     D_text, D_audio, D_vision,
    #     hidden_dim=cfg.model["hidden_dim"],
    #     dropout=   cfg.model["dropout"]
    # ).to(device)

    model = LateFusionTransformer(
        D_text, D_audio, D_vision,
        hidden_dim=cfg.model["hidden_dim"],
        n_heads=   cfg.model["n_heads"],
        n_layers=  cfg.model["n_layers"],
        dropout=   cfg.model["dropout"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    # criterion = LabelSmoothingCrossEntropy(eps=0.1)
    optimizer = optim.Adam(model.parameters(), lr=cfg.train["lr"])
    # optimizer = optim.AdamW(
    #     model.parameters(),
    #     lr=cfg.train["lr"],
    #     weight_decay=1e-4
    # )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    out = []
    for epoch in range(1, cfg.train["epochs"] + 1):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, criterion,
            device, cfg.train["max_grad_norm"]
        )
        val_loss, val_acc = eval_epoch(
            model, val_loader, criterion, device
        )
        te_loss, te_acc = eval_epoch(
            model, test_loader, criterion, device
        )

        out.append({
          "epoch": epoch,
          "train_loss": tr_loss,
          "train_acc":  tr_acc,
          "val_loss":   val_loss,
          "val_acc":    val_acc,
          "test_loss": te_loss,
          "test_acc": te_acc
        })

        scheduler.step(val_loss)
        print(f"Epoch {epoch:02d}  "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
              f"val_loss={val_loss:.4f}   val_acc={val_acc:.4f}")

    te_loss, te_acc = eval_epoch(
        model, test_loader, criterion, device
    )
    print(f"\nTest ▶ loss={te_loss:.4f} acc={te_acc:.4f}")
    torch.save(model.state_dict(), "models/baseline1.pth")
    dump_csv(out, "csv/baseline1_metrics.csv")



    # import torch
    # model.eval()
    # preds, trues = [], []
    # with torch.no_grad():
    #     for t,a,v,y in val_loader:
    #         ŷ = model(t.to(device),a.to(device),v.to(device)).argmax(1).cpu()
    #         preds.extend(ŷ.tolist())
    #         trues.extend(y.tolist())
    # from sklearn.metrics import confusion_matrix
    # print(confusion_matrix(trues, preds))


def main2(cfg, data):
    train_loader, val_loader, test_loader = get_data_loaders(
        data, cfg.train["batch_size"]
    )

    # infer dims from one batch
    sample = next(iter(train_loader))
    D_text, D_audio, D_vision = (
        sample["text"].shape[-1],
        sample["audio"].shape[-1],
        sample["vision"].shape[-1]
    )

    # model = BasicLateFusionModel(
    #     D_text, D_audio, D_vision,
    #     hidden_dim=cfg.model["hidden_dim"],
    #     dropout=   cfg.model["dropout"]
    # ).to(device)

    model = LateFusionWithCrossModal(
        D_text, D_audio, D_vision,
        hidden_dim=cfg.model["hidden_dim"],
        n_heads=   cfg.model["n_heads"],
        n_layers=  cfg.model["n_layers"],
        dropout=   cfg.model["dropout"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    # criterion = LabelSmoothingCrossEntropy(eps=0.1)
    optimizer = optim.Adam(model.parameters(), lr=cfg.train["lr"])
    # optimizer = optim.AdamW(
    #     model.parameters(),
    #     lr=cfg.train["lr"],
    #     weight_decay=1e-4
    # )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    out = []
    for epoch in range(1, cfg.train["epochs"] + 1):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, criterion,
            device, cfg.train["max_grad_norm"]
        )
        val_loss, val_acc = eval_epoch(
            model, val_loader, criterion, device
        )
        
        te_loss, te_acc = eval_epoch(
            model, test_loader, criterion, device
        )

        out.append({
          "epoch": epoch,
          "train_loss": tr_loss,
          "train_acc":  tr_acc,
          "val_loss":   val_loss,
          "val_acc":    val_acc,
          "test_loss": te_loss,
          "test_acc": te_acc
        })

        scheduler.step(val_loss)
        print(f"Epoch {epoch:02d}  "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
              f"val_loss={val_loss:.4f}   val_acc={val_acc:.4f}")

    te_loss, te_acc = eval_epoch(
        model, test_loader, criterion, device
    )
    print(f"\nTest ▶ loss={te_loss:.4f} acc={te_acc:.4f}")
    torch.save(model.state_dict(), "models/baseline2.pth")
    dump_csv(out, "csv/baseline2_metrics.csv")



    # import torch
    # model.eval()
    # preds, trues = [], []
    # with torch.no_grad():
    #     for t,a,v,y in val_loader:
    #         ŷ = model(t.to(device),a.to(device),v.to(device)).argmax(1).cpu()
    #         preds.extend(ŷ.tolist())
    #         trues.extend(y.tolist())
    # from sklearn.metrics import confusion_matrix
    # print(confusion_matrix(trues, preds))

def main3(cfg, data):
    
    train_loader, val_loader, test_loader = get_data_loaders(
        data, cfg.train["batch_size"]
    )
    # infer dims
    sample = next(iter(train_loader))
    D_text, D_audio, D_vision = (
        sample["text"].shape[-1],
        sample["audio"].shape[-1],
        sample["vision"].shape[-1]
    )

    model = LateFusionWithCrossModalOrtho(
        D_text, D_audio, D_vision,
        hidden_dim=cfg.model["hidden_dim"],
        n_heads=   cfg.model["n_heads"],
        n_layers=  cfg.model["n_layers"],
        dropout=   cfg.model["dropout"]
    ).to(device)

    # classification loss
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.train["lr"])
    # optimizer with weight decay
    # optimizer = optim.AdamW(
    #     model.parameters(),
    #     lr=cfg.train["lr"],
    #     weight_decay=cfg.train.get("weight_decay", 1e-4)
    # )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    ortho_weight = cfg.train.get("ortho_weight", 0.01)
    max_grad_norm = cfg.train.get("max_grad_norm", 1.0)

    print(f'Ortho Weight = {ortho_weight}')

    best_val = float("inf")
    out = []
    for epoch in range(1, cfg.train["epochs"] + 1):
        tr_loss, tr_acc, ortho = train_epoch_ortho(
            model, train_loader, optimizer, criterion,
            device, ortho_weight, max_grad_norm
        )
        val_loss, val_acc = eval_epoch_ortho(
            model, val_loader, criterion, device
        )

        te_loss, te_acc = eval_epoch_ortho(
            model, test_loader, criterion, device
        )

        out.append({
          "epoch": epoch,
          "train_loss": tr_loss,
          "train_acc":  tr_acc,
          "ortho_loss":  ortho,
          "val_loss":   val_loss,
          "val_acc":    val_acc,
          "test_loss": te_loss,
          "test_acc": te_acc
        })

        scheduler.step(val_loss)

        print(f"Epoch {epoch:02d}  "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
              f"val_loss={val_loss:.4f}   val_acc={val_acc:.4f}")

        # if val_loss < best_val:
        #     best_val = val_loss
        #     torch.save(model.state_dict(), "improved_ortho.pth")
            # print("  ↳ Saved new best model")

        te_loss, te_acc = eval_epoch_ortho(
            model, test_loader, criterion, device
        )
        print(f"\nTest loss={te_loss:.4f} acc={te_acc:.4f}")

    te_loss, te_acc = eval_epoch_ortho(
        model, test_loader, criterion, device
    )
    print(f"\nTest ▶ loss={te_loss:.4f} acc={te_acc:.4f}")

    # final save
    torch.save(model.state_dict(), "models/improved_ortho.pth")
    dump_csv(out, "csv/ortho_metrics.csv")

def main4(cfg, data):
    train_loader, val_loader, test_loader = get_data_loaders(
        data, cfg.train["batch_size"]
    )

    # infer dims from one batch
    sample = next(iter(train_loader))
    D_text, D_audio, D_vision = (
        sample["text"].shape[-1],
        sample["audio"].shape[-1],
        sample["vision"].shape[-1]
    )

    # model = BasicLateFusionModel(
    #     D_text, D_audio, D_vision,
    #     hidden_dim=cfg.model["hidden_dim"],
    #     dropout=   cfg.model["dropout"]
    # ).to(device)

    model = LateFusionWithCrossModalAuxHeads(
        D_text, D_audio, D_vision,
        hidden_dim=cfg.model["hidden_dim"],
        n_heads=   cfg.model["n_heads"],
        n_layers=  cfg.model["n_layers"],
        dropout=   cfg.model["dropout"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    # criterion = LabelSmoothingCrossEntropy(eps=0.1)
    optimizer = optim.Adam(model.parameters(), lr=cfg.train["lr"])
    # optimizer = optim.AdamW(
    #     model.parameters(),
    #     lr=cfg.train["lr"],
    #     weight_decay=1e-4
    # )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    out = []
    for epoch in range(1, cfg.train["epochs"] + 1):
        tr_loss, tr_acc, text_l, audio_l, video_l = train_epoch_aux(
            model, train_loader, optimizer, criterion,
            device, cfg.train["aux_weight"], cfg.train["max_grad_norm"]
        )
        val_loss, val_acc = eval_epoch_aux(
            model, val_loader, criterion, device
        )
        
        te_loss, te_acc = eval_epoch_aux(
            model, test_loader, criterion, device
        )
        out.append({
          "epoch":         epoch,
          "train_loss":    tr_loss,
          "train_acc":     tr_acc,
          "aux_text_loss": text_l,
          "aux_audio_loss":audio_l,
          "aux_video_loss":video_l,
          "val_loss":      val_loss,
          "val_acc":       val_acc,
          "test_loss": te_loss,
          "test_acc": te_acc
        })

        scheduler.step(val_loss)
        print(f"Epoch {epoch:02d}  "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}  "
              f"val_loss={val_loss:.4f}   val_acc={val_acc:.4f}")

    te_loss, te_acc = eval_epoch_aux(
        model, test_loader, criterion, device
    )
    print(f"\nTest ▶ loss={te_loss:.4f} acc={te_acc:.4f}")
    torch.save(model.state_dict(), "models/improved_aux.pth")
    dump_csv(out, "csv/aux_metrics.csv")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--data",   default="data/aligned_mosei_dataset.npy")
    parser.add_argument('--run1', action='store_true', help='Run the Late Fusion model (baseline1)')
    parser.add_argument('--run2', action='store_true', help='Run the Cross Attention model (baseline2)')
    parser.add_argument('--run3', action='store_true', help='Run the Orthogonality model (improved3)')
    parser.add_argument('--run4', action='store_true', help='Run the Aux Heads model (improved4)')
    args = parser.parse_args()

    cfg = Config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.run1:
        print("Latefusion")
        main1(cfg, args.data)
    if args.run2:
        print("CrossAttention")
        main2(cfg, args.data)
    if args.run3:
        print("Ortho")
        main3(cfg, args.data)
    if args.run4:
        print("AuxHeads")
        main4(cfg, args.data)
