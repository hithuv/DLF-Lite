import argparse
import torch, torch.nn as nn, torch.optim as optim
from utils import get_data_loaders
from model import LateFusionTransformerMultiHead
from train import train_epoch, eval_epoch

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npy_path',   default='aligned_mosei_dataset.npy')
    p.add_argument('--batch_size', type=int,   default=32)
    p.add_argument('--hidden',     type=int,   default=128)
    p.add_argument('--heads',      type=int,   default=4)
    p.add_argument('--layers',     type=int,   default=2)
    p.add_argument('--lr',         type=float, default=1e-5,
                   help='lower LR to avoid explosion')
    p.add_argument('--epochs',     type=int,   default=15)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader = get_data_loaders(
        args.npy_path, args.batch_size
    )

    # infer dims
    sample   = next(iter(train_loader))
    D_text   = sample['text'].shape[-1]
    D_audio  = sample['audio'].shape[-1]
    D_vision = sample['vision'].shape[-1]

    model = LateFusionTransformerMultiHead(
        D_text, D_audio, D_vision,
        hidden_dim=args.hidden,
        n_heads=args.heads,
        n_layers=args.layers
    ).to(device)

    crit7 = nn.CrossEntropyLoss()
    crit2 = nn.CrossEntropyLoss()
    opt   = optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float('inf')
    for ep in range(1, args.epochs+1):
        tr_loss, tr_acc7, tr_acc2 = train_epoch(
            model, train_loader, opt, crit7, crit2, device
        )
        val_loss, val_acc7, val_acc2 = eval_epoch(
            model, val_loader, crit7, crit2, device
        )
        print(f"Epoch {ep:02d}  "
              f"train_loss={tr_loss:.4f} train_acc7={tr_acc7:.4f} train_acc2={tr_acc2:.4f}  "
              f"val_loss={val_loss:.4f}   val_acc7={val_acc7:.4f}   val_acc2={val_acc2:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), 'best_model.pt')

    # final test
    te_loss, te_acc7, te_acc2 = eval_epoch(
        model, test_loader, crit7, crit2, device
    )
    print(f"\nTest ▶ loss={te_loss:.4f} acc7={te_acc7:.4f} acc2={te_acc2:.4f}")

if __name__ == '__main__':
    main()
