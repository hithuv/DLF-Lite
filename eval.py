import torch

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = total_correct = 0
    n_samples = 0

    with torch.no_grad():
        for batch in loader:
            x_t = batch["text"].to(device)
            x_a = batch["audio"].to(device)
            x_v = batch["vision"].to(device)
            y7  = batch["label7"].to(device)

            logits = model(x_t, x_a, x_v)
            loss = criterion(logits, y7)

            bs = x_t.size(0)
            total_loss    += loss.item() * bs
            total_correct += (logits.argmax(1) == y7).sum().item()
            n_samples     += bs

    return total_loss / n_samples, total_correct / n_samples
