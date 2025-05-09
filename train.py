import torch
from torch.nn.utils import clip_grad_norm_

def train_epoch(model, loader, optimizer, scheduler,
                criterion, max_grad_norm, device):
    model.train()
    total_loss = total_correct = 0
    n_samples = 0

    for batch in loader:
        optimizer.zero_grad()
        x_t = batch["text"].to(device)
        x_a = batch["audio"].to(device)
        x_v = batch["vision"].to(device)
        y7  = batch["label7"].to(device)

        logits = model(x_t, x_a, x_v)
        loss = criterion(logits, y7)

        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        bs = x_t.size(0)
        total_loss    += loss.item() * bs
        total_correct += (logits.argmax(1) == y7).sum().item()
        n_samples     += bs

    return total_loss / n_samples, total_correct / n_samples
