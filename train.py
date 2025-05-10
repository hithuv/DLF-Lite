import torch
from torch.nn.utils import clip_grad_norm_

def train_epoch(model, loader, optimizer, criterion, device, max_grad_norm):
    model.train()
    total_loss = 0.0
    total_acc  = 0
    n_samples  = 0

    for batch in loader:
        t = batch["text"].to(device)
        a = batch["audio"].to(device)
        v = batch["vision"].to(device)
        y = batch["label7"].to(device)

        logits = model(t, a, v)
        loss   = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_acc  += (preds == y).sum().item()
        n_samples  += bs

    return total_loss / n_samples, total_acc / n_samples



def train_epoch_ortho(
    model,
    loader,
    optimizer,
    criterion,
    device,
    ortho_weight: float = 0.1,
    max_grad_norm: float = 1.0
):
    model.train()
    total_loss = 0.0
    total_acc  = 0
    n_samples  = 0

    for batch in loader:
        # unpack batch
        t = batch["text"].to(device)
        a = batch["audio"].to(device)
        v = batch["vision"].to(device)
        y = batch["label7"].to(device)

        # forward pass returns (logits, text_feat, av_feat)
        logits, t_feat, a_feat, v_feat = model(t, a, v)

        # classification loss
        cls_loss = criterion(logits, y)

        # orthogonality loss: mean of squared dot-products
        dot1        = (t_feat * a_feat).sum(dim=1)  # (B,)
        dot2        = (a_feat * v_feat).sum(dim=1)  # (B,)
        ortho_loss = ((dot1+dot2) ** 2).mean()              # scalar

        # total loss
        loss = cls_loss + ortho_weight * ortho_loss

        # backward + clip + step
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        # metrics
        preds = logits.argmax(dim=1)
        bs    = y.size(0)
        total_loss += loss.item() * bs
        total_acc  += (preds == y).sum().item()
        n_samples  += bs

    return total_loss / n_samples, total_acc / n_samples
