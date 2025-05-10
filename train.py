import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

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
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        preds = logits.argmax(dim=1)
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_acc  += (preds == y).sum().item()
        n_samples  += bs

    return total_loss / n_samples, total_acc / n_samples



def train_epoch_ortho(model, loader, optimizer, criterion, device, ortho_weight: float = 0.1, max_grad_norm: float = 1.0):
    model.train()
    total_loss = 0.0
    total_acc  = 0
    n_samples  = 0
    ortho_loss_avg = 0.0

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
        t_feat = F.normalize(t_feat, p=2, dim=1)
        a_feat = F.normalize(a_feat, p=2, dim=1)
        v_feat = F.normalize(v_feat, p=2, dim=1)
        dot1 = (t_feat * a_feat).sum(dim=1)  # (B,)
        dot2 = (t_feat * v_feat).sum(dim=1)  # (B,)
        dot3 = (a_feat * v_feat).sum(dim=1)  # (B,)
        ortho_loss = ((dot1) ** 2).mean() + ((dot2) ** 2).mean() + ((dot3) ** 2).mean() #scalar

    
        # total loss
        loss = cls_loss + ortho_weight * ortho_loss

        # backward + clip + step
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        # metrics
        preds = logits.argmax(dim=1)
        bs = y.size(0)
        total_loss += loss.item() * bs
        ortho_loss_avg += ortho_loss * bs
        total_acc += (preds == y).sum().item()
        n_samples += bs

    print(f'ortho_loss = {(ortho_loss_avg/n_samples)*ortho_weight}')

    return total_loss / n_samples, total_acc / n_samples


def train_epoch_aux(model, loader, optimizer, criterion, device, max_grad_norm):
    model.train()
    total_loss = 0.0
    total_acc  = 0
    n_samples  = 0

    for batch in loader:
        t = batch["text"].to(device)
        a = batch["audio"].to(device)
        v = batch["vision"].to(device)
        y = batch["label7"].to(device)

        logits, logits_text, logits_audio, logits_video = model(t, a, v)
        loss   = criterion(logits, y)
        loss_text = criterion(logits_text, y)
        loss_audio = criterion(logits_audio, y)
        loss_video = criterion(logits_video, y)
        # Combine losses
        loss = loss + 0.2 * (loss_text + loss_audio + loss_video)

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        preds = logits.argmax(dim=1)
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_acc  += (preds == y).sum().item()
        n_samples  += bs

    return total_loss / n_samples, total_acc / n_samples
