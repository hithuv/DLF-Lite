import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

def train_epoch(model, loader, optimizer, criterion, device, max_grad_norm):
    model.train()
    total_loss = 0.0
    total_acc = 0
    n_samples = 0

    for batch in loader:
        t = batch["text"].to(device)
        a = batch["audio"].to(device)
        v = batch["vision"].to(device)
        y = batch["label3"].to(device)

        logits = model(t, a, v)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        preds = logits.argmax(dim=1)
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_acc += (preds == y).sum().item()
        n_samples += bs

    return total_loss / n_samples, total_acc / n_samples


def train_epoch_ortho(model, loader, optimizer, criterion, device, ortho_weight: float = 0.1, max_grad_norm: float = 1.0):
    model.train()
    total_loss = 0.0
    total_acc = 0
    n_samples = 0
    ortho_loss_sum = 0.0

    for batch in loader:
        t = batch["text"].to(device)
        a = batch["audio"].to(device)
        v = batch["vision"].to(device)
        y = batch["label3"].to(device)

        # Forward pass
        logits, t_feat, a_feat, v_feat = model(t, a, v)

        # Classification loss
        cls_loss = criterion(logits, y)

        # Orthogonality loss
        t_feat = F.normalize(t_feat, p=2, dim=1)
        a_feat = F.normalize(a_feat, p=2, dim=1)
        v_feat = F.normalize(v_feat, p=2, dim=1)
        dot1 = (t_feat * a_feat).sum(dim=1)
        dot2 = (t_feat * v_feat).sum(dim=1)
        dot3 = (a_feat * v_feat).sum(dim=1)
        ortho_loss = ((dot1) ** 2).mean() + ((dot2) ** 2).mean() + ((dot3) ** 2).mean()

        # Total loss
        loss = cls_loss + ortho_weight * ortho_loss

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        preds = logits.argmax(dim=1)
        bs = y.size(0)
        total_loss += loss.item() * bs
        ortho_loss_sum += ortho_loss.item() * bs
        total_acc += (preds == y).sum().item()
        n_samples += bs

    avg_ortho_loss = ortho_loss_sum / n_samples
    print(f'ortho_loss = {avg_ortho_loss * ortho_weight}')

    return total_loss / n_samples, total_acc / n_samples, avg_ortho_loss * ortho_weight


def train_epoch_aux(model, loader, optimizer, criterion, device, aux_weight: float = 0.05, max_grad_norm: float = 1.0):
    model.train()
    total_loss = 0.0
    loss_text_sum = 0.0
    loss_audio_sum = 0.0
    loss_video_sum = 0.0
    total_acc = 0
    n_samples = 0

    for batch in loader:
        t = batch["text"].to(device)
        a = batch["audio"].to(device)
        v = batch["vision"].to(device)
        y = batch["label3"].to(device)

        # Forward pass
        logits, logits_text, logits_audio, logits_video = model(t, a, v)

        # Losses
        loss = criterion(logits, y)
        loss_text = criterion(logits_text, y)
        loss_audio = criterion(logits_audio, y)
        loss_video = criterion(logits_video, y)

        # Combine losses
        loss = loss + aux_weight * (loss_text + loss_audio + loss_video)

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        preds = logits.argmax(dim=1)
        bs = y.size(0)
        total_loss += loss.item() * bs
        loss_text_sum += loss_text.item() * bs
        loss_audio_sum += loss_audio.item() * bs
        loss_video_sum += loss_video.item() * bs
        total_acc += (preds == y).sum().item()
        n_samples += bs

    avg_text = loss_text_sum / n_samples
    avg_audio = loss_audio_sum / n_samples
    avg_video = loss_video_sum / n_samples

    return total_loss / n_samples, total_acc / n_samples, avg_text, avg_audio, avg_video