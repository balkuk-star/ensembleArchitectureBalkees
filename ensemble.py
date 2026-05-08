from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from utils import compute_metrics_from_logits, dice_bce_loss, ensure_binary_output


class WeightedEnsemble(nn.Module):
    def __init__(self, resunetpp: nn.Module, transfuse: nn.Module, wdffnet: nn.Module):
        super().__init__()
        self.r = resunetpp.eval()
        self.t = transfuse.eval()
        self.w = wdffnet.eval()

        for m in [self.r, self.t, self.w]:
            for p in m.parameters():
                p.requires_grad = False

        self.weight_head = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=1),
        )

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            r = ensure_binary_output(self.r(x))
            t = ensure_binary_output(self.t(x))
            w = ensure_binary_output(self.w(x))

        stack = torch.cat([r, t, w], dim=1)
        weights = torch.softmax(self.weight_head(stack), dim=1)
        fused = (weights * stack).sum(dim=1, keepdim=True)
        return fused


def train_ensemble_head(model: WeightedEnsemble, train_loader, val_loader, device: torch.device, epochs: int = 10, lr: float = 1e-4):
    model.to(device)
    optimizer = torch.optim.Adam(model.weight_head.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "val_dice": []}
    best_dice = -1.0
    best_path = "best_ensemble.pth"

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = dice_bce_loss(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        val_losses = []
        dices = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_losses.append(dice_bce_loss(logits, y).item())
                m = compute_metrics_from_logits(logits, y)
                dices.append(m["Dice"])

        train_loss = float(np.mean(losses))
        val_loss = float(np.mean(val_losses))
        val_dice = float(np.mean(dices))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        print(f"[Ensemble] Epoch {ep}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_dice={val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    return history, best_path
