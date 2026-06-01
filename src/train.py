"""Common Trainer for both ECFP-MLP and GraphConv models.

Identical training conditions (optimizer, scheduler, early stopping, logging)
are applied to both model families so the comparison in notebook 04 is fair.

The Trainer is model-agnostic via a `forward_batch` callback that knows how to
unpack either a TensorDataset batch (x, y, mask) or a PyG Batch.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from .losses import MaskedBCEWithLogitsLoss


# -----------------------------------------------------------------------------
# Batch adapters
# -----------------------------------------------------------------------------
def ecfp_forward(model, batch, device):
    """Unpack (x, y, mask) tensor batch -> (logits, y, mask)."""
    x, y, mask = batch
    x, y, mask = x.to(device), y.to(device), mask.to(device)
    logits = model(x)
    return logits, y, mask


def graph_forward(model, batch, device):
    """Unpack a PyG Batch -> (logits, y, mask)."""
    batch = batch.to(device)
    logits = model(batch)
    y = batch.y.view(logits.shape[0], -1)
    mask = batch.mask.view(logits.shape[0], -1)
    return logits, y, mask


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------
@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 100
    patience: int = 10
    ckpt_dir: str = "/content/drive/MyDrive/tox21/"
    ckpt_name: str = "model_best.pt"
    grad_clip: float = 5.0
    monitor: str = "val_loss"   # lower is better
    verbose: bool = True


@dataclass
class History:
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        forward_batch: Callable,
        config: Optional[TrainConfig] = None,
        device: Optional[str] = None,
    ):
        self.cfg = config or TrainConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loss_fn = loss_fn.to(self.device)
        self.forward_batch = forward_batch
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )
        self.history = History()
        os.makedirs(self.cfg.ckpt_dir, exist_ok=True)
        self.ckpt_path = os.path.join(self.cfg.ckpt_dir, self.cfg.ckpt_name)

    # -- internal epoch loops -------------------------------------------------
    def _run_epoch(self, loader, train: bool) -> float:
        self.model.train(train)
        total, n_batches = 0.0, 0
        torch.set_grad_enabled(train)
        for batch in loader:
            logits, y, mask = self.forward_batch(self.model, batch, self.device)
            loss = self.loss_fn(logits, y, mask)
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip
                )
                self.optimizer.step()
            total += float(loss.detach().cpu())
            n_batches += 1
        torch.set_grad_enabled(True)
        return total / max(n_batches, 1)

    # -- public API -----------------------------------------------------------
    def fit(self, train_loader, valid_loader) -> History:
        best = float("inf")
        bad_epochs = 0
        for epoch in range(1, self.cfg.max_epochs + 1):
            t0 = time.time()
            tr = self._run_epoch(train_loader, train=True)
            va = self._run_epoch(valid_loader, train=False)
            self.scheduler.step(va)
            self.history.train_loss.append(tr)
            self.history.val_loss.append(va)

            improved = va < best - 1e-5
            if improved:
                best = va
                bad_epochs = 0
                torch.save(self.model.state_dict(), self.ckpt_path)
            else:
                bad_epochs += 1

            if self.cfg.verbose:
                print(
                    f"Epoch {epoch:03d} | train {tr:.4f} | val {va:.4f} "
                    f"| best {best:.4f} | {time.time()-t0:.1f}s"
                    + ("  *" if improved else "")
                )
            if bad_epochs >= self.cfg.patience:
                if self.cfg.verbose:
                    print(f"Early stopping at epoch {epoch} (patience reached).")
                break

        # Restore best weights.
        if os.path.exists(self.ckpt_path):
            self.model.load_state_dict(torch.load(self.ckpt_path, map_location=self.device))
        return self.history

    @torch.no_grad()
    def predict(self, loader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (probs, y_true, mask) over a loader (probs = sigmoid logits)."""
        self.model.eval()
        ps, ys, ms = [], [], []
        for batch in loader:
            logits, y, mask = self.forward_batch(self.model, batch, self.device)
            ps.append(torch.sigmoid(logits).cpu().numpy())
            ys.append(y.cpu().numpy())
            ms.append(mask.cpu().numpy())
        return (np.concatenate(ps), np.concatenate(ys), np.concatenate(ms))

    def save_curves(self, path: str, title: str = "Learning curve"):
        """Save a train/val loss curve figure to `path`."""
        import matplotlib.pyplot as plt

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        plt.figure(figsize=(7, 5))
        plt.plot(self.history.train_loss, label="train")
        plt.plot(self.history.val_loss, label="val")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title(title)
        plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        return path
