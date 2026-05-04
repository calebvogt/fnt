"""MAD U-Net training pipeline.

Trains a binary segmentation model on painted spectrogram tiles. Loss is
masked BCE + Dice — pixels outside committed bands contribute zero
supervision, so label-sparse files cost nothing.

Heavy deps (``torch``, ``segmentation_models_pytorch``) are imported
lazily inside :func:`train_unet` so the module is safe to import from
the GUI even when those packages aren't installed.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
@dataclass
class UNetTrainingConfig:
    project_dir: str
    run_name: str = ""               # auto-filled from timestamp if empty
    encoder_name: str = "resnet18"
    encoder_weights: Optional[str] = "imagenet"
    n_epochs: int = 30
    batch_size: int = 8
    learning_rate: float = 1e-3
    val_fraction: float = 0.20
    device: str = "auto"             # 'auto' | 'cuda' | 'mps' | 'cpu'

    # SLEAP-style early stopping: halt when val_loss fails to improve by
    # more than ``early_stop_min_delta`` for ``early_stop_patience``
    # consecutive epochs. Set patience=0 to disable.
    early_stop_patience: int = 8
    early_stop_min_delta: float = 1e-4

    # Spectrogram / tile params — filled from MADProjectConfig.
    nperseg: int = 512
    noverlap: int = 384
    nfft: int = 1024
    db_min: float = -100.0
    db_max: float = -20.0
    tile_time_frames: int = 256
    tile_freq_bins: int = 512
    tile_overlap_fraction: float = 0.25

    wav_paths: List[str] = field(default_factory=list)

    def resolve_run_dir(self) -> str:
        name = self.run_name or datetime.now().strftime("unet_%Y%m%d_%H%M%S")
        return str(Path(self.project_dir) / "models" / name)


# ----------------------------------------------------------------------
# Device selection
# ----------------------------------------------------------------------
def _resolve_device(pref: str) -> str:
    import torch
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "mps":
        return "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    # auto
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ----------------------------------------------------------------------
# Loss: masked BCE + Dice (binary)
# ----------------------------------------------------------------------
def masked_bce_dice_loss(logits, target, weight, eps: float = 1e-6):
    """Masked BCE + (1 - soft-Dice) on logits.

    Args:
        logits: (N, 1, H, W)
        target: (N, 1, H, W) in {0, 1}
        weight: (N, 1, H, W) in {0, 1} — 0 means ignore
    """
    import torch
    import torch.nn.functional as F

    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    w_sum = weight.sum().clamp_min(1.0)
    bce_masked = (bce * weight).sum() / w_sum

    probs = torch.sigmoid(logits)
    p = probs * weight
    t = target * weight
    inter = (p * t).sum()
    union = p.sum() + t.sum()
    dice = 1 - (2 * inter + eps) / (union + eps)

    return bce_masked + dice


# ----------------------------------------------------------------------
# Main trainer
# ----------------------------------------------------------------------
ProgressFn = Callable[[int, int, Dict], None]
# progress(epoch, total_epochs, metrics_dict)


def train_unet(
    cfg: UNetTrainingConfig,
    progress: Optional[ProgressFn] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> Dict:
    """Train a U-Net on the project's painted tiles.

    Returns a summary dict including ``model_path`` pointing at the
    saved ``weights.pt``. Raises :class:`RuntimeError` if no labeled
    tiles are available.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    try:
        import segmentation_models_pytorch as smp
    except Exception as e:
        raise RuntimeError(
            "segmentation_models_pytorch is required for MAD U-Net training. "
            "Install with:\n    pip install segmentation-models-pytorch"
        ) from e

    from .mad_dataset import collect_training_tiles

    # ---- collect tiles ----
    if progress:
        progress(0, cfg.n_epochs, {'status': 'collecting_tiles'})
    specs, targets, weights = collect_training_tiles(
        cfg.wav_paths,
        nperseg=cfg.nperseg, noverlap=cfg.noverlap, nfft=cfg.nfft,
        db_min=cfg.db_min, db_max=cfg.db_max,
        tile_time_frames=cfg.tile_time_frames,
        tile_freq_bins=cfg.tile_freq_bins,
        overlap_fraction=cfg.tile_overlap_fraction,
        progress=(
            lambda i, n, name: progress(0, cfg.n_epochs, {
                'status': 'collecting_tiles', 'file_i': i,
                'file_n': n, 'file_name': name,
            }) if progress else None
        ),
    )

    n_total = specs.shape[0]
    if n_total == 0:
        raise RuntimeError(
            "No labeled tiles found. Paint at least one positive pixel in "
            "one or more files before training."
        )

    # ---- train/val split ----
    rng = np.random.default_rng(42)
    indices = rng.permutation(n_total)
    n_val = max(1, int(n_total * cfg.val_fraction))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:] if n_total > n_val else indices

    def to_tensor(arr):
        return torch.from_numpy(arr).float().unsqueeze(1)  # (N, 1, H, W)

    train_ds = TensorDataset(
        to_tensor(specs[train_idx]),
        to_tensor(targets[train_idx]),
        to_tensor(weights[train_idx]),
    )
    val_ds = TensorDataset(
        to_tensor(specs[val_idx]),
        to_tensor(targets[val_idx]),
        to_tensor(weights[val_idx]),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    # ---- model ----
    device = _resolve_device(cfg.device)
    model = smp.Unet(
        encoder_name=cfg.encoder_name,
        encoder_weights=cfg.encoder_weights,
        in_channels=1,
        classes=1,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    run_dir = Path(cfg.resolve_run_dir())
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- train ----
    history: List[Dict] = []
    best_val = float('inf')
    best_path = run_dir / 'weights.pt'
    epochs_without_improvement = 0
    early_stopped = False
    global_batch = 0
    batches_per_epoch = max(1, len(train_loader))

    for epoch in range(1, cfg.n_epochs + 1):
        if should_stop and should_stop():
            break

        model.train()
        train_loss_sum, train_n = 0.0, 0
        for bi, (xb, yb, wb) in enumerate(train_loader):
            if should_stop and should_stop():
                break
            xb = xb.to(device); yb = yb.to(device); wb = wb.to(device)
            optim.zero_grad()
            logits = model(xb)
            loss = masked_bce_dice_loss(logits, yb, wb)
            loss.backward()
            optim.step()
            batch_loss = float(loss.item())
            train_loss_sum += batch_loss * xb.size(0)
            train_n += xb.size(0)
            global_batch += 1
            if progress:
                progress(epoch, cfg.n_epochs, {
                    'status': 'batch',
                    'epoch': epoch, 'total_epochs': cfg.n_epochs,
                    'batch_i': bi + 1,
                    'batches_per_epoch': batches_per_epoch,
                    'global_batch': global_batch,
                    'batch_loss': batch_loss,
                })
        train_loss = train_loss_sum / max(1, train_n)

        model.eval()
        val_loss_sum, val_n = 0.0, 0
        dice_sum = 0.0
        with torch.no_grad():
            for xb, yb, wb in val_loader:
                xb = xb.to(device); yb = yb.to(device); wb = wb.to(device)
                logits = model(xb)
                loss = masked_bce_dice_loss(logits, yb, wb)
                val_loss_sum += float(loss.item()) * xb.size(0)
                val_n += xb.size(0)

                probs = torch.sigmoid(logits)
                pred = (probs > 0.5).float() * wb
                tgt = yb * wb
                inter = (pred * tgt).sum()
                union = pred.sum() + tgt.sum()
                dice = (2 * inter / (union + 1e-6)).item() if union.item() > 0 else float('nan')
                dice_sum += dice * xb.size(0)
        val_loss = val_loss_sum / max(1, val_n)
        val_dice = dice_sum / max(1, val_n)

        improved = val_loss < best_val - cfg.early_stop_min_delta
        if improved:
            best_val = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        metrics = {
            'status': 'training',
            'epoch': epoch, 'total_epochs': cfg.n_epochs,
            'train_loss': train_loss, 'val_loss': val_loss,
            'val_dice': val_dice,
            'best_val_loss': best_val,
            'epochs_without_improvement': epochs_without_improvement,
            'patience': cfg.early_stop_patience,
            'n_train_tiles': train_n, 'n_val_tiles': val_n,
            'global_batch': global_batch,
        }
        history.append(metrics)
        if progress:
            progress(epoch, cfg.n_epochs, metrics)

        if improved:
            torch.save(
                {
                    'state_dict': model.state_dict(),
                    'encoder_name': cfg.encoder_name,
                    'in_channels': 1,
                    'classes': 1,
                    'tile_freq_bins': cfg.tile_freq_bins,
                    'tile_time_frames': cfg.tile_time_frames,
                    'nperseg': cfg.nperseg,
                    'noverlap': cfg.noverlap,
                    'nfft': cfg.nfft,
                    'db_min': cfg.db_min,
                    'db_max': cfg.db_max,
                },
                best_path,
            )

        if (cfg.early_stop_patience > 0 and
                epochs_without_improvement >= cfg.early_stop_patience):
            early_stopped = True
            if progress:
                progress(epoch, cfg.n_epochs, {
                    'status': 'early_stop',
                    'epoch': epoch, 'total_epochs': cfg.n_epochs,
                    'best_val_loss': best_val,
                    'epochs_without_improvement': epochs_without_improvement,
                    'patience': cfg.early_stop_patience,
                })
            break

    # ---- summary ----
    summary = {
        'model_path': str(best_path),
        'run_dir': str(run_dir),
        'best_val_loss': best_val,
        'n_epochs_run': len(history),
        'early_stopped': early_stopped,
        'n_train_tiles': n_total - n_val,
        'n_val_tiles': n_val,
        'history': history,
        'config': asdict(cfg),
    }
    with open(run_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    if progress:
        progress(cfg.n_epochs, cfg.n_epochs, {
            'status': 'done', **{k: v for k, v in summary.items() if k != 'history'}
        })
    return summary
