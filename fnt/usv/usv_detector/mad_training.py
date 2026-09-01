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
    # Decoder architecture (all via segmentation_models_pytorch):
    #   'unet'   : smp.Unet (baseline)
    #   'unetpp' : smp.UnetPlusPlus (denser skips, finer detail)
    #   'manet'  : smp.MAnet (attention decoder)
    #   'hrnet'  : smp.Unet with a timm HRNet encoder (high-res features)
    model_arch: str = "unet"
    encoder_name: str = "resnet18"
    encoder_weights: Optional[str] = "imagenet"
    n_epochs: int = 30
    batch_size: int = 8
    learning_rate: float = 1e-3
    val_fraction: float = 0.20
    # Seed for the train/val split. Recorded in the run summary so a reported
    # number can be reproduced exactly.
    split_seed: int = 42
    device: str = "auto"             # 'auto' | 'cuda' | 'mps' | 'cpu'

    # SLEAP-style early stopping: halt when val_loss fails to improve by
    # more than ``early_stop_min_delta`` for ``early_stop_patience``
    # consecutive epochs. Set patience=0 to disable.
    early_stop_patience: int = 8
    early_stop_min_delta: float = 1e-4

    # Live per-epoch prediction previews: when on, each epoch emits a small
    # rotating set of (spectrogram, ground-truth, predicted-mask) tiles for the
    # GUI preview window. Cheap (one extra forward pass on ``preview_count``
    # val tiles, under no_grad); off in headless/CLI use.
    emit_previews: bool = False
    preview_count: int = 6

    # On-the-fly training augmentation (train tiles only; val is never
    # augmented). SpecAugment-style time/freq masking + mild noise/gain jitter
    # + a small time shift. The single biggest lever against overfitting on a
    # modest label set. Turn off to reproduce un-augmented behavior.
    augment: bool = True

    # Segmentation loss: 'bce_dice' (default, symmetric) or 'focal_tversky'
    # (recall-weighted via beta>alpha — better for thin/faint structure that a
    # symmetric loss under-segments). gamma focuses on hard examples.
    loss: str = "bce_dice"
    tversky_alpha: float = 0.3   # false-positive penalty
    tversky_beta: float = 0.7    # false-negative penalty (>alpha => favor recall)
    tversky_gamma: float = 1.3333  # focal exponent

    # Spectrogram / tile params — filled from MADProjectConfig.
    nperseg: int = 512
    noverlap: int = 384
    nfft: int = 1024
    db_min: float = -100.0
    db_max: float = -20.0
    tile_time_frames: int = 256
    tile_freq_bins: int = 512
    tile_overlap_fraction: float = 0.25

    # Self-contained per-call example store the model trains from. When set,
    # training reads patches from here instead of (WAV + sibling-PNG) pairs.
    training_data_dir: str = ""

    wav_paths: List[str] = field(default_factory=list)

    def resolve_run_dir(self, n_examples: int = 0) -> str:
        if self.run_name:
            name = self.run_name
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            arch = (self.model_arch or "unet").lower()
            name = f"{ts}_{arch}_n={n_examples}"
        return str(Path(self.project_dir) / "models" / name)


# ----------------------------------------------------------------------
# Train/val split
# ----------------------------------------------------------------------
def grouped_split(
    group_keys: List[str], val_fraction: float, seed: int = 42,
) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
    """Hold out whole groups. Returns ``(train_idx, val_idx, val_groups)``, or
    None when there are fewer than two groups to split.

    Splitting over tiles is what a segmentation pipeline must not do. Tiles are
    cut from the same patch (a long call spans several) and neighbouring tiles
    overlap, so a tile-level shuffle puts near-duplicate — sometimes literally
    the same — pixels in both train and val. The model then scores itself on
    data it fit, and validation Dice reads high for a model that has learned one
    recording's noise floor rather than what a call looks like.

    Groups are consumed in a seeded shuffle until the validation set reaches
    ``val_fraction`` of tiles, and at least one group is always left for
    training. Because groups differ in size, the realized fraction is
    approximate — correctness of the split beats hitting the ratio exactly.
    """
    order: List[str] = []
    members: Dict[str, List[int]] = {}
    for i, key in enumerate(group_keys):
        if key not in members:
            members[key] = []
            order.append(key)
        members[key].append(i)
    if len(order) < 2:
        return None

    rng = np.random.default_rng(seed)
    shuffled = [order[i] for i in rng.permutation(len(order))]

    n_total = len(group_keys)
    target = max(1, int(round(n_total * float(val_fraction))))
    val_groups: List[str] = []
    n_val = 0
    # Leave the final group for training no matter what, so neither side is ever
    # empty. The first iteration always appends (n_val starts at 0 and target is
    # at least 1), and there are at least two groups here, so val_groups cannot
    # come out empty either — no fallback needed.
    for key in shuffled[:-1]:
        if n_val >= target:
            break
        val_groups.append(key)
        n_val += len(members[key])

    val_set = set(val_groups)
    val_idx = np.array(
        [i for i, k in enumerate(group_keys) if k in val_set], dtype=np.int64)
    train_idx = np.array(
        [i for i, k in enumerate(group_keys) if k not in val_set], dtype=np.int64)
    return train_idx, val_idx, val_groups


def _split_tiles(
    groups: List[Tuple[str, str]], n_total: int, val_fraction: float,
    seed: int = 42,
) -> Dict:
    """Pick the strongest split the label set supports, and say which one.

    Preference order, because each level is a weaker guarantee than the last:

    * ``file`` — whole recordings held out. The only split that measures what
      the user cares about: does this model work on a recording it has never
      seen? Needs labels on ≥2 recordings.
    * ``call`` — whole calls held out, but train and val share recordings. Stops
      a call's own overlapping tiles straddling the split; does NOT stop the
      model exploiting one recording's noise floor, mic, or animal. Reported so
      the number is read with that caveat.
    * ``tile`` — last resort for a single labeled call. Train and val overlap;
      validation is not a held-out measurement at all.
    """
    if groups and len(groups) == n_total:
        for level, keys in (('file', [g[0] for g in groups]),
                            ('call', [g[1] for g in groups])):
            split = grouped_split(keys, val_fraction, seed=seed)
            if split is not None:
                train_idx, val_idx, val_groups = split
                n_groups = len(set(keys))
                return {
                    'train_idx': train_idx, 'val_idx': val_idx,
                    'split_level': level, 'val_groups': sorted(val_groups),
                    'n_groups': n_groups,
                    'n_val_groups': len(val_groups),
                    # Deliberately narrow: True means "validated on a RECORDING
                    # the model never saw", the only claim worth reporting
                    # unqualified. A call-level split still shares recordings
                    # with training, so it reads False and every consumer warns.
                    'val_held_out': level == 'file',
                }
    # One call (or no provenance): nothing can be held out.
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    n_val = max(1, int(n_total * val_fraction))
    return {
        'train_idx': indices[n_val:] if n_total > n_val else indices,
        'val_idx': indices[:n_val],
        'split_level': 'tile', 'val_groups': [], 'n_groups': 1,
        'n_val_groups': 0, 'val_held_out': False,
    }


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
# Model factory — shared by training and inference
# ----------------------------------------------------------------------
# Encoder forced for the HRNet architecture option (high-resolution
# features for crisp thin-structure outlines).
HRNET_ENCODER = "tu-hrnet_w18"


def build_model(arch: str, encoder_name: str, encoder_weights,
                in_channels: int = 1, classes: int = 1):
    """Construct a binary segmentation model for the given architecture.

    Used by both :func:`train_unet` and inference so a checkpoint always
    rebuilds with the architecture it was trained on. ``arch`` is one of
    ``'unet' | 'unetpp' | 'manet' | 'hrnet'`` (unknown values fall back to
    ``'unet'``). The ``'hrnet'`` option ignores ``encoder_name`` and uses a
    timm HRNet encoder under the standard U-Net decoder.
    """
    import segmentation_models_pytorch as smp

    arch = (arch or "unet").lower()
    kw = dict(encoder_weights=encoder_weights, in_channels=in_channels,
              classes=classes)
    if arch == "unetpp":
        return smp.UnetPlusPlus(encoder_name=encoder_name, **kw)
    if arch == "manet":
        return smp.MAnet(encoder_name=encoder_name, **kw)
    if arch == "hrnet":
        return smp.Unet(encoder_name=HRNET_ENCODER, **kw)
    return smp.Unet(encoder_name=encoder_name, **kw)


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


def masked_focal_tversky_loss(logits, target, weight, alpha=0.3, beta=0.7,
                              gamma=1.3333, eps: float = 1e-6):
    """Masked BCE + Focal-Tversky over the supervised region.

    The Tversky index TP/(TP + alpha*FP + beta*FN) generalizes Dice; with
    ``beta > alpha`` it penalizes false negatives (missed pixels) more than
    false positives, pushing recall up — useful for thin, faint call structure
    that a symmetric BCE+Dice tends to under-segment. The focal exponent
    ``gamma`` concentrates gradient on hard (low-TI) tiles. A small BCE term is
    kept for gradient stability early on (mirrors :func:`masked_bce_dice_loss`).
    """
    import torch
    import torch.nn.functional as F

    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    bce_masked = (bce * weight).sum() / weight.sum().clamp_min(1.0)

    probs = torch.sigmoid(logits)
    t = target * weight
    tp = (probs * t).sum()
    fp = (probs * (1 - target) * weight).sum()
    fn = ((1 - probs) * t).sum()
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    focal_tversky = (1.0 - tversky).clamp_min(0.0) ** gamma

    return bce_masked + focal_tversky


def make_loss_fn(cfg):
    """Return ``loss(logits, target, weight)`` for the configured loss."""
    if getattr(cfg, 'loss', 'bce_dice') == 'focal_tversky':
        a = float(getattr(cfg, 'tversky_alpha', 0.3))
        b = float(getattr(cfg, 'tversky_beta', 0.7))
        g = float(getattr(cfg, 'tversky_gamma', 1.3333))
        return lambda logits, target, weight: masked_focal_tversky_loss(
            logits, target, weight, a, b, g)
    return masked_bce_dice_loss


class _AugmentedTileDataset:
    """Training-tile dataset that augments on the fly (val never uses this).

    Geometric ops (time shift) move spec + target + weight together; spectral
    ops (SpecAugment time/freq masking, Gaussian noise, gain jitter) corrupt the
    *input spec only* — the target/weight are untouched, so the model learns to
    predict the call even when the input is partly masked or noisy. Each op fires
    independently with its own probability, so most tiles get a mild mix."""

    def __init__(self, specs, targets, weights, seed: int = 0,
                 p_time_shift=0.5, max_shift_frac=0.1,
                 p_freq_mask=0.5, max_freq_frac=0.15, n_freq_masks=2,
                 p_time_mask=0.5, max_time_frac=0.15, n_time_masks=2,
                 p_noise=0.5, noise_std=0.03,
                 p_gain=0.5, gain_range=(0.9, 1.1)):
        import torch
        self._torch = torch
        self.specs = specs      # (N, H, W) float32, ~[0,1]
        self.targets = targets
        self.weights = weights
        self.rng = np.random.default_rng(seed)
        self.p_time_shift, self.max_shift_frac = p_time_shift, max_shift_frac
        self.p_freq_mask, self.max_freq_frac, self.n_freq_masks = \
            p_freq_mask, max_freq_frac, n_freq_masks
        self.p_time_mask, self.max_time_frac, self.n_time_masks = \
            p_time_mask, max_time_frac, n_time_masks
        self.p_noise, self.noise_std = p_noise, noise_std
        self.p_gain, self.gain_range = p_gain, gain_range

    def __len__(self):
        return self.specs.shape[0]

    def _bands(self, n_masks, length, max_frac):
        for _ in range(n_masks):
            span = int(self.rng.integers(0, max(1, int(length * max_frac)) + 1))
            if span <= 0:
                continue
            start = int(self.rng.integers(0, max(1, length - span + 1)))
            yield start, start + span

    def __getitem__(self, i):
        s = self.specs[i].copy()      # (H, W)
        t = self.targets[i].copy()
        w = self.weights[i].copy()
        H, W = s.shape
        # --- geometric: whole-tile time shift (spec + target + weight) ---
        if self.rng.random() < self.p_time_shift:
            sh = int(self.rng.integers(-int(W * self.max_shift_frac),
                                       int(W * self.max_shift_frac) + 1))
            if sh:
                s = np.roll(s, sh, axis=1)
                t = np.roll(t, sh, axis=1)
                w = np.roll(w, sh, axis=1)
        # --- spectral (input spec only) ---
        if self.rng.random() < self.p_freq_mask:
            for a, b in self._bands(self.n_freq_masks, H, self.max_freq_frac):
                s[a:b, :] = 0.0
        if self.rng.random() < self.p_time_mask:
            for a, b in self._bands(self.n_time_masks, W, self.max_time_frac):
                s[:, a:b] = 0.0
        if self.rng.random() < self.p_noise:
            s = s + self.rng.normal(0.0, self.noise_std, s.shape).astype(s.dtype)
        if self.rng.random() < self.p_gain:
            lo, hi = self.gain_range
            s = s * float(self.rng.uniform(lo, hi))
        s = np.clip(s, 0.0, 1.0)
        torch = self._torch
        return (torch.from_numpy(s).float().unsqueeze(0),
                torch.from_numpy(t).float().unsqueeze(0),
                torch.from_numpy(w).float().unsqueeze(0))


# ----------------------------------------------------------------------
# Live per-epoch prediction previews
# ----------------------------------------------------------------------
def _shrink_for_preview(a: np.ndarray, max_h: int = 160, max_w: int = 192):
    """Nearest-neighbour decimate a tile so preview payloads stay small."""
    h, w = a.shape
    sh = max(1, int(np.ceil(h / max_h)))
    sw = max(1, int(np.ceil(w / max_w)))
    return a[::sh, ::sw]


def _build_epoch_previews(model, device, specs, targets, gt_pool, nogt_pool,
                          rng, k):
    """Run the current model on a fresh random handful of val tiles and return
    lightweight preview payloads. ``gt_pool``/``nogt_pool`` are index lists into
    ``specs``/``targets`` for tiles that do / don't contain a labelled call.
    Roughly 2/3 are call tiles (with a ground-truth mask) and 1/3 background
    tiles (no call), so the user sees both fit quality and false positives.

    Each payload is a dict of small uint8 arrays: ``spec`` (grayscale 0-255),
    ``pred`` (predicted probability 0-255), ``gt`` (0/1 mask, or None for
    background tiles), ``dice`` (float over the supervised tile, or None), and
    ``has_gt``. Must be called under ``model.eval()`` / ``torch.no_grad()``.
    """
    import torch

    def _sample(pool, n):
        if not pool or n <= 0:
            return []
        replace = n > len(pool)
        return [int(i) for i in rng.choice(pool, size=n, replace=replace)]

    if gt_pool:
        n_gt = min(len(gt_pool), max(1, (k * 2) // 3)) if nogt_pool else \
            min(len(gt_pool), k)
    else:
        n_gt = 0
    n_nogt = (k - n_gt) if nogt_pool else 0
    picks = [(i, True) for i in _sample(gt_pool, n_gt)]
    picks += [(i, False) for i in _sample(nogt_pool, n_nogt)]
    if not picks:
        return []

    batch = np.stack([specs[i] for i, _ in picks])[:, None, :, :]  # (n,1,H,W)
    xb = torch.from_numpy(batch.astype(np.float32)).to(device)
    with torch.no_grad():  # self-contained — safe even outside an eval context
        probs = torch.sigmoid(model(xb))[:, 0].detach().cpu().numpy()  # [0,1]

    tiles = []
    for (idx, has_gt), prob in zip(picks, probs):
        spec = specs[idx]
        payload = {
            'has_gt': bool(has_gt),
            'spec': (np.clip(_shrink_for_preview(spec), 0.0, 1.0) * 255
                     ).astype(np.uint8),
            'pred': (np.clip(_shrink_for_preview(prob), 0.0, 1.0) * 255
                     ).astype(np.uint8),
            'gt': None,
            'dice': None,
        }
        if has_gt:
            gt = targets[idx] > 0.5
            payload['gt'] = _shrink_for_preview(gt.astype(np.uint8))
            pred_bin = prob > 0.5
            inter = float(np.logical_and(pred_bin, gt).sum())
            denom = float(pred_bin.sum() + gt.sum())
            payload['dice'] = (2.0 * inter / denom) if denom > 0 else 1.0
        tiles.append(payload)
    return tiles


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

    from .mad_examples import collect_training_examples

    # ---- collect tiles from the self-contained example store ----
    if progress:
        progress(0, cfg.n_epochs, {'status': 'collecting_tiles'})
    specs, targets, weights, groups = collect_training_examples(
        cfg.training_data_dir,
        tile_time_frames=cfg.tile_time_frames,
        tile_freq_bins=cfg.tile_freq_bins,
        progress=(
            lambda i, n, name: progress(0, cfg.n_epochs, {
                'status': 'collecting_tiles', 'file_i': i,
                'file_n': n, 'file_name': name,
            }) if progress else None
        ),
        return_groups=True,
    )

    n_total = specs.shape[0]
    if n_total == 0:
        raise RuntimeError(
            "No confirmed training examples found. Label and confirm at least "
            "one call (Enter) before training."
        )

    # ---- train/val split (grouped — see _split_tiles) ----
    split = _split_tiles(groups, n_total, cfg.val_fraction, seed=cfg.split_seed)
    train_idx, val_idx = split['train_idx'], split['val_idx']
    n_val = int(len(val_idx))
    if progress:
        progress(0, cfg.n_epochs, {
            'status': 'split',
            'split_level': split['split_level'],
            'val_held_out': split['val_held_out'],
            'n_groups': split['n_groups'],
            'n_val_groups': split['n_val_groups'],
            'val_groups': split['val_groups'],
            'n_train_tiles': int(len(train_idx)), 'n_val_tiles': n_val,
        })

    def to_tensor(arr):
        return torch.from_numpy(arr).float().unsqueeze(1)  # (N, 1, H, W)

    # Train tiles are augmented on the fly when enabled; val is always raw.
    if getattr(cfg, 'augment', False):
        train_ds = _AugmentedTileDataset(
            specs[train_idx], targets[train_idx], weights[train_idx])
    else:
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

    loss_fn = make_loss_fn(cfg)

    # ---- model ----
    device = _resolve_device(cfg.device)
    # Report the resolved compute device so the session log makes it obvious
    # whether training runs on GPU or CPU (and which GPU).
    dev_desc = device
    try:
        if device == 'cuda':
            dev_desc = f"cuda — {torch.cuda.get_device_name(0)}"
        elif device == 'mps':
            dev_desc = "mps — Apple GPU"
        else:
            cuda_seen = bool(getattr(torch, 'cuda', None)
                             and torch.cuda.is_available())
            dev_desc = ("cpu" if cuda_seen
                        else "cpu (no CUDA GPU detected by PyTorch)")
    except Exception:
        pass
    if progress:
        progress(0, cfg.n_epochs, {'status': 'device', 'device': device,
                                   'device_desc': dev_desc,
                                   'requested': cfg.device})
    model = build_model(
        cfg.model_arch, cfg.encoder_name, cfg.encoder_weights,
        in_channels=1, classes=1,
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # Name the run by the number of accepted labels (confirmed calls), which is
    # what the user tracks — not the tile count (long calls split into several
    # tiles, so n_total > n_labels). The tile count is recorded in the summary.
    try:
        from .mad_examples import count_examples
        n_labels = count_examples(cfg.training_data_dir) or n_total
        # Hard negatives harvested from rejections. They train the model but are
        # not calls, so they stay out of the run name and are reported
        # separately — a silent change to what the model is taught would be
        # impossible to notice later.
        n_negatives = count_examples(cfg.training_data_dir, kind='negative')
    except Exception:
        n_labels, n_negatives = n_total, 0
    if progress and n_negatives:
        progress(0, cfg.n_epochs, {
            'status': 'negatives', 'n_negatives': n_negatives})
    run_dir = Path(cfg.resolve_run_dir(n_examples=n_labels))
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- live-preview pools (rotating val tiles) ----
    # Kept as numpy so we can re-run the *current* model on a fresh random
    # handful each epoch. Split into call tiles (non-empty mask) and background
    # tiles (empty mask → "no ground truth", surfaces false positives).
    prev_specs = specs[val_idx]
    prev_targets = targets[val_idx]
    prev_gt_pool = [i for i in range(len(prev_targets))
                    if prev_targets[i].sum() > 0]
    prev_nogt_pool = [i for i in range(len(prev_targets))
                      if prev_targets[i].sum() == 0]
    prev_rng = np.random.default_rng()  # unseeded → rotates each epoch

    # ---- train ----
    history: List[Dict] = []
    best_val = float('inf')
    best_epoch = 0
    best_metrics: Dict = {}
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
            loss = loss_fn(logits, yb, wb)
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
        # Pixel confusion (TP/FP/FN) over the supervised region at several
        # thresholds — cheap, nothing stored — for precision/recall and a
        # Dice-vs-threshold sweep. These separate "masks too tight / missing
        # faint structure" (low recall) from "masks bloated" (low precision),
        # and reveal whether a lower inference threshold would recover the call.
        _THRS = (0.3, 0.4, 0.5, 0.6, 0.7)
        _tp = {t: 0.0 for t in _THRS}
        _fp = {t: 0.0 for t in _THRS}
        _fn = {t: 0.0 for t in _THRS}
        pos_px = 0.0
        sup_px = 0.0
        with torch.no_grad():
            for xb, yb, wb in val_loader:
                xb = xb.to(device); yb = yb.to(device); wb = wb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb, wb)
                val_loss_sum += float(loss.item()) * xb.size(0)
                val_n += xb.size(0)

                probs = torch.sigmoid(logits)
                tgt = yb * wb
                pred = (probs > 0.5).float() * wb
                inter = (pred * tgt).sum()
                union = pred.sum() + tgt.sum()
                dice = (2 * inter / (union + 1e-6)).item() if union.item() > 0 else float('nan')
                dice_sum += dice * xb.size(0)
                pos_px += float(tgt.sum())
                sup_px += float(wb.sum())
                for t in _THRS:
                    pbt = (probs > t).float() * wb
                    _tp[t] += float((pbt * tgt).sum())
                    _fp[t] += float((pbt * (1 - yb) * wb).sum())
                    _fn[t] += float(((wb - pbt) * tgt).sum())
        val_loss = val_loss_sum / max(1, val_n)
        val_dice = dice_sum / max(1, val_n)

        def _prd(t):
            prec = _tp[t] / (_tp[t] + _fp[t] + 1e-6)
            rec = _tp[t] / (_tp[t] + _fn[t] + 1e-6)
            d = 2 * _tp[t] / (2 * _tp[t] + _fp[t] + _fn[t] + 1e-6)
            return prec, rec, d
        val_precision, val_recall, _ = _prd(0.5)
        dice_sweep = {t: round(_prd(t)[2], 4) for t in _THRS}
        best_thr = max(dice_sweep, key=dice_sweep.get)
        pos_frac = pos_px / max(1.0, sup_px)  # how sparse the masks are

        improved = val_loss < best_val - cfg.early_stop_min_delta
        if improved:
            best_val = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        metrics = {
            'status': 'training',
            'epoch': epoch, 'total_epochs': cfg.n_epochs,
            'train_loss': train_loss, 'val_loss': val_loss,
            'val_dice': val_dice,
            'val_precision': round(val_precision, 4),
            'val_recall': round(val_recall, 4),
            'val_dice_sweep': dice_sweep,
            'best_threshold': best_thr,
            'val_pos_frac': round(pos_frac, 5),
            'best_val_loss': best_val,
            'epochs_without_improvement': epochs_without_improvement,
            'patience': cfg.early_stop_patience,
            'n_train_tiles': train_n, 'n_val_tiles': val_n,
            'global_batch': global_batch,
        }
        if improved:
            best_metrics = metrics
        history.append(metrics)
        if progress:
            progress(epoch, cfg.n_epochs, metrics)

        # Live prediction preview: a fresh rotating handful of val tiles, run
        # through the just-updated model (still in eval/no_grad from validation).
        if progress and cfg.emit_previews and len(prev_specs):
            try:
                with torch.no_grad():
                    tiles = _build_epoch_previews(
                        model, device, prev_specs, prev_targets,
                        prev_gt_pool, prev_nogt_pool, prev_rng,
                        cfg.preview_count)
                if tiles:
                    progress(epoch, cfg.n_epochs, {
                        'status': 'epoch_preview', 'epoch': epoch,
                        'total_epochs': cfg.n_epochs, 'val_dice': val_dice,
                        'tiles': tiles,
                    })
            except Exception:
                pass  # previews are best-effort; never break a run

        if improved:
            torch.save(
                {
                    'state_dict': model.state_dict(),
                    'model_arch': cfg.model_arch,
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
                    # Provenance: travels with the weights, so a checkpoint
                    # handed to someone else still says what its val score
                    # actually measured.
                    'split_level': split['split_level'],
                    'val_held_out': split['val_held_out'],
                    'val_groups': split['val_groups'],
                    'n_train_tiles': int(len(train_idx)),
                    'n_val_tiles': n_val,
                    'trained': datetime.now().isoformat(timespec='seconds'),
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
    # ``best_metrics`` are the validation stats at the saved (best-val-loss)
    # checkpoint — the model that actually gets deployed — so precision/recall
    # and the threshold sweep describe the weights on disk, not the last epoch.
    summary = {
        'model_path': str(best_path),
        'run_dir': str(run_dir),
        'best_val_loss': best_val,
        'best_epoch': best_epoch,
        'best_val_dice': best_metrics.get('val_dice'),
        'best_val_precision': best_metrics.get('val_precision'),
        'best_val_recall': best_metrics.get('val_recall'),
        'best_val_dice_sweep': best_metrics.get('val_dice_sweep'),
        'best_threshold': best_metrics.get('best_threshold'),
        'val_pos_frac': best_metrics.get('val_pos_frac'),
        'n_epochs_run': len(history),
        'early_stopped': early_stopped,
        'n_train_tiles': int(len(train_idx)),
        'n_val_tiles': n_val,
        'n_labels': n_labels,
        # Rejected detections reused as explicit hard negatives.
        'n_negatives': n_negatives,
        # How the split was made. Every val_* number above must be read through
        # this: only 'file' means "measured on a recording the model never saw".
        'split_level': split['split_level'],
        'val_held_out': split['val_held_out'],
        'n_groups': split['n_groups'],
        'n_val_groups': split['n_val_groups'],
        'val_groups': split['val_groups'],
        'split_seed': cfg.split_seed,
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
