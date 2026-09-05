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
    # How train/val is grouped:
    #   'call' — DEFAULT. Hold out whole calls (masks), regardless of which
    #            recording they came from. Every labelled file trains. This is
    #            the correct default for MAD's actual workflow: label a file,
    #            train, run inference on new audio, correct the junk and the
    #            low-confidence guesses, retrain. Each round the label set grows
    #            across files, and a split that reserves whole recordings keeps
    #            pulling the newest corrections out of training — the exact data
    #            the next round exists to learn from. Validation shares
    #            recordings with training, so val_dice runs optimistic; every
    #            consumer prints the caveat next to the number.
    #   'auto' — 'file' once >= 2 recordings carry calls, else 'call'.
    #   'file' — always hold out whole recordings. The strictest generalization
    #            measure ("does this work on a recording it has never seen") and
    #            the right choice for a final, honest number once labelling has
    #            settled. Needs calls on >= 2 recordings.
    split_mode: str = "call"    # 'call' | 'auto' | 'file'
    # How many times each labeled call is placed, at a random position, into a
    # training tile. See mad_examples.collect_training_examples for why this
    # is what makes the model usable on full recordings.
    tile_placements: int = 3
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
    db_norm: str = 'fixed'      # see MADProjectConfig.db_norm
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
    positives: Optional[np.ndarray] = None,
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
    training. A group that would push validation past **half the data** is
    skipped in favour of a smaller one: per-recording label counts are very
    uneven in practice, and without that guard one dominant recording is
    swallowed whole, leaving the model fit on a small remainder and scored on
    the bulk. Because groups differ in size the realized fraction is still
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
    # Never hand more than half the data to validation. Label counts per
    # recording are wildly uneven in practice — one long, heavily-labeled file
    # against several sparse ones — and a plain "consume groups until you reach
    # the target" rule will swallow the dominant group whole. On real VoleCosm
    # data that produced a 4152/668 val/train split: the reported score was
    # measured on 86% of the labels, and the model was fit on 14% of them.
    cap = max(target, n_total // 2)
    # The largest group always trains. That is what guarantees the train side
    # is never empty *and* never trivial: reserving an arbitrary group (say,
    # the last one drawn) satisfies "non-empty" while still allowing the bulk
    # of the labels into validation.
    # Positive tiles per group. "Largest group trains" keeps the train side
    # non-empty but says nothing about it being *learnable*: on a label set
    # where one recording holds every call and another contributes only hard
    # negatives, the negative-only recording is the larger one, so it anchored
    # training and every call went to validation. That trains a segmentation
    # model on zero positive pixels — it can only ever predict background.
    # Anchor on the largest group that actually contains calls instead.
    pos_count: Dict[str, int] = {k: 0 for k in order}
    if positives is not None:
        for i, key in enumerate(group_keys):
            if positives[i]:
                pos_count[key] += 1
    total_pos = sum(pos_count.values())

    if total_pos:
        largest = max((k for k in order if pos_count[k]),
                      key=lambda k: len(members[k]))
    else:
        largest = max(order, key=lambda k: len(members[k]))
    candidates = [k for k in shuffled if k != largest]

    def _drains_positives(chosen: List[str]) -> bool:
        """True if handing ``chosen`` to validation leaves training with no calls."""
        if not total_pos:
            return False
        return (total_pos - sum(pos_count[k] for k in chosen)) <= 0

    val_groups: List[str] = []
    n_val = 0
    for key in candidates:
        if n_val >= target:
            break
        size = len(members[key])
        if n_val + size > cap:
            continue        # would overshoot; try a smaller one
        if _drains_positives(val_groups + [key]):
            continue        # would leave training with nothing to learn from
        val_groups.append(key)
        n_val += size
    if not val_groups:
        # Every candidate exceeds the cap on its own. Take the smallest that
        # still leaves calls in training, so validation is a held-out recording
        # rather than nothing — but never at the cost of a trainable split.
        usable = [k for k in candidates if not _drains_positives([k])]
        if not usable:
            return None     # no viable split at this grouping level
        smallest = min(usable, key=lambda k: len(members[k]))
        val_groups = [smallest]
        n_val = len(members[smallest])

    val_set = set(val_groups)
    val_idx = np.array(
        [i for i, k in enumerate(group_keys) if k in val_set], dtype=np.int64)
    train_idx = np.array(
        [i for i, k in enumerate(group_keys) if k not in val_set], dtype=np.int64)
    return train_idx, val_idx, val_groups


def _split_tiles(
    groups: List[Tuple[str, str]], n_total: int, val_fraction: float,
    seed: int = 42, positives: Optional[np.ndarray] = None,
    mode: str = "call",
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
    def _usable(train_idx, val_idx) -> bool:
        """A split is only usable when BOTH sides contain calls.

        Zero positives in train means the model cannot learn a call at all;
        zero positives in val means the score is measured on background only.
        Either way the run is wasted, so a degenerate split at one grouping
        level falls through to the next weaker level rather than being used.
        """
        if positives is None:
            return True
        return bool(positives[train_idx].any()) and bool(positives[val_idx].any())

    degraded: List[str] = []
    levels = [('file', [g[0] for g in groups] if groups else []),
              ('call', [g[1] for g in groups] if groups else [])]
    if mode == 'file':
        levels = levels[:1]
    elif mode == 'call':
        levels = levels[1:]

    if groups and len(groups) == n_total:
        for level, keys in levels:
            # Holding out a recording only measures "works on an unseen
            # recording" when at least two recordings actually carry calls.
            # With calls on one file the held-out side has either all of them
            # or none — the first starves training, the second makes the score
            # meaningless. That is the normal state early in the label-one-file
            # -then-expand workflow, so auto drops to a call-level split rather
            # than forcing a split the label set cannot support.
            if level == 'file' and mode == 'auto' and positives is not None:
                files_with_calls = {k for k, p in zip(keys, positives) if p}
                if len(files_with_calls) < 2:
                    degraded.append('file')
                    continue
            split = grouped_split(keys, val_fraction, seed=seed,
                                  positives=positives)
            if split is not None:
                train_idx, val_idx, val_groups = split
                if not _usable(train_idx, val_idx):
                    degraded.append(level)
                    continue
                n_groups = len(set(keys))
                return {
                    'degraded_from': degraded,
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
    # One call (or no provenance): nothing can be held out. Stratify on
    # positives so the last-resort split still puts calls on both sides —
    # an unstratified shuffle can hand every call to one side by chance.
    rng = np.random.default_rng(seed)
    if positives is not None and positives.any() and not positives.all():
        pos_idx = rng.permutation(np.flatnonzero(positives))
        neg_idx = rng.permutation(np.flatnonzero(~positives))
        n_pos_val = max(1, int(len(pos_idx) * val_fraction))
        n_neg_val = max(1, int(len(neg_idx) * val_fraction))
        # Never take the last positive away from training.
        n_pos_val = min(n_pos_val, max(0, len(pos_idx) - 1))
        val_idx = np.concatenate([pos_idx[:n_pos_val], neg_idx[:n_neg_val]])
        train_idx = np.concatenate([pos_idx[n_pos_val:], neg_idx[n_neg_val:]])
    else:
        indices = rng.permutation(n_total)
        n_val = max(1, int(n_total * val_fraction))
        val_idx = indices[:n_val]
        train_idx = indices[n_val:] if n_total > n_val else indices
    return {
        'train_idx': train_idx,
        'val_idx': val_idx,
        'split_level': 'tile', 'val_groups': [], 'n_groups': 1,
        'n_val_groups': 0, 'val_held_out': False,
        'degraded_from': degraded,
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
        placements=cfg.tile_placements,
        seed=cfg.split_seed,
    )

    # Refuse to train on a mix of normalization rules. A patch is normalized
    # and quantized when it is saved, so examples from different rules are on
    # two incompatible intensity scales and the result would look like a merely
    # mediocre model rather than a broken setup.
    try:
        from .mad_examples import db_norm_breakdown
        norms = db_norm_breakdown(cfg.training_data_dir)
    except Exception:
        norms = {}
    if len(norms) > 1:
        detail = ", ".join(f"{k}: {v}" for k, v in sorted(norms.items()))
        raise RuntimeError(
            "This project's labels were saved under more than one dB "
            f"normalization rule ({detail}). A saved patch is already "
            "normalized, so the two sets cannot be reconciled. Either set "
            "db_norm back to the rule the majority were labeled with, or "
            "re-label the minority under the current rule."
        )
    if norms and cfg.db_norm not in norms:
        only = next(iter(norms))
        raise RuntimeError(
            f"Labels were saved with db_norm='{only}' but this project "
            f"is set to db_norm='{cfg.db_norm}'. Training on them would "
            "feed the model an intensity scale the labels were never "
            f"drawn on. Set db_norm back to '{only}', or re-label under "
            f"'{cfg.db_norm}'."
        )

    n_total = specs.shape[0]
    if n_total == 0:
        raise RuntimeError(
            "No confirmed training examples found. Label and confirm at least "
            "one call (Enter) before training."
        )

    # ---- train/val split (grouped — see _split_tiles) ----
    # Which tiles actually contain a labelled call. The split needs this: a
    # split that is balanced by tile count can still be empty of calls on one
    # side, which silently produces a model that only ever predicts background.
    tile_positive = (targets.reshape(n_total, -1) > 0.5).any(axis=1)
    split = _split_tiles(groups, n_total, cfg.val_fraction, seed=cfg.split_seed,
                         positives=tile_positive,
                         mode=getattr(cfg, 'split_mode', 'call'))
    train_idx, val_idx = split['train_idx'], split['val_idx']

    # Refuse to burn a GPU run on an unlearnable split. Reaching here means no
    # grouping level could put calls on both sides — the label set itself is
    # the problem, and only the user can fix it.
    if tile_positive.any() and not tile_positive[train_idx].any():
        raise RuntimeError(
            "The train/validation split left ZERO labelled calls in the "
            "training set, so the model would have nothing to learn from.\n\n"
            "This happens when the calls are concentrated on recordings that "
            "all end up held out — typically when one recording contributes "
            "only rejected/hard-negative examples and another holds every "
            "confirmed call.\n\n"
            "Label confirmed calls on at least two recordings, then retrain."
        )
    n_val = int(len(val_idx))

    # Positive-pixel balance, measured over the supervised region only (the
    # weight mask) because that is exactly what the loss sees. When this is
    # tiny, an all-background prediction is a cheap minimum for a symmetric
    # loss and the run silently collapses; reporting it makes that predictable
    # instead of something you infer from three zeroed metrics.
    def _balance(idx):
        if len(idx) == 0:
            return {'pos_frac': 0.0, 'n_call_tiles': 0, 'n_tiles': 0}
        t = targets[idx] > 0.5
        w = weights[idx] > 0.5
        sup = float(w.sum())
        pos = float(np.logical_and(t, w).sum())
        per_tile = t.reshape(len(idx), -1).sum(axis=1)
        return {
            'pos_frac': (pos / sup) if sup > 0 else 0.0,
            'n_call_tiles': int((per_tile > 0).sum()),
            'n_tiles': int(len(idx)),
        }

    train_balance = _balance(train_idx)
    val_balance = _balance(val_idx)

    if progress:
        progress(0, cfg.n_epochs, {
            'status': 'split',
            'train_balance': train_balance,
            'val_balance': val_balance,
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
        dice_n = 0
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
                # A batch with neither a predicted nor a labelled pixel has no
                # meaningful Dice. It used to contribute nan, and one nan in
                # the running sum makes val_dice nan for the whole run — which
                # is what happens the moment background-only tiles enter
                # validation and the model correctly predicts nothing on them.
                # Skip those batches instead of poisoning the average.
                if union.item() > 0:
                    dice = (2 * inter / (union + 1e-6)).item()
                    dice_sum += dice * xb.size(0)
                    dice_n += xb.size(0)
                pos_px += float(tgt.sum())
                sup_px += float(wb.sum())
                for t in _THRS:
                    pbt = (probs > t).float() * wb
                    _tp[t] += float((pbt * tgt).sum())
                    _fp[t] += float((pbt * (1 - yb) * wb).sum())
                    _fn[t] += float(((wb - pbt) * tgt).sum())
        val_loss = val_loss_sum / max(1, val_n)
        # Averaged over the batches that HAD something to score, not all
        # of them — otherwise background-only batches drag it toward 0.
        val_dice = (dice_sum / dice_n) if dice_n else 0.0

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
                    # Inference reads this back: feeding a per-file-normalized
                    # model fixed-range input (or vice versa) silently degrades
                    # every detection.
                    'db_norm': cfg.db_norm,
                    # Provenance: travels with the weights, so a checkpoint
                    # handed to someone else still says what its val score
                    # actually measured.
                    'split_level': split['split_level'],
                    'val_held_out': split['val_held_out'],
                    'val_groups': split['val_groups'],
                    'n_train_tiles': int(len(train_idx)),
                    'n_val_tiles': n_val,
                    'trained': datetime.now().isoformat(timespec='seconds'),
                    # The threshold this checkpoint actually scores best at.
                    # 0.5 is only a convention, and on sparse masks the best
                    # operating point is routinely elsewhere — carrying it with
                    # the weights means inference does not have to guess.
                    'best_threshold': best_thr,
                    'val_dice_sweep': dice_sweep,
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
        'db_norm': cfg.db_norm,
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
        # Positive-pixel balance — the number that explains an all-background
        # collapse after the fact (see the _balance note above).
        'train_balance': train_balance,
        'val_balance': val_balance,
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
