"""MAD inference pipeline.

Runs a trained U-Net checkpoint over full WAV files, stitches a
per-pixel probability mask, thresholds it, extracts connected-component
blobs, and writes sibling CSV + PNG artifacts next to each wav.

Heavy deps (``torch``, ``segmentation_models_pytorch``, ``scipy.ndimage``)
are imported lazily inside the run functions so the module is safe to
import from the GUI even when those packages aren't installed.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .mad_dataset import compute_full_spec_image
from .mad_labels import (
    committed_columns, load_mask_png, mask_sibling_path,
    pred_csv_sibling_path, pred_mask_sibling_path, save_mask_png,
)
from .spectrogram import load_audio


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
@dataclass
class MADInferenceConfig:
    model_path: str
    threshold: float = 0.5
    min_blob_pixels: int = 8
    tile_time_frames: int = 256
    tile_freq_bins: int = 512
    tile_overlap_fraction: float = 0.25
    device: str = "auto"
    save_mask_png: bool = True
    save_blob_csv: bool = True
    # If True (default), the probability mask is zeroed out in any time
    # column that already contains a user-painted label. Those time
    # regions are treated as 'owned' by the human annotator, so inference
    # never overwrites manual work. Committed columns come from
    # :func:`fnt.usv.usv_detector.mad_labels.committed_columns`.
    preserve_labels: bool = True
    # Optional per-wav processing parameters — filled from model checkpoint
    # when not specified.
    nperseg: Optional[int] = None
    noverlap: Optional[int] = None
    nfft: Optional[int] = None
    db_min: Optional[float] = None
    db_max: Optional[float] = None


# ----------------------------------------------------------------------
# Device selection (duplicated from training to avoid import cycle)
# ----------------------------------------------------------------------
def _resolve_device(pref: str) -> str:
    import torch
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "mps":
        return "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------
def load_model(model_path: str, device: str = "auto"):
    """Load a U-Net checkpoint saved by :func:`train_unet`.

    Returns ``(model, checkpoint_dict, resolved_device)``.
    """
    import torch
    try:
        import segmentation_models_pytorch as smp
    except Exception as e:
        raise RuntimeError(
            "segmentation_models_pytorch is required for MAD inference. "
            "Install with:\n    pip install segmentation-models-pytorch"
        ) from e

    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    encoder_name = ckpt.get('encoder_name', 'resnet18')
    in_channels = int(ckpt.get('in_channels', 1))
    classes = int(ckpt.get('classes', 1))

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=in_channels,
        classes=classes,
    )
    model.load_state_dict(ckpt['state_dict'])
    resolved = _resolve_device(device)
    model.to(resolved).eval()
    return model, ckpt, resolved


# ----------------------------------------------------------------------
# Full-file probability mask via tiled inference
# ----------------------------------------------------------------------
def _sliding_tile_starts(total: int, tile: int, overlap_fraction: float) -> List[int]:
    if total <= tile:
        return [0]
    step = max(1, int(round(tile * (1.0 - overlap_fraction))))
    starts = list(range(0, max(1, total - tile + 1), step))
    last_start = total - tile
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _hann_weight_1d(n: int) -> np.ndarray:
    """Return a Hann-ish weighting with small floor so tiles always contribute."""
    if n <= 1:
        return np.ones(n, dtype=np.float32)
    x = np.linspace(0.0, np.pi, n, dtype=np.float32)
    w = 0.5 - 0.5 * np.cos(2 * x)  # same shape as np.hanning, but vectorized
    w = np.maximum(w, 0.05)
    return w.astype(np.float32)


def infer_probability_mask(
    model, spec_image: np.ndarray,
    tile_freq_bins: int, tile_time_frames: int,
    overlap_fraction: float, device: str,
    batch_size: int = 4,
    progress: Optional[Callable[[int, int], None]] = None,
) -> np.ndarray:
    """Tile-and-stitch inference over a full-file spectrogram image.

    Returns a float32 probability mask with the **same shape** as
    ``spec_image`` (n_freq_bins, n_time_frames). Tiles are blended with
    a cosine weighting along time so tile seams don't show.
    """
    import torch

    H, W = spec_image.shape
    prob_sum = np.zeros((H, W), dtype=np.float32)
    weight_sum = np.zeros((H, W), dtype=np.float32)

    # Freq axis: we centered tiles at bin 0 during training, so crop the
    # top tile_freq_bins here too. If tile_freq_bins < H, we miss bins
    # above — but training used the same crop so predictions live in
    # the same subband.
    f_crop = min(tile_freq_bins, H)

    t_starts = _sliding_tile_starts(W, tile_time_frames, overlap_fraction)
    time_w = _hann_weight_1d(tile_time_frames)

    total_batches = (len(t_starts) + batch_size - 1) // batch_size
    batch_i = 0

    for b0 in range(0, len(t_starts), batch_size):
        starts = t_starts[b0:b0 + batch_size]
        tiles = np.zeros((len(starts), 1, tile_freq_bins, tile_time_frames), dtype=np.float32)
        for k, t0 in enumerate(starts):
            t1 = min(W, t0 + tile_time_frames)
            tiles[k, 0, :f_crop, :t1 - t0] = spec_image[:f_crop, t0:t1]
        xb = torch.from_numpy(tiles).to(device)
        with torch.no_grad():
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()[:, 0]  # (B, H, W)

        for k, t0 in enumerate(starts):
            t1 = min(W, t0 + tile_time_frames)
            tp = probs[k, :f_crop, :t1 - t0]
            w = time_w[:t1 - t0]
            prob_sum[:f_crop, t0:t1] += tp * w[None, :]
            weight_sum[:f_crop, t0:t1] += w[None, :]

        batch_i += 1
        if progress is not None:
            progress(batch_i, total_batches)

    out = np.zeros_like(prob_sum)
    nz = weight_sum > 0
    out[nz] = prob_sum[nz] / weight_sum[nz]
    return out


# ----------------------------------------------------------------------
# Blob extraction
# ----------------------------------------------------------------------
def extract_blobs(
    prob_mask: np.ndarray, threshold: float,
    min_blob_pixels: int = 8,
) -> List[Dict]:
    """Return connected-component blobs from a thresholded prob mask.

    Each blob is a dict:
        {
          't_start': int, 't_end_exclusive': int,
          'f_low': int, 'f_high_exclusive': int,
          'area_pixels': int, 'score': float,  # mean prob inside blob
        }
    """
    from scipy import ndimage as ndi
    binary = (prob_mask >= threshold).astype(np.uint8)
    if binary.sum() == 0:
        return []

    # 8-connectivity via 3x3 structuring element.
    structure = np.ones((3, 3), dtype=np.uint8)
    labels, n_labels = ndi.label(binary, structure=structure)
    if n_labels == 0:
        return []

    blobs: List[Dict] = []
    # objects[i] is the slice tuple for label i+1
    slices = ndi.find_objects(labels)
    for i, sl in enumerate(slices, start=1):
        if sl is None:
            continue
        fs, ts = sl  # (freq_slice, time_slice)
        sub_labels = labels[fs, ts]
        sub_mask = sub_labels == i
        area = int(sub_mask.sum())
        if area < min_blob_pixels:
            continue
        sub_probs = prob_mask[fs, ts]
        score = float(sub_probs[sub_mask].mean())
        blobs.append({
            't_start': int(ts.start),
            't_end_exclusive': int(ts.stop),
            'f_low': int(fs.start),
            'f_high_exclusive': int(fs.stop),
            'area_pixels': area,
            'score': score,
        })
    # Sort by time.
    blobs.sort(key=lambda b: (b['t_start'], b['f_low']))
    return blobs


# ----------------------------------------------------------------------
# Blob index → time / freq conversion
# ----------------------------------------------------------------------
def _time_per_frame(nperseg: int, noverlap: int, sr: int) -> float:
    return (nperseg - noverlap) / float(sr)


def _freq_per_bin(nfft: int, sr: int) -> float:
    return (sr / 2.0) / (nfft // 2)


def blobs_to_rows(
    blobs: List[Dict], nperseg: int, noverlap: int, nfft: int, sr: int,
) -> List[Dict]:
    """Convert pixel-index blobs to second / Hz rows for CSV output."""
    dt = _time_per_frame(nperseg, noverlap, sr)
    df = _freq_per_bin(nfft, sr)
    rows: List[Dict] = []
    for i, b in enumerate(blobs):
        rows.append({
            'blob_id': i,
            'start_s': round(b['t_start'] * dt, 6),
            'stop_s': round(b['t_end_exclusive'] * dt, 6),
            'min_freq_hz': round(b['f_low'] * df, 2),
            'max_freq_hz': round(b['f_high_exclusive'] * df, 2),
            'area_pixels': b['area_pixels'],
            'score': round(b['score'], 4),
            'status': 'pending',  # for user review (accept / reject)
        })
    return rows


def write_blob_csv(path: str, rows: List[Dict]) -> None:
    fieldnames = [
        'blob_id', 'start_s', 'stop_s',
        'min_freq_hz', 'max_freq_hz',
        'area_pixels', 'score', 'status',
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def read_blob_csv(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                'blob_id': int(r['blob_id']),
                'start_s': float(r['start_s']),
                'stop_s': float(r['stop_s']),
                'min_freq_hz': float(r['min_freq_hz']),
                'max_freq_hz': float(r['max_freq_hz']),
                'area_pixels': int(r['area_pixels']),
                'score': float(r['score']),
                'status': r.get('status', 'pending') or 'pending',
            })
    return rows


# ----------------------------------------------------------------------
# End-to-end per-file run
# ----------------------------------------------------------------------
def run_inference_on_file(
    wav_path: str,
    cfg: MADInferenceConfig,
    model=None, ckpt=None, device: Optional[str] = None,
    progress: Optional[Callable[[str, int, int], None]] = None,
) -> Dict:
    """Run inference on one wav, write sibling PNG + CSV, return summary.

    ``progress`` is invoked with ``(stage, i, n)`` where ``stage`` is one
    of ``'spec'``, ``'infer'``, ``'blobs'`` so the GUI can show a live
    bar even for files that take a while.
    """
    if model is None:
        model, ckpt, device = load_model(cfg.model_path, cfg.device)
    assert ckpt is not None
    assert device is not None

    # Fall back to checkpoint-saved spec params when cfg leaves them None.
    nperseg = int(cfg.nperseg if cfg.nperseg is not None else ckpt.get('nperseg', 512))
    noverlap = int(cfg.noverlap if cfg.noverlap is not None else ckpt.get('noverlap', 384))
    nfft = int(cfg.nfft if cfg.nfft is not None else ckpt.get('nfft', 1024))
    db_min = float(cfg.db_min if cfg.db_min is not None else ckpt.get('db_min', -100.0))
    db_max = float(cfg.db_max if cfg.db_max is not None else ckpt.get('db_max', -20.0))
    tile_freq_bins = int(ckpt.get('tile_freq_bins', cfg.tile_freq_bins))
    tile_time_frames = int(ckpt.get('tile_time_frames', cfg.tile_time_frames))

    if progress:
        progress('spec', 0, 1)
    audio, sr = load_audio(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    spec = compute_full_spec_image(
        audio.astype(np.float32), sr,
        nperseg=nperseg, noverlap=noverlap, nfft=nfft,
        db_min=db_min, db_max=db_max,
    )
    if progress:
        progress('spec', 1, 1)

    prob = infer_probability_mask(
        model, spec,
        tile_freq_bins=tile_freq_bins,
        tile_time_frames=tile_time_frames,
        overlap_fraction=cfg.tile_overlap_fraction,
        device=device,
        progress=(lambda i, n: progress('infer', i, n)) if progress else None,
    )

    # Preserve user-painted labels: zero out the probability mask in any
    # time column that already has a committed manual label. Predictions
    # never overwrite manual annotations.
    if cfg.preserve_labels:
        label_png = mask_sibling_path(wav_path)
        if Path(label_png).exists():
            try:
                user_mask = load_mask_png(label_png)
                # Normalise to fit the prob mask's (f, t) layout.
                if user_mask.shape == prob.shape:
                    committed = committed_columns(user_mask)
                elif user_mask.T.shape == prob.shape:
                    committed = committed_columns(user_mask.T)
                else:
                    # Sizes mismatch (rare — spec params differ) — just
                    # align on the overlap along the time axis.
                    t_len = min(user_mask.shape[1], prob.shape[1])
                    crop = user_mask[:, :t_len]
                    committed = np.zeros(prob.shape[1], dtype=bool)
                    committed[:t_len] = (crop == 1).any(axis=0)
                if committed.any():
                    prob[:, committed] = 0.0
            except Exception:
                # Don't let a bad label PNG block inference.
                pass

    if progress:
        progress('blobs', 0, 1)
    blobs = extract_blobs(prob, threshold=cfg.threshold, min_blob_pixels=cfg.min_blob_pixels)
    rows = blobs_to_rows(blobs, nperseg=nperseg, noverlap=noverlap, nfft=nfft, sr=sr)

    mask_png = pred_mask_sibling_path(wav_path)
    csv_path = pred_csv_sibling_path(wav_path)

    if cfg.save_mask_png:
        # Save thresholded binary mask (0 or 255) for easy viewing.
        binary = (prob >= cfg.threshold).astype(np.uint8) * 255
        save_mask_png(mask_png, binary)
    if cfg.save_blob_csv:
        write_blob_csv(csv_path, rows)
    if progress:
        progress('blobs', 1, 1)

    return {
        'wav_path': wav_path,
        'mask_png': mask_png if cfg.save_mask_png else None,
        'csv_path': csv_path if cfg.save_blob_csv else None,
        'n_blobs': len(rows),
        'prob_shape': list(prob.shape),
        'sample_rate': sr,
        'nperseg': nperseg, 'noverlap': noverlap, 'nfft': nfft,
    }


def run_inference_on_files(
    wav_paths: List[str],
    cfg: MADInferenceConfig,
    progress: Optional[Callable[[int, int, str, str, int, int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> List[Dict]:
    """Run inference on a batch of wavs. Loads the model once.

    ``progress`` is invoked as ``(file_i, file_n, wav_name, stage, stage_i, stage_n)``.
    """
    model, ckpt, device = load_model(cfg.model_path, cfg.device)
    results: List[Dict] = []
    n = len(wav_paths)
    for i, wav in enumerate(wav_paths):
        if should_stop and should_stop():
            break
        name = Path(wav).name

        def _inner(stage: str, si: int, sn: int, _i=i, _n=n, _name=name):
            if progress:
                progress(_i, _n, _name, stage, si, sn)
        try:
            summary = run_inference_on_file(
                wav, cfg, model=model, ckpt=ckpt, device=device, progress=_inner,
            )
            results.append(summary)
        except Exception as e:
            results.append({'wav_path': wav, 'error': str(e)})
    return results
