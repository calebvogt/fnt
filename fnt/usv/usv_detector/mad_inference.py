"""MAD inference pipeline.

Runs a trained U-Net checkpoint over full WAV files, stitches a
per-pixel probability mask, thresholds it, extracts connected-component
blobs, and writes a sibling CSV (blob boxes/scores) plus the small per-blob
pixel-mask crops into the sibling ``_FNT_masks.h5``. The full-resolution
probability grid is NOT persisted — MAD does not re-threshold after the fact
(re-run inference to change the threshold), so storing ~1 GB/file just to
re-derive call shapes on load is wasteful. See MAD_README.md.

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
from .mad_labels import pred_csv_sibling_path
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
    save_blob_csv: bool = True
    # If True (default), the probability mask is zeroed out in any time
    # column that already contains a confirmed call for this file (rebuilt
    # from the example store via
    # :func:`fnt.usv.usv_detector.mad_examples.reconstruct_file_mask`), so
    # inference never overwrites human-confirmed annotations.
    preserve_labels: bool = True
    # Example store used to look up confirmed labels for preserve_labels.
    training_data_dir: str = ""
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
        import segmentation_models_pytorch as smp  # noqa: F401 — presence check
    except Exception as e:
        raise RuntimeError(
            "segmentation_models_pytorch is required for MAD inference. "
            "Install with:\n    pip install segmentation-models-pytorch"
        ) from e

    from .mad_training import build_model

    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    # Older checkpoints predate selectable architectures — default to U-Net.
    model_arch = ckpt.get('model_arch', 'unet')
    encoder_name = ckpt.get('encoder_name', 'resnet18')
    in_channels = int(ckpt.get('in_channels', 1))
    classes = int(ckpt.get('classes', 1))

    model = build_model(
        model_arch, encoder_name, encoder_weights=None,
        in_channels=in_channels, classes=classes,
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
    wait_if_paused: Optional[Callable[[], None]] = None,
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
        if wait_if_paused is not None:
            wait_if_paused()  # blocks here while the user has paused the run
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
    include_mask: bool = False,
    spec: Optional[np.ndarray] = None,
) -> List[Dict]:
    """Return connected-component blobs from a thresholded prob mask.

    Each blob is a dict:
        {
          't_start': int, 't_end_exclusive': int,
          'f_low': int, 'f_high_exclusive': int,
          'area_pixels': int, 'score': float,  # mean prob inside blob
        }

    When ``include_mask`` is True, each blob also carries ``'mask'`` — the
    cropped boolean pixel mask of that blob (shape
    ``[f_high_exclusive - f_low, t_end_exclusive - t_start]``), so callers can
    persist the exact call shape without re-thresholding the full grid later.
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
        blob = {
            't_start': int(ts.start),
            't_end_exclusive': int(ts.stop),
            'f_low': int(fs.start),
            'f_high_exclusive': int(fs.stop),
            'area_pixels': area,
            'score': score,
        }
        # Keep the bbox mask so blobs_to_rows can compute the metric set (it
        # also has the full spectrogram for per-frame spectral features).
        if include_mask or spec is not None:
            blob['mask'] = np.ascontiguousarray(sub_mask)
        blobs.append(blob)
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


# Adjacent-frame frequency change above this counts as a "jump" (step call).
_FREQ_JUMP_HZ = 5000.0

# Per-call metric columns computed by ``compute_call_metrics`` — shared by the
# prediction (blobs_to_rows) and hand-label (GUI) paths so both row types are
# directly comparable. (call_number, inter_call_interval_ms, call_rate_hz are
# derived across calls at CSV-write time; model_name/threshold/min_blob_pixels
# are provenance set by the caller.)
CALL_METRIC_KEYS = [
    'peak_freq_hz', 'freq_bandwidth_hz',
    'start_freq_hz', 'end_freq_hz', 'mean_freq_hz', 'freq_std_hz',
    'freq_slope_hz_per_s', 'freq_excursion_hz', 'num_freq_jumps', 'sinuosity',
    'spectral_centroid_hz', 'spectral_entropy', 'tonality',
    'max_power_db', 'mean_power_db', 'total_energy_db', 'snr_db',
    'peak_time_frac', 'amplitude_modulation', 'fill_ratio', 'aspect_ratio',
]


# ±N frequency bins around the per-frame peak counted as "tonal" (CAD parity).
_TONALITY_HALF_BINS = 2


def compute_call_metrics(
    spec_db_cols: np.ndarray, mask: np.ndarray, f_low: int,
    df: float, dt: float, db_min: float, db_max: float,
) -> Dict:
    """Quantify one call.

    ``spec_db_cols`` is the **full-frequency** spectrogram (dB) for the call's
    time columns, shape ``(F_full, W)``; ``mask`` is the call's bounding-box
    pixel mask, shape ``(H, W)``; ``f_low`` is the global frequency-bin index of
    the mask's top row (so the band crop is ``spec_db_cols[f_low:f_low+H]``).
    dB is clipped to [db_min, db_max] so predictions (clipped spec) and
    hand-labels (raw dB) compute on the same scale. Per-frame spectral entropy
    and tonality use the full column (matching CAD's `dsp_detector`). Returns a
    dict keyed by :data:`CALL_METRIC_KEYS`; degenerate metrics are omitted.
    """
    m: Dict = {}
    if mask is None or mask.size == 0 or not mask.any():
        return m
    H, W = mask.shape
    full = np.clip(np.asarray(spec_db_cols, dtype=np.float64), db_min, db_max)
    F_full = full.shape[0]
    f_hi = f_low + H
    if f_hi > F_full or full.shape[1] != W:
        return m
    bb = full[f_low:f_hi, :]                  # call-band crop (H, W), dB
    Pbb = np.power(10.0, bb / 10.0)           # linear power in the band
    ys, xs = np.where(mask)
    vals_db = bb[mask]

    # --- power / energy (over the call's pixels) ---
    m['max_power_db'] = round(float(vals_db.max()), 2)
    m['mean_power_db'] = round(float(vals_db.mean()), 2)
    m['total_energy_db'] = round(float(10.0 * np.log10(Pbb[mask].sum() + 1e-12)), 2)
    peak_pix = int(np.argmax(vals_db))        # loudest call pixel → peak freq
    m['peak_freq_hz'] = round(float((f_low + ys[peak_pix]) * df), 2)

    # --- frequency contour: peak-power freq per masked time column ---
    cols, cfreq = [], []
    for t in range(W):
        rows_t = np.where(mask[:, t])[0]
        if rows_t.size == 0:
            continue
        peak_row = rows_t[int(np.argmax(bb[rows_t, t]))]
        cols.append(t)
        cfreq.append((f_low + peak_row) * df)
    if cfreq:
        cols_a = np.asarray(cols, dtype=np.float64)
        cf = np.asarray(cfreq, dtype=np.float64)
        m['start_freq_hz'] = round(float(cf[0]), 2)
        m['end_freq_hz'] = round(float(cf[-1]), 2)
        m['mean_freq_hz'] = round(float(cf.mean()), 2)
        m['freq_std_hz'] = round(float(cf.std()), 2)
        t_s = cols_a * dt
        if cf.size >= 2 and np.ptp(t_s) > 0:
            slope = float(np.polyfit(t_s, cf, 1)[0])
        else:
            slope = 0.0
        m['freq_slope_hz_per_s'] = round(slope, 2)
        dcf = np.abs(np.diff(cf))
        m['freq_excursion_hz'] = round(float(dcf.sum()), 2)
        m['num_freq_jumps'] = int((dcf > _FREQ_JUMP_HZ).sum())
        # sinuosity: contour path length / chord length, in (frame, bin) space
        fb = cf / df
        seg = np.hypot(np.diff(cols_a), np.diff(fb))
        chord = float(np.hypot(cols_a[-1] - cols_a[0], fb[-1] - fb[0]))
        m['sinuosity'] = round(float(seg.sum()) / chord, 3) if chord > 1e-6 else 1.0

    # --- frequency bandwidth from mask extent ---
    m['freq_bandwidth_hz'] = round(float((ys.max() - ys.min() + 1) * df), 2)

    # --- spectral centroid: power-weighted mean freq over the call's pixels ---
    Pm = np.where(mask, Pbb, 0.0)
    row_power = Pm.sum(axis=1)                 # power per band freq bin (masked)
    tot = float(row_power.sum())
    if tot > 0:
        freqs = (f_low + np.arange(H)) * df
        m['spectral_centroid_hz'] = round(float((freqs * row_power).sum() / tot), 2)

    # --- per-frame spectral entropy + tonality over the FULL column (CAD) ---
    Pfull = np.power(10.0, full / 10.0)
    max_ent = np.log2(F_full) if F_full > 1 else 1.0
    ton = np.zeros(W)
    ent = np.zeros(W)
    half = _TONALITY_HALF_BINS
    for t in range(W):
        col = Pfull[:, t]
        s = float(col.sum())
        if s <= 0:
            ent[t] = 1.0       # empty column → maximally "noisy"
            continue
        pk = int(np.argmax(col))
        lo, hi = max(0, pk - half), min(F_full, pk + half + 1)
        ton[t] = float(col[lo:hi].sum()) / s
        p = col / s
        p = p[p > 0]
        ent[t] = float(-(p * np.log2(p)).sum()) / max_ent
    m['tonality'] = round(float(ton.mean()), 4)
    m['spectral_entropy'] = round(float(ent.mean()), 4)

    # --- amplitude envelope over time (masked band energy per frame) ---
    col_energy = Pm.sum(axis=0)
    if W > 1:
        m['peak_time_frac'] = round(int(np.argmax(col_energy)) / float(W - 1), 3)
    env = col_energy[col_energy > 0]
    if env.size:
        emax, emin = float(env.max()), float(env.min())
        denom = emax + emin
        m['amplitude_modulation'] = round((emax - emin) / denom, 3) if denom > 0 else 0.0

    # --- morphology ---
    area = int(mask.sum())
    m['fill_ratio'] = round(area / float(H * W), 3) if H * W else 0.0
    m['aspect_ratio'] = round(W / float(H), 3) if H else 0.0

    # --- SNR: peak call power minus the local noise floor (CAD: max − floor).
    # Floor = median dB of the out-of-band rows at the call's time columns (the
    # background spectrum flanking the call), always available; fall back to the
    # off-mask pixels inside the bbox if the call spans the whole band. ---
    band = np.zeros(F_full, dtype=bool)
    band[f_low:f_hi] = True
    if (~band).any():
        noise = float(np.median(full[~band, :]))
    else:
        off = ~mask
        noise = float(np.median(bb[off])) if off.any() else db_min
    m['snr_db'] = round(float(m['max_power_db'] - noise), 2)
    return m


def blobs_to_rows(
    blobs: List[Dict], nperseg: int, noverlap: int, nfft: int, sr: int,
    db_min: Optional[float] = None, db_max: Optional[float] = None,
    spec: Optional[np.ndarray] = None,
) -> List[Dict]:
    """Convert pixel-index blobs to second / Hz rows for CSV output. When the
    (normalized) full ``spec`` + db range are supplied, attach the full per-call
    metric set via :func:`compute_call_metrics`."""
    dt = _time_per_frame(nperseg, noverlap, sr)
    df = _freq_per_bin(nfft, sr)
    span = (float(db_max) - float(db_min)
            if db_min is not None and db_max is not None else None)
    rows: List[Dict] = []
    for i, b in enumerate(blobs):
        min_f = round(b['f_low'] * df, 2)
        max_f = round(b['f_high_exclusive'] * df, 2)
        row = {
            'blob_id': i,
            'class': '',
            'start_s': round(b['t_start'] * dt, 6),
            'stop_s': round(b['t_end_exclusive'] * dt, 6),
            'min_freq_hz': min_f,
            'max_freq_hz': max_f,
            'freq_bandwidth_hz': round(max_f - min_f, 2),
            'area_pixels': b['area_pixels'],
            'score': round(b['score'], 4),
            'status': 'pending',  # for user review (accept / reject)
            'source': 'prediction',
        }
        if (spec is not None and b.get('mask') is not None and span is not None):
            # Full-frequency dB columns for this call's time span.
            cols_db = spec[:, b['t_start']:b['t_end_exclusive']] * span + db_min
            row.update(compute_call_metrics(
                cols_db, b['mask'], b['f_low'], df, dt, db_min, db_max))
        rows.append(row)
    return rows


# Unified per-wav detections CSV — column names/order mirror CAD's
# ``_FNT_CAD_detections.csv`` so the two tools' outputs are cross-readable.
# CAD-shared: call_number, call_id, start_seconds, stop_seconds, duration_ms,
# min/max/peak_freq_hz, freq_bandwidth_hz, max/mean_power_db, status, source.
# MAD-specific extras: class (call type), score (mean prob), area_pixels.
# (Internally rows use blob_id/start_s/stop_s keys; this layer translates.)
# Columns carried verbatim (key == column name) beyond the CAD-shared core.
_EXTRA_COLS = [
    'inter_call_interval_ms', 'call_rate_hz',
    'start_freq_hz', 'end_freq_hz', 'mean_freq_hz', 'freq_std_hz',
    'freq_slope_hz_per_s', 'freq_excursion_hz', 'num_freq_jumps', 'sinuosity',
    'spectral_centroid_hz', 'spectral_entropy', 'tonality',
    'total_energy_db', 'snr_db', 'peak_time_frac', 'amplitude_modulation',
    'fill_ratio', 'aspect_ratio',
    'model_name', 'threshold', 'min_blob_pixels',
]

# CAD-shared core (first 16, matching _FNT_CAD_detections.csv naming/order) +
# MAD-specific quantification columns appended.
CSV_FIELDNAMES = [
    'call_number', 'call_id', 'start_seconds', 'stop_seconds', 'duration_ms',
    'min_freq_hz', 'max_freq_hz', 'peak_freq_hz', 'freq_bandwidth_hz',
    'max_power_db', 'mean_power_db', 'class', 'score', 'area_pixels',
    'status', 'source',
] + _EXTRA_COLS

# Local-window half-width (seconds) for the call_rate_hz density estimate.
_CALL_RATE_WINDOW_S = 0.5


def _safe_float(v, default=0.0):
    try:
        if v is None or v == '':
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _coerce_blob_id(v):
    """Call ids are ints for predictions but stable strings for hand-labels."""
    s = str(v).strip()
    try:
        return int(s)
    except (TypeError, ValueError):
        return s


def write_blob_csv(path: str, rows: List[Dict]) -> None:
    """Write the unified detections CSV from internal row dicts (keys:
    blob_id, class, start_s, stop_s, min/max_freq_hz, area_pixels, score,
    status, and optionally peak_freq_hz/freq_bandwidth_hz/max_power_db/
    mean_power_db/source). call_number is (re)assigned by time order;
    duration/bandwidth are derived if absent."""
    ordered = sorted(
        enumerate(rows), key=lambda kv: (_safe_float(kv[1].get('start_s')), kv[0]))
    starts = [_safe_float(r.get('start_s')) for _, r in ordered]
    stops = [_safe_float(r.get('stop_s')) for _, r in ordered]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES,
                                extrasaction='ignore')
        writer.writeheader()
        for n, (_, r) in enumerate(ordered, start=1):
            start, stop = starts[n - 1], stops[n - 1]
            minf = _safe_float(r.get('min_freq_hz'))
            maxf = _safe_float(r.get('max_freq_hz'))
            bw = r.get('freq_bandwidth_hz')
            bw = round(maxf - minf, 2) if bw in (None, '') else bw
            # Cross-call: gap to previous offset, and local emission rate.
            ici = '' if n == 1 else round((start - stops[n - 2]) * 1000.0, 2)
            lo, hi = start - _CALL_RATE_WINDOW_S, start + _CALL_RATE_WINDOW_S
            rate = round(sum(1 for s in starts if lo <= s <= hi)
                         / (2.0 * _CALL_RATE_WINDOW_S), 2)
            out = {
                'call_number': n,
                'call_id': r.get('blob_id'),
                'start_seconds': start,
                'stop_seconds': stop,
                'duration_ms': round((stop - start) * 1000.0, 2),
                'min_freq_hz': minf,
                'max_freq_hz': maxf,
                'peak_freq_hz': r.get('peak_freq_hz', ''),
                'freq_bandwidth_hz': bw,
                'max_power_db': r.get('max_power_db', ''),
                'mean_power_db': r.get('mean_power_db', ''),
                'class': r.get('class', '') or '',
                'score': r.get('score', ''),
                'area_pixels': r.get('area_pixels', ''),
                'status': r.get('status', 'pending') or 'pending',
                'source': r.get('source', '') or '',
            }
            for col in _EXTRA_COLS:
                out[col] = r.get(col, '')
            out['inter_call_interval_ms'] = ici
            out['call_rate_hz'] = rate
            writer.writerow(out)


def read_blob_csv(path: str) -> List[Dict]:
    """Read the unified CSV into internal row dicts. Tolerant of the legacy
    column names (blob_id/start_s/stop_s) and of missing optional columns."""
    rows: List[Dict] = []
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            def g(*keys, default=None):
                for k in keys:
                    v = r.get(k)
                    if v not in (None, ''):
                        return v
                return default
            row = {
                'blob_id': _coerce_blob_id(g('call_id', 'blob_id')),
                'class': (g('class', default='') or '').strip(),
                'start_s': _safe_float(g('start_seconds', 'start_s')),
                'stop_s': _safe_float(g('stop_seconds', 'stop_s')),
                'min_freq_hz': _safe_float(g('min_freq_hz')),
                'max_freq_hz': _safe_float(g('max_freq_hz')),
                'area_pixels': int(_safe_float(g('area_pixels'))),
                'score': _safe_float(g('score')),
                'status': g('status', default='pending') or 'pending',
                'source': g('source', default='') or '',
                'peak_freq_hz': g('peak_freq_hz', default=''),
                'max_power_db': g('max_power_db', default=''),
                'mean_power_db': g('mean_power_db', default=''),
            }
            # Carry the appended quantification/provenance columns through
            # verbatim, so a read-modify-write (e.g. status change) preserves
            # them. inter_call_interval_ms / call_rate_hz are recomputed on
            # write, so they needn't round-trip.
            for col in _EXTRA_COLS:
                row[col] = g(col, default='')
            rows.append(row)
    return rows


# ----------------------------------------------------------------------
# End-to-end per-file run
# ----------------------------------------------------------------------
def run_inference_on_file(
    wav_path: str,
    cfg: MADInferenceConfig,
    model=None, ckpt=None, device: Optional[str] = None,
    progress: Optional[Callable[[str, int, int], None]] = None,
    wait_if_paused: Optional[Callable[[], None]] = None,
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

    import time as _time
    if progress:
        progress('spec', 0, 1)
    _t_spec0 = _time.perf_counter()
    audio, sr = load_audio(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio_dur = len(audio) / float(sr) if sr else 0.0
    spec = compute_full_spec_image(
        audio.astype(np.float32), sr,
        nperseg=nperseg, noverlap=noverlap, nfft=nfft,
        db_min=db_min, db_max=db_max,
    )
    t_spec = _time.perf_counter() - _t_spec0
    if progress:
        progress('spec', 1, 1)

    _t_inf0 = _time.perf_counter()
    prob = infer_probability_mask(
        model, spec,
        tile_freq_bins=tile_freq_bins,
        tile_time_frames=tile_time_frames,
        overlap_fraction=cfg.tile_overlap_fraction,
        device=device,
        progress=(lambda i, n: progress('infer', i, n)) if progress else None,
        wait_if_paused=wait_if_paused,
    )
    t_infer = _time.perf_counter() - _t_inf0

    # Preserve confirmed labels: zero out the probability mask in any time
    # column that already contains a human-confirmed call for this file, so
    # predictions never overwrite confirmed annotations.
    if cfg.preserve_labels and cfg.training_data_dir:
        try:
            from .mad_examples import reconstruct_file_mask
            user_mask = reconstruct_file_mask(
                cfg.training_data_dir, Path(wav_path).name, prob.shape
            )
            cols = (user_mask > 0).any(axis=0)
            if cols.any():
                prob[:, cols] = 0.0
        except Exception:
            # Don't let a label-store hiccup block inference.
            pass

    if progress:
        progress('blobs', 0, 1)
    _t_blob0 = _time.perf_counter()
    blobs = extract_blobs(prob, threshold=cfg.threshold,
                          min_blob_pixels=cfg.min_blob_pixels, include_mask=True,
                          spec=spec)
    rows = blobs_to_rows(blobs, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                         sr=sr, db_min=db_min, db_max=db_max, spec=spec)
    # Provenance: which model + settings produced these predictions.
    model_name = Path(cfg.model_path).stem if cfg.model_path else ''
    for r in rows:
        r['model_name'] = model_name
        r['threshold'] = cfg.threshold
        r['min_blob_pixels'] = cfg.min_blob_pixels

    csv_path = pred_csv_sibling_path(wav_path)
    if cfg.save_blob_csv:
        # The CSV is unified: hand-labels (stable string blob_ids) live here
        # alongside predictions (int blob_ids). A run replaces only the
        # prediction rows; existing hand-label rows are preserved.
        preserved = []
        if Path(csv_path).is_file():
            try:
                preserved = [r for r in read_blob_csv(csv_path)
                             if not isinstance(r.get('blob_id'), int)]
            except Exception:
                preserved = []
        write_blob_csv(csv_path, preserved + rows)
    # Persist each blob's small cropped mask (NOT the multi-GB /prob grid):
    # blob_id matches the CSV row's blob_id (both come from enumerate over the
    # same time-sorted blob list). On file switch these few-MB crops are read
    # directly, so predictions redraw without decompressing the full grid.
    # MAD intentionally drops /prob — re-run inference to change the threshold.
    h5_path = None
    try:
        from .fnt_mask_store import (masks_sibling_path, write_pred_masks,
                                     set_grid_attrs, delete_prob)
        h5_path = masks_sibling_path(wav_path)
        set_grid_attrs(h5_path, sample_rate=sr, nperseg=nperseg,
                       noverlap=noverlap, nfft=nfft,
                       n_freq_bins=prob.shape[0], n_time_frames=prob.shape[1])
        crops = [
            {'blob_id': i, 'mask': b['mask'],
             'f_off': b['f_low'], 't_off': b['t_start']}
            for i, b in enumerate(blobs)
        ]
        write_pred_masks(h5_path, crops)
        # Reclaim disk from any legacy full-grid prob map for this file.
        delete_prob(h5_path)
    except Exception:
        h5_path = None
    t_blobs = _time.perf_counter() - _t_blob0
    if progress:
        progress('blobs', 1, 1)

    total = t_spec + t_infer + t_blobs
    # Realtime factor: seconds of audio scanned per wall-second of inference
    # (the tile-scan stage). <1 means slower than realtime — typical on CPU.
    rt_factor = (audio_dur / t_infer) if t_infer > 0 else 0.0
    return {
        'wav_path': wav_path,
        'csv_path': csv_path if cfg.save_blob_csv else None,
        'h5_path': h5_path,
        'n_blobs': len(rows),
        'prob_shape': list(prob.shape),
        'sample_rate': sr,
        'nperseg': nperseg, 'noverlap': noverlap, 'nfft': nfft,
        'timing': {
            'device': device,
            'audio_dur_s': round(audio_dur, 1),
            't_spec': round(t_spec, 2),
            't_infer': round(t_infer, 2),
            't_blobs': round(t_blobs, 2),
            't_total': round(total, 2),
            'realtime_factor': round(rt_factor, 2),
        },
    }


def run_inference_on_files(
    wav_paths: List[str],
    cfg: MADInferenceConfig,
    progress: Optional[Callable[[int, int, str, str, int, int], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    wait_if_paused: Optional[Callable[[], None]] = None,
    on_device: Optional[Callable[[str], None]] = None,
    on_file_done: Optional[Callable[[Dict], None]] = None,
) -> List[Dict]:
    """Run inference on a batch of wavs. Loads the model once.

    ``progress`` is invoked as ``(file_i, file_n, wav_name, stage, stage_i, stage_n)``.
    ``wait_if_paused``, if given, is called between files and between inference
    tiles; it should block while the run is paused and return on resume/stop.
    ``on_device(device)`` is called once after the model loads; ``on_file_done``
    is called with each file's summary (incl. timing) as it completes — both let
    the GUI log device + per-file speed live.
    """
    model, ckpt, device = load_model(cfg.model_path, cfg.device)
    if on_device is not None:
        try:
            on_device(device)
        except Exception:
            pass
    results: List[Dict] = []
    n = len(wav_paths)
    for i, wav in enumerate(wav_paths):
        if wait_if_paused is not None:
            wait_if_paused()
        if should_stop and should_stop():
            break
        name = Path(wav).name

        def _inner(stage: str, si: int, sn: int, _i=i, _n=n, _name=name):
            if progress:
                progress(_i, _n, _name, stage, si, sn)
        try:
            summary = run_inference_on_file(
                wav, cfg, model=model, ckpt=ckpt, device=device, progress=_inner,
                wait_if_paused=wait_if_paused,
            )
            results.append(summary)
        except Exception as e:
            results.append({'wav_path': wav, 'error': str(e)})
        if on_file_done is not None:
            try:
                on_file_done(results[-1])
            except Exception:
                pass
    return results
