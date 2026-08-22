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
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .mad_dataset import compute_full_spec_image
from .mad_labels import pred_csv_sibling_path
from .spectrogram import load_audio


# Default accumulator budget, in time frames. The float32 accumulator for one
# chunk costs n_freq_bins x chunk_frames x 4 bytes: at 513 bins this is ~270 MB
# for the default, regardless of how long the recording is. See
# infer_probability_mask for why the whole grid is no longer materialized.
DEFAULT_CHUNK_FRAMES = 131072


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
    # Tiles per forward pass. 8 keeps a modest GPU busy; raise to 16-32 on a
    # card with plenty of VRAM for a near-linear throughput win, lower it on
    # out-of-memory. Has no effect on results.
    batch_size: int = 8
    # Mixed precision (fp16 autocast). CUDA only — ignored on CPU/MPS. Roughly
    # 1.5-2x faster on modern NVIDIA cards; the probability grid is quantized to
    # 1/255 anyway, so fp16 costs nothing that survives to the output.
    amp: bool = True
    # Time-column budget for the per-chunk float32 accumulator; bounds peak RAM
    # independently of recording length. See infer_probability_mask.
    chunk_frames: int = DEFAULT_CHUNK_FRAMES
    save_blob_csv: bool = True
    # If True (default), the probability mask is zeroed out in any time
    # column that already contains a confirmed call for this file (rebuilt
    # from the example store via
    # :func:`fnt.usv.usv_detector.mad_examples.reconstruct_file_mask`), so
    # inference never overwrites human-confirmed annotations.
    preserve_labels: bool = True
    # Example store used to look up confirmed labels for preserve_labels.
    training_data_dir: str = ""
    # Merge consecutive detections that belong to one call but surfaced as
    # separate blobs (tile seams, brief sub-threshold dips). Off by default.
    # ``merge_max_gap_s`` is the largest time gap bridged; freq-overlap gating
    # keeps time-adjacent but frequency-separated calls (harmonics) distinct.
    merge_consecutive: bool = False
    merge_max_gap_s: float = 0.01
    merge_require_freq_overlap: bool = True
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


def prob_threshold_u8(threshold: float) -> int:
    """A [0,1] probability threshold in the uint8 domain used by prob grids."""
    return int(round(max(0.0, min(1.0, threshold)) * 255.0))


def infer_probability_mask(
    model, spec_image: np.ndarray,
    tile_freq_bins: int, tile_time_frames: int,
    overlap_fraction: float, device: str,
    batch_size: int = 8,
    use_amp: bool = True,
    chunk_frames: int = DEFAULT_CHUNK_FRAMES,
    out_dtype=np.uint8,
    progress: Optional[Callable[[int, int], None]] = None,
    wait_if_paused: Optional[Callable[[], None]] = None,
) -> np.ndarray:
    """Tile-and-stitch inference over a full-file spectrogram image.

    Returns a probability mask with the **same shape** as ``spec_image``
    (n_freq_bins, n_time_frames). Tiles are blended with a cosine weighting
    along time so tile seams don't show.

    ``out_dtype`` is ``np.uint8`` by default, storing probability x 255. A
    10-minute 250 kHz recording is ~1.17M frames x 513 bins; as float32 that
    single grid is 2.4 GB, and the old implementation held three of them plus a
    same-shape float32 ``weight_sum`` — ~9.6 GB for one file, which is what put
    long recordings at risk of OOM. uint8 costs 600 MB, and 1/255 resolution is
    far finer than anything thresholding or mean-probability scores need. Pass
    ``out_dtype=np.float32`` for small windows (the interactive view path) where
    the grid is a few MB and exact probabilities are convenient.

    Memory is bounded three ways:
      * the full-resolution float32 accumulator is per **chunk** of at most
        ``chunk_frames`` columns, not per file;
      * ``weight_sum`` is 1-D — the Hann blend varies only along time, so the
        old (n_freq, n_time) weight array stored the same value in every row;
      * normalization happens in place.

    Chunks are cut on tile boundaries and the un-finalized tail is carried into
    the next chunk, so the result is bit-identical to accumulating the whole
    grid at once — no seam artifacts at chunk edges.
    """
    import torch

    H, W = spec_image.shape
    out = np.zeros((H, W), dtype=out_dtype)
    quantize = np.dtype(out_dtype) == np.uint8

    # Freq axis: we centered tiles at bin 0 during training, so crop the
    # top tile_freq_bins here too. If tile_freq_bins < H, we miss bins
    # above — but training used the same crop so predictions live in
    # the same subband.
    f_crop = min(tile_freq_bins, H)

    t_starts = _sliding_tile_starts(W, tile_time_frames, overlap_fraction)
    time_w = _hann_weight_1d(tile_time_frames)

    # Group tiles into chunks whose accumulator stays within the frame budget.
    # A chunk always holds at least one tile, however small the budget.
    chunk_frames = max(int(chunk_frames), tile_time_frames)
    groups: List[List[int]] = []
    cur: List[int] = []
    for t0 in t_starts:
        if cur and (min(W, t0 + tile_time_frames) - cur[0]) > chunk_frames:
            groups.append(cur)
            cur = []
        cur.append(t0)
    if cur:
        groups.append(cur)

    amp_on = bool(use_amp) and str(device).startswith('cuda')
    total_batches = sum(
        (len(g) + batch_size - 1) // batch_size for g in groups) or 1
    batch_i = 0

    # Un-finalized tail of the previous chunk: columns the next chunk's tiles
    # still contribute to. At most one tile wide.
    carry_acc: Optional[np.ndarray] = None
    carry_w: Optional[np.ndarray] = None

    for gi, starts in enumerate(groups):
        g0 = starts[0]
        g1 = min(W, starts[-1] + tile_time_frames)
        acc = np.zeros((f_crop, g1 - g0), dtype=np.float32)
        wacc = np.zeros(g1 - g0, dtype=np.float32)
        if carry_acc is not None:
            cw = carry_acc.shape[1]
            acc[:, :cw] += carry_acc
            wacc[:cw] += carry_w
            carry_acc = carry_w = None

        for b0 in range(0, len(starts), batch_size):
            if wait_if_paused is not None:
                wait_if_paused()  # blocks while the user has paused the run
            bstarts = starts[b0:b0 + batch_size]
            tiles = np.zeros(
                (len(bstarts), 1, tile_freq_bins, tile_time_frames),
                dtype=np.float32)
            for k, t0 in enumerate(bstarts):
                t1 = min(W, t0 + tile_time_frames)
                tiles[k, 0, :f_crop, :t1 - t0] = spec_image[:f_crop, t0:t1]
            xb = torch.from_numpy(tiles).to(device)
            with torch.no_grad():
                if amp_on:
                    with torch.autocast(device_type='cuda',
                                        dtype=torch.float16):
                        logits = model(xb)
                    probs = torch.sigmoid(logits.float()).cpu().numpy()[:, 0]
                else:
                    logits = model(xb)
                    probs = torch.sigmoid(logits).cpu().numpy()[:, 0]

            for k, t0 in enumerate(bstarts):
                t1 = min(W, t0 + tile_time_frames)
                n = t1 - t0
                a0 = t0 - g0
                acc[:, a0:a0 + n] += probs[k, :f_crop, :n] * time_w[None, :n]
                wacc[a0:a0 + n] += time_w[:n]

            batch_i += 1
            if progress is not None:
                progress(batch_i, total_batches)

        # Columns before the next chunk's first tile are final; the rest still
        # awaits that chunk's contributions and becomes the carry.
        final_end = groups[gi + 1][0] if gi + 1 < len(groups) else g1
        fw = final_end - g0
        done = acc[:, :fw]
        w = wacc[:fw]
        nz = w > 0
        if nz.any():
            done[:, nz] /= w[nz]
        if quantize:
            np.clip(done, 0.0, 1.0, out=done)
            done *= 255.0
            out[:f_crop, g0:final_end] = done.astype(np.uint8)
        else:
            out[:f_crop, g0:final_end] = done
        if fw < acc.shape[1]:
            carry_acc = acc[:, fw:].copy()
            carry_w = wacc[fw:].copy()
        del acc, wacc

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

    ``prob_mask`` may be float in [0,1] or the memory-efficient uint8 grid
    (probability x 255) returned by :func:`infer_probability_mask`; ``threshold``
    is a [0,1] probability either way and ``score`` always comes back in [0,1].
    """
    from scipy import ndimage as ndi
    u8 = prob_mask.dtype == np.uint8
    cut = prob_threshold_u8(threshold) if u8 else threshold
    score_scale = (1.0 / 255.0) if u8 else 1.0
    binary = (prob_mask >= cut).astype(np.uint8)
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
        score = float(sub_probs[sub_mask].mean()) * score_scale
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


def _freq_overlap(a: Dict, b: Dict) -> bool:
    return (a['f_low'] < b['f_high_exclusive']
            and b['f_low'] < a['f_high_exclusive'])


def merge_consecutive_blobs(
    blobs: List[Dict], max_gap_frames: int,
    require_freq_overlap: bool = True,
) -> List[Dict]:
    """Merge runs of consecutive blobs into single detections.

    A long call split across tile seams (or broken by a brief sub-threshold
    dip) surfaces as several adjacent blobs; this stitches them back into one.
    Two blobs join when the time gap between the running cluster's offset and
    the next blob's onset is ``<= max_gap_frames`` (a negative gap means they
    already overlap in time) and — when ``require_freq_overlap`` — their
    frequency bands overlap, so calls stacked in time but separated in
    frequency (e.g. a harmonic vs. its fundamental) are left distinct.

    Inspired by BirdNET's ``--merge_consecutive``. Blobs must carry the
    ``'mask'`` bbox crop (as :func:`extract_blobs` produces with
    ``include_mask=True``); merged masks are OR-composited into the union
    bounding box, and the merged ``score`` is area-weighted across components
    so a big confident blob isn't diluted by a small faint neighbour. Returns
    new blob dicts in onset order; input is left untouched.
    """
    if max_gap_frames < 0 or len(blobs) < 2:
        return list(blobs)
    ordered = sorted(blobs, key=lambda b: (b['t_start'], b['f_low']))

    def _flush(group: List[Dict]) -> Dict:
        if len(group) == 1:
            return dict(group[0])
        f_low = min(b['f_low'] for b in group)
        f_high = max(b['f_high_exclusive'] for b in group)
        t_start = min(b['t_start'] for b in group)
        t_end = max(b['t_end_exclusive'] for b in group)
        mask = np.zeros((f_high - f_low, t_end - t_start), dtype=bool)
        for b in group:
            bm = b.get('mask')
            if bm is None:
                continue
            fo, to = b['f_low'] - f_low, b['t_start'] - t_start
            mask[fo:fo + bm.shape[0], to:to + bm.shape[1]] |= bm
        area = int(mask.sum()) or sum(b['area_pixels'] for b in group)
        w = float(sum(b['area_pixels'] for b in group)) or 1.0
        score = sum(b['score'] * b['area_pixels'] for b in group) / w
        return {
            't_start': t_start, 't_end_exclusive': t_end,
            'f_low': f_low, 'f_high_exclusive': f_high,
            'area_pixels': area, 'score': float(score),
            'mask': np.ascontiguousarray(mask),
        }

    merged: List[Dict] = []
    group = [ordered[0]]
    cur_end = ordered[0]['t_end_exclusive']
    for b in ordered[1:]:
        gap = b['t_start'] - cur_end
        joins = gap <= max_gap_frames and (
            not require_freq_overlap
            or any(_freq_overlap(b, g) for g in group))
        if joins:
            group.append(b)
            cur_end = max(cur_end, b['t_end_exclusive'])
        else:
            merged.append(_flush(group))
            group = [b]
            cur_end = b['t_end_exclusive']
    merged.append(_flush(group))
    merged.sort(key=lambda b: (b['t_start'], b['f_low']))
    return merged


# ----------------------------------------------------------------------
# Blob index → time / freq conversion
# ----------------------------------------------------------------------
def _time_per_frame(nperseg: int, noverlap: int, sr: int) -> float:
    return (nperseg - noverlap) / float(sr)


def _frame_time_origin(nperseg: int, sr: int) -> float:
    """Seconds at spectrogram frame 0.

    ``scipy.signal.spectrogram`` centres its first frame half a window in, so
    frame ``i`` sits at ``(nperseg/2 + i*hop)/sr`` — **not** ``i*hop/sr``.
    Dropping this term biased every exported onset early by ``nperseg/(2*sr)``
    (~1 ms at nperseg=512, sr=250 kHz — two whole hops) and put MAD's times on a
    different axis from CAD's, which reads ``times[idx]`` straight out of scipy.
    See MAD_README.md.
    """
    return nperseg / (2.0 * float(sr))


def frames_to_seconds(idx, nperseg: int, noverlap: int, sr: int):
    """Spectrogram frame index → seconds (scipy/CAD convention)."""
    return (_frame_time_origin(nperseg, sr)
            + idx * _time_per_frame(nperseg, noverlap, sr))


def seconds_to_frames(sec, nperseg: int, noverlap: int, sr: int):
    """Seconds → spectrogram frame index (inverse of :func:`frames_to_seconds`)."""
    dt = _time_per_frame(nperseg, noverlap, sr)
    if dt <= 0:
        return 0.0
    return (sec - _frame_time_origin(nperseg, sr)) / dt


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
    t_org = _frame_time_origin(nperseg, sr)
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
            'start_s': round(t_org + b['t_start'] * dt, 6),
            'stop_s': round(t_org + b['t_end_exclusive'] * dt, 6),
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
            # ``starts`` is sorted, so the window population is two bisects
            # instead of a full scan — the scan made this writer O(N^2), which
            # dominated every bulk review action on files with 1000+ calls.
            ici = '' if n == 1 else round((start - stops[n - 2]) * 1000.0, 2)
            lo, hi = start - _CALL_RATE_WINDOW_S, start + _CALL_RATE_WINDOW_S
            rate = round((bisect_right(starts, hi) - bisect_left(starts, lo))
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
# Call embeddings — fixed-length feature vectors per detection.
#
# Each detection's spectrogram patch is pushed through the trained model's
# encoder and the deepest feature map is global-average-pooled to a single
# vector (512-d for a ResNet18 encoder). These embeddings support clustering
# calls by similarity, surfacing novel call types, or comparing repertoires
# across animals — the use cases BirdNET's ``embeddings`` command targets,
# but computed from the model you trained on your own calls.
# ----------------------------------------------------------------------

def extract_embeddings(
    model, spec_image: np.ndarray, boxes: List[Tuple[int, int, int, int]],
    device: str, patch_size: int = 128, batch_size: int = 16,
) -> np.ndarray:
    """Global-average-pooled encoder features for each pixel box.

    ``boxes`` are ``(f_low, f_high_exclusive, t_start, t_end_exclusive)`` in
    spectrogram-pixel coordinates. Each box is cropped from ``spec_image``,
    resized to ``patch_size`` (a multiple of 32 for the encoder's downsampling),
    and run through ``model.encoder``; the deepest feature map is pooled over
    space. Returns a ``(len(boxes), D)`` float32 array (D=512 for ResNet18).
    """
    import torch
    import torch.nn.functional as F

    if not boxes:
        return np.zeros((0, 0), dtype=np.float32)
    H, W = spec_image.shape
    out: List[np.ndarray] = []
    for b0 in range(0, len(boxes), batch_size):
        chunk = boxes[b0:b0 + batch_size]
        patches = np.zeros((len(chunk), 1, patch_size, patch_size),
                           dtype=np.float32)
        for k, (f0, f1, t0, t1) in enumerate(chunk):
            f0, f1 = max(0, f0), min(H, max(f0 + 1, f1))
            t0, t1 = max(0, t0), min(W, max(t0 + 1, t1))
            crop = spec_image[f0:f1, t0:t1]
            ten = torch.from_numpy(np.ascontiguousarray(crop))[None, None]
            rs = F.interpolate(ten, size=(patch_size, patch_size),
                               mode='bilinear', align_corners=False)
            patches[k, 0] = rs[0, 0].numpy()
        xb = torch.from_numpy(patches).to(device)
        with torch.no_grad():
            feats = model.encoder(xb)
            deepest = feats[-1]  # (B, C, h, w)
            pooled = deepest.mean(dim=(2, 3))  # (B, C)
        out.append(pooled.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


def embed_file(
    wav_path: str, cfg: MADInferenceConfig,
    model=None, ckpt=None, device: Optional[str] = None,
) -> Dict:
    """Compute a per-detection embedding for one wav's saved detections.

    Reads the file's sibling detections CSV (produced by inference), rebuilds
    the spectrogram with the checkpoint's params, and embeds each detection's
    box. Returns ``{'wav_path', 'blob_id', 'start_s', 'stop_s', 'class',
    'embeddings'}`` where ``embeddings`` is ``(N, D)``. Raises if the file has
    no saved detections."""
    if model is None:
        model, ckpt, device = load_model(cfg.model_path, cfg.device)
    assert ckpt is not None and device is not None

    csv_path = pred_csv_sibling_path(wav_path)
    if not Path(csv_path).is_file():
        raise RuntimeError(
            f"No detections CSV for {Path(wav_path).name} — run inference "
            "(analyze) first.")
    rows = [r for r in read_blob_csv(csv_path)
            if (r.get('status') or 'pending') != 'rejected']
    if not rows:
        return {'wav_path': wav_path, 'blob_id': [], 'start_s': [],
                'stop_s': [], 'class': [],
                'embeddings': np.zeros((0, 0), dtype=np.float32)}

    nperseg = int(cfg.nperseg if cfg.nperseg is not None else ckpt.get('nperseg', 512))
    noverlap = int(cfg.noverlap if cfg.noverlap is not None else ckpt.get('noverlap', 384))
    nfft = int(cfg.nfft if cfg.nfft is not None else ckpt.get('nfft', 1024))
    db_min = float(cfg.db_min if cfg.db_min is not None else ckpt.get('db_min', -100.0))
    db_max = float(cfg.db_max if cfg.db_max is not None else ckpt.get('db_max', -20.0))

    audio, sr = load_audio(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    spec = compute_full_spec_image(
        audio.astype(np.float32), sr, nperseg=nperseg, noverlap=noverlap,
        nfft=nfft, db_min=db_min, db_max=db_max)

    df = _freq_per_bin(nfft, sr)
    boxes: List[Tuple[int, int, int, int]] = []
    for r in rows:
        t0 = int(seconds_to_frames(
            _safe_float(r.get('start_s')), nperseg, noverlap, sr))
        t1 = int(round(seconds_to_frames(
            _safe_float(r.get('stop_s')), nperseg, noverlap, sr)))
        f0 = int(_safe_float(r.get('min_freq_hz')) / df) if df else 0
        f1 = int(round(_safe_float(r.get('max_freq_hz')) / df)) if df else 0
        boxes.append((f0, f1, t0, t1))

    emb = extract_embeddings(model, spec, boxes, device)
    return {
        'wav_path': wav_path,
        'blob_id': [r.get('blob_id') for r in rows],
        'start_s': [_safe_float(r.get('start_s')) for r in rows],
        'stop_s': [_safe_float(r.get('stop_s')) for r in rows],
        'class': [(r.get('class') or '').strip() for r in rows],
        'embeddings': emb,
    }


def write_embeddings_npz(path: str, results: List[Dict]) -> int:
    """Write per-detection embeddings from one or more :func:`embed_file`
    results into a single ``.npz``. Arrays: ``wav`` (source file per row),
    ``blob_id``, ``start_s``, ``stop_s``, ``class``, and ``embeddings``
    ``(total_N, D)``. Returns the total row count."""
    wav, bid, t0, t1, cls, embs = [], [], [], [], [], []
    for res in results:
        e = res['embeddings']
        if e.size == 0:
            continue
        n = e.shape[0]
        wav.extend([res['wav_path']] * n)
        bid.extend(res['blob_id'])
        t0.extend(res['start_s'])
        t1.extend(res['stop_s'])
        cls.extend(res['class'])
        embs.append(e)
    stacked = (np.concatenate(embs, axis=0) if embs
               else np.zeros((0, 0), dtype=np.float32))
    np.savez(
        path,
        wav=np.array(wav, dtype=object),
        blob_id=np.array(bid, dtype=object),
        start_s=np.array(t0, dtype=np.float32),
        stop_s=np.array(t1, dtype=np.float32),
        **{'class': np.array(cls, dtype=object)},
        embeddings=stacked,
    )
    return stacked.shape[0]


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
        batch_size=cfg.batch_size,
        use_amp=cfg.amp,
        chunk_frames=cfg.chunk_frames,
        progress=(lambda i, n: progress('infer', i, n)) if progress else None,
        wait_if_paused=wait_if_paused,
    )
    t_infer = _time.perf_counter() - _t_inf0

    # --- Carry reviewed decisions across re-runs -----------------------
    # Read the file's prior detections so a normal re-run only ever
    # (re)generates *pending* predictions. Two classes of prior row survive:
    #   • hand-labels        — string blob_id (painted / SAM), always kept, and
    #   • reviewed predictions — int blob_id already Accepted or Rejected.
    # Reviewed calls' regions are blanked from the probability map below so the
    # model can't resurface them as fresh pending blobs. Deleted calls left no
    # row, so their region re-opens for detection.
    #
    # ``preserve_labels`` is the master switch: when the user picks "Re-detect
    # from scratch" it is False, so we ignore reviewed decisions (discard those
    # rows, no region shielding) and re-detect everywhere. Hand-label ROWS are
    # never deleted regardless, but with preserve off their regions aren't
    # shielded either, so predictions may appear over them.
    dt = _time_per_frame(nperseg, noverlap, sr)
    df = _freq_per_bin(nfft, sr)
    csv_path = pred_csv_sibling_path(wav_path)
    prior_rows: List[Dict] = []
    if Path(csv_path).is_file():
        try:
            prior_rows = read_blob_csv(csv_path)
        except Exception:
            prior_rows = []
    handlabel_rows = [r for r in prior_rows
                      if not isinstance(r.get('blob_id'), int)]
    reviewed_rows = ([r for r in prior_rows
                      if isinstance(r.get('blob_id'), int)
                      and r.get('status') in ('accepted', 'rejected')]
                     if cfg.preserve_labels else [])

    # Prior per-blob crops — used both to redraw preserved calls as exact masks
    # and to blank their exact mask (dilated) below.
    from .fnt_mask_store import (masks_sibling_path as _masks_path,
                                 read_all_pred_masks)
    try:
        old_crops = read_all_pred_masks(_masks_path(wav_path))
    except Exception:
        old_crops = {}

    # Blank each reviewed call from the probability map. Prefer the exact stored
    # mask dilated by a few px — the dilation swallows the sub-threshold
    # probability halo just outside the old mask (so the call can't re-form,
    # even at a modestly lower threshold) while staying tight to the call's
    # actual shape, so a genuinely different call nearby is NOT over-suppressed.
    # Fall back to the second/Hz bounding box only when no crop is stored.
    _SUPPRESS_DILATE_PX = 3
    if reviewed_rows:
        from scipy import ndimage
    Fb, Tb = prob.shape
    for r in reviewed_rows:
        c = old_crops.get(str(r.get('blob_id')))
        m = None if c is None else (np.asarray(c.get('mask')) > 0)
        if m is not None and m.any():
            pad = _SUPPRESS_DILATE_PX
            f_off, t_off = int(c['f_off']), int(c['t_off'])
            mh, mw = m.shape
            # Window on the full grid, padded so the dilation has room.
            wf0, wf1 = max(0, f_off - pad), min(Fb, f_off + mh + pad)
            wt0, wt1 = max(0, t_off - pad), min(Tb, t_off + mw + pad)
            if wf1 <= wf0 or wt1 <= wt0:
                continue
            win = np.zeros((wf1 - wf0, wt1 - wt0), dtype=bool)
            # Portion of the mask that lands inside the (clipped) window.
            sf0, sf1 = max(0, wf0 - f_off), min(mh, wf1 - f_off)
            st0, st1 = max(0, wt0 - t_off), min(mw, wt1 - t_off)
            if sf1 > sf0 and st1 > st0:
                win[(f_off + sf0) - wf0:(f_off + sf1) - wf0,
                    (t_off + st0) - wt0:(t_off + st1) - wt0] = m[sf0:sf1, st0:st1]
                win = ndimage.binary_dilation(win, iterations=pad)
                prob[wf0:wf1, wt0:wt1][win] = 0.0
        else:  # no stored crop — fall back to the CSV's second/Hz box
            f0 = int(r.get('min_freq_hz', 0.0) / df) if df else 0
            f1 = int(round(r.get('max_freq_hz', 0.0) / df)) if df else Fb
            t0 = int(seconds_to_frames(
                r.get('start_s', 0.0), nperseg, noverlap, sr))
            t1 = int(round(seconds_to_frames(
                r.get('stop_s', 0.0), nperseg, noverlap, sr)))
            f0, f1 = max(0, min(Fb, f0)), max(0, min(Fb, f1))
            t0, t1 = max(0, min(Tb, t0)), max(0, min(Tb, t1))
            if f1 > f0 and t1 > t0:
                prob[f0:f1, t0:t1] = 0.0

    # Preserve confirmed labels: zero out the probability mask in any time
    # column that already contains a human-confirmed call for this file, so
    # predictions never overwrite confirmed annotations. Skipped when the user
    # asked to re-detect from scratch (preserve_labels False).
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
    if cfg.merge_consecutive:
        # dt (seconds/frame) → gap in frames; both derived above.
        max_gap_frames = int(round(cfg.merge_max_gap_s / dt)) if dt else 0
        blobs = merge_consecutive_blobs(
            blobs, max_gap_frames=max_gap_frames,
            require_freq_overlap=cfg.merge_require_freq_overlap)
    rows = blobs_to_rows(blobs, nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                         sr=sr, db_min=db_min, db_max=db_max, spec=spec)
    # Re-key the fresh predictions so their int blob_ids never collide with the
    # reviewed predictions we're keeping (whose ids came from an earlier run).
    kept_int_ids = [r['blob_id'] for r in reviewed_rows]
    id_offset = (max(kept_int_ids) + 1) if kept_int_ids else 0
    for i, r in enumerate(rows):
        r['blob_id'] = id_offset + i
    # Provenance: which model + settings produced these predictions.
    model_name = Path(cfg.model_path).stem if cfg.model_path else ''
    for r in rows:
        r['model_name'] = model_name
        r['threshold'] = cfg.threshold
        r['min_blob_pixels'] = cfg.min_blob_pixels

    if cfg.save_blob_csv:
        # The CSV is unified: hand-labels (string blob_ids) and reviewed
        # predictions (Accepted/Rejected) are carried over verbatim; only
        # pending predictions are regenerated.
        write_blob_csv(csv_path, handlabel_rows + reviewed_rows + rows)
    # Persist each blob's small cropped mask (NOT the multi-GB /prob grid):
    # blob_id matches the CSV row's blob_id. On file switch these few-MB crops
    # are read directly, so predictions redraw without decompressing the full
    # grid. MAD intentionally drops /prob — re-run inference to change the
    # threshold. Reviewed predictions keep their original crop so they still
    # draw as exact masks (write_pred_masks replaces the whole crop group).
    h5_path = None
    try:
        from .fnt_mask_store import (masks_sibling_path, write_pred_masks,
                                     set_grid_attrs, delete_prob)
        h5_path = masks_sibling_path(wav_path)
        set_grid_attrs(h5_path, sample_rate=sr, nperseg=nperseg,
                       noverlap=noverlap, nfft=nfft,
                       n_freq_bins=prob.shape[0], n_time_frames=prob.shape[1])
        crops = []
        for r in reviewed_rows:
            c = old_crops.get(str(r['blob_id']))
            if c is not None:
                crops.append({'blob_id': r['blob_id'], 'mask': c['mask'],
                              'f_off': c['f_off'], 't_off': c['t_off']})
        for i, b in enumerate(blobs):
            crops.append({'blob_id': id_offset + i, 'mask': b['mask'],
                          'f_off': b['f_low'], 't_off': b['t_start']})
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
