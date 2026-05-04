"""MAD mask label utilities.

Pixel-mask value conventions used throughout MAD:

    0  unlabeled  — outside a committed band
    1  positive   — user-painted target pixel
    2  negative   — certified non-target (inside a committed band but not painted)

On-disk sibling PNGs store only **0 and 1** — the raw paint. Certified
negatives are derived at training / inference / display time by scanning
for "committed columns": any time frame that contains a painted pixel is
considered fully reviewed along the frequency axis. Contiguous runs of
such time frames form committed bands; unpainted pixels inside those
bands become certified negatives.

This sparse supervision trick is the core MAD idea: the user only paints
positives and the tool infers where they have implicitly labeled the
absence of calls.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    from PIL import Image as PILImage
    _HAS_PIL = True
    # MAD mask PNGs can reach >500M pixels for long high-sample-rate
    # recordings (e.g. 600s @ 250kHz → 513 × ~1.17M frames). We trust
    # our own sibling files, so disable Pillow's DecompressionBomb guard.
    PILImage.MAX_IMAGE_PIXELS = None
except Exception:
    _HAS_PIL = False


LABEL_SUFFIX = "_FNT_MAD_labels.png"
PRED_SUFFIX = "_FNT_MAD_predictions.csv"
PRED_MASK_SUFFIX = "_FNT_MAD_predictions.png"


def mask_sibling_path(wav_path: str) -> str:
    """Return the sibling PNG path that stores paint labels for ``wav_path``."""
    stem = Path(wav_path).stem
    return str(Path(wav_path).with_name(stem + LABEL_SUFFIX))


def pred_csv_sibling_path(wav_path: str) -> str:
    stem = Path(wav_path).stem
    return str(Path(wav_path).with_name(stem + PRED_SUFFIX))


def pred_mask_sibling_path(wav_path: str) -> str:
    stem = Path(wav_path).stem
    return str(Path(wav_path).with_name(stem + PRED_MASK_SUFFIX))


def save_mask_png(path: str, mask: np.ndarray) -> None:
    if not _HAS_PIL:
        raise RuntimeError("Pillow is required to save MAD label PNGs.")
    arr = np.clip(mask.astype(np.int16), 0, 255).astype(np.uint8)
    PILImage.fromarray(arr, mode='L').save(path)


def load_mask_png(path: str) -> np.ndarray:
    if not _HAS_PIL:
        raise RuntimeError("Pillow is required to load MAD label PNGs.")
    img = PILImage.open(path).convert('L')
    return np.array(img, dtype=np.uint8)


def committed_columns(mask: np.ndarray) -> np.ndarray:
    """Return a 1-D bool array marking time frames that are inside a
    committed band.

    A column is committed iff at least one positive (value 1) pixel lives
    in it. Contiguous True runs form committed bands. Overlapping blob
    x-ranges are trivially merged by the projection: this is equivalent
    to "find min/max time of each connected blob, merge overlapping
    x-intervals" since every painted pixel contributes.
    """
    return (mask == 1).any(axis=0)


def committed_band_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return sorted ``[(t_start, t_end_exclusive), …]`` runs over time."""
    col_any = committed_columns(mask)
    runs: List[Tuple[int, int]] = []
    n = col_any.shape[0]
    i = 0
    while i < n:
        if not col_any[i]:
            i += 1
            continue
        j = i
        while j < n and col_any[j]:
            j += 1
        runs.append((i, j))
        i = j
    return runs


def apply_committed_negatives(mask: np.ndarray) -> np.ndarray:
    """Return a new mask with ``0 → 2`` inside committed columns.

    Painted positives (1) stay 1. Unpainted pixels outside committed
    columns stay 0. Unpainted pixels inside committed columns become 2
    (certified negative).
    """
    out = mask.copy()
    col = committed_columns(mask)  # shape (T,)
    if not col.any():
        return out
    # Broadcast: mark all unpainted pixels in committed columns as negative.
    in_band = np.broadcast_to(col[np.newaxis, :], out.shape)
    out = np.where((out == 0) & in_band, np.uint8(2), out)
    return out


def supervision_weight(mask: np.ndarray) -> np.ndarray:
    """Return a bool weight mask: True where the label has supervision.

    True for positives AND for pixels inside committed columns. False
    elsewhere. Used to zero out the loss outside labeled regions.
    """
    col = committed_columns(mask)
    return np.broadcast_to(col[np.newaxis, :], mask.shape).copy() | (mask == 1)


def positive_target(mask: np.ndarray) -> np.ndarray:
    """Return a float32 binary target: 1.0 where painted, 0.0 elsewhere."""
    return (mask == 1).astype(np.float32)
