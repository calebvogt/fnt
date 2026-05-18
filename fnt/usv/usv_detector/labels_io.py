"""
Sibling-CSV label I/O for Deep Audio Detector.

DAD labels live next to each audio file as `<wav_stem>_FNT_DAD_detections.csv`.
This keeps labels portable and shareable across projects — a DAD project
references source folders; it does not own labels.

Schema (one row per box):
    call_number, start_seconds, stop_seconds, duration_ms,
    min_freq_hz, max_freq_hz, peak_freq_hz, freq_bandwidth_hz,
    max_power_db, mean_power_db, confidence, status, source

status: 'accepted' | 'rejected' | 'pending' | 'negative' | 'user_drawn'
source: 'user' | 'ml' | 'cad_import' | 'manual'
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd


DAD_SUFFIX = "_FNT_DAD_detections"  # current canonical suffix
LEGACY_SUFFIXES = (             # recognized but not written
    "_dad",
    "_yoloDetection",
    "_usv_yolo",
)

# Columns we try to preserve on disk; extras pass through.
STD_COLUMNS = [
    "call_number", "start_seconds", "stop_seconds", "duration_ms",
    "min_freq_hz", "max_freq_hz", "peak_freq_hz", "freq_bandwidth_hz",
    "max_power_db", "mean_power_db", "confidence", "status", "source",
]

USER_STATUSES = {"accepted", "user_drawn", "negative"}
USER_SOURCES = {"user", "manual"}


def sibling_csv_path(wav_path) -> Path:
    """Return the canonical sibling-CSV path for a wav (written form)."""
    p = Path(wav_path)
    return p.parent / f"{p.stem}{DAD_SUFFIX}.csv"


def find_existing_sibling_csv(wav_path) -> Optional[Path]:
    """Find any existing sibling label CSV (canonical or legacy).

    Returns the canonical `_FNT_DAD_detections.csv` path if present, else the
    first legacy match, else None.
    """
    p = Path(wav_path)
    parent = p.parent
    stem = p.stem

    canonical = parent / f"{stem}{DAD_SUFFIX}.csv"
    if canonical.exists():
        return canonical
    for suffix in LEGACY_SUFFIXES:
        legacy = parent / f"{stem}{suffix}.csv"
        if legacy.exists():
            return legacy
    return None


def load_labels(wav_path) -> pd.DataFrame:
    """Load labels for a wav from its sibling CSV.

    Returns an empty DataFrame with the standard columns if no CSV exists.
    """
    existing = find_existing_sibling_csv(wav_path)
    if existing is None:
        return pd.DataFrame(columns=STD_COLUMNS)

    try:
        df = pd.read_csv(existing)
    except Exception:
        return pd.DataFrame(columns=STD_COLUMNS)

    # Ensure required columns exist with sensible defaults so downstream code
    # can count on them.
    if "status" not in df.columns:
        df["status"] = "pending"
    if "source" not in df.columns:
        df["source"] = "unknown"
    return df


def save_labels(wav_path, df: pd.DataFrame) -> Optional[Path]:
    """Write labels to the canonical sibling CSV. Returns the path written.

    If ``df`` is empty or None, any existing canonical CSV is removed so file
    browsers stay in sync with reality. Legacy CSVs (``_yoloDetection.csv``)
    are left alone.
    """
    canonical = sibling_csv_path(wav_path)
    if df is None or len(df) == 0:
        if canonical.exists():
            try:
                canonical.unlink()
            except OSError:
                pass
        return None

    df = df.copy()
    # Renumber call_number so CSVs are self-consistent.
    df["call_number"] = range(1, len(df) + 1)

    # Reorder: known cols first, extras after.
    cols = [c for c in STD_COLUMNS if c in df.columns]
    extras = [c for c in df.columns if c not in cols]
    df = df[cols + extras]

    # Atomic write
    tmp = canonical.with_suffix(canonical.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(canonical)
    return canonical


def _overlaps(a, b, t_tol: float = 0.005, f_tol: float = 50.0) -> bool:
    """Heuristic duplicate test: boxes within 5ms / 50Hz on all edges."""
    return (
        abs(float(a["start_seconds"]) - float(b["start_seconds"])) <= t_tol
        and abs(float(a["stop_seconds"]) - float(b["stop_seconds"])) <= t_tol
        and abs(float(a.get("min_freq_hz", 0)) - float(b.get("min_freq_hz", 0))) <= f_tol
        and abs(float(a.get("max_freq_hz", 0)) - float(b.get("max_freq_hz", 0))) <= f_tol
    )


def merge_with_inference(
    existing: pd.DataFrame,
    inference: List[dict],
) -> pd.DataFrame:
    """Merge fresh inference results into an existing label set.

    Rules:
      - All user-sourced rows (status in USER_STATUSES or source in USER_SOURCES)
        are preserved verbatim. Inference never deletes user work.
      - Previous inference rows (status='pending' / source='ml') are replaced
        wholesale by the new inference pass.
      - Dedupe: inference rows that overlap a preserved user row are dropped.
    """
    if existing is None or len(existing) == 0:
        keep = pd.DataFrame(columns=STD_COLUMNS)
    else:
        status_mask = existing["status"].isin(USER_STATUSES)
        source_col = existing["source"] if "source" in existing.columns else ""
        source_mask = existing.get("source", pd.Series([""] * len(existing))).isin(USER_SOURCES)
        keep = existing[status_mask | source_mask].copy()

    new_rows = []
    for det in inference:
        drop = False
        for _, kept in keep.iterrows():
            if _overlaps(det, kept):
                drop = True
                break
        if not drop:
            new_rows.append(det)

    new_df = pd.DataFrame(new_rows) if new_rows else pd.DataFrame(columns=STD_COLUMNS)

    combined = pd.concat([keep, new_df], ignore_index=True, sort=False)
    if len(combined) == 0:
        return pd.DataFrame(columns=STD_COLUMNS)

    combined = combined.sort_values("start_seconds").reset_index(drop=True)
    combined["call_number"] = range(1, len(combined) + 1)
    return combined


def scan_folders_for_wavs(folders) -> List[Path]:
    """Recursively collect *.wav files across one or more folders.

    Results are sorted by path for deterministic ordering. Hidden files
    (dotfiles) and hidden subdirectories are skipped.
    """
    seen = set()
    wavs: List[Path] = []
    for folder in folders:
        root = Path(folder)
        if not root.exists() or not root.is_dir():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() != ".wav":
                continue
            if any(part.startswith(".") for part in p.parts):
                continue
            if p in seen:
                continue
            seen.add(p)
            wavs.append(p)
    wavs.sort()
    return wavs
