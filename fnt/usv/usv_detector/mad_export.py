"""Folder-wide tables of what MAD found: recordings, detections and labelled calls.

MAD keeps everything about a recording in the ``.mad`` beside its wav, and File >
Export Annotations to CSV turns each one into a CSV beside it. An analysis of a
whole run -- tens of thousands of recordings -- wants a few tables instead, one
row per recording or per call, read in one go by R or pandas. :func:`export`
writes them into one output folder:

    mad_recordings.csv        one row per recording whose last inference was the
                              chosen model (by default the most common one)
    mad_detections.csv        one row per detection of those recordings: score,
                              review status, harmonic number, the contour
                              measures of its mask and its brightness ridge;
                              fundamentals only unless harmonics are asked for
    mad_labelled_calls.csv    one row per labelled (accepted) call, whatever
                              model found it, measured with MAD's call metrics
                              (recomputed from the audio, as Export Annotations
                              does), the contour measures and the ridge
    mad_call_images/<id>.png  one per labelled call: the call and 10 ms either
                              side, every frequency bin (high at the top), grey =
                              level as MAD stores it, alpha = the call's own mask
    mad_export.json           how and when each table was written

The columns are generic: where a recording is, what its file name says (an
Avisoft name gives its start time and channel) and what its .mad holds.
Anything particular to an experiment -- trial, treatment, which animals were
present -- is a join in the analysis code, not a column here.

Built for runs of ~40,000 recordings on a network share:

* Rows stream to a temporary file that replaces the table only once it is
  complete, so a crash or a stop leaves the previous table whole.
* Exporting again into the same folder, with the same settings, copies the rows
  of every recording whose .mad is unchanged (same modification time) from the
  previous tables, so only new and changed recordings are read. A recording
  whose rows in the previous table do not add up to its count is read again.
* A .mad written in the last few minutes (MAD may still be writing it), or one
  that cannot be opened, is left out until the next export.

The GUI (File > Export...) and ``mad export`` both call :func:`export`.
"""
from __future__ import annotations

import csv
import dataclasses
import io
import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import fnt_mask_store as ms
from .mad_inference import CALL_METRIC_KEYS, frames_to_seconds
from .mad_ridge import RIDGE_ATTRS, grid_units
from .mad_sampling import _AVISOFT_NAME, parse_channel

__all__ = [
    "ExportOptions", "ExportStopped", "Store", "export", "find_stores",
    "scan_models", "previous_scan", "model_counts", "resolve_model",
    "contour_measures", "call_image",
    "name_fields", "RECORDINGS_CSV", "DETECTIONS_CSV", "LABELS_CSV",
    "IMAGES_DIR", "SETTINGS_JSON",
]

RECORDINGS_CSV = "mad_recordings.csv"
DETECTIONS_CSV = "mad_detections.csv"
LABELS_CSV = "mad_labelled_calls.csv"
IMAGES_DIR = "mad_call_images"
SETTINGS_JSON = "mad_export.json"

#: Bumped whenever a column's meaning or a measure changes, so the next export
#: reads every recording again rather than copying rows computed the old way.
EXPORT_VERSION = 1

#: A .mad written more recently than this (seconds) may still be open in MAD.
SETTLE_S = 600.0

#: Context either side of a call in its image.
IMAGE_MARGIN_S = 0.01

#: Contour measures of a call's own mask (see :func:`contour_measures`).
CONTOUR_COLS = ["duration_ms", "f_mean_khz", "f_min_khz", "f_max_khz",
                "f_start_khz", "f_end_khz", "slope_khz_ms", "contour_sd_khz",
                "turns", "max_jump_khz", "thickness_khz", "area_px"]
RIDGE_COLS = list(RIDGE_ATTRS)

REC_COLS = ["path", "folder", "recording", "channel", "start", "duration_s",
            "model", "inferred_at", "threshold", "detections", "calls",
            "mad_mtime"]
# folder + recording lead, as the key that joins a detection to its recording
# and as the key the next export finds each recording's block of rows by.
DET_COLS = (["folder", "recording", "blob_id", "start_s", "score", "status",
             "harmonic_n", "harmonic_call_id"] + CONTOUR_COLS + RIDGE_COLS)
IMAGE_COLS = ["image", "img_col0", "img_col1", "f_bin0", "f_bin1"]
LABEL_COLS = (["path", "folder", "recording", "channel", "rec_start", "id",
               "blob_id", "painted", "class", "start_s", "stop_s",
               "min_freq_hz", "max_freq_hz", "harmonic_n", "harmonic_call_id",
               "f0_hz", "score", "model_name", "note", "tags"]
              + list(CALL_METRIC_KEYS) + CONTOUR_COLS + RIDGE_COLS
              + IMAGE_COLS + ["mad_mtime"])

# As cli._walk_wavs: project internals hold no recordings of their own.
_SKIP_DIRS = {"models", "datasets", "batch_runs", ".scratch", "legacy_pre_h5"}
# Newest naming first: "_FNT.mad" also ends with ".mad", so each file takes the
# first suffix it ends with (as mad_csv_rebuild.rebuild_folder does).
_SUFFIXES = tuple(s.lower() for s in (ms.MAD_SUFFIX,) + tuple(ms.LEGACY_MAD_SUFFIXES))
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


class ExportStopped(Exception):
    """Raised inside :func:`export` when ``should_stop`` says so; the tables
    being written are discarded and the previous ones stay."""


@dataclass
class ExportOptions:
    """What :func:`export` writes, and from where."""
    out_dir: str
    folders: Sequence[str] = ()          # walked to the bottom for .mad files
    wavs: Sequence[str] = ()             # recordings named one by one (a project's)
    detections: bool = True              # mad_recordings.csv + mad_detections.csv
    labels: bool = False                 # mad_labelled_calls.csv
    images: bool = False                 # mad_call_images/ (labelled calls)
    model: Optional[str] = None          # None/'': the most common
    harmonics: bool = False              # include harmonic_n > 1
    incremental: bool = True             # copy unchanged recordings' rows
    settle_s: float = SETTLE_S


@dataclass(frozen=True)
class Store:
    """One recording's .mad. ``wav`` is where the recording is (or was: MAD
    references audio by path, and a missing wav does not stop its .mad being
    exported)."""
    wav: str
    mad: str
    mtime: float = field(compare=False)

    @property
    def folder(self) -> str:
        return os.path.basename(os.path.dirname(self.wav))

    @property
    def recording(self) -> str:
        return os.path.basename(self.wav)


# ----------------------------------------------------------------------
# Finding the .mad files
# ----------------------------------------------------------------------
def _stores_in(d: str, entries, stems=None) -> List[Store]:
    """The stores among one directory's entries, one per recording (newest
    naming), restricted to the lower-case wav ``stems`` when given."""
    best: Dict[str, Tuple[int, str, object]] = {}
    wav_names: Dict[str, str] = {}
    for e in entries:
        low = e.name.lower()
        if low.endswith(".wav"):
            wav_names[low[:-4]] = e.name
            continue
        for rank, s in enumerate(_SUFFIXES):
            if low.endswith(s):
                stem = e.name[:-len(s)]
                cur = best.get(stem.lower())
                if cur is None or rank < cur[0]:
                    best[stem.lower()] = (rank, stem, e)
                break
    out = []
    for key, (_rank, stem, e) in best.items():
        if stems is not None and key not in stems:
            continue
        try:
            if not e.is_file():
                continue
            mtime = e.stat().st_mtime
        except OSError:
            continue
        wav = os.path.join(d, (stems or {}).get(key) or wav_names.get(key)
                           or stem + ".wav")
        out.append(Store(wav=wav, mad=e.path, mtime=mtime))
    return out


def find_stores(folders: Sequence[str] = (), wavs: Sequence[str] = (),
                should_stop: Optional[Callable[[], bool]] = None) -> List[Store]:
    """Every recording's .mad under ``folders`` (walked to the bottom) and
    beside each of ``wavs``, one per recording, sorted by recording path.

    One directory listing per folder, never a stat per file: on Windows a
    listing carries each file's modification time, and a stat per recording
    over a network share is the slow part of finding 40,000 of them."""
    found: Dict[str, Store] = {}

    def add(stores):
        for s in stores:
            found.setdefault(_key(s.wav), s)

    stack = [os.path.abspath(f) for f in folders]
    while stack:
        if should_stop and should_stop():
            raise ExportStopped()
        d = stack.pop()
        try:
            entries = list(os.scandir(d))
        except OSError:
            continue                        # unreachable share, ACLs
        for e in entries:
            try:
                if (e.is_dir(follow_symlinks=False) and not e.name.startswith(".")
                        and e.name.lower() not in _SKIP_DIRS):
                    stack.append(e.path)
            except OSError:
                continue
        add(_stores_in(d, entries))

    by_dir: Dict[str, Dict[str, str]] = {}
    for w in wavs:
        d, name = os.path.split(os.path.abspath(w))
        by_dir.setdefault(d, {})[os.path.splitext(name)[0].lower()] = name
    for d, stems in by_dir.items():
        try:
            entries = list(os.scandir(d))
        except OSError:
            continue                        # a moved recording is soft
        add(_stores_in(d, entries, stems))
    return sorted(found.values(), key=lambda s: os.path.normcase(s.wav))


def _same_mtime(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-3
    except (TypeError, ValueError):
        return False


# ----------------------------------------------------------------------
# Which model each recording was last analysed with
# ----------------------------------------------------------------------
def scan_models(stores: Sequence[Store], known: Optional[Dict[str, Dict]] = None,
                progress: Optional[Callable] = None,
                should_stop: Optional[Callable[[], bool]] = None) -> Dict[str, Dict]:
    """``{mad path: {mtime, model}}`` for every store: the model of its last
    inference ('' if it has had none; None if it could not be opened).

    Opens only files not in ``known`` (an earlier scan, or the previous
    recordings table) at the same modification time, and reads only root
    attributes. ``progress(stage, i, n, name)`` with stage ``'scan'``. Stops
    early, with what it has, when ``should_stop`` says so."""
    out: Dict[str, Dict] = {}
    known = known or {}
    n = len(stores)
    for i, s in enumerate(stores):
        if should_stop and should_stop():
            break
        k = known.get(s.mad)
        if (k is not None and k.get("model") is not None
                and _same_mtime(k.get("mtime"), s.mtime)):
            out[s.mad] = k
            continue
        if progress:
            progress("scan", i, n, s.recording)
        try:
            run = ms.read_store_summary(s.mad)["run"]
            out[s.mad] = {"mtime": s.mtime,
                          "model": str(run.get("last_infer_model") or "")}
        except Exception:
            out[s.mad] = {"mtime": s.mtime, "model": None}
    return out


def previous_scan(out_dir: str, stores: Sequence[Store]) -> Dict[str, Dict]:
    """The model of each store as the recordings table already in ``out_dir``
    recorded it, keyed and shaped as :func:`scan_models`' ``known`` (it only
    counts where the .mad's modification time still matches)."""
    rows = _read_table(os.path.join(out_dir, RECORDINGS_CSV), REC_COLS) or ()
    by_wav = {_key(s.wav): s for s in stores}
    out = {}
    for r in rows:
        s = by_wav.get(_key(r["path"]))
        if s is not None:
            out[s.mad] = {"mtime": r["mad_mtime"], "model": r["model"]}
    return out


def _key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def model_counts(scan: Dict[str, Dict]) -> Counter:
    """Recordings per model, most common first (recordings never analysed and
    files that could not be opened are not counted)."""
    return Counter(v["model"] for v in scan.values() if v.get("model"))


def resolve_model(name: Optional[str], counts: Counter) -> str:
    """The model to export: ``name`` exactly, else the one model whose name
    starts with it (``20261014_181834`` for ``20261014_181834_unet_n=3366``),
    else -- with no name -- the most common. Raises ValueError otherwise."""
    if not counts:
        raise ValueError("No .mad file in the input has been through inference.")
    if not name:
        return counts.most_common(1)[0][0]
    if name in counts:
        return name
    hits = [m for m in counts if m.startswith(name)]
    if len(hits) == 1:
        return hits[0]
    found = ", ".join(f"{m} ({n})" for m, n in counts.most_common())
    raise ValueError(f"No .mad file's last inference was model '{name}'. "
                     f"Models found: {found}")


# ----------------------------------------------------------------------
# Measures
# ----------------------------------------------------------------------
def name_fields(recording: str) -> Tuple[Optional[int], str]:
    """``(channel, start)`` from a recording's file name.

    An Avisoft name (``ch<N><yyyy-mm-dd>_<HH-MM-SS>_<n>.wav``) gives both, the
    start as 'yyyy-mm-dd HH:MM:SS' (the recorder's clock); any other name
    carrying a channel token (``_ch3_``) gives the channel. Otherwise None and
    ''. The Avisoft pattern is tried first because its channel digit runs into
    the date (``ch12025-...``)."""
    m = _AVISOFT_NAME.match(os.path.basename(recording))
    if m:
        try:
            start = datetime.strptime(f"{m.group(2)} {m.group(3)}",
                                      "%Y-%m-%d %H-%M-%S").isoformat(sep=" ")
        except ValueError:
            start = ""
        return int(m.group(1)), start
    ch = parse_channel(recording)
    return (int(ch[2:]) if ch else None), ""


def contour_measures(mask, f_off: int, df_khz: float, dt_ms: float) -> Optional[Dict]:
    """Contour measures of one call from its own mask alone (no audio needed).

    ``mask`` is frequency x time, as MAD stores it, and its first row is
    frequency bin ``f_off``. The contour is the mask's mean frequency in each
    time frame that holds any of it; ``turns`` counts reversals of the contour
    by more than one frequency bin, ``max_jump_khz`` its largest frame-to-frame
    step, ``thickness_khz`` the mean height of the mask per frame. ``duration_ms``
    is the mask's width. The measures of 2021_8x8_Mus R/01_build_usv.R, as the
    2025_VoleCosm analysis used them. None for an empty mask."""
    m = np.asarray(mask, bool).T                    # time x frequency
    if not m.any():
        return None
    px = m.sum(1)
    on = px > 0
    cont = ((m @ np.arange(m.shape[1]))[on] / px[on] + f_off) * df_khz
    d = np.diff(cont)
    steps = np.sign(d[np.abs(d) > df_khz])
    bins = np.flatnonzero(m.any(0))
    dur = m.shape[0] * dt_ms
    return {"duration_ms": dur, "f_mean_khz": cont.mean(),
            "f_min_khz": (bins[0] + f_off) * df_khz,
            "f_max_khz": (bins[-1] + 1 + f_off) * df_khz,
            "f_start_khz": cont[0], "f_end_khz": cont[-1],
            "slope_khz_ms": (cont[-1] - cont[0]) / dur,
            "contour_sd_khz": cont.std(ddof=1) if len(cont) > 1 else 0.0,
            "turns": int((np.diff(steps) != 0).sum()) if len(steps) > 1 else 0,
            "max_jump_khz": np.abs(d).max() if len(d) else 0.0,
            "thickness_khz": px[on].mean() * df_khz, "area_px": int(m.sum())}


def _rounded(d: Optional[Dict], digits: int = 4) -> Dict:
    return {k: (round(float(v), digits) if isinstance(v, (float, np.floating)) else v)
            for k, v in (d or {}).items()}


def _units(grid: Dict) -> Optional[Tuple[float, float]]:
    """(kHz per bin, ms per frame) of a grid dict, or None."""
    try:
        return grid_units(float(grid["sample_rate"]), int(grid["nperseg"]),
                          int(grid["noverlap"]), int(grid["nfft"]))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


def _is_fundamental(n) -> bool:
    """A call not grouped as a harmonic (no harmonic number, or 1)."""
    if n in (None, ""):
        return True
    try:
        return int(float(n)) <= 1
    except (TypeError, ValueError):
        return True


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "TRUE" if v else "FALSE"
    if isinstance(v, np.integer):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return repr(v) if math.isfinite(v) else ""
    # one line per row: the next export finds a recording's rows by line
    return str(v).replace("\r", " ").replace("\n", " ")


def _blob_sort(b) -> Tuple[int, str]:
    s = str(b)
    return (int(s), "") if s.lstrip("-").isdigit() else (0, s)


# ----------------------------------------------------------------------
# Recordings and detections
# ----------------------------------------------------------------------
def _detection_rows(store: Store, summary: Dict, harmonics: bool
                    ) -> Tuple[Dict, List[Dict]]:
    """(recordings row, detection rows) of one recording from its
    :func:`fnt_mask_store.read_store_summary` (with preds and example_meta)."""
    grid, run = summary["grid"], summary["run"]
    units = _units(grid)
    try:
        timing = (int(grid["nperseg"]), int(grid["noverlap"]), int(grid["sample_rate"]))
    except (KeyError, TypeError, ValueError):
        timing = None
    # Review status from the examples, as the per-recording CSV derives it: an
    # accepted label outranks a rejection of the same detection (the later
    # decision), and a label's harmonic grouping is the current one.
    decided: Dict[str, Tuple[str, Dict]] = {}
    for meta in summary.get("example_meta") or ():
        bid = meta.get("blob_id")
        if bid is None:
            continue
        status = ("rejected" if ms.example_kind(meta) in ms.REJECTED_KINDS
                  else "accepted")
        prev = decided.get(str(bid))
        if prev is None or (prev[0] == "rejected" and status == "accepted"):
            decided[str(bid)] = (status, meta)
    rows, n_det, n_calls = [], 0, 0
    for key, crop in (summary.get("preds") or {}).items():
        n_det += 1
        status, meta = decided.get(str(key), ("pending", {}))
        src = meta if status == "accepted" and meta.get("harmonic_n") not in (None, "") else crop
        hn, hc = src.get("harmonic_n"), src.get("harmonic_call_id")
        if _is_fundamental(hn):
            n_calls += 1
        elif not harmonics:
            continue
        t_off = int(crop.get("t_off", 0))
        row = {"folder": store.folder, "recording": store.recording,
               "blob_id": int(key) if str(key).isdigit() else key,
               "start_s": (round(float(frames_to_seconds(t_off, *timing)), 6)
                           if timing else None),
               "score": (round(float(crop["score"]), 4)
                         if isinstance(crop.get("score"), (int, float)) else None),
               "status": status, "harmonic_n": hn, "harmonic_call_id": hc,
               "_t": t_off}
        if units is not None:
            row.update(_rounded(contour_measures(crop["mask"], int(crop.get("f_off", 0)),
                                                 *units)))
        for k in RIDGE_COLS:
            v = crop.get(k)
            row[k] = round(v, 4) if isinstance(v, float) else v
        rows.append(row)
    rows.sort(key=lambda r: (r["_t"], _blob_sort(r["blob_id"])))
    channel, start = name_fields(store.recording)
    try:
        dur = round(int(grid["n_time_frames"]) * (int(grid["nperseg"]) - int(grid["noverlap"]))
                    / float(grid["sample_rate"]), 3)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        dur = None
    thr = run.get("last_infer_threshold")
    rec = {"path": store.wav, "folder": store.folder, "recording": store.recording,
           "channel": channel, "start": start, "duration_s": dur,
           "model": run.get("last_infer_model", ""),
           "inferred_at": run.get("last_infer_at", ""),
           "threshold": float(thr) if isinstance(thr, (int, float)) else thr,
           "detections": n_det, "calls": n_calls, "mad_mtime": store.mtime}
    return rec, rows


def _row_prefix(folder: str, recording: str) -> bytes:
    """How a detections row of this recording starts, as csv writes it."""
    buf = io.StringIO()
    csv.writer(buf).writerow([_cell(folder), _cell(recording), ""])
    return buf.getvalue().rstrip("\r\n").encode("utf-8")


def _index_blocks(path: str) -> Optional[Dict[Tuple[str, str], Tuple[int, int]]]:
    """``{(folder, recording): (byte offset of its first row, row count)}`` of
    a previous detections table with this module's columns, or None.

    Read once, line by line, so the previous table never has to be held in
    memory (a full run's is millions of rows). Rows are written grouped by
    recording; a recording whose rows are not together is left out, which
    makes it be read again."""
    out: Dict[Tuple[str, str], Tuple[int, int]] = {}
    bad = set()
    try:
        with open(path, "rb") as f:
            header = f.readline()
            if next(csv.reader([header.decode("utf-8")]), None) != DET_COLS:
                return None
            pos = len(header)
            key, start, n, prefix = None, 0, 0, None

            def close():
                if key is None:
                    return
                if key in out:
                    bad.add(key)
                else:
                    out[key] = (start, n)

            for line in iter(f.readline, b""):
                if prefix is not None and line.startswith(prefix):
                    n += 1
                else:
                    row = next(csv.reader([line.decode("utf-8")]), None)
                    if row and len(row) >= 2:
                        close()
                        key, start, n = (row[0], row[1]), pos, 1
                        prefix = _row_prefix(row[0], row[1])
                pos += len(line)
            close()
    except (OSError, UnicodeDecodeError, csv.Error):
        return None
    for k in bad:
        out.pop(k, None)
    return out


def _read_table(path: str, cols: List[str]) -> Optional[List[Dict]]:
    """The rows of a previous (small) table with exactly ``cols``, or None."""
    try:
        with open(path, newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            if next(r, None) != cols:
                return None
            return [dict(zip(cols, row)) for row in r if len(row) == len(cols)]
    except (OSError, UnicodeDecodeError, csv.Error):
        return None


def _discard(*paths) -> None:
    for p in paths:
        try:
            if p and os.path.isfile(p):
                os.remove(p)
        except OSError:
            pass


def _export_detections(stores: List[Store], opts: ExportOptions, known: Dict,
                       prev_settings: Dict, rep: Dict, progress, log, stop) -> None:
    out_rec = os.path.join(opts.out_dir, RECORDINGS_CSV)
    out_det = os.path.join(opts.out_dir, DETECTIONS_CSV)

    # Which model each unchanged recording had is known from the previous
    # table; only the rest are opened to find out.
    for mad, k in previous_scan(opts.out_dir, stores).items():
        known.setdefault(mad, k)
    # The previous tables' rows are copied only when written by this version
    # with the same settings bar the model (checked per recording, below).
    old: Dict[str, Dict] = {}
    blocks: Dict[Tuple[str, str], Tuple[int, int]] = {}
    ps = prev_settings.get("detections") or {}
    if (opts.incremental and prev_settings.get("export_version") == EXPORT_VERSION
            and ps.get("harmonics") == opts.harmonics):
        rows = _read_table(out_rec, REC_COLS)
        idx = _index_blocks(out_det) if rows is not None else None
        if rows is not None and idx is not None:
            old = {_key(r["path"]): r for r in rows}
            blocks = idx

    log(f"Reading which model each of {len(stores):,} recording(s) was last "
        "analysed with…")
    scan = scan_models(stores, known, progress, stop)
    if stop():
        raise ExportStopped()
    counts = model_counts(scan)
    rep["models"] = dict(counts)
    rep["unreadable"] += sum(1 for v in scan.values() if v.get("model") is None)
    model = resolve_model(opts.model, counts)
    rep["model"] = model
    chosen = [s for s in stores if scan.get(s.mad, {}).get("model") == model]
    others = sum(n for m, n in counts.items() if m != model)
    log(f"Model {model}: {len(chosen):,} recording(s)"
        + (f" ({others:,} with another model left out)" if others else ""))

    # A detection names its recording by folder and file name; two recordings
    # sharing both cannot be told apart in the table, nor found again in it.
    keys = Counter((s.folder, s.recording) for s in chosen)
    dup = {k for k, n in keys.items() if n > 1}
    if dup:
        log(f"Warning: {sum(keys[k] for k in dup)} recording(s) share a folder "
            "name and file name with another; their detections cannot be told "
            "apart by folder and recording.")
    count_col = "detections" if opts.harmonics else "calls"

    def reusable(s: Store) -> bool:
        r = old.get(_key(s.wav))
        if r is None or r["model"] != model or not _same_mtime(r["mad_mtime"], s.mtime):
            return False
        key = (s.folder, s.recording)
        if key in dup:
            return False
        try:
            want = int(r[count_col])
        except ValueError:
            return False
        return blocks.get(key, (0, 0))[1] == want

    reuse = {s.mad for s in chosen if reusable(s)}
    log(f"{len(reuse):,} recording(s) unchanged since the last export, "
        f"{len(chosen) - len(reuse):,} to read")

    tmp_det, tmp_rec = out_det + ".tmp", out_rec + ".tmp"
    rec_rows: List[Dict] = []
    n_rows = 0
    try:
        with open(tmp_det, "w", newline="", encoding="utf-8") as fd:
            w = csv.writer(fd)
            w.writerow(DET_COLS)
            fo = open(out_det, "rb") if reuse else None
            try:
                for i, s in enumerate(chosen):
                    if stop():
                        raise ExportStopped()
                    progress("detections", i, len(chosen), s.recording)
                    if s.mad in reuse:
                        r = old[_key(s.wav)]
                        start, n = blocks.get((s.folder, s.recording), (0, 0))
                        if n:
                            fo.seek(start)
                            for _ in range(n):
                                fd.write(fo.readline().decode("utf-8"))
                        n_rows += n
                        rec_rows.append(r)
                        rep["reused"] += 1
                        continue
                    try:
                        summary = ms.read_store_summary(s.mad, preds=True,
                                                        example_meta=True)
                    except Exception:
                        rep["unreadable"] += 1       # MAD writing it, or damaged
                        continue
                    if summary["run"].get("last_infer_model") != model:
                        continue                     # re-analysed since the scan
                    rec, dets = _detection_rows(s, summary, opts.harmonics)
                    for d in dets:
                        w.writerow([_cell(d.get(c)) for c in DET_COLS])
                    n_rows += len(dets)
                    rec_rows.append(rec)
                    rep["read"] += 1
            finally:
                if fo is not None:
                    fo.close()
        with open(tmp_rec, "w", newline="", encoding="utf-8") as fr:
            w = csv.writer(fr)
            w.writerow(REC_COLS)
            for r in rec_rows:
                w.writerow([_cell(r.get(c)) for c in REC_COLS])
        # Detections first: a crash between the two leaves recordings whose
        # counts no longer match their rows, and those are read again.
        os.replace(tmp_det, out_det)
        os.replace(tmp_rec, out_rec)
    except BaseException:
        _discard(tmp_det, tmp_rec)
        raise
    progress("detections", len(chosen), len(chosen), "")
    rep["tables"] += [RECORDINGS_CSV, DETECTIONS_CSV]
    rep["recordings"] = len(rec_rows)
    rep["detection_rows"] = n_rows
    log(f"Wrote {RECORDINGS_CSV} ({len(rec_rows):,} recording(s)) and "
        f"{DETECTIONS_CSV} ({n_rows:,} "
        f"{'detection' if opts.harmonics else 'call'}(s))")
    _save_settings(opts, "detections", {
        "model": model, "harmonics": opts.harmonics,
        "recordings": len(rec_rows), "rows": n_rows})


# ----------------------------------------------------------------------
# Labelled calls
# ----------------------------------------------------------------------
def call_image(spec_u8: np.ndarray, own: np.ndarray, f0: int, t0: int,
               meta: Dict, dt_s: float, margin_s: float = IMAGE_MARGIN_S
               ) -> Optional[Tuple[np.ndarray, Dict]]:
    """``(image, fields)`` of one labelled call.

    ``spec_u8`` is the example's stored patch (frequency x time, 0-255) and
    ``own`` the call's own mask with its first pixel at full-grid bin ``f0``,
    frame ``t0``. The image is the patch from ``margin_s`` before the call to
    ``margin_s`` after it, every frequency bin with the highest at the top:
    channel 0 the level, channel 1 (alpha) 255 on the call's own pixels.
    ``fields`` places the call in it, counting from 1 as R does: columns
    ``img_col0``..``img_col1`` and frequency bins ``f_bin0``..``f_bin1`` (bin 1
    is 0 Hz)."""
    n_f, n_t = spec_u8.shape
    h, w = own.shape
    lf0 = f0 - int(meta.get("patch_f_off") or 0)
    lt0 = t0 - int(meta.get("patch_t_off") or 0)
    if lf0 < 0 or lt0 < 0 or lf0 + h > n_f or lt0 + w > n_t or dt_s <= 0:
        return None
    mf = int(round(margin_s / dt_s))
    c0, c1 = max(0, lt0 - mf), min(n_t, lt0 + w + mf)
    alpha = np.zeros_like(spec_u8)
    alpha[lf0:lf0 + h, lt0:lt0 + w] = np.where(own, 255, 0)
    img = np.stack([spec_u8[::-1, c0:c1], alpha[::-1, c0:c1]], axis=-1)
    return np.ascontiguousarray(img), {
        "img_col0": lt0 - c0 + 1, "img_col1": lt0 + w - c0,
        "f_bin0": f0 + 1, "f_bin1": f0 + h}


def _label_rows(store: Store, opts: ExportOptions, img_dir: str,
                rep: Dict) -> List[Dict]:
    """One row per labelled call of one recording, its images written."""
    from .mad_csv_rebuild import _recompute_features
    from .mad_examples import _examples_to_annotations

    # One open answers it for the many recordings with no label at all.
    summary = ms.read_store_summary(store.mad, example_meta=True)
    if not any(ms.example_kind(m) == "label" for m in summary["example_meta"]):
        return []
    grid = summary["grid"]
    ridges = ms.read_example_ridges(store.mad)
    channel, rec_start = name_fields(store.recording)
    rows: List[Dict] = []
    calc: List[Dict] = []                   # what _recompute_features reads
    masks: Dict[str, Tuple[np.ndarray, int]] = {}
    for ex in ms.td_iter_file_examples(store.mad, store.recording,
                                       with_spec=opts.images, kinds=("label",)):
        meta = ex["meta"]
        if not opts.harmonics and not _is_fundamental(meta.get("harmonic_n")):
            continue
        # The call's own pixels on the full grid, as the per-recording export
        # measures them (a stored patch also holds its neighbours' context).
        anns = list(_examples_to_annotations([ex], meta.get("source_wav", ""),
                                             (10 ** 9, 10 ** 9), kinds=("label",)))
        if not anns:
            continue                        # nothing painted
        a = anns[0]
        eid = str(meta.get("id") or a.get("id") or "")
        bid = meta.get("blob_id")
        key = str(bid) if bid is not None else eid
        row = {"path": store.wav, "folder": store.folder,
               "recording": store.recording, "channel": channel,
               "rec_start": rec_start, "id": eid, "blob_id": bid,
               "painted": bid is None, "class": meta.get("class", ""),
               "start_s": meta.get("t_start_s"), "stop_s": meta.get("t_stop_s"),
               "min_freq_hz": meta.get("f_low_hz"),
               "max_freq_hz": meta.get("f_high_hz"),
               "mad_mtime": store.mtime}
        for k in ("harmonic_n", "harmonic_call_id", "f0_hz", "score",
                  "model_name", "note", "tags"):
            v = meta.get(k)
            row[k] = ",".join(map(str, v)) if isinstance(v, (list, tuple)) else v
        units = _units(meta) or _units(grid)
        if units is not None:
            row.update(_rounded(contour_measures(a["mask"], int(a["f0"]), *units)))
        for k, v in (ridges.get(eid) or {}).items():
            if k in RIDGE_COLS:
                row[k] = round(v, 4) if isinstance(v, float) else v
        if opts.images and ex.get("spec") is not None and units is not None:
            spec_u8 = np.round(np.asarray(ex["spec"]) * 255.0).astype(np.uint8)
            got = call_image(spec_u8, np.asarray(a["mask"], bool), int(a["f0"]),
                             int(a["t0"]), meta, units[1] / 1000.0)
            if got is not None:
                img, fields = got
                name = _UNSAFE.sub("_", eid or key) + ".png"
                _save_png(os.path.join(img_dir, name), img)
                row.update(fields, image=name)
        rows.append(row)
        calc.append({"blob_id": key, "start_s": row["start_s"],
                     "stop_s": row["stop_s"], "min_freq_hz": row["min_freq_hz"]})
        masks[key] = (a["mask"], int(a["f0"]))
    if rows:
        # MAD's call metrics, from the audio with today's code -- the same
        # function File > Export Annotations to CSV uses.
        if not os.path.isfile(store.wav):
            rep["labels_no_audio"] += 1
        elif _recompute_features(store.wav, calc, grid, masks):
            for row, c in zip(rows, calc):
                row.update({k: c[k] for k in CALL_METRIC_KEYS if k in c})
    rows.sort(key=lambda r: (float(r["start_s"] or 0.0), str(r["id"])))
    return rows


def _save_png(path: str, img: np.ndarray) -> None:
    from PIL import Image
    Image.fromarray(img).save(path)          # (h, w, 2) uint8: grey + alpha


def _export_labels(stores: List[Store], opts: ExportOptions,
                   prev_settings: Dict, rep: Dict, progress, log, stop) -> None:
    out_csv = os.path.join(opts.out_dir, LABELS_CSV)
    img_dir = os.path.join(opts.out_dir, IMAGES_DIR)
    if opts.images:
        os.makedirs(img_dir, exist_ok=True)
    old: Dict[str, List[Dict]] = {}
    ps = prev_settings.get("labelled_calls") or {}
    if (opts.incremental and prev_settings.get("export_version") == EXPORT_VERSION
            and ps.get("harmonics") == opts.harmonics
            and ps.get("images") == opts.images):
        for r in _read_table(out_csv, LABEL_COLS) or ():
            old.setdefault(_key(r["path"]), []).append(r)

    def reusable(s: Store) -> Optional[List[Dict]]:
        prev = old.get(_key(s.wav))
        if not prev or not all(_same_mtime(r["mad_mtime"], s.mtime) for r in prev):
            return None
        if opts.images and not all(
                r["image"] and os.path.isfile(os.path.join(img_dir, r["image"]))
                for r in prev):
            return None
        return prev

    log(f"Reading the labelled calls of {len(stores):,} recording(s)…")
    tmp = out_csv + ".tmp"
    images = set()
    n_rows = n_recs = 0
    try:
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(LABEL_COLS)
            for i, s in enumerate(stores):
                if stop():
                    raise ExportStopped()
                progress("labels", i, len(stores), s.recording)
                rows = reusable(s)
                if rows is not None:
                    rep["labels_reused"] += 1
                else:
                    try:
                        rows = _label_rows(s, opts, img_dir, rep)
                    except Exception:
                        rep["unreadable_labels"] += 1
                        continue
                for r in rows:
                    w.writerow([_cell(r.get(c)) for c in LABEL_COLS])
                    if r.get("image"):
                        images.add(r["image"])
                n_rows += len(rows)
                n_recs += bool(rows)
        os.replace(tmp, out_csv)
    except BaseException:
        _discard(tmp)
        raise
    progress("labels", len(stores), len(stores), "")
    stale = 0
    if opts.images:
        # Images of calls no longer labelled (rejected, deleted, re-drawn
        # under a new id) go; the folder holds this table's images only.
        for name in os.listdir(img_dir):
            if name.lower().endswith(".png") and name not in images:
                _discard(os.path.join(img_dir, name))
                stale += 1
    rep["tables"].append(LABELS_CSV)
    rep["label_recordings"] = n_recs
    rep["label_rows"] = n_rows
    log(f"Wrote {LABELS_CSV} ({n_rows:,} labelled call(s) from {n_recs:,} "
        "recording(s))" + (f" and {len(images):,} image(s) in {IMAGES_DIR}/"
                           if opts.images else "")
        + (f"; {stale:,} stale image(s) removed" if stale else ""))
    _save_settings(opts, "labelled_calls", {
        "harmonics": opts.harmonics, "images": opts.images,
        "recordings": n_recs, "rows": n_rows})


# ----------------------------------------------------------------------
# Settings record
# ----------------------------------------------------------------------
def _load_settings(out_dir: str) -> Dict:
    try:
        with open(os.path.join(out_dir, SETTINGS_JSON), encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_settings(opts: ExportOptions, table: str, settings: Dict) -> None:
    """Record how ``table`` was written (and that it was), for the next export
    and for whoever reads the tables later. Written after the table, so a
    table replaced under new settings by an export that died before this
    point is read in full next time rather than copied from."""
    d = _load_settings(opts.out_dir)
    if d.get("export_version") != EXPORT_VERSION:
        d = {k: v for k, v in d.items() if k not in ("detections", "labelled_calls")}
    d["export_version"] = EXPORT_VERSION
    d["fnt_version"] = ms._fnt_version()
    inputs = [os.path.abspath(f) for f in opts.folders]
    if opts.wavs:
        inputs.append(f"{len(opts.wavs)} recording(s) named one by one")
    d[table] = dict(settings, written=datetime.now().isoformat(timespec="seconds"),
                    inputs=inputs)
    path = os.path.join(opts.out_dir, SETTINGS_JSON)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    os.replace(path + ".tmp", path)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def export(opts: ExportOptions, progress: Optional[Callable] = None,
           log: Optional[Callable[[str], None]] = None,
           should_stop: Optional[Callable[[], bool]] = None,
           known: Optional[Dict[str, Dict]] = None) -> Dict:
    """Write the tables ``opts`` asks for into ``opts.out_dir``; returns a report.

    ``progress(stage, i, n, name)`` -- stage 'scan', 'detections' or 'labels'
    -- is called per recording; ``log(message)`` with a line for each step;
    ``should_stop()`` is checked between recordings, and when it says stop
    the table being written is discarded (the previous one stays) and the
    report carries ``stopped``. ``known`` is an earlier :func:`scan_models` of
    the same files, so a scan the caller already ran is not repeated.

    Raises ValueError when there is nothing to export: no table asked for, or
    only detections asked for and no recording analysed with the model asked
    for. (With labelled calls asked for too, that is reported in ``error`` and
    the labelled calls are still written.)"""
    progress = progress or (lambda *a: None)
    log = log or (lambda m: None)
    stop = should_stop or (lambda: False)
    if not (opts.detections or opts.labels or opts.images):
        raise ValueError("Nothing to export: ask for detections, labelled "
                         "calls or images.")
    if opts.images and not opts.labels:
        opts = dataclasses.replace(opts, labels=True)  # images are of labelled calls
    rep = {"stores": 0, "recent": 0, "unreadable": 0, "reused": 0, "read": 0,
           "recordings": 0, "detection_rows": 0, "labels_reused": 0,
           "label_recordings": 0, "label_rows": 0, "labels_no_audio": 0,
           "unreadable_labels": 0, "stopped": False, "error": "",
           "tables": [], "out_dir": opts.out_dir}
    os.makedirs(opts.out_dir, exist_ok=True)
    try:
        log("Looking for .mad files…")
        stores = find_stores(opts.folders, opts.wavs, stop)
        rep["stores"] = len(stores)
        settled = time.time() - max(0.0, float(opts.settle_s))
        recent = [s for s in stores if s.mtime > settled]
        if recent:
            stores = [s for s in stores if s.mtime <= settled]
            rep["recent"] = len(recent)
        log(f"{rep['stores']:,} recording(s) with a .mad file"
            + (f"; {len(recent):,} written in the last "
               f"{opts.settle_s / 60:g} min left out (MAD may still be writing "
               "them; the next export includes them)" if recent else ""))
        prev = _load_settings(opts.out_dir)
        if opts.detections:
            try:
                _export_detections(stores, opts, dict(known or {}), prev, rep,
                                   progress, log, stop)
            except ValueError as e:
                if not opts.labels:
                    raise
                rep["error"] = str(e)
                log(f"No detection tables: {e}")
        if opts.labels:
            _export_labels(stores, opts, prev, rep, progress, log, stop)
    except ExportStopped:
        rep["stopped"] = True
        log("Stopped; the table being written was discarded and the previous "
            "one kept.")
    if rep["unreadable"] or rep["unreadable_labels"]:
        log(f"{rep['unreadable'] + rep['unreadable_labels']:,} .mad file(s) could "
            "not be read (open in MAD, or damaged) and were left out.")
    if rep["labels_no_audio"]:
        log(f"{rep['labels_no_audio']:,} recording(s) with labelled calls have no "
            "audio where the .mad says; their call metrics are blank.")
    return rep
