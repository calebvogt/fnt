"""Regenerate a recording's annotations CSV from its h5, rather than patching it.

MAD has been writing two files on every review action: the per-wav h5 (the
pixels, and now score and model provenance) and the per-wav CSV (the table). Two
writable copies of the same facts drift, and on a real project they did -- one
file's CSV listed 891 calls where its h5 held 906. Incremental patching cannot
fix that, because each patch assumes the two were in step to begin with.

So the h5 becomes the source of truth and the CSV becomes a projection of it.
Every review state is already recorded in the h5 as a side effect of the action:

    accepted   an /examples entry (kind 'label'), carrying the blob_id it came from
    rejected   an /examples entry retagged kind 'rejected' (a confirmed call the
               user took back) or kind 'negative' (a prediction they refused)
    pending    a /pred_calls crop with no example against it

Nothing derived status from that until now, which is precisely why it drifted.

**Rows with no h5 record are kept, not dropped.** Accepting from the gallery
with no audio loaded records the decision in the CSV and mints no example on
purpose -- the label is created later, when the file is next opened. Treating
"absent from the h5" as "deleted" would silently discard real decisions, so
those rows survive and are reported instead.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

__all__ = ["rebuild_annotations_csv", "rebuild_folder", "rows_for_wav"]

#: Fields the h5 is authoritative for. Everything else in an existing row --
#: the derived acoustic features, the harmonic grouping -- is carried through
#: untouched: nothing reads it back, and recomputing it needs the audio.
_AUTHORITATIVE = ("status", "class", "start_s", "stop_s",
                  "min_freq_hz", "max_freq_hz", "score",
                  "model_name", "threshold", "min_blob_pixels",
                  "harmonic_call_id", "harmonic_n", "f0_hz")


def _status_for_kind(kind: str) -> str:
    if kind == "rejected" or kind == "negative":
        return "rejected"
    return "accepted"


def _example_rows(h5_path: str) -> Dict[str, Dict]:
    """CSV-key -> row fields, for every stored example."""
    from . import fnt_mask_store as _ms
    out: Dict[str, Dict] = {}
    for meta in _ms.td_iter_meta(h5_path):
        eid = meta.get("id")
        if not eid:
            continue
        kind = _ms.example_kind(meta)
        # A hand-drawn label is keyed by its own id; one minted by accepting a
        # prediction keeps the detection row it came from.
        key = str(meta.get("blob_id", eid) if meta.get("blob_id") is not None
                  else eid)
        row = {
            "blob_id": meta.get("blob_id", eid) if meta.get("blob_id") is not None else eid,
            "status": _status_for_kind(kind),
            "class": meta.get("class", "") or "",
            "start_s": meta.get("t_start_s"),
            "stop_s": meta.get("t_stop_s"),
            "min_freq_hz": meta.get("f_low_hz"),
            "max_freq_hz": meta.get("f_high_hz"),
            "source": "label",
        }
        # Provenance, plus the harmonic grouping — which lives on the call
        # now rather than in a CSV column, so it survives a close and comes
        # back out on export.
        for k in ("score", "model_name", "threshold", "min_blob_pixels",
                  "harmonic_call_id", "harmonic_n", "f0_hz"):
            if meta.get(k) not in (None, ""):
                row[k] = meta[k]
        # A confirmed label outranks a negative at the same key: accepting after
        # rejecting leaves both behind, and the label is the later decision.
        prev = out.get(key)
        if prev is None or (prev["status"] == "rejected"
                            and row["status"] == "accepted"):
            out[key] = row
    return out


def _pending_rows(h5_path: str, seen: set, nperseg: int, noverlap: int,
                  nfft: int, sr: int) -> Dict[str, Dict]:
    """CSV-key -> row fields for prediction crops nobody has reviewed."""
    from . import fnt_mask_store as _ms
    from .mad_inference import frames_to_seconds
    out: Dict[str, Dict] = {}
    if not sr or not nperseg:
        return out
    df = (sr / 2.0) / max(1, (nfft // 2))
    for key, crop in _ms.read_all_pred_masks(h5_path).items():
        if key in seen:
            continue                       # already accepted or rejected
        h, w = crop["mask"].shape
        t0, t1 = crop["t_off"], crop["t_off"] + w
        f0, f1 = crop["f_off"], crop["f_off"] + h
        row = {
            "blob_id": int(key) if str(key).isdigit() else key,
            "status": "pending",
            "class": crop.get("class", "") or "",
            "start_s": frames_to_seconds(t0, nperseg, noverlap, sr),
            "stop_s": frames_to_seconds(t1, nperseg, noverlap, sr),
            "min_freq_hz": round(f0 * df, 2),
            "max_freq_hz": round(f1 * df, 2),
            "area_pixels": int(crop["mask"].sum()),
            "source": "prediction",
        }
        for k in ("score", "model_name", "threshold", "min_blob_pixels"):
            if crop.get(k) not in (None, ""):
                row[k] = crop[k]
        out[key] = row
    return out


def _recompute_features(wav_path: str, rows, grid: Dict, masks) -> int:
    """Fill each row's derived acoustic metrics from the audio.

    These ~25 columns (sinuosity, spectral entropy, SNR, tonality...) are the
    only thing in the CSV that the store cannot supply, because they are
    measured from the recording rather than from the label. Computing them on
    export rather than at label time is the point: a metric written when a call
    was confirmed is frozen at whatever the feature code did that day, and
    improving that code silently leaves every old CSV stale with no signal.
    Recomputing means the analysis always reflects current code, on labels that
    never had to move.

    Returns how many rows were filled; 0 if the recording is unavailable, in
    which case the caller keeps whatever the previous CSV held.
    """
    if not os.path.isfile(wav_path):
        return 0
    try:
        import numpy as np
        from .mad_dataset import compute_spectrogram
        from .mad_inference import (
            _freq_per_bin, _time_per_frame, compute_call_metrics,
            seconds_to_frames)
        from .spectrogram import load_audio
    except Exception:
        return 0
    nperseg = int(grid.get("nperseg", 512))
    noverlap = int(grid.get("noverlap", 384))
    nfft = int(grid.get("nfft", 1024))
    db_min = float(grid.get("db_min", -100.0))
    db_max = float(grid.get("db_max", -20.0))
    try:
        audio, sr = load_audio(wav_path)
        _f, _t, spec_db = compute_spectrogram(
            audio, sr=sr, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    except Exception:
        return 0
    df = _freq_per_bin(nfft, sr)
    dt = _time_per_frame(nperseg, noverlap, sr)
    n = 0
    for r in rows:
        try:
            t0 = int(round(seconds_to_frames(
                float(r.get("start_s") or 0.0), nperseg, noverlap, sr)))
            t1 = int(round(seconds_to_frames(
                float(r.get("stop_s") or 0.0), nperseg, noverlap, sr)))
            f0 = int(round(float(r.get("min_freq_hz") or 0.0) / df))
            t0, t1 = max(0, t0), min(spec_db.shape[1], t1)
            if t1 <= t0:
                continue
            entry = (masks or {}).get(str(r.get("blob_id")))
            if entry is None:
                continue     # no pixels to measure; leave the row's columns be
            mask, m_f0 = entry
            mask = np.asarray(mask) > 0
            if mask.shape[1] != (t1 - t0):
                continue
            r.update(compute_call_metrics(
                spec_db[:, t0:t1], mask, int(m_f0), df, dt, db_min, db_max))
            n += 1
        except Exception:
            continue
    return n


def rows_for_wav(wav_path: str):
    """Every call on ``wav_path``, as CSV-shaped rows, straight from the store.

    The same derivation the export uses, without writing anything — for callers
    that want a recording's calls and should not depend on a CSV existing.
    """
    from . import fnt_mask_store as _ms
    h5_path = _ms.masks_sibling_path(wav_path)
    if not os.path.isfile(h5_path):
        return []
    grid = _ms.get_grid_attrs(h5_path) or {}
    out = _example_rows(h5_path)
    out.update(_pending_rows(
        h5_path, set(out),
        int(grid.get("nperseg", 512)), int(grid.get("noverlap", 384)),
        int(grid.get("nfft", 1024)), int(grid.get("sample_rate", 0))))
    return list(out.values())


def rebuild_annotations_csv(wav_path: str, *, dry_run: bool = False,
                            recompute_features: bool = False) -> Dict:
    """Rewrite ``wav_path``'s annotations CSV from its h5.

    Returns a report: how many rows came from each source, and how many
    existing rows had no h5 record (kept, never dropped -- see the module
    docstring). ``dry_run`` computes the report without writing.
    """
    from . import fnt_mask_store as _ms
    from .mad_inference import read_blob_csv, write_blob_csv
    from .mad_labels import annotations_csv_sibling_path

    h5_path = _ms.masks_sibling_path(wav_path)
    csv_path = annotations_csv_sibling_path(wav_path)
    report = {"wav": os.path.basename(wav_path), "csv": csv_path,
              "accepted": 0, "rejected": 0, "pending": 0,
              "kept_without_h5": 0, "written": 0, "h5_missing": False}
    if not os.path.isfile(h5_path):
        report["h5_missing"] = True
        return report

    existing = {}
    if os.path.isfile(csv_path):
        try:
            for r in read_blob_csv(csv_path):
                existing[str(r.get("blob_id"))] = r
        except Exception:
            existing = {}

    grid = _ms.get_grid_attrs(h5_path) or {}
    from_h5 = _example_rows(h5_path)
    from_h5.update(_pending_rows(
        h5_path, set(from_h5),
        int(grid.get("nperseg", 512)), int(grid.get("noverlap", 384)),
        int(grid.get("nfft", 1024)), int(grid.get("sample_rate", 0))))

    # A crop with no example AND no CSV row is an unreviewed prediction the CSV
    # never knew about — normally impossible, but real in files whose CSV was
    # replaced or cleared while the h5 kept its /pred_calls group. They are
    # emitted (the h5 is the source of truth) but counted separately, because a
    # rebuild that silently adds hundreds of detections to a reviewed file is
    # exactly the kind of surprise this whole change exists to prevent.
    # Masks for the metric pass, gathered only when it is actually going to
    # run — reading every crop is the expensive part of an export.
    masks: Dict[str, tuple] = {}
    if recompute_features:
        try:
            # A stored example holds the whole PATCH — the call plus a wide
            # margin of context. The metrics want the call's own bounding box,
            # which is what _examples_to_annotations carves out, positioned on
            # the full-file grid. Prediction crops are already tight.
            from .mad_examples import _examples_to_annotations
            wav_name = os.path.basename(wav_path)
            big = (10 ** 9, 10 ** 9)
            for kind in ("label", "rejected", "negative"):
                for a in _examples_to_annotations(
                        _ms.td_iter_file_examples(h5_path, wav_name,
                                                  with_spec=False,
                                                  kinds=(kind,)),
                        wav_name, big, kinds=(kind,)):
                    meta_id = str(a.get("id"))
                    key = str(a.get("blob_id") or meta_id)
                    masks[key] = (a["mask"], int(a["f0"]))
            for k, crop in _ms.read_all_pred_masks(h5_path).items():
                masks.setdefault(str(k), (crop["mask"], int(crop["f_off"])))
        except Exception:
            masks = {}

    report["pending_new_to_csv"] = sum(
        1 for k, v in from_h5.items()
        if v["status"] == "pending" and k not in existing)

    rows = []
    for key, fields in from_h5.items():
        # Start from the existing row so the derived feature columns and the
        # harmonic grouping survive; the h5 then overwrites what it owns.
        row = dict(existing.get(key, {}))
        row.update({k: v for k, v in fields.items()
                    if k in _AUTHORITATIVE or k in ("blob_id", "source",
                                                    "area_pixels")})
        rows.append(row)
        report[fields["status"]] = report.get(fields["status"], 0) + 1

    for key, row in existing.items():
        if key in from_h5:
            continue
        # No pixels behind it, but a decision the user made — see the module
        # docstring. Kept and counted rather than quietly removed.
        rows.append(row)
        report["kept_without_h5"] += 1

    if recompute_features:
        report["features_recomputed"] = _recompute_features(
            wav_path, rows, grid, masks)

    report["written"] = len(rows)
    if not dry_run:
        write_blob_csv(csv_path, rows)
    return report


def rebuild_folder(folder: str, *, recursive: bool = True,
                   dry_run: bool = False, recompute_features: bool = False,
                   progress=None):
    """Rebuild every recording's CSV under ``folder`` — the bulk export."""
    from . import fnt_mask_store as _ms
    root = Path(folder)
    # A folder can hold stores from before and after the renames. Note the
    # suffixes nest — "_FNT.mad" ends with ".mad" — so each file's suffix is
    # resolved newest-first rather than by which glob happened to match it,
    # or a store would be visited twice under two different wav names.
    order = (_ms.MAD_SUFFIX,) + _ms.LEGACY_MAD_SUFFIXES
    found = {}
    for suffix in order:
        pat = ("**/*" if recursive else "*") + suffix
        for path in root.glob(pat):
            found.setdefault(str(path), path)
    stores = []
    for path in found.values():
        suffix = next((s for s in order if path.name.endswith(s)), None)
        if suffix is not None:
            stores.append((path, suffix))
    reports = []
    seen = set()
    h5s = sorted(stores, key=lambda kv: str(kv[0]))
    for i, (h5, suffix) in enumerate(h5s):
        wav = h5.with_name(h5.name[: -len(suffix)] + ".wav")
        if str(wav) in seen:
            continue            # a recording with several namings: rebuild once
        seen.add(str(wav))
        if progress is not None:
            progress(i, len(h5s), wav.name)
        try:
            reports.append(rebuild_annotations_csv(
                str(wav), dry_run=dry_run,
                recompute_features=recompute_features))
        except Exception as e:            # one bad file must not stop the run
            reports.append({"wav": wav.name, "error": str(e)})
    return reports
