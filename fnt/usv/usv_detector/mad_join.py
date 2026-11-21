"""Join fragments: pending detections that are pieces of one call, made one.

A faint call can cross the model's threshold in pieces, and inference run with
``--no-infer-merge`` (as the 2021_8x8 production run was) stores each piece as
its own detection. That inflates the call count and splits every shape
measure. Inference has a mask-level join (``merge_consecutive_blobs``), but it
gates on frequency *bands*; this step runs afterwards, on the detections a
``.mad`` already holds -- no model, no GPU, no audio -- and gates on whether the
pitch carries on across the gap (2026-11-09).

The rule. Two detections A (earlier onset) and B join when

1. both are pending (no example in the .mad names their blob_id, and their box
   does not overlap any example's: a reviewed call is never touched) and both
   are fundamentals (``harmonic_n`` 1 or unset);
2. the gap from A's last mask frame to B's first is at most ``max_gap_s``
   (converted to frames as inference converts ``merge_max_gap_s``: 10 ms is 20
   frames, 10.24 ms); a negative gap, a time overlap, is allowed;
3. B's start frequency is within ``max_df_khz`` of A's end frequency. These
   are the brightness ridge (mad_ridge.ridge_path: the brightest continuous
   path through the mask's own pixels, one bin per frame) averaged over A's
   last ``end_frames`` and B's first ``end_frames`` mask frames;
4. if they overlap in time by more than ``concurrent_fraction`` of the shorter
   one they are sounding together, and must also share a frequency band --
   which keeps a harmonic, or a second animal's call, off its neighbour. This
   is ``_bands_joinable`` with no limit on the sequential band gap;
   ``band_rule`` applies inference's own limit as well (``band_max_gap_bins``).

Each detection joins at most one predecessor and one successor, the best
(smallest frequency step, then shortest gap) pairs first, so a call broken
into A, B, C comes back as the chain A-B-C and one piece cannot bridge two
simultaneous calls. Pairs are found in onset order.

A joined call: the union of the pieces' masks on their union box (the gaps
stay empty), the spectrogram assembled from the pieces' crops (0 between
them), the earliest piece's blob_id and attrs, ``joined_from`` (the blob_ids,
comma-separated) and ``n_pieces``, the score area-weighted as
``merge_consecutive_blobs`` weights it, and its ridge recomputed (one per
piece; mad_ridge). Harmonic grouping is then recomputed around the joins
(mad_harmonics.group_calls over the runs of time-overlapping elements that
hold a joined call, which gives what grouping the whole recording would),
but stored only on the pending detections
the joins affected -- the joined calls and any detection whose call id named
one of the pieces, before or after. Every other detection, and every example,
keeps the grouping it had.

Reversible. Before a detection is first replaced, deleted or regrouped, its
``pred_calls`` / ``pred_specs`` / ``pred_ridge`` entries are copied, as they
are, into ``/unjoined``. A rerun first puts those back, so it always starts
from the detections inference wrote; ``undo`` puts them back and stops. A
joined call reviewed (or deleted) since the join is left as it is, with its
pieces kept in ``/unjoined``. A re-inference since the join makes the copies
stale: they are dropped, never restored over the new detections.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import fnt_mask_store as ms

#: The group holding the originals of every detection a join replaced, deleted
#: or regrouped, as ``/unjoined/{pred_calls,pred_specs,pred_ridge}/<blob_id>``.
UNJOINED_GROUP = "unjoined"
_PARTS = (ms.PRED_GROUP, ms.PRED_SPEC_GROUP, ms.PRED_RIDGE_GROUP)
#: Root attrs a join sets (and undo removes).
JOIN_ATTRS = ("join_gap_s", "join_max_df_khz", "join_end_frames",
              "join_concurrent_fraction", "join_band_rule", "joined_at",
              "join_fnt_version", "join_n_calls", "join_n_pieces")
#: Root attrs a join changes; their values before it are kept on /unjoined.
_CHANGED_ROOT_ATTRS = ("n_pred_blobs",)
_HARM_KEYS = ("harmonic_call_id", "harmonic_n", "f0_hz")


@dataclass(frozen=True)
class JoinConfig:
    """The join rule's numbers (see the module docstring)."""
    #: Largest gap bridged, seconds (10 ms; inference's merge_max_gap_s).
    max_gap_s: float = 0.010
    #: Largest step between A's end and B's start frequency, kHz.
    max_df_khz: float = 5.0
    #: Mask frames averaged at each end for the end/start frequency.
    end_frames: int = 3
    #: Above this time overlap (fraction of the shorter) two pieces are
    #: concurrent and must share a band (inference's merge_concurrent_fraction).
    concurrent_fraction: float = 0.5
    #: Also apply inference's band gating (merge_max_freq_gap_bins for
    #: sequential pieces).
    band_rule: bool = False
    band_max_gap_bins: int = 8


# ----------------------------------------------------------------------
# Pieces and the pair test
# ----------------------------------------------------------------------
def _grid(attrs) -> Optional[Tuple[float, float, int, int]]:
    """(kHz per bin, s per frame, hop, nperseg) from a file's root attrs."""
    try:
        sr = float(attrs["sample_rate"])
        nperseg, noverlap, nfft = (int(attrs["nperseg"]), int(attrs["noverlap"]),
                                   int(attrs["nfft"]))
    except KeyError:
        return None
    hop = nperseg - noverlap
    if sr <= 0 or nfft <= 0 or hop <= 0:
        return None
    return sr / nfft / 1000.0, hop / sr, hop, nperseg


def _contour_khz(mask: np.ndarray, rec: Dict, df_khz: float) -> np.ndarray:
    """The detection's frequency (kHz) in each of its mask frames, in time
    order: the stored brightness ridge (/pred_ridge) when it is one row; else
    the ridge recomputed from the stored spectrogram; else, with neither, the
    mask's centroid per frame."""
    from .mad_ridge import ridge_path
    on = np.flatnonzero(mask.any(axis=0))
    stored = rec.get("ridge_khz")
    if stored is not None:
        k = np.asarray(stored, float)
        if k.ndim == 1 and k.shape[0] == mask.shape[1] and np.isfinite(k[on]).all():
            return k[on]
    f_off = int(rec.get("f_off", 0))
    spec = rec.get("spec")
    if spec is not None and np.asarray(spec).shape == mask.shape:
        _frames, bins = ridge_path(spec, mask)
        return (np.asarray(bins, float) + f_off) * df_khz
    rows = np.arange(mask.shape[0], dtype=float)[:, None]
    w = mask[:, on].astype(float)
    return ((rows * w).sum(axis=0) / np.maximum(w.sum(axis=0), 1.0) + f_off) * df_khz


def make_piece(blob_id: str, rec: Dict, df_khz: float, end_frames: int = 3) -> Optional[Dict]:
    """A stored detection as the join sees it: its tight box on the full grid
    (``t_start``/``t_end_exclusive``/``f_low``/``f_high_exclusive``, the keys
    ``_bands_joinable`` reads), its end/start frequencies (kHz) and score
    weight. None for an empty mask."""
    m = np.asarray(rec["mask"]) > 0
    cols = np.flatnonzero(m.any(axis=0))
    rows = np.flatnonzero(m.any(axis=1))
    if not len(cols):
        return None
    f_off, t_off = int(rec.get("f_off", 0)), int(rec.get("t_off", 0))
    khz = _contour_khz(m, rec, df_khz)
    n = max(1, int(end_frames))
    try:
        fundamental = int(rec.get("harmonic_n") or 1) == 1
    except (TypeError, ValueError):
        fundamental = True
    return {
        "blob_id": str(blob_id),
        "t_start": t_off + int(cols[0]), "t_end_exclusive": t_off + int(cols[-1]) + 1,
        "f_low": f_off + int(rows[0]), "f_high_exclusive": f_off + int(rows[-1]) + 1,
        "f_start_khz": float(khz[:n].mean()), "f_end_khz": float(khz[-n:].mean()),
        "area": int(m.sum()),
        "score": float(rec["score"]) if isinstance(rec.get("score"), (int, float)) else None,
        "fundamental": fundamental,
    }


def gap_frames_limit(max_gap_s: float, dt_s: float) -> int:
    """``max_gap_s`` in frames, rounded as inference rounds merge_max_gap_s."""
    return int(round(max_gap_s / dt_s)) if dt_s else 0


def pair_test(a: Dict, b: Dict, cfg: JoinConfig, max_gap_frames: int
              ) -> Optional[Tuple[float, int]]:
    """``(frequency step kHz, gap frames)`` when ``b`` (the later onset) may
    continue ``a`` under ``cfg``, else None."""
    from .mad_inference import _bands_joinable
    gap = b["t_start"] - a["t_end_exclusive"]
    if gap > max_gap_frames:
        return None
    step = abs(b["f_start_khz"] - a["f_end_khz"])
    if step > cfg.max_df_khz:
        return None
    max_bins = cfg.band_max_gap_bins if cfg.band_rule else 10 ** 9
    if not _bands_joinable(a, b, max_bins, cfg.concurrent_fraction):
        return None
    return step, gap


def band_rule_test(a: Dict, b: Dict, max_gap_frames: int, max_freq_gap_bins: int = 8,
                   concurrent_fraction: float = 0.5) -> bool:
    """Inference's own pair test (merge_consecutive_blobs): gap and band
    gating only, no pitch continuity. For comparing the two rules."""
    from .mad_inference import _bands_joinable
    return (b["t_start"] - a["t_end_exclusive"] <= max_gap_frames
            and _bands_joinable(a, b, max_freq_gap_bins, concurrent_fraction))


def plan_chains(pieces: List[Dict], cfg: JoinConfig, dt_s: float) -> List[List[Dict]]:
    """The chains of two or more pieces to join, each in onset order."""
    order = sorted(pieces, key=lambda p: (p["t_start"], p["f_low"], p["blob_id"]))
    gmax = gap_frames_limit(cfg.max_gap_s, dt_s)
    edges = []
    for i, a in enumerate(order):
        for j in range(i + 1, len(order)):
            b = order[j]
            if b["t_start"] - a["t_end_exclusive"] > gmax:
                break                       # onset-sorted: none later can qualify
            res = pair_test(a, b, cfg, gmax)
            if res is not None:
                edges.append((res[0], res[1], i, j))
    edges.sort()
    succ: Dict[int, int] = {}
    pred: Dict[int, int] = {}
    for _step, _gap, i, j in edges:
        if i in succ or j in pred:
            continue
        succ[i], pred[j] = j, i
    chains = []
    for i in range(len(order)):
        if i in pred or i not in succ:
            continue
        chain = [i]
        while chain[-1] in succ:
            chain.append(succ[chain[-1]])
        chains.append([order[k] for k in chain])
    return chains


# ----------------------------------------------------------------------
# Reading a file's state
# ----------------------------------------------------------------------
def _infer_stamp(f) -> str:
    a = f.attrs
    return f"{a.get('last_infer_model', '')}@{a.get('last_infer_at', '')}"


def _attr(v):
    return v.decode() if isinstance(v, bytes) else v.item() if hasattr(v, "item") else v


def _record(grp_calls, grp_specs, key: str, grp_ridge=None) -> Dict:
    ds = grp_calls[key]
    rec = {"mask": ds[()] > 0, "f_off": int(ds.attrs.get("f_off", 0)),
           "t_off": int(ds.attrs.get("t_off", 0))}
    for k in ds.attrs:
        rec[k] = _attr(ds.attrs[k])
    if grp_specs is not None and key in grp_specs:
        rec["spec"] = grp_specs[key][()]
    if grp_ridge is not None and key in grp_ridge:
        rec["ridge_khz"] = grp_ridge[key][()]
    return rec


def _example_state(f, grid) -> Tuple[set, List[Tuple[int, int, int, int]]]:
    """(blob_ids the examples name, the examples' boxes as (t0, t1, f0, f1)
    grid cells). A detection in either is reviewed and never joined."""
    claimed, boxes = set(), []
    if grid is None:
        return claimed, boxes
    df_khz, dt_s, _hop, nperseg = grid
    # Example bounds are seconds on scipy's axis, where frame 0 sits half a
    # window in (mad_inference._frame_time_origin).
    t_org = nperseg / (2.0 * float(f.attrs["sample_rate"]))
    for meta in ms._iter_meta(f):
        bid = meta.get("blob_id")
        if bid is not None:
            claimed.add(str(bid))
        try:
            t0 = int(np.floor((float(meta["t_start_s"]) - t_org) / dt_s))
            t1 = int(np.ceil((float(meta["t_stop_s"]) - t_org) / dt_s))
            f0 = int(np.floor(float(meta["f_low_hz"]) / 1000.0 / df_khz))
            f1 = int(np.ceil(float(meta["f_high_hz"]) / 1000.0 / df_khz))
        except (KeyError, TypeError, ValueError):
            continue
        boxes.append((t0, t1, f0, f1))
    return claimed, boxes


def _overlaps_any(p: Dict, boxes) -> bool:
    for t0, t1, f0, f1 in boxes:
        if (p["t_start"] < t1 and t0 < p["t_end_exclusive"]
                and p["f_low"] < f1 and f0 < p["f_high_exclusive"]):
            return True
    return False


def _backup_meta(f) -> Dict:
    g = f.get(UNJOINED_GROUP)
    if g is None:
        return {}
    try:
        return json.loads(g.attrs.get("joins_json", "{}"))
    except Exception:
        return {}


def _restore_plan(f) -> Dict:
    """What putting the originals back would do, read-only.

    ``{stale, joins (restorable joined id -> record), kept (joined ids left
    joined: reviewed or deleted since), regrouped (restorable ids whose only
    change was their grouping)}``."""
    out = {"stale": False, "joins": {}, "kept": {}, "regrouped": [], "backup": False}
    g = f.get(UNJOINED_GROUP)
    if g is None:
        return out
    out["backup"] = True
    if str(g.attrs.get("infer_run", "")) != _infer_stamp(f):
        out["stale"] = True
        return out
    grid = _grid(f.attrs)
    claimed, boxes = _example_state(f, grid)
    preds = f.get(ms.PRED_GROUP)
    joins = _backup_meta(f)
    in_join = set()
    for jid, rec in joins.items():
        ids = [str(x) for x in rec.get("pieces", [])]
        in_join.update(ids)
        cur = preds.get(jid) if preds is not None else None
        reviewed = cur is None or any(i in claimed for i in ids)
        if not reviewed and boxes:
            reviewed = _overlaps_any(_box(cur), boxes)
        (out["kept"] if reviewed else out["joins"])[jid] = rec
    bcalls = g.get(ms.PRED_GROUP)
    for key in (bcalls or {}):
        if key in in_join or key in claimed:
            continue
        if preds is not None and key in preds:
            out["regrouped"].append(key)
    return out


def _box(ds) -> Dict:
    fo, to = int(ds.attrs.get("f_off", 0)), int(ds.attrs.get("t_off", 0))
    h, w = ds.shape
    return {"t_start": to, "t_end_exclusive": to + w, "f_low": fo, "f_high_exclusive": fo + h}


def _current_records(f, plan: Optional[Dict] = None) -> Dict[str, Dict]:
    """Every detection as it stands -- or, given a restore plan, as it would
    stand once the originals in it are back (a dry run never writes)."""
    preds, specs, ridges = (f.get(ms.PRED_GROUP), f.get(ms.PRED_SPEC_GROUP),
                            f.get(ms.PRED_RIDGE_GROUP))
    out = {k: _record(preds, specs, k, ridges) for k in (preds or {})}
    if plan and not plan["stale"] and plan["backup"]:
        b = f[UNJOINED_GROUP]
        bcalls, bspecs, bridges = (b.get(ms.PRED_GROUP), b.get(ms.PRED_SPEC_GROUP),
                                   b.get(ms.PRED_RIDGE_GROUP))
        for jid, rec in plan["joins"].items():
            out.pop(jid, None)
            for k in rec.get("pieces", []):
                if bcalls is not None and str(k) in bcalls:
                    out[str(k)] = _record(bcalls, bspecs, str(k), bridges)
        for k in plan["regrouped"]:
            out[k] = _record(bcalls, bspecs, k, bridges)
    return out


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------
def _drop(f, key: str) -> None:
    for part in _PARTS:
        g = f.get(part)
        if g is not None and key in g:
            del g[key]


def _back_up(f, key: str) -> None:
    """Copy a detection's current entries into /unjoined, once: the first copy
    is the original, and a later one would overwrite it with a joined call."""
    b = f.require_group(UNJOINED_GROUP)
    bcalls = b.get(ms.PRED_GROUP)
    if bcalls is not None and key in bcalls:
        return
    for part in _PARTS:
        src = f.get(part)
        if src is not None and key in src:
            f.copy(src[key], b.require_group(part), name=key)


def _put_back(f, key: str) -> None:
    """Restore one detection from /unjoined exactly (data, attrs, filters),
    then forget the copy."""
    _drop(f, key)
    b = f[UNJOINED_GROUP]
    for part in _PARTS:
        src = b.get(part)
        if src is not None and key in src:
            f.copy(src[key], f.require_group(part), name=key)
            del src[key]


def _apply_restore(f, plan: Dict) -> Dict:
    """Put the originals in ``plan`` back. Returns counts."""
    n = {"joins": 0, "restored": 0, "kept": len(plan["kept"]), "stale": plan["stale"]}
    if not plan["backup"]:
        return n
    b = f[UNJOINED_GROUP]
    if plan["stale"]:
        # A re-inference wrote new detections over the joined ones: these
        # copies belong to detections that no longer exist. Never restore
        # them over the new ones; and the old join's root attrs describe
        # nothing in the file any more.
        del f[UNJOINED_GROUP]
        for k in JOIN_ATTRS:
            if k in f.attrs:
                del f.attrs[k]
        return n
    joins = _backup_meta(f)
    for jid, rec in plan["joins"].items():
        _drop(f, jid)
        for k in rec.get("pieces", []):
            _put_back(f, str(k))
            n["restored"] += 1
        joins.pop(jid, None)
        n["joins"] += 1
    for k in plan["regrouped"]:
        _put_back(f, k)
        n["restored"] += 1
    if joins:
        b.attrs["joins_json"] = json.dumps(joins)
        preds = f.get(ms.PRED_GROUP)
        if preds is not None:
            f.attrs["n_pred_blobs"] = int(len(preds))
    else:
        # The originals' own values, even where they were already stale (the
        # GUI does not always keep n_pred_blobs in step): undo means as it was.
        orig = json.loads(b.attrs.get("root_attrs_json", "{}"))
        del f[UNJOINED_GROUP]
        for k in JOIN_ATTRS:
            if k in f.attrs:
                del f.attrs[k]
        for k, v in orig.items():
            if v is None:
                if k in f.attrs:
                    del f.attrs[k]
            else:
                f.attrs[k] = v
    return n


def _start_backup(f) -> None:
    """Create /unjoined for a first join, noting the inference run its copies
    belong to and the root attrs a join changes."""
    if UNJOINED_GROUP in f:
        return
    b = f.create_group(UNJOINED_GROUP)
    b.attrs["infer_run"] = _infer_stamp(f)
    orig = {}
    for k in _CHANGED_ROOT_ATTRS:
        orig[k] = _attr(f.attrs[k]) if k in f.attrs else None
    b.attrs["root_attrs_json"] = json.dumps(orig)
    b.attrs["joins_json"] = "{}"


def _joined_crop(chain: List[Dict], recs: Dict[str, Dict]) -> Dict:
    """Mask, spectrogram, offsets and score of one chain joined."""
    rs = [recs[p["blob_id"]] for p in chain]
    f_lo = min(r["f_off"] for r in rs)
    t_lo = min(r["t_off"] for r in rs)
    f_hi = max(r["f_off"] + r["mask"].shape[0] for r in rs)
    t_hi = max(r["t_off"] + r["mask"].shape[1] for r in rs)
    mask = np.zeros((f_hi - f_lo, t_hi - t_lo), dtype=bool)
    spec = np.zeros(mask.shape, dtype=np.uint8)
    have_spec = all(r.get("spec") is not None and np.asarray(r["spec"]).shape == r["mask"].shape
                    for r in rs)
    for r in rs:
        fo, to = r["f_off"] - f_lo, r["t_off"] - t_lo
        h, w = r["mask"].shape
        mask[fo:fo + h, to:to + w] |= r["mask"]
        if have_spec:
            # The pieces are crops of one spectrogram, so where two crops
            # overlap they hold the same levels; max keeps either.
            np.maximum(spec[fo:fo + h, to:to + w], np.asarray(r["spec"], np.uint8),
                       out=spec[fo:fo + h, to:to + w])
    # Area-weighted, as merge_consecutive_blobs scores a merge: a big
    # confident piece is not diluted by a small faint one.
    scored = [p for p in chain if p["score"] is not None]
    w = float(sum(p["area"] for p in scored))
    score = (sum(p["score"] * p["area"] for p in scored) / w) if w else None
    return {"mask": mask, "spec": spec if have_spec else None,
            "f_off": f_lo, "t_off": t_lo, "score": score}


def _write_joined(f, chain: List[Dict], recs: Dict[str, Dict], units) -> str:
    keep = chain[0]["blob_id"]
    crop = _joined_crop(chain, recs)
    attrs = {k: v for k, v in recs[keep].items()
             if k not in ("mask", "spec", "ridge_khz", "f_off", "t_off", "score")
             and k not in ms._RIDGE_ATTRS}
    for p in chain:
        _back_up(f, p["blob_id"])
    for p in chain:
        _drop(f, p["blob_id"])
    m = crop["mask"].astype(np.uint8)
    ds = f.require_group(ms.PRED_GROUP).create_dataset(
        keep, data=m, compression="gzip", compression_opts=4)
    ds.attrs["f_off"] = int(crop["f_off"])
    ds.attrs["t_off"] = int(crop["t_off"])
    for k, v in attrs.items():
        ds.attrs[k] = v
    if crop["score"] is not None:
        ds.attrs["score"] = round(float(crop["score"]), 4)
    ds.attrs["joined_from"] = ",".join(p["blob_id"] for p in chain)
    ds.attrs["n_pieces"] = int(len(chain))
    if crop["spec"] is not None:
        f.require_group(ms.PRED_SPEC_GROUP).create_dataset(
            keep, data=crop["spec"], compression="gzip", compression_opts=4)
    ms._store_ridge(f, keep, ds, m, crop["spec"], units)
    return keep


# ----------------------------------------------------------------------
# One file
# ----------------------------------------------------------------------
def join_file(h5_path: str, cfg: JoinConfig = JoinConfig(), *, dry_run: bool = False,
              harmonics: bool = True) -> Dict:
    """Join the fragments among one ``.mad``'s pending detections (see the
    module docstring). Starts from the stored originals when the file was
    joined before. ``dry_run`` reads only. Returns ``{status, before, after,
    n_joined, pieces (n_pieces per joined call), kept, restored, regrouped}``,
    where ``before`` counts the detections as inference wrote them."""
    import h5py
    out = {"path": h5_path, "status": "ok", "before": 0, "after": 0, "n_joined": 0,
           "pieces": [], "kept": 0, "restored": 0, "regrouped": 0}
    mode = "r" if dry_run else "a"
    with h5py.File(h5_path, mode) as f:
        grid = _grid(f.attrs)
        if grid is None:
            out["status"] = "no_grid"
            return out
        df_khz, dt_s = grid[0], grid[1]
        plan = _restore_plan(f)
        if dry_run:
            recs = _current_records(f, plan)
        else:
            r = _apply_restore(f, plan)
            out["restored"] = r["restored"]
            recs = _current_records(f)
        out["kept"] = len(plan["kept"])
        claimed, boxes = _example_state(f, grid)
        kept_pieces = sum(len(r.get("pieces", [])) - 1 for r in plan["kept"].values())
        out["before"] = len(recs) + kept_pieces
        pieces = []
        for key, rec in recs.items():
            if key in claimed:
                continue
            p = make_piece(key, rec, df_khz, cfg.end_frames)
            if p is None or not p["fundamental"] or _overlaps_any(p, boxes):
                continue
            pieces.append(p)
        chains = plan_chains(pieces, cfg, dt_s)
        out["n_joined"] = len(chains)
        out["pieces"] = [len(c) for c in chains]
        out["after"] = len(recs) - sum(len(c) - 1 for c in chains) + kept_pieces
        if dry_run or not chains:
            if not dry_run:
                _finish_root(f, cfg)
            return out
        _start_backup(f)
        units = ms._ridge_units(f)
        joins = _backup_meta(f)
        for chain in chains:
            jid = _write_joined(f, chain, recs, units)
            joins[jid] = {"pieces": [p["blob_id"] for p in chain],
                          "gap_s": cfg.max_gap_s, "max_df_khz": cfg.max_df_khz}
        f[UNJOINED_GROUP].attrs["joins_json"] = json.dumps(joins)
        _finish_root(f, cfg)
        source_wav = str(_attr(f.attrs.get("source_wav", "")) or "")
    if harmonics:
        out["regrouped"] = _regroup(h5_path, source_wav, chains)
    return out


def _finish_root(f, cfg: JoinConfig) -> None:
    if UNJOINED_GROUP not in f:
        return                      # nothing joined, nothing left joined: as it was
    preds = f.get(ms.PRED_GROUP)
    if preds is not None:
        f.attrs["n_pred_blobs"] = int(len(preds))
    joins = _backup_meta(f)
    f.attrs["join_gap_s"] = float(cfg.max_gap_s)
    f.attrs["join_max_df_khz"] = float(cfg.max_df_khz)
    f.attrs["join_end_frames"] = int(cfg.end_frames)
    f.attrs["join_concurrent_fraction"] = float(cfg.concurrent_fraction)
    f.attrs["join_band_rule"] = int(bool(cfg.band_rule))
    f.attrs["joined_at"] = datetime.now().isoformat(timespec="seconds")
    f.attrs["join_fnt_version"] = ms._fnt_version()
    f.attrs["join_n_calls"] = int(len(joins))
    f.attrs["join_n_pieces"] = int(sum(len(r.get("pieces", [])) for r in joins.values()))


def _regroup(h5_path: str, wav_name: str, chains) -> int:
    """Recompute the harmonic grouping around the joins and store it on the pending
    detections the joins affected (the joined calls, and any whose call id
    named a piece before or after). Their originals go to /unjoined first.
    Everything else keeps the grouping it had: on these files the grouping
    was computed at inference, before review, and regrouping the whole file
    would rewrite calls the join never touched. Returns the number changed."""
    import h5py
    from .mad_harmonics import group_calls, grouping_fields, overlap_clusters, recording_elements
    joined = {c[0]["blob_id"] for c in chains}
    absorbed = {p["blob_id"] for c in chains for p in c[1:]}
    got = recording_elements(h5_path, wav_name)
    if got is None:
        return 0
    elements, _example_ids, forced = got
    # Only the runs of time-overlapping elements that hold a joined call can
    # change (a harmonic overlaps its fundamental; the pieces a joined call
    # absorbed overlap it). Grouping just those gives the same answer for
    # them as the whole recording, which on a 3,334-detection file took
    # 160 s. A manual pin pulls its partner's run in too.
    runs = overlap_clusters(elements)
    run_of = {e.id: i for i, r in enumerate(runs) for e in r}
    want = {run_of[j] for j in joined if j in run_of}
    links: Dict[str, set] = {}
    for k, v in (forced or {}).items():
        if v is not None:
            links.setdefault(k, set()).add(v)
            links.setdefault(v, set()).add(k)
    grew = True
    while grew:
        grew = False
        for i in list(want):
            for e in runs[i]:
                for o in links.get(e.id, ()):
                    if o in run_of and run_of[o] not in want:
                        want.add(run_of[o])
                        grew = True
    subset = [e for i in sorted(want) for e in runs[i]]
    if not subset:
        return 0
    fields = grouping_fields(group_calls(subset, forced=forced))
    n = 0
    with h5py.File(h5_path, "a") as f:
        preds = f.get(ms.PRED_GROUP)
        if preds is None:
            return 0
        claimed, _boxes = _example_state(f, _grid(f.attrs))
        touched = joined | absorbed
        for key in preds:
            if key in claimed:
                continue
            ds = preds[key]
            old = {k: _attr(ds.attrs[k]) for k in _HARM_KEYS if k in ds.attrs}
            new = fields.get(key)
            if new is None:
                continue
            if not (key in joined or str(old.get("harmonic_call_id", "")) in touched
                    or str(new.get("harmonic_call_id", "")) in touched):
                continue
            same = (str(old.get("harmonic_call_id", "")) == str(new["harmonic_call_id"])
                    and int(old.get("harmonic_n", 1) or 1) == int(new["harmonic_n"])
                    and abs(float(old.get("f0_hz") or 0) - float(new["f0_hz"] or 0)) < 0.01)
            if same:
                continue
            if key not in joined:
                _back_up(f, key)        # a joined call's original is its pieces
            for k in _HARM_KEYS:
                v = new.get(k)
                if v is None or v == "":
                    if k in ds.attrs:
                        del ds.attrs[k]
                else:
                    ds.attrs[k] = v
            n += 1
    return n


def undo_file(h5_path: str, *, dry_run: bool = False) -> Dict:
    """Put back every original a join replaced (see the module docstring).
    Returns ``{status, joins, restored, kept, stale}``."""
    import h5py
    with h5py.File(h5_path, "r" if dry_run else "a") as f:
        plan = _restore_plan(f)
        if not plan["backup"]:
            return {"path": h5_path, "status": "not_joined", "joins": 0, "restored": 0,
                    "kept": 0, "stale": False}
        if dry_run:
            n = {"joins": len(plan["joins"]),
                 "restored": sum(len(r.get("pieces", [])) for r in plan["joins"].values())
                 + len(plan["regrouped"]), "kept": len(plan["kept"]), "stale": plan["stale"]}
        else:
            n = _apply_restore(f, plan)
    return {"path": h5_path, "status": "stale" if n["stale"] else "ok", **n}


def join_status(h5_path: str) -> Dict:
    """``{joined, n_calls, n_pieces, gap_s, max_df_khz}`` from root attrs."""
    import h5py
    with h5py.File(h5_path, "r") as f:
        a = f.attrs
        return {"joined": UNJOINED_GROUP in f,
                "n_calls": int(a.get("join_n_calls", 0)),
                "n_pieces": int(a.get("join_n_pieces", 0)),
                "gap_s": float(a["join_gap_s"]) if "join_gap_s" in a else None,
                "max_df_khz": float(a["join_max_df_khz"]) if "join_max_df_khz" in a else None}


# ----------------------------------------------------------------------
# Finding files
# ----------------------------------------------------------------------
_SKIP_DIRS = {"models", "datasets", "batch_runs", ".scratch", "legacy_pre_h5"}


def find_stores(inputs: List[str]) -> List[str]:
    """``.mad`` files from a mix of .mad files, wav files (their .mad) and
    folders (walked to the bottom), sorted and de-duplicated."""
    out, seen = [], set()

    def add(p):
        ap = os.path.normcase(os.path.abspath(p))
        if ap not in seen and os.path.isfile(p):
            seen.add(ap)
            out.append(p)

    for item in inputs:
        if os.path.isdir(item):
            for dirpath, dirnames, filenames in os.walk(item):
                dirnames[:] = sorted(d for d in dirnames if not d.startswith(".")
                                     and d.lower() not in _SKIP_DIRS)
                for fn in sorted(filenames):
                    if ms.is_mad_store(fn) and not fn.startswith("."):
                        add(os.path.join(dirpath, fn))
        elif item.lower().endswith(".wav"):
            add(ms.masks_sibling_path(item))
        else:
            add(item)
    return out
