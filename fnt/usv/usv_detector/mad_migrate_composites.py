"""Undo the composited-mask era: split ``mask`` back into one call + neighbours.

Between 2026-09-05 (``929490c``) and this module, every confirmed call was
saved with **every confirmed pixel in its patch window** ORed into its mask.
The intent was training-only — a neighbouring labelled call must not be
supervised as background — but the same array is what the overlay, the
confirmed-mask gallery, mask editing and the CSV geometry read as *this call's
shape*. So adjacent calls read as one detection, and re-confirming a call minted
a larger composite whose bounding box painted over its neighbour.

The split is recoverable because each example still records the call's own
extent in its metadata: ``t_start_s``/``t_stop_s`` and ``f_low_hz``/``f_high_hz``
were always written from the single call, never from the composite. So:

* label the mask's connected components;
* keep the ones overlapping the metadata box as ``mask``;
* move the rest to ``neighbors``.

**Ambiguity.** A component that overlaps the box but extends far outside it may
be two genuinely touching calls that the accumulator fused into one blob — no
amount of metadata separates those, because the pixels are connected. Those are
reported, not guessed at, so they can be re-traced by hand.

Run via :func:`migrate_store` on one file or :func:`migrate_project` on a whole
project (training store + every recording's ``.mad``).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import fnt_mask_store as _ms


@dataclass
class ExampleReport:
    """What the split did to one example."""
    example_id: str
    source_wav: str
    kind: str
    t_start_s: float
    px_before: int
    px_after: int
    components_before: int
    components_kept: int
    px_to_neighbors: int
    ambiguous: bool = False
    reason: str = ""

    @property
    def changed(self) -> bool:
        return self.px_to_neighbors > 0


@dataclass
class StoreReport:
    path: str
    examples: int = 0
    changed: int = 0
    ambiguous: int = 0
    emptied: int = 0
    px_moved: int = 0
    backup: Optional[str] = None
    error: str = ""
    rows: List[ExampleReport] = field(default_factory=list)
    #: (id_a, id_b, iou, shared_px) — two traces of one call, left for the
    #: user to resolve. Merging them is a judgement about the recording.
    duplicates: List[tuple] = field(default_factory=list)

    def summary(self) -> str:
        if self.error:
            return f"{os.path.basename(self.path)}: ERROR {self.error}"
        return (f"{os.path.basename(self.path)}: {self.changed}/{self.examples} "
                f"split, {self.px_moved} px moved to neighbours"
                + (f", {self.ambiguous} ambiguous" if self.ambiguous else "")
                + (f", {self.emptied} would empty (left alone)"
                   if self.emptied else "")
                + (f", {len(self.duplicates)} duplicate pair(s)"
                   if self.duplicates else ""))


def _label(mask: np.ndarray):
    """Connected components, 8-connected. scipy is already a dependency."""
    from scipy import ndimage
    structure = np.ones((3, 3), dtype=bool)
    return ndimage.label(mask, structure=structure)


def _own_box(meta: Dict, shape) -> Optional[tuple]:
    """The call's own (f0, f1, t0, t1) inside the patch, from its metadata.

    Returns None when the metadata cannot place the call — an example written
    before these fields existed, or one whose patch parameters are missing.
    """
    H, W = shape
    try:
        nfft = int(meta["nfft"])
        sr = float(meta["sample_rate"])
        hop = int(meta["nperseg"]) - int(meta["noverlap"])
        df = (sr / 2.0) / (nfft // 2)
        dt = hop / sr
        # Patch-local frame indices. patch_t0_s and t_start_s share the same
        # frame-time origin, so it cancels in the difference.
        lt0 = int(round((float(meta["t_start_s"]) - float(meta["patch_t0_s"])) / dt))
        lt1 = int(round((float(meta["t_stop_s"]) - float(meta["patch_t0_s"])) / dt))
        f0 = int(round(float(meta["f_low_hz"]) / df)) - int(meta.get("patch_f_off") or 0)
        f1 = int(round(float(meta["f_high_hz"]) / df)) - int(meta.get("patch_f_off") or 0)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    lt0, lt1 = max(0, min(lt0, W)), max(0, min(lt1, W))
    f0, f1 = max(0, min(f0, H)), max(0, min(f1, H))
    if lt1 <= lt0:
        lt1 = min(W, lt0 + 1)
    if f1 <= f0:
        f1 = min(H, f0 + 1)
    return f0, f1, lt0, lt1


#: A kept component wider than this multiple of the call's own time span is
#: reported as ambiguous — most likely two calls the accumulator fused into one
#: connected blob, which no metadata can separate.
FUSE_WIDTH_FACTOR = 2.5


def split_example(mask: np.ndarray, meta: Dict) -> Dict:
    """Split one composited mask into ``own`` / ``neighbors``.

    Returns ``{'own', 'neighbors', 'components', 'kept', 'ambiguous',
    'reason'}``. On anything it cannot reason about it keeps the mask whole
    and says why — this runs over the only copy of a user's labels, so the
    failure mode has to be "changed nothing".
    """
    mask = np.asarray(mask) > 0
    out = {"own": mask, "neighbors": np.zeros_like(mask),
           "components": 0, "kept": 0, "ambiguous": False, "reason": ""}
    if not mask.any():
        return out
    lab, n = _label(mask)
    out["components"] = n
    # A rejected PREDICTION ('negative') has no traced mask of its own — it is
    # a hard negative, supervised with an all-zero target. So every pixel in
    # its composite arrived from a neighbouring confirmed call, and all of it
    # belongs in `neighbors`. This is the worse half of the composite bug:
    # collect_training_examples zeroes a negative's mask, so those confirmed
    # pixels were training the model that a call a human had just accepted was
    # background. ('rejected' is different — a demoted confirmed call that
    # keeps its own trace — so it goes through the geometry path below.)
    if _ms.example_kind(meta) == "negative":
        out["own"] = np.zeros_like(mask)
        out["neighbors"] = mask
        out["kept"] = 0
        return out
    box = _own_box(meta, mask.shape)
    if box is None:
        out["reason"] = "no usable geometry in metadata"
        out["ambiguous"] = True
        return out
    f0, f1, t0, t1 = box
    if n > 1:
        ids = np.unique(lab[f0:f1, t0:t1])
        ids = ids[ids > 0]
        if ids.size == 0:
            # Every component sits outside the call's own box. The traced
            # pixels for this call are gone (an edit that moved them, most
            # likely), and picking one of the neighbours would invent a call.
            out["reason"] = "no component overlaps the call's own box"
            out["ambiguous"] = True
            return out
        own = np.isin(lab, ids)
        out["own"] = own
        out["neighbors"] = mask & ~own
        out["kept"] = int(ids.size)
    else:
        own = mask
        out["kept"] = n
    # A kept component far wider than the call itself is two fused calls. This
    # runs for a single component too: two calls that touch are ONE connected
    # blob, which is precisely the case no component split can catch.
    cols = np.where(own.any(axis=0))[0]
    if cols.size:
        span = int(cols[-1] - cols[0] + 1)
        if span > FUSE_WIDTH_FACTOR * max(1, t1 - t0):
            out["ambiguous"] = True
            out["reason"] = (f"kept blob spans {span} frames vs the call's "
                             f"{t1 - t0} - likely two fused calls")
    return out


#: A component of one label and the whole of another label are "the same
#: object" when each covers this much of the other. Both directions matter: the
#: first alone would strip a big blob that merely contains a small one, the
#: second alone would strip a small blob that sits inside a big label.
SAME_OBJECT_FRAC = 0.9

#: Two labels overlapping by more than this are near-duplicates of one call —
#: reported, never merged, because which trace to keep is the user's call.
DUPLICATE_IOU = 0.7


def _abs_points(mask: np.ndarray, meta: Dict):
    """A label's pixels as absolute (freq_bin, frame) on the full-file grid.

    Patches have different widths and offsets, so nothing can be compared
    between two labels until both are put back on the shared grid.
    """
    f_off = int(meta.get("patch_f_off") or 0)
    t_off = int(meta.get("patch_t_off") or 0)
    fs, ts = np.nonzero(mask)
    return set(zip((fs + f_off).tolist(), (ts + t_off).tolist()))


def subtract_sibling_labels(entries: List[Dict]) -> List[Dict]:
    """Move components that ARE another label out of each label's own mask.

    The geometry pass in :func:`split_example` fails whenever an example's own
    ``t_start``/``t_stop`` were themselves derived from a composite — an edited
    mask recomputes them from whatever it is holding, so a 30 ms call can end up
    recorded as 125 ms and its box then covers every neighbour. The other
    labels are the way out: if a component of A is, pixel for pixel, some other
    label B, then it belongs to B and A never drew it.

    ``entries`` are ``{'meta', 'mask'}`` dicts for the LABELS of one recording.
    Returns ``{'index', 'own', 'neighbors', 'moved', 'foreign'}`` per change.
    """
    from scipy import ndimage
    structure = np.ones((3, 3), dtype=bool)
    pts = [_abs_points(e["mask"], e["meta"]) for e in entries]
    out: List[Dict] = []
    for i, e in enumerate(entries):
        mask = np.asarray(e["mask"]) > 0
        if not mask.any():
            continue
        lab, n = ndimage.label(mask, structure=structure)
        if n <= 1:
            continue
        f_off = int(e["meta"].get("patch_f_off") or 0)
        t_off = int(e["meta"].get("patch_t_off") or 0)
        foreign = np.zeros_like(mask)
        names: List[str] = []
        for cid in range(1, n + 1):
            comp = lab == cid
            fs, ts = np.nonzero(comp)
            cpts = set(zip((fs + f_off).tolist(), (ts + t_off).tolist()))
            for j, other in enumerate(pts):
                if j == i or not other:
                    continue
                inter = len(cpts & other)
                if not inter:
                    continue
                if (inter / len(cpts) >= SAME_OBJECT_FRAC
                        and inter / len(other) >= SAME_OBJECT_FRAC):
                    foreign |= comp
                    names.append(str(entries[j]["meta"].get("id") or j))
                    break
        if not foreign.any():
            continue
        own = mask & ~foreign
        if not own.any():
            continue          # never strip a label down to nothing
        out.append({"index": i, "own": own, "neighbors": foreign,
                    "moved": int(foreign.sum()), "foreign": names})
    return out


def duplicate_pairs(entries: List[Dict], iou: float = DUPLICATE_IOU):
    """Label pairs overlapping enough to be two traces of one call.

    Not merged: which trace to keep is a judgement about the recording, not
    about the data.
    """
    pts = [_abs_points(e["mask"], e["meta"]) for e in entries]
    hits = []
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            a, b = pts[i], pts[j]
            if not a or not b:
                continue
            inter = len(a & b)
            if not inter:
                continue
            score = inter / len(a | b)
            if score >= iou:
                hits.append((i, j, score, inter))
    return hits


def migrate_store(h5_path: str, *, dry_run: bool = True,
                  backup: bool = True) -> StoreReport:
    """Split every composited example in one store.

    Two passes, because one signal is not enough:

    1. **Geometry** — keep the components over the call's own recorded
       time/frequency box (:func:`split_example`).
    2. **Siblings** — move any component that *is* another label out of this
       one (:func:`subtract_sibling_labels`). This catches what geometry
       cannot: an example whose own ``t_start``/``t_stop`` were themselves
       recomputed from a composite, so its box already covers its neighbours.

    ``dry_run`` reports without writing, which is the only sane default for an
    in-place rewrite of a label store. A backup is taken before the first write.
    """
    _ms._require_h5()
    import h5py
    rep = StoreReport(path=h5_path)
    if not os.path.isfile(h5_path):
        rep.error = "not found"
        return rep

    # Pass 0: read everything, holding the file read-only.
    items = []
    try:
        with h5py.File(h5_path, "r") as f:
            ex = f.get("examples")
            if ex is None:
                return rep
            for key in ex:
                g = ex[key]
                try:
                    meta = json.loads(g.attrs.get("meta_json", "{}"))
                    mask = g["mask"][()] > 0
                    nb = (g["neighbors"][()] > 0) if "neighbors" in g \
                        else np.zeros_like(mask)
                except Exception:
                    continue
                rep.examples += 1
                items.append({"key": key, "meta": meta, "mask": mask,
                              "own": mask, "neighbors": nb,
                              "was_split": "neighbors" in g,
                              "kind": _ms.example_kind(meta),
                              "dirty": False, "ambiguous": False,
                              "reason": "", "components": 0, "kept": 0})
    except Exception as exc:
        rep.error = f"{type(exc).__name__}: {exc}"
        return rep

    # Pass 1: geometry. Skips anything already split — that pass is settled.
    for it in items:
        if it["was_split"]:
            continue
        res = split_example(it["mask"], it["meta"])
        it["components"] = res["components"]
        it["kept"] = res["kept"]
        it["ambiguous"] = bool(res["ambiguous"])
        it["reason"] = res["reason"]
        if int(res["neighbors"].sum()) <= 0:
            continue
        if not res["own"].any() and it["kind"] != "negative":
            rep.emptied += 1          # never leave a call with no mask
            continue
        it["own"] = res["own"]
        it["neighbors"] = res["neighbors"]
        it["dirty"] = True

    # Pass 2: siblings, per recording — a label can only be explained by
    # another label on the same audio.
    by_wav: Dict[str, List[int]] = {}
    for i, it in enumerate(items):
        if it["kind"] != "label":
            continue
        by_wav.setdefault(os.path.basename(str(it["meta"].get("source_wav", ""))),
                          []).append(i)
    for idxs in by_wav.values():
        entries = [{"meta": items[i]["meta"], "mask": items[i]["own"]}
                   for i in idxs]
        for change in subtract_sibling_labels(entries):
            it = items[idxs[change["index"]]]
            it["own"] = change["own"]
            it["neighbors"] = it["neighbors"] | change["neighbors"]
            it["dirty"] = True
            it["ambiguous"] = False       # resolved by an exact match
            it["reason"] = (f"contained label(s) "
                            f"{', '.join(change['foreign'][:3])}")
        for i, j, score, inter in duplicate_pairs(entries):
            rep.duplicates.append((
                str(items[idxs[i]]["meta"].get("id") or i),
                str(items[idxs[j]]["meta"].get("id") or j),
                float(score), int(inter)))

    # Report + plan.
    plans = []
    for it in items:
        moved = int((it["mask"] & ~it["own"]).sum())
        rep.rows.append(ExampleReport(
            example_id=it["key"],
            source_wav=str(it["meta"].get("source_wav", "")),
            kind=it["kind"],
            t_start_s=float(it["meta"].get("t_start_s") or 0.0),
            px_before=int(it["mask"].sum()),
            px_after=int(it["own"].sum()),
            components_before=it["components"],
            components_kept=it["kept"],
            px_to_neighbors=moved,
            ambiguous=it["ambiguous"],
            reason=it["reason"],
        ))
        if it["ambiguous"]:
            rep.ambiguous += 1
        if not it["dirty"]:
            continue
        rep.changed += 1
        rep.px_moved += moved
        plans.append((it["key"], it["own"], it["neighbors"]))

    if dry_run or not plans:
        return rep

    if backup:
        rep.backup = _ms.backup_store(h5_path)
    try:
        with h5py.File(h5_path, "a") as f:
            ex = f["examples"]
            for key, own, nb in plans:
                g = ex[key]
                del g["mask"]
                g.create_dataset("mask", data=own.astype(np.uint8),
                                 compression="gzip", compression_opts=4)
                if "neighbors" in g:
                    del g["neighbors"]
                g.create_dataset("neighbors", data=nb.astype(np.uint8),
                                 compression="gzip", compression_opts=4)
    except Exception as exc:
        rep.error = f"{type(exc).__name__}: {exc}"
    return rep


def project_stores(project_dir: str,
                   extra_dirs: Sequence[str] = ()) -> List[str]:
    """Every store a project's labels live in: the training store, plus the
    ``.mad`` beside each recording the project references."""
    paths: List[str] = []
    td = os.path.join(project_dir, "models", "training_data",
                      _ms.TRAINING_STORE_NAME)
    if os.path.isfile(td):
        paths.append(td)
    info = os.path.join(project_dir, "mad_project_info.json")
    wavs: List[str] = []
    if os.path.isfile(info):
        try:
            with open(info) as fh:
                cfg = json.load(fh)
            wavs = [str(w) for w in (cfg.get("audio_files") or [])]
        except Exception:
            wavs = []
    for wav in wavs:
        for cand in _ms.store_paths_for(wav):
            if os.path.isfile(cand) and cand not in paths:
                paths.append(cand)
                break
    for d in extra_dirs:
        for root, _dirs, files in os.walk(d):
            for name in files:
                if name.endswith(_ms.MAD_SUFFIX):
                    p = os.path.join(root, name)
                    if p not in paths:
                        paths.append(p)
    return paths


def migrate_project(project_dir: str, *, dry_run: bool = True,
                    backup: bool = True,
                    extra_dirs: Sequence[str] = ()) -> List[StoreReport]:
    """Split composited masks across a project's whole label set."""
    return [migrate_store(p, dry_run=dry_run, backup=backup)
            for p in project_stores(project_dir, extra_dirs)]


def _main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("project", help="MAD project directory")
    ap.add_argument("--audio-root", action="append", default=[],
                    help="also scan this tree for *_FNT.mad (repeatable)")
    ap.add_argument("--apply", action="store_true",
                    help="write the split (default is a dry run)")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="list every changed example")
    a = ap.parse_args(argv)
    reps = migrate_project(a.project, dry_run=not a.apply,
                           backup=not a.no_backup,
                           extra_dirs=a.audio_root)
    tot = dict(ex=0, ch=0, am=0, px=0)
    for r in reps:
        print(r.summary())
        tot["ex"] += r.examples
        tot["ch"] += r.changed
        tot["am"] += r.ambiguous
        tot["px"] += r.px_moved
        if a.verbose:
            for row in r.rows:
                if not row.changed and not row.ambiguous:
                    continue
                flag = "AMBIGUOUS " if row.ambiguous else ""
                print(f"    {flag}{row.source_wav} t={row.t_start_s:8.3f} "
                      f"{row.kind:8} {row.px_before:6} -> {row.px_after:6} px "
                      f"({row.components_before} comp, kept {row.components_kept})"
                      + (f"  [{row.reason}]" if row.reason else ""))
    print(f"\n{len(reps)} store(s): {tot['ch']}/{tot['ex']} examples split, "
          f"{tot['px']} px moved, {tot['am']} ambiguous"
          + ("" if a.apply else "   (DRY RUN — pass --apply to write)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
