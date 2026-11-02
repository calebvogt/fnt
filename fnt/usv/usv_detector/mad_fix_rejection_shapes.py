"""Give every stored rejection its own shape back.

Until 2026-10-11 a rejected prediction (``kind='negative'``) was saved with the
wrong thing in ``mask``: the OTHER confirmed calls in its patch window, with the
rejected call's own pixels zeroed. The rejected shape itself was stored nowhere.
That broke the store's contract (``mask`` is this call and nothing else, see
:mod:`fnt_mask_store`) in three visible ways:

* **Rejections reopened as boxes.** With no shape to draw, the review overlay
  rebuilt a filled rectangle from the metadata (1,533 of 2,125 in the anosmia
  project) -- or, when the window held a confirmed call, drew that call's
  outline as the rejection (592).
* **"Rejected calls stay rejected" did not.** Re-runs suppress detections that
  land on a stored rejection by painting each rejection's ``mask``: nothing, or
  the neighbouring confirmed calls.
* **Confirmed neighbours trained as background.** ``neighbors`` is written
  exclusive of ``mask``, so with the neighbours in ``mask`` it came out empty.
  Training zeroes a negative's ``mask`` and excludes only ``neighbors`` from the
  loss -- so the confirmed calls the old code meant to protect were taught as
  noise after all.

The repair, per legacy negative:

1. **Neighbours** -- whatever ``mask`` held moves to ``neighbors`` (it was only
   ever confirmed calls), merged with any ``neighbors`` already there.
2. **Own shape** -- in order of fidelity:

   * ``pred_crop``: the prediction crop with the same ``blob_id`` is still in
     the recording's ``.mad`` (rejections made since that recording's last
     inference). That IS the shape the model proposed and the reviewer saw.
   * ``spec_trace``: otherwise it is re-traced from the stored spectrogram
     patch, inside the box the metadata records (:func:`trace_shape`). An
     approximation of the bright ridge, not the model's blob -- good enough to
     draw and to suppress (re-runs dilate it), and never used as a training
     target: a negative's target is all background regardless.

Training targets do not change except for the third bullet above, which is a
fix. Every repaired example is stamped ``mask_is_own=True`` plus
``shape_source``; new rejections are saved that way from the start.

Dry run by default. ``--apply`` copies each store aside first
(``<store>.pre_rejection_shapes.bak``) and must only run while MAD is closed:
two writers on one HDF5 file over SMB is how a store gets corrupted.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import fnt_mask_store as _ms

BACKUP_SUFFIX = ".pre_rejection_shapes.bak"


def needs_repair(meta: Dict) -> bool:
    """A rejected prediction saved before rejections kept their own shape."""
    return (_ms.example_kind(meta) == "negative"
            and not (meta or {}).get("mask_is_own"))


def _box(meta: Dict, shape) -> Optional[tuple]:
    """The rejection's own (f0, f1, t0, t1) in patch coordinates, clipped."""
    from .mad_examples import _bbox_from_meta
    box = _bbox_from_meta(meta)
    if box is None:
        return None
    H, W = shape
    f0, f1, t0, t1 = box
    f0, f1 = max(0, min(f0, H)), max(0, min(f1, H))
    t0, t1 = max(0, min(t0, W)), max(0, min(t1, W))
    if f1 <= f0 or t1 <= t0:
        return None
    return f0, f1, t0, t1


def shape_from_crop(crop: Dict, meta: Dict, shape) -> Optional[np.ndarray]:
    """Place a stored prediction crop on the example's patch grid.

    Returns None unless the crop actually lands on the rejection's recorded
    box: a re-run that kept the id but not the detection must not lend its
    shape to a different call.
    """
    m = np.asarray(crop.get("mask")) > 0
    if not m.any():
        return None
    box = _box(meta, shape)
    if box is None:
        return None
    H, W = shape
    f_off = int(crop.get("f_off", 0)) - int(meta.get("patch_f_off") or 0)
    t_off = int(crop.get("t_off", 0)) - int(meta.get("patch_t_off") or 0)
    out = np.zeros(shape, dtype=bool)
    h, w = m.shape
    a0, a1 = max(0, f_off), min(H, f_off + h)
    b0, b1 = max(0, t_off), min(W, t_off + w)
    if a1 <= a0 or b1 <= b0:
        return None
    out[a0:a1, b0:b1] = m[a0 - f_off:a1 - f_off, b0 - t_off:b1 - t_off]
    f0, f1, t0, t1 = box
    inside = out[f0:f1, t0:t1].sum()
    # Most of the crop must sit in the recorded box -- the two describe the
    # same detection, so anything less means a different one.
    if inside < 0.8 * out.sum():
        return None
    return out


#: Smallest contrast (normalized spectrogram units, ~0.6 dB on the default
#: 80 dB range) a pixel needs over its column's median to count as part of
#: the traced ridge.
MIN_CONTRAST = 0.0075


def trace_shape(spec: np.ndarray, box, avoid: Optional[np.ndarray] = None,
                ) -> np.ndarray:
    """Re-trace a rejected call from its spectrogram patch, inside ``box``.

    Contrast is taken against each time column's median over frequency, not
    each row's median over time: the rejections this matters most for are a
    stationary ~21 kHz electrical tone, which a per-row floor would subtract
    away. Inside the box, Otsu's threshold on that contrast picks the bright
    structure; a 3x3 closing joins its speckle into one outline, and specks
    much smaller than the main body are dropped.

    ``avoid`` (confirmed calls sharing the window) is never claimed. The result
    is never empty when the box is: if nothing clears the threshold, the
    brightest quarter of the box is taken -- a shape, never a rectangle.
    """
    from scipy import ndimage
    spec = np.asarray(spec, dtype=np.float32)
    if spec.max() > 1.5:                     # raw uint8 patch
        spec = spec / 255.0
    H, W = spec.shape
    out = np.zeros((H, W), dtype=bool)
    if box is None:
        return out
    f0, f1, t0, t1 = box
    contrast = spec - np.median(spec, axis=0, keepdims=True)
    v = contrast[f0:f1, t0:t1]
    if v.size == 0:
        return out
    allowed = np.ones_like(v, dtype=bool)
    if avoid is not None:
        allowed = ~(np.asarray(avoid)[f0:f1, t0:t1] > 0)
    fg = (v > max(_otsu(v[allowed] if allowed.any() else v), MIN_CONTRAST))
    fg &= allowed
    if fg.any():
        # Edge-padded so the closing does not nibble pixels at the box edge
        # (the erosion half treats outside-the-array as background).
        fg = ndimage.binary_closing(np.pad(fg, 1, mode="edge"),
                                    structure=np.ones((3, 3), bool)
                                    )[1:-1, 1:-1] & allowed
        lab, n = ndimage.label(fg, structure=np.ones((3, 3), bool))
        if n > 1:
            sizes = ndimage.sum(fg, lab, index=np.arange(1, n + 1))
            keep = np.flatnonzero(sizes >= max(3, 0.1 * sizes.max())) + 1
            fg = np.isin(lab, keep)
    if not fg.any():
        pool = v[allowed] if allowed.any() else v
        fg = (v >= np.percentile(pool, 75)) & allowed
        if not fg.any():
            fg = v >= np.percentile(v, 75)
    out[f0:f1, t0:t1] = fg
    return out


def _otsu(values: np.ndarray, bins: int = 64) -> float:
    """Otsu's threshold over a 1-D sample. No skimage dependency for one call."""
    x = np.asarray(values, dtype=np.float64).ravel()
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return hi
    hist, edges = np.histogram(x, bins=bins, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2.0
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    mu0 = np.divide(m0, w0, out=np.zeros_like(m0), where=w0 > 0)
    mu1 = np.divide(m0[-1] - m0, w1, out=np.zeros_like(m0), where=w1 > 0)
    between = w0 * w1 * (mu0 - mu1) ** 2
    return float(centers[int(np.argmax(between))])


def repair_example(spec: np.ndarray, mask: np.ndarray,
                   neighbors: Optional[np.ndarray], meta: Dict,
                   crop: Optional[Dict] = None) -> Optional[Dict]:
    """The repaired ``mask``/``neighbors``/meta for one legacy negative.

    Returns None when the metadata cannot place the rejection at all (nothing
    to trace inside) -- left untouched and reported, never guessed at.
    """
    mask = np.asarray(mask) > 0
    nb = mask.copy()
    if neighbors is not None:
        nb |= np.asarray(neighbors) > 0
    own, source = None, ""
    if crop is not None:
        own = shape_from_crop(crop, meta, mask.shape)
        if own is not None:
            own &= ~nb
            source = "pred_crop" if own.any() else ""
    if not source:
        box = _box(meta, mask.shape)
        if box is None:
            return None
        own = trace_shape(spec, box, avoid=nb)
        if not own.any():
            return None
        source = "spec_trace"
    nb &= ~own
    new_meta = dict(meta)
    new_meta.update({
        "mask_is_own": True, "shape_source": source,
        "shape_repaired": datetime.now().isoformat(timespec="seconds"),
    })
    return {"own": own, "neighbors": nb, "meta": new_meta, "source": source}


@dataclass
class StoreReport:
    path: str
    negatives: int = 0
    repaired: int = 0
    from_crop: int = 0
    traced: int = 0
    neighbors_moved: int = 0
    unplaceable: List[str] = field(default_factory=list)
    backup: Optional[str] = None
    error: str = ""

    def summary(self) -> str:
        if self.error:
            return f"{os.path.basename(self.path)}: ERROR {self.error}"
        return (f"{os.path.basename(self.path)}: {self.repaired}/{self.negatives}"
                f" rejections repaired ({self.from_crop} from the model's crop,"
                f" {self.traced} re-traced; {self.neighbors_moved} carried"
                f" confirmed neighbours)"
                + (f", {len(self.unplaceable)} unplaceable"
                   if self.unplaceable else ""))


def repair_store(h5_path: str, *, dry_run: bool = True,
                 backup: bool = True) -> StoreReport:
    """Repair every legacy negative in one store.

    Reads everything read-only first and writes only if there is something to
    write, backing the file up before the first write.
    """
    _ms._require_h5()
    import h5py
    rep = StoreReport(path=h5_path)
    if not os.path.isfile(h5_path):
        rep.error = "not found"
        return rep
    plans = []
    try:
        with h5py.File(h5_path, "r") as f:
            ex = f.get("examples")
            if ex is None:
                return rep
            preds = f.get(_ms.PRED_GROUP)
            for key in ex:
                g = ex[key]
                try:
                    meta = json.loads(g.attrs.get("meta_json", "{}"))
                except Exception:
                    continue
                if _ms.example_kind(meta) != "negative":
                    continue
                rep.negatives += 1
                if not needs_repair(meta):
                    continue
                spec = g["spec"][()]
                mask = g["mask"][()]
                nb = g["neighbors"][()] if "neighbors" in g else None
                crop = None
                bid = meta.get("blob_id")
                if preds is not None and bid is not None and str(bid) in preds:
                    ds = preds[str(bid)]
                    crop = {"mask": ds[()], "f_off": int(ds.attrs.get("f_off", 0)),
                            "t_off": int(ds.attrs.get("t_off", 0))}
                res = repair_example(spec, mask, nb, meta, crop)
                if res is None:
                    rep.unplaceable.append(key)
                    continue
                rep.repaired += 1
                rep.from_crop += res["source"] == "pred_crop"
                rep.traced += res["source"] == "spec_trace"
                rep.neighbors_moved += bool(res["neighbors"].any())
                plans.append((key, res))
    except Exception as exc:
        rep.error = f"{type(exc).__name__}: {exc}"
        return rep
    if dry_run or not plans:
        return rep
    if backup:
        dest = h5_path + BACKUP_SUFFIX
        try:
            shutil.copy2(h5_path, dest)
            rep.backup = dest
        except Exception as exc:
            # Unlike a routine save, this rewrites hundreds of examples in one
            # go: no backup, no write.
            rep.error = f"backup failed, nothing written: {exc}"
            return rep
    try:
        with h5py.File(h5_path, "a") as f:
            ex = f["examples"]
            for key, res in plans:
                g = ex[key]
                del g["mask"]
                g.create_dataset("mask", data=res["own"].astype(np.uint8),
                                 compression="gzip", compression_opts=4)
                if "neighbors" in g:
                    del g["neighbors"]
                if res["neighbors"].any():
                    g.create_dataset("neighbors",
                                     data=res["neighbors"].astype(np.uint8),
                                     compression="gzip", compression_opts=4)
                g.attrs["meta_json"] = json.dumps(res["meta"])
                _ms.store_example_ridge(g)
    except Exception as exc:
        rep.error = f"{type(exc).__name__}: {exc}"
    return rep


def repair_project(project_dir: str, *, dry_run: bool = True,
                   backup: bool = True,
                   extra_dirs: Sequence[str] = ()) -> List[StoreReport]:
    """Repair every store a project's labels live in."""
    from .mad_migrate_composites import project_stores
    return [repair_store(p, dry_run=dry_run, backup=backup)
            for p in project_stores(project_dir, extra_dirs)]


def _main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("project", help="MAD project directory")
    ap.add_argument("--audio-root", action="append", default=[],
                    help="also scan this tree for *_FNT.mad (repeatable)")
    ap.add_argument("--apply", action="store_true",
                    help="write the repair (default is a dry run). Close MAD "
                         "first.")
    ap.add_argument("--no-backup", action="store_true")
    a = ap.parse_args(argv)
    reps = repair_project(a.project, dry_run=not a.apply,
                          backup=not a.no_backup, extra_dirs=a.audio_root)
    tot = dict(neg=0, rep=0, crop=0, trace=0, nb=0, bad=0, err=0)
    for r in reps:
        if r.repaired or r.unplaceable or r.error:
            print(r.summary())
        tot["neg"] += r.negatives
        tot["rep"] += r.repaired
        tot["crop"] += r.from_crop
        tot["trace"] += r.traced
        tot["nb"] += r.neighbors_moved
        tot["bad"] += len(r.unplaceable)
        tot["err"] += bool(r.error)
    print(f"\n{len(reps)} store(s): {tot['rep']} of {tot['neg']} rejections "
          f"repaired -- {tot['crop']} from the model's crop, {tot['trace']} "
          f"re-traced; {tot['nb']} had confirmed neighbours moved out of "
          f"`mask`; {tot['bad']} unplaceable; {tot['err']} store error(s)"
          + ("" if a.apply else "   (DRY RUN -- pass --apply to write)"))
    return 1 if tot["err"] else 0


if __name__ == "__main__":
    raise SystemExit(_main())
