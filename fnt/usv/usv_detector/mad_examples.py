"""MAD per-call training-example store.

Each confirmed call is saved as a small, self-contained example under the
project's ``training_data/`` directory — a normalized spectrogram
**patch** (full frequency height x a time window around the call), its binary
mask, and a metadata sidecar. Training reads these patches directly, so the
project no longer needs the original WAVs (inference still runs on whatever
recordings the user points at).

This replaces the old WAV + sibling-PNG + committed-bands path
(:mod:`fnt.usv.usv_detector.mad_dataset`) for the GUI labeling flow.

Layout (per example ``<id>``)::

    <dataset_dir>/<id>.png        normalized spec patch, uint8 (freq x time)
    <dataset_dir>/<id>_mask.png   binary mask, uint8 {0, 255}
    <dataset_dir>/<id>.json       metadata (class, source wav, time/freq, params)
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .mad_dataset import TILE_FREQ_BINS, TILE_TIME_FRAMES, _crop_or_pad

try:
    from PIL import Image as _PILImage
    _PILImage.MAX_IMAGE_PIXELS = None
    _HAS_PIL = True
except Exception:  # pragma: no cover - PIL is a project dep but be safe
    _HAS_PIL = False


def _require_pil():
    if not _HAS_PIL:
        raise RuntimeError(
            "PIL/Pillow is required to read/write MAD training examples. "
            "pip install pillow"
        )


# ----------------------------------------------------------------------
# Write
# ----------------------------------------------------------------------
def save_example(
    dataset_dir: str,
    spec_patch: np.ndarray,
    mask_patch: np.ndarray,
    meta: Dict,
    example_id: Optional[str] = None,
    neighbors_patch: Optional[np.ndarray] = None,
) -> str:
    """Persist one confirmed-call example.

    Args:
        dataset_dir: ``<project>/training_data``.
        spec_patch:  normalized spectrogram patch, float in [0, 1] OR uint8.
        mask_patch:  binary mask of THIS call only, same shape as
                     ``spec_patch`` (any nonzero = positive).
        meta:        metadata dict (class, source_wav, time/freq, params, ...).
        example_id:  optional explicit id; a uuid-based one is generated
                     otherwise.
        neighbors_patch: other confirmed calls falling inside this patch
                     window. Kept out of ``mask_patch`` so per-call readers
                     (overlay, gallery, CSV) see one call; recombined by
                     :func:`collect_training_examples`.

    Returns the example id.

    New examples are written to the consolidated ``training_data.h5`` store;
    legacy PNG/JSON triplets remain readable (see :func:`iter_examples`).
    """
    from . import fnt_mask_store as _ms
    d = Path(dataset_dir)
    d.mkdir(parents=True, exist_ok=True)
    return _ms.td_save_example(_store_path(dataset_dir), spec_patch, mask_patch,
                               meta, example_id,
                               neighbors_patch=neighbors_patch)


def _store_path(dataset_dir: str) -> str:
    """Path to the consolidated HDF5 example store for a dataset dir."""
    return str(Path(dataset_dir) / "training_data.h5")


# ----------------------------------------------------------------------
# Read
# ----------------------------------------------------------------------
def _load_png_gray(path: Path) -> np.ndarray:
    _require_pil()
    return np.asarray(_PILImage.open(path).convert("L"))


def _iter_legacy_examples(dataset_dir: str) -> Iterator[Dict]:
    """Yield examples from legacy PNG/JSON triplets (pre-h5 storage)."""
    d = Path(dataset_dir)
    if not d.is_dir():
        return
    for meta_path in sorted(d.glob("*.json")):
        ex_id = meta_path.stem
        spec_path = d / f"{ex_id}.png"
        mask_path = d / f"{ex_id}_mask.png"
        if not spec_path.is_file() or not mask_path.is_file():
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            spec = _load_png_gray(spec_path).astype(np.float32) / 255.0
            mask = (_load_png_gray(mask_path) > 0).astype(np.float32)
        except Exception:
            continue
        yield {"meta": meta, "spec": spec, "mask": mask}


#: dataset_source_names() results, keyed by the store's files and versions.
_SOURCE_NAME_CACHE: Dict[tuple, frozenset] = {}


def dataset_source_names(dataset_dir: str) -> frozenset:
    """Names of the recordings with at least one example in the project store
    (the h5 and any legacy triplets), read from metadata only.

    reconstruct_file_mask reads the whole store to find one recording's
    examples. Inference asks that for every recording without a sidecar --
    on a folder run, every recording -- and against a 600 MB store it cost
    ~8 s a file: ~4 days of a 42,000-file run spent finding nothing. With this
    index a recording the store doesn't hold is skipped without a read.
    Cached per store version (each file's size and mtime), so a store rebuilt
    at the start of training is read afresh.
    """
    from . import fnt_mask_store as _ms
    d = Path(dataset_dir)
    h5 = Path(_store_path(dataset_dir))
    legacy = sorted(d.glob("*.json")) if d.is_dir() else []
    files = [p for p in [h5, *legacy] if p.is_file()]
    key = (str(d), tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns)
                         for p in files))
    hit = _SOURCE_NAME_CACHE.get(key)
    if hit is not None:
        return hit
    names = set()
    if h5.is_file():
        for meta in _ms.td_iter_meta(str(h5)):
            names.add(Path(str(meta.get("source_wav", ""))).name)
    for j in legacy:
        try:
            with open(j) as f:
                names.add(Path(str(json.load(f).get("source_wav", ""))).name)
        except Exception:
            continue
    _SOURCE_NAME_CACHE[key] = frozenset(names)
    return _SOURCE_NAME_CACHE[key]


def iter_examples(dataset_dir: str) -> Iterator[Dict]:
    """Yield example dicts ``{meta, spec, mask}`` from the consolidated h5 store
    **and** any legacy PNG/JSON triplets (h5 wins on id collision).

    ``spec`` is float32 in [0, 1]; ``mask`` is float32 {0, 1}.
    """
    from . import fnt_mask_store as _ms
    seen = set()
    for ex in _ms.td_iter_examples(_store_path(dataset_dir)):
        seen.add(ex["meta"].get("id"))
        yield ex
    for ex in _iter_legacy_examples(dataset_dir):
        if ex["meta"].get("id") not in seen:
            yield ex


def count_examples(dataset_dir: str, kind: Optional[str] = "label") -> int:
    """Examples in the consolidated store. Defaults to **labels only** — the
    counts that appear in run names and the UI mean "calls you drew", and hard
    negatives harvested from rejections would otherwise inflate them. Pass
    ``kind=None`` for the raw total or ``'negative'`` for the negatives."""
    from . import fnt_mask_store as _ms
    n = _ms.td_count(_store_path(dataset_dir), kind=kind)
    # add legacy triplets not already in the h5 (ids are unique uuids)
    d = Path(dataset_dir)
    if d.is_dir():
        n += sum(
            1 for j in d.glob("*.json")
            if (d / f"{j.stem}.png").is_file()
            and (d / f"{j.stem}_mask.png").is_file()
        )
    return n


def count_by_source_wav(dataset_dir: str,
                        kind: Optional[str] = "label") -> Dict[str, int]:
    """Return ``{wav_basename: n_confirmed_examples}`` cheaply (metadata only,
    no spec/mask decompression). Combines the consolidated h5 store with any
    legacy PNG/JSON triplets."""
    from . import fnt_mask_store as _ms
    out = dict(_ms.td_count_by_source_wav(_store_path(dataset_dir), kind=kind))
    d = Path(dataset_dir)
    if d.is_dir():
        for meta_path in d.glob("*.json"):
            ex_id = meta_path.stem
            if not (d / f"{ex_id}.png").is_file() or \
                    not (d / f"{ex_id}_mask.png").is_file():
                continue
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except Exception:
                continue
            bn = Path(str(meta.get("source_wav", ""))).name
            if bn:
                out[bn] = out.get(bn, 0) + 1
    return out


def list_classes(dataset_dir: str) -> List[str]:
    """Distinct class labels present in the example store (in first-seen order)."""
    seen: List[str] = []
    for ex in iter_examples(dataset_dir):
        c = ex["meta"].get("class")
        if c and c not in seen:
            seen.append(c)
    return seen


# ----------------------------------------------------------------------
# Edit / delete
# ----------------------------------------------------------------------
def delete_example(dataset_dir: str, example_id: str) -> None:
    """Remove an example from the h5 store and/or its legacy triplet."""
    from . import fnt_mask_store as _ms
    _ms.td_delete(_store_path(dataset_dir), example_id)
    d = Path(dataset_dir)
    for suffix in (".png", "_mask.png", ".json"):
        p = d / f"{example_id}{suffix}"
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass


def update_example_class(dataset_dir: str, example_id: str, new_class: str) -> None:
    """Rewrite an example's class label (h5 store, else legacy sidecar)."""
    from . import fnt_mask_store as _ms
    sp = _store_path(dataset_dir)
    if example_id in _ms.td_list_ids(sp):
        _ms.td_update_class(sp, example_id, new_class)
        return
    p = Path(dataset_dir) / f"{example_id}.json"
    if not p.is_file():
        return
    with open(p) as f:
        meta = json.load(f)
    meta["class"] = new_class
    with open(p, "w") as f:
        json.dump(meta, f, indent=2)


def update_example_mask(
    dataset_dir: str, example_id: str, mask_patch: np.ndarray,
    meta_updates: Optional[Dict] = None,
) -> None:
    """Overwrite an example's mask (h5 store, else legacy PNG) + merge metadata."""
    from . import fnt_mask_store as _ms
    sp = _store_path(dataset_dir)
    if example_id in _ms.td_list_ids(sp):
        _ms.td_update_mask(sp, example_id, mask_patch, meta_updates)
        return
    _require_pil()
    d = Path(dataset_dir)
    mask = (np.asarray(mask_patch) > 0).astype(np.uint8) * 255
    _PILImage.fromarray(mask, mode="L").save(d / f"{example_id}_mask.png")
    if meta_updates:
        p = d / f"{example_id}.json"
        meta = {}
        if p.is_file():
            with open(p) as f:
                meta = json.load(f)
        meta.update(meta_updates)
        with open(p, "w") as f:
            json.dump(meta, f, indent=2)


# ----------------------------------------------------------------------
# Legacy → HDF5 migration
# ----------------------------------------------------------------------
def has_legacy_examples(dataset_dir: str) -> bool:
    """True if any legacy PNG/JSON example triplets are present."""
    d = Path(dataset_dir)
    if not d.is_dir():
        return False
    for j in d.glob("*.json"):
        if (d / f"{j.stem}.png").is_file() and (d / f"{j.stem}_mask.png").is_file():
            return True
    return False


def migrate_legacy_to_h5(dataset_dir: str) -> int:
    """Move legacy triplets into ``training_data.h5`` and archive the originals
    to ``legacy_pre_h5/``. Returns the number of examples migrated."""
    from . import fnt_mask_store as _ms
    import shutil
    d = Path(dataset_dir)
    if not d.is_dir():
        return 0
    sp = _store_path(dataset_dir)
    archive = d / "legacy_pre_h5"
    n = 0
    for ex in list(_iter_legacy_examples(dataset_dir)):
        meta = ex["meta"]
        ex_id = meta.get("id")
        try:
            _ms.td_save_example(sp, ex["spec"], ex["mask"], meta, ex_id)
            archive.mkdir(exist_ok=True)
            for suffix in (".png", "_mask.png", ".json"):
                p = d / f"{ex_id}{suffix}"
                if p.exists():
                    shutil.move(str(p), str(archive / p.name))
            n += 1
        except Exception:
            continue
    return n


# ----------------------------------------------------------------------
# Per-file annotations (one object per saved example)
# ----------------------------------------------------------------------
def _bbox_from_meta(meta):
    """Patch-local (f0, f1, t0, t1) for an example whose mask is blank.

    Rebuilt from the absolute time/frequency span every example records, so a
    rejection can be drawn even though its mask carries no pixels. Returns
    None when the metadata cannot place it, which is what distinguishes a
    rejected detection from a painted hard negative.
    """
    try:
        sr = float(meta["sample_rate"])
        nperseg = float(meta["nperseg"])
        noverlap = float(meta["noverlap"])
        nfft = float(meta["nfft"])
        hop = nperseg - noverlap
        dt = hop / sr
        df = (sr / 2.0) / (nfft // 2)
        t0_s, t1_s = float(meta["t_start_s"]), float(meta["t_stop_s"])
        lo_hz, hi_hz = float(meta["f_low_hz"]), float(meta["f_high_hz"])
        patch_t0 = float(meta.get("patch_t0_s", 0.0))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    if dt <= 0 or df <= 0 or t1_s <= t0_s or hi_hz <= lo_hz:
        return None
    t0 = int(round((t0_s - patch_t0) / dt))
    t1 = max(t0 + 1, int(round((t1_s - patch_t0) / dt)))
    f0 = int(round(lo_hz / df))
    f1 = max(f0 + 1, int(round(hi_hz / df)))
    if t0 < 0 or f0 < 0:
        return None
    return f0, f1, t0, t1


def rebuild_training_store(training_data_dir: str, wav_paths) -> int:
    """Rebuild the consolidated training store from the per-recording sidecars.

    The ``.mad`` next to each recording is the source of truth; this store is
    only a training cache, rebuilt fresh every run so a removed or re-drawn
    label can never linger in it.

    Lives here rather than in the GUI because the CLI needs it just as much:
    ``mad train`` used to feed ``train_unet`` whatever the last GUI session
    happened to leave behind, so a headless run could train on a stale label
    set and report success. A caller that skips this is not training on the
    labels the user can see.

    Writes to a temp file and swaps it in with ``os.replace`` only after a
    complete, error-free pass, so an interruption (full disk, dropped network
    share, unreadable sidecar) leaves the previous store untouched rather than
    truncated. Raises on any failure — a partial label set must never be
    trained on silently.

    Returns the number of examples written.
    """
    import os
    from .fnt_mask_store import (
        masks_sibling_path, td_iter_examples, td_count, td_save_example)

    os.makedirs(training_data_dir, exist_ok=True)
    store = _store_path(training_data_dir)
    tmp = store + ".rebuild.tmp"

    def _drop_tmp():
        try:
            if os.path.isfile(tmp):
                os.remove(tmp)
        except Exception:
            pass

    _drop_tmp()
    n, failed = 0, []
    try:
        for fp in wav_paths:
            h5 = masks_sibling_path(fp)
            try:
                if td_count(h5) == 0:
                    continue
                examples = list(td_iter_examples(h5))
            except Exception as e:
                failed.append(f"{os.path.basename(fp)}: {e}")
                continue
            for ex in examples:
                meta = ex["meta"]
                try:
                    # neighbors MUST come across. It holds the other confirmed
                    # calls sharing this patch window, and collect_training_
                    # examples folds it back into the target. Dropping it here
                    # rebuilt a store where each tile's ground truth was one
                    # call while the tile plainly contained several — so the
                    # model was penalised for correctly finding calls the user
                    # had already labelled, and val_dice sat near zero on
                    # exactly the tiles it got right.
                    td_save_example(tmp, ex["spec"], ex["mask"], meta,
                                    meta.get("id") or None,
                                    neighbors_patch=ex.get("neighbors"))
                    n += 1
                except Exception as e:
                    failed.append(f"{os.path.basename(fp)}: {e}")
        if failed:
            raise RuntimeError(
                f"{len(failed)} label(s) could not be copied. "
                f"First error — {failed[0]}")
        if n:
            os.replace(tmp, store)      # atomic swap; old store until now
        else:
            _drop_tmp()
            if os.path.isfile(store):
                os.remove(store)
    except Exception:
        _drop_tmp()                     # keep the previous store as-is
        raise
    return n


def _examples_to_annotations(examples, wav_name, grid_shape, kinds=("label",)):
    """Shared: convert example dicts into annotation dicts clipped to *grid_shape*.

    ``kinds`` selects which example kinds to emit. It matters because a demoted
    ``'rejected'`` example still carries a real mask -- that is the whole point
    of demoting rather than deleting -- so unlike a ``'negative'`` it would not
    be filtered out by the empty-mask check below, and would otherwise come back
    as a confirmed call.
    """
    from . import fnt_mask_store as _ms
    n_freq, n_time = grid_shape
    target = Path(str(wav_name)).name
    for ex in examples:
        meta = ex["meta"]
        if Path(str(meta.get("source_wav", ""))).name != target:
            continue
        if kinds is not None and _ms.example_kind(meta) not in kinds:
            continue
        m = ex["mask"] > 0
        if not m.any():
            # A rejection saved before 2026-10-11 has no shape of its own (see
            # mad_fix_rejection_shapes, which repairs them). It is still a
            # decision the reviewer made and wants to see, so rebuild the box
            # from the metadata every example carries. A true hand-painted
            # negative has no such box and is still skipped.
            box = _bbox_from_meta(meta)
            if box is None:
                continue
            m = np.zeros_like(m)
            m[box[0]:box[1], box[2]:box[3]] = True
        else:
            # Trim the stored mask back to the call this example is about.
            #
            # A saved patch deliberately carries EVERY confirmed pixel in its
            # window, not just this call's: the patch is a time-crop with
            # context, so it routinely contains neighbouring USVs, and storing
            # only one component made the others supervised as background.
            # That union is the right training target — but it is not this
            # call's identity, and the box below is derived from the mask. Left
            # alone, one label's box grew to span every call sharing its
            # window: separate USVs came back as one connected detection, with
            # a frequency range and pixel count to match.
            #
            # The metadata still records what was actually drawn, so intersect
            # with it. Examples saved before compositing existed have a mask
            # that already equals the call, and for them this is a no-op.
            box = _bbox_from_meta(meta)
            if box is not None:
                own = np.zeros_like(m)
                own[box[0]:box[1], box[2]:box[3]] = True
                trimmed = m & own
                if trimmed.any():
                    m = trimmed
        t_off = int(meta.get("patch_t_off") or 0)
        f_off = int(meta.get("patch_f_off") or 0)
        fs = np.where(m.any(axis=1))[0]
        ts = np.where(m.any(axis=0))[0]
        lf0, lf1 = int(fs[0]), int(fs[-1]) + 1
        lt0, lt1 = int(ts[0]), int(ts[-1]) + 1
        f0 = max(0, f_off + lf0)
        f1 = min(n_freq, f_off + lf1)
        t0 = max(0, t_off + lt0)
        t1 = min(n_time, t_off + lt1)
        if f1 <= f0 or t1 <= t0:
            continue
        local = m[lf0:lf0 + (f1 - f0), lt0:lt0 + (t1 - t0)]
        ann = {
            "id": meta.get("id", ex.get("meta", {}).get("id")),
            "category": meta.get("class", ""),
            "f0": f0, "f1": f1, "t0": t0, "t1": t1,
            "mask": np.ascontiguousarray(local),
        }
        # Examples created by accepting a model prediction remember the blob_id
        # of the row they came from, so the reviewer can tell "this CSV row is
        # already on screen as a confirmed label" from "this is an accepted row
        # with no example behind it" and not draw the call twice.
        if meta.get("blob_id") is not None:
            ann["blob_id"] = meta["blob_id"]
        yield ann


def iter_file_annotations(
    dataset_dir: str, wav_name: str, grid_shape: Tuple[int, int],
):
    """Yield one annotation dict per example belonging to ``wav_name``.

    Uses a fast path that skips heavy array reads for non-matching examples.
    """
    from . import fnt_mask_store as _ms
    h5_path = _store_path(dataset_dir)
    fast = list(_ms.td_iter_file_examples(
        h5_path, Path(str(wav_name)).name, with_spec=False))
    legacy = [ex for ex in _iter_legacy_examples(dataset_dir)
              if Path(str(ex["meta"].get("source_wav", ""))).name
              == Path(str(wav_name)).name]
    seen = {ex["meta"].get("id") for ex in fast}
    combined = fast + [ex for ex in legacy
                       if ex["meta"].get("id") not in seen]
    yield from _examples_to_annotations(combined, wav_name, grid_shape)


def iter_file_rejected_annotations(
    dataset_dir: str, wav_name: str, grid_shape: Tuple[int, int],
):
    """Yield annotation dicts for this file's *demoted* (rejected) examples.

    These keep their traced mask, so a rejected call renders as a real red
    outline instead of a bounding box, and re-accepting it restores the shape
    the user actually drew rather than a filled rectangle.
    """
    from . import fnt_mask_store as _ms
    h5_path = _store_path(dataset_dir)
    fast = list(_ms.td_iter_file_examples(
        h5_path, Path(str(wav_name)).name, with_spec=False,
        kinds=_ms.REJECTED_KINDS))
    yield from _examples_to_annotations(fast, wav_name, grid_shape,
                                        kinds=_ms.REJECTED_KINDS)


# ----------------------------------------------------------------------
# Per-file confirmed-mask reconstruction (for the GUI overlay)
# ----------------------------------------------------------------------
def reconstruct_file_mask(
    dataset_dir: str, wav_name: str, grid_shape: Tuple[int, int],
    extra_stores: Optional[Sequence[str]] = None,
    kinds: Sequence[str] = ("label",),
) -> np.ndarray:
    """Rebuild the confirmed-positive mask for one source file.

    Pastes every matching example's mask back onto the full-file spec grid at
    its saved time offset. ``grid_shape`` is ``(n_freq_bins, n_time_frames)``.
    Returns a uint8 {0, 1} array.

    ``extra_stores`` are additional example stores to union in — in practice
    the recording's own ``.mad``, which is the MASTER label set: labels always
    save there, project or not, and ``rebuild_training_store`` regenerates
    ``dataset_dir`` from the sidecars at the start of every run. So the project
    store is a cache and is stale between runs — measured on one v5 recording,
    39 labels in the ``.mad`` against 28 in the cache.

    Pass ``dataset_dir=""`` to read the given stores ONLY. Callers that must
    reflect the labels as they are right now (inference's ``preserve_labels``)
    should do that when the sidecar exists, because a label deleted from the
    sidecar lingers in the cache until the next rebuild, and unioning the two
    would honour a label the user has already removed.
    """
    from . import fnt_mask_store as _ms
    n_freq, n_time = grid_shape
    out = np.zeros((n_freq, n_time), dtype=np.uint8)
    target = Path(str(wav_name)).name

    def _sources():
        # Only when the store holds this recording at all (an index lookup),
        # rather than reading the whole store to find nothing.
        if dataset_dir and target in dataset_source_names(dataset_dir):
            yield from iter_examples(dataset_dir)
        for store in (extra_stores or ()):
            if store and Path(store).is_file():
                yield from _ms.td_iter_file_examples(store, target,
                                                     with_spec=False)

    seen_ids = set()
    for ex in _sources():
        meta = ex["meta"]
        # The same example is routinely in both stores; paint it once.
        ex_id = meta.get("id")
        if ex_id is not None:
            if ex_id in seen_ids:
                continue
            seen_ids.add(ex_id)
        if Path(str(meta.get("source_wav", ""))).name != target:
            continue
        # Only real labels. A demoted ('rejected') example keeps its mask, so
        # without this it would paint in here — showing as a confirmed call in
        # the overlay, and worse, shielding its pixels from re-detection via
        # inference's preserve_labels. A rejected region is exactly a region
        # the model should be allowed to look at again.
        # ``kinds`` lets inference ask for the rejected set separately: those
        # are suppressed on a re-run (see mad_inference), but never painted as
        # labels.
        if _ms.example_kind(meta) not in kinds:
            continue
        t_off = int(meta.get("patch_t_off") or 0)
        f_off = int(meta.get("patch_f_off") or 0)
        m = ex["mask"] > 0
        h, w = m.shape
        t0 = max(0, t_off)
        t1 = min(n_time, t_off + w)
        f0 = max(0, f_off)
        f1 = min(n_freq, f_off + h)
        if t1 <= t0 or f1 <= f0:
            continue
        sub = m[(f0 - f_off):(f1 - f_off), (t0 - t_off):(t1 - t_off)]
        out[f0:f1, t0:t1][sub] = 1
    return out


# ----------------------------------------------------------------------
# Training-tile assembly
# ----------------------------------------------------------------------
def db_norm_breakdown(dataset_dir: str) -> Dict[str, int]:
    """``{db_norm: n_examples}`` across the store, from metadata only.

    A patch is normalized and quantized when it is saved, so it can never be
    re-normalized afterwards. Mixing examples saved under different rules trains
    the model on two incompatible intensity scales at once, which looks like a
    merely mediocre model rather than a broken setup — so callers check this and
    say so out loud.
    """
    from . import fnt_mask_store as _ms
    out: Dict[str, int] = {}
    for ex in _ms.td_iter_meta(_store_path(dataset_dir)):
        k = str(ex.get("db_norm") or "fixed")
        out[k] = out.get(k, 0) + 1
    return out


def _background_fill(spec: np.ndarray, mask: np.ndarray, width: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Columns that look like this patch's own noise floor.

    Used to fill the part of a training tile the patch does not cover. Filling
    with zeros was the bug: real audio always has a noise bed, so a flat-zero
    region is something the model never meets at inference, and because the
    padding also carried zero weight the model was free to output anything
    there — it learned to fire in it. Per-row median of the patch's call-free
    columns, plus that row's own jitter, gives background the model WILL see.
    """
    m = np.asarray(mask) > 0
    bg_cols = ~m.any(axis=0)
    src = spec[:, bg_cols] if int(bg_cols.sum()) >= 4 else spec
    row_med = np.median(src, axis=1)
    row_std = src.std(axis=1) * 0.7 + 1e-3
    fill = (row_med[:, None]
            + rng.normal(0.0, 1.0, (spec.shape[0], width)) * row_std[:, None])
    return np.clip(fill, 0.0, 1.0).astype(np.float32)


def _place_in_tile(spec: np.ndarray, mask: np.ndarray, tile_h: int,
                   tile_w: int, t_off: int, rng: np.random.Generator,
                   ignore: Optional[np.ndarray] = None):
    """Put a (narrower-than-tile) patch at column ``t_off`` of a fresh tile.

    Everything outside the patch is noise-floor fill supervised as background
    (weight 1, target 0). Rows are cropped to ``tile_h`` from the bottom, the
    same convention inference uses.

    ``ignore`` marks patch pixels to drop from the loss (weight 0) — a
    confirmed call sitting inside a rejection's window, which must be taught
    as neither foreground nor background.
    """
    H, W = spec.shape
    # Clamp rather than trust the caller: an offset that overruns the tile
    # would otherwise raise deep inside a training run.
    W = min(W, tile_w)
    t_off = max(0, min(int(t_off), tile_w - W))
    s = _background_fill(spec, mask, tile_w, rng)
    t = np.zeros((H, tile_w), dtype=np.float32)
    s[:, t_off:t_off + W] = spec[:, :W]
    t[:, t_off:t_off + W] = (np.asarray(mask)[:, :W] > 0)
    w = np.ones((H, tile_w), dtype=np.float32)
    if ignore is not None and np.any(ignore):
        w[:, t_off:t_off + W] = (np.asarray(ignore)[:, :W] <= 0)
    return (_crop_or_pad(s, tile_h, tile_w, 0, 0),
            _crop_or_pad(t, tile_h, tile_w, 0, 0),
            _crop_or_pad(w, tile_h, tile_w, 0, 0))


def cap_examples_per_recording(examples: List[Dict], cap: int, *,
                               seed: int = 0,
                               report: Optional[Dict] = None) -> List[Dict]:
    """Keep at most ``cap`` examples per (recording, label-or-negative).

    Order is preserved for what is kept, and the choice is seeded, so the same
    store and seed always train on the same subset.
    """
    import os as _os
    from . import fnt_mask_store as _ms
    if cap <= 0:
        return list(examples)
    buckets: Dict[Tuple[str, str], List[int]] = {}
    for i, ex in enumerate(examples):
        meta = ex.get("meta") or {}
        rec = _os.path.basename(str(meta.get("source_wav", "") or ""))
        kind = ("negative" if _ms.example_kind(meta) in ("negative", "rejected")
                else "label")
        buckets.setdefault((rec, kind), []).append(i)
    # Separate stream from tile placement, so turning the cap on or off does
    # not also reshuffle where every kept call lands in its tile.
    rng = np.random.default_rng(int(seed) + 7919)
    keep = set()
    for (rec, kind), idx in sorted(buckets.items()):
        if len(idx) <= cap:
            keep.update(idx)
            continue
        chosen = rng.choice(len(idx), size=cap, replace=False)
        keep.update(idx[j] for j in chosen)
        if report is not None:
            report.setdefault(rec or "?", {})[kind] = (cap, len(idx))
    return [ex for i, ex in enumerate(examples) if i in keep]


#: Context saved on each side of a call, in spectrogram frames, when a project
#: sets no ``context_ms``: ~65 ms at hop 128 / 250 kHz.
DEFAULT_CONTEXT_FRAMES = 128


def context_frames(context_ms, sample_rate, hop) -> int:
    """Frames of context for ``context_ms`` on a grid (``hop`` samples per
    frame); the legacy :data:`DEFAULT_CONTEXT_FRAMES` when no context or no
    usable grid is given."""
    try:
        if context_ms is None or not sample_rate or not hop or hop <= 0:
            return DEFAULT_CONTEXT_FRAMES
        return max(0, int(round(float(context_ms) / 1000.0
                                * float(sample_rate) / float(hop))))
    except (TypeError, ValueError):
        return DEFAULT_CONTEXT_FRAMES


def crop_to_context(ex: Dict, context_ms) -> Dict:
    """Narrow an example's patch to its own call +/- ``context_ms``.

    Patches are saved with the legacy context (or wider); the project's
    context decides how much of it trains. Cropping, never padding: an example
    saved narrower than the context keeps what it has. Measured from the
    example's own ``mask`` (a label's call, a rejection's own shape), so an
    example with no own pixels is left alone.
    """
    if context_ms is None:
        return ex
    m = np.asarray(ex.get("mask")) > 0
    if m.ndim != 2 or not m.any():
        return ex
    meta = dict(ex.get("meta") or {})
    hop = (meta.get("nperseg") or 0) - (meta.get("noverlap") or 0)
    ctx = context_frames(context_ms, meta.get("sample_rate"), hop)
    cols = np.flatnonzero(m.any(axis=0))
    lo = max(0, int(cols[0]) - ctx)
    hi = min(m.shape[1], int(cols[-1]) + 1 + ctx)
    if lo == 0 and hi == m.shape[1]:
        return ex
    out = dict(ex)
    for k in ("spec", "mask", "neighbors"):
        if out.get(k) is not None:
            out[k] = np.asarray(out[k])[:, lo:hi]
    meta["patch_t_off"] = int(meta.get("patch_t_off") or 0) + lo
    out["meta"] = meta
    return out


#: Spectrogram grid an example's frame offsets are counted on. Two examples'
#: patches line up only when all of these agree.
_GRID_KEYS = ("nperseg", "noverlap", "nfft", "sample_rate")


def _neighbour_placeable(ex: Dict) -> bool:
    """Can this example's window be laid over its recording's other calls?
    Needs its patch offset, its recording, and a mask."""
    import os as _os
    meta = ex.get("meta") or {}
    return (meta.get("patch_t_off") is not None
            and bool(_os.path.basename(str(meta.get("source_wav", "") or "")))
            and ex.get("mask") is not None)


def confirmed_neighbours(examples: Sequence[Dict]) -> List[Optional[np.ndarray]]:
    """For each example, the OTHER confirmed calls on its recording that fall
    inside its patch window: a bool array shaped like the patch, or ``None``.

    The ``neighbors`` an example stores is a snapshot of the calls confirmed
    at the moment it was saved. A call accepted later in the same bout is
    therefore missing from every earlier call's window, and training taught it
    as background -- the very thing ``neighbors`` exists to prevent, and
    worst in dense bouts. Measured on the 8x8 mouse project
    (mad_v1_VoleCosm, 2026-10-11): 48 of 413 labels, 6.7% of all neighbour
    pixels, 22% on one C57 trial. Rebuilding the windows from the whole store
    at training time makes the order calls were accepted in irrelevant.

    Only labels (painted calls and accepted detections) are neighbours;
    rejections never are. A call's own pixels are placed by its stored
    ``patch_t_off``/``patch_f_off``, so only examples cut on the same
    recording and the same grid are compared. Recordings are matched by
    basename, the same key the train/val split groups on.
    """
    import bisect
    import os as _os
    from . import fnt_mask_store as _ms

    def _rec(meta):
        return _os.path.basename(str(meta.get("source_wav", "") or ""))

    def _placed(meta):
        return meta.get("patch_t_off") is not None

    groups: Dict[tuple, List[int]] = {}
    for i, ex in enumerate(examples):
        meta = ex.get("meta") or {}
        if (_ms.example_kind(meta) == "label" and _placed(meta) and _rec(meta)
                and ex.get("mask") is not None):
            key = (_rec(meta),) + tuple(meta.get(k) for k in _GRID_KEYS)
            groups.setdefault(key, []).append(i)
    # Per group: labels sorted by start frame, so each window only visits the
    # labels that can reach it rather than every label on the recording.
    index = {}
    for key, idx in groups.items():
        idx = sorted(idx, key=lambda j: int(examples[j]["meta"]["patch_t_off"]))
        starts = [int(examples[j]["meta"]["patch_t_off"]) for j in idx]
        widest = max(np.asarray(examples[j]["mask"]).shape[1] for j in idx)
        index[key] = (idx, starts, widest)

    out: List[Optional[np.ndarray]] = [None] * len(examples)
    for i, ex in enumerate(examples):
        meta = ex.get("meta") or {}
        if not _neighbour_placeable(ex):
            continue
        found = index.get((_rec(meta),) + tuple(meta.get(k) for k in _GRID_KEYS))
        if found is None:
            continue
        idx, starts, widest = found
        H, W = np.asarray(ex["mask"]).shape
        a_t = int(meta["patch_t_off"])
        a_f = int(meta.get("patch_f_off") or 0)
        win = None
        lo = bisect.bisect_left(starts, a_t - widest + 1)
        hi = bisect.bisect_left(starts, a_t + W)
        for j in idx[lo:hi]:
            if j == i:
                continue
            bm = examples[j]["meta"]
            b = np.asarray(examples[j]["mask"]) > 0
            b_t = int(bm["patch_t_off"])
            b_f = int(bm.get("patch_f_off") or 0)
            t0, t1 = max(a_t, b_t), min(a_t + W, b_t + b.shape[1])
            f0, f1 = max(a_f, b_f), min(a_f + H, b_f + b.shape[0])
            if t1 <= t0 or f1 <= f0:
                continue
            piece = b[f0 - b_f:f1 - b_f, t0 - b_t:t1 - b_t]
            if not piece.any():
                continue
            if win is None:
                win = np.zeros((H, W), dtype=bool)
            win[f0 - a_f:f1 - a_f, t0 - a_t:t1 - a_t] |= piece
        out[i] = win
    return out


def collect_training_examples(
    dataset_dir: str,
    tile_time_frames: int = TILE_TIME_FRAMES,
    tile_freq_bins: int = TILE_FREQ_BINS,
    progress=None,
    return_groups: bool = False,
    placements: int = 3,
    seed: int = 0,
    max_per_recording: int = 0,
    cap_report: Optional[Dict] = None,
    context_ms: Optional[float] = None,
):
    """Assemble ``(specs, targets, weights)`` stacks from saved examples.

    ``context_ms`` crops every patch to its call +/- that much (see
    :func:`crop_to_context`); None trains each patch as saved.

    ``max_per_recording`` (0 = off) keeps at most that many examples from any
    one recording, labels and hard negatives counted separately, chosen at
    random under ``seed``. Pass a dict as ``cap_report`` to have it filled with
    ``{recording: {'label': (kept, total), ...}}`` for each recording the cap
    actually trimmed. See ``UNetTrainingConfig.max_examples_per_recording``
    for why whole examples, and never pixels, are what gets dropped.

    Each example patch is sliced into one or more tiles of shape
    ``(tile_freq_bins, tile_time_frames)``. The whole patch is supervised, so
    ``weight`` is 1 everywhere a tile overlaps the patch (and 0 in any pad).
    Mirrors the return contract of
    :func:`fnt.usv.usv_detector.mad_dataset.collect_training_tiles`.

    With ``return_groups``, a fourth value is returned: a list of
    ``(source_wav, example_id)`` pairs, one per tile, identifying where each
    tile came from. A train/val split MUST be made over those groups rather
    than over tiles — one call yields several tiles, so a tile-level shuffle
    would put the same call on both sides of the split. See
    :func:`fnt.usv.usv_detector.mad_training.grouped_split`.

    **Placement.** Each patch is placed ``placements`` times at a random column
    offset within the tile (or, for patches wider than a tile, as random
    windows into it). This is not decoration. Tiles used to be cut from column
    0 with zero padding on the right, so every call sat at the same offset with
    a flat-zero region beside it — a layout that never occurs in a real
    recording. The model learned the task (Dice ~0.9 inside the patch) but
    position-locked: on full-file inference it recognized a call only when it
    happened to land at that one offset, and fired mostly into the padding it
    had never been penalized for. In-sample F1 was 0.09. Random placement over
    a noise-floor background is what makes a model trained here usable outside
    the tile it was trained on.
    """
    from . import fnt_mask_store as _ms
    examples = list(iter_examples(dataset_dir))
    # Neighbours from the whole store, not the snapshot each example took when
    # it was saved (see confirmed_neighbours) -- and before the cap: a call the
    # cap leaves out of training is still a confirmed call in its neighbours'
    # windows.
    #
    # The rebuild REPLACES the snapshot wherever an example can be placed on
    # its recording. Snapshots were read from the canvas, which also held
    # pending and rejected detections, so they carried pixels nobody accepted
    # -- 15,368 of 836,066 snapshot pixels in mad_v5_VC (60 labels, 2,895 on
    # the 21 kHz tone) -- and a label's neighbours train as calls. Every pixel
    # of a call confirmed now is in the rebuild; anything only the snapshot
    # holds was pending, rejected, or a label since deleted. An example that
    # cannot be placed keeps its snapshot.
    for ex, rebuilt in zip(examples, confirmed_neighbours(examples)):
        if not _neighbour_placeable(ex):
            continue
        ex["neighbors"] = rebuilt
    # After the neighbours, which place calls by the full patch's offsets.
    if context_ms is not None:
        examples = [crop_to_context(ex, context_ms) for ex in examples]
    if max_per_recording and max_per_recording > 0:
        examples = cap_examples_per_recording(
            examples, int(max_per_recording), seed=seed, report=cap_report)
    specs: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    weights: List[np.ndarray] = []
    groups: List[Tuple[str, str]] = []
    rng = np.random.default_rng(seed)
    placements = max(1, int(placements))
    n = len(examples)
    for i, ex in enumerate(examples):
        meta = ex["meta"]
        if progress is not None:
            progress(i, n, meta.get("id", ""))
        spec = ex["spec"]
        mask = ex["mask"]
        # Other confirmed calls sharing this patch window. Stored apart from
        # ``mask`` so the overlay sees one call per example, and folded back in
        # here because training is the one consumer that wants the whole window.
        nb = ex.get("neighbors")
        nb = None if nb is None else (np.asarray(nb) > 0)
        ignore = None
        # A rejected call trains as a HARD NEGATIVE: its patch is supervised
        # with an all-zero target, so every bright pixel a human refused is
        # explicitly taught as "not a call".
        #
        # This is the only thing in the training set that can teach *shape*.
        # With positives alone, "bright ⇒ call" fits the data perfectly and the
        # model has no reason to learn that a blob of noise is the wrong shape
        # for a call -- the surrounding quiet only ever teaches "quiet ⇒ not a
        # call". Rejections are bright things that are not calls, which is
        # exactly the counterexample that forces shape to matter.
        if _ms.example_kind(meta) in ("negative", "rejected"):
            mask = np.zeros_like(mask)
            # ...but a rejection's window can contain a CONFIRMED call, and
            # zeroing the whole patch taught the model that a call a human had
            # just accepted was background. Those pixels are excluded from the
            # loss instead: the rejection still teaches its own shape, and the
            # confirmed call in the corner is simply not asked about.
            ignore = nb
        elif nb is not None:
            # A neighbour's traced pixels are confirmed foreground, so they are
            # supervised positive rather than ignored. Leaving them at 0 was
            # the original problem: the weight mask is 1 across the patch, so
            # an unpainted neighbour trained as background.
            mask = np.maximum(mask, nb.astype(mask.dtype))
        # Provenance for the split. Fall back to the positional index for
        # examples with no id, so distinct calls never collapse into one group.
        src = Path(str(meta.get("source_wav", ""))).name or "<unknown>"
        eid = str(meta.get("id") or f"<example {i}>")
        H, W = spec.shape
        for _k in range(placements):
            if W <= tile_time_frames:
                off = int(rng.integers(0, tile_time_frames - W + 1))
                s_t, t_t, w_t = _place_in_tile(
                    spec, mask, tile_freq_bins, tile_time_frames, off, rng,
                    ignore=ignore)
            else:
                # Wider than a tile (a long call): a random window into it,
                # fully supervised — inference cuts long calls this way too.
                start = int(rng.integers(0, W - tile_time_frames + 1))
                s_t = _crop_or_pad(spec, tile_freq_bins, tile_time_frames, 0, start)
                t_t = _crop_or_pad(mask, tile_freq_bins, tile_time_frames, 0, start)
                w_t = np.ones((tile_freq_bins, tile_time_frames), dtype=np.float32)
                if ignore is not None and ignore.any():
                    ig = _crop_or_pad(ignore.astype(np.float32),
                                      tile_freq_bins, tile_time_frames, 0, start)
                    w_t = w_t * (ig <= 0)
            specs.append(s_t)
            targets.append(t_t)
            weights.append(w_t)
            groups.append((src, eid))
    if progress is not None:
        progress(n, n, "done")
    if not specs:
        empty = np.zeros((0, tile_freq_bins, tile_time_frames), dtype=np.float32)
        out = (empty, empty.copy(), empty.copy())
        return (*out, []) if return_groups else out
    out = (
        np.stack(specs, axis=0),
        np.stack(targets, axis=0),
        np.stack(weights, axis=0),
    )
    return (*out, groups) if return_groups else out
