"""Batch inference of a trained Mask Tracker model over whole videos.

Every ``step``-th frame of each video is segmented and the masks are written to a
sidecar next to the video, ``<video stem>_masks.h5`` (plain HDF5):

    attrs     video, model, step, conf, iou, imgsz, fps, n_frames, width, height,
              created, complete (True once every sampled frame is written)
    frames/   frame_idx (int32), time_s (float64), n_det (int16)  -- one row per sampled frame
    det/      frame_idx (int32), score (float32), bbox (float32, x0 y0 x1 y1), area (int32),
              cx, cy (float32, mask centroid), rle (uint32 runs, COCO column-major,
              starting with a 0-run; decode with size = (height, width))

A video whose sidecar is complete for the same model and settings is skipped, so a
run resumes where it stopped. With a 250-frame step on the 15-fps zone cameras every
sample is a keyframe, which cv2 reaches without decoding the frames between.

Used by ``python -m fnt.videoTracking.mask_tracker_cli infer-videos``, which shows
the run in a window (progress, the last frame with its masks, Pause / Stop).
"""
from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Callable, List, Optional, Sequence

import cv2
import numpy as np

SIDECAR_SUFFIX = "_masks.h5"
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def sidecar_path(video: str) -> str:
    return os.path.splitext(video)[0] + SIDECAR_SUFFIX


def resolve_weights(model: str) -> str:
    """A model run folder (weights_best.pt, else weights.pt) or a .pt file."""
    if os.path.isdir(model):
        for name in ("weights_best.pt", "weights.pt"):
            p = os.path.join(model, name)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"no weights_best.pt / weights.pt in {model}")
    if not os.path.exists(model):
        raise FileNotFoundError(model)
    return model


def model_name(model: str) -> str:
    return os.path.basename(os.path.normpath(model if os.path.isdir(model) else os.path.dirname(model)))


def sidecar_done(video: str, name: str, step: int, conf: float, imgsz: int) -> bool:
    p = sidecar_path(video)
    if not os.path.exists(p):
        return False
    import h5py
    try:
        with h5py.File(p, "r") as f:
            a = f.attrs
            return (bool(a.get("complete", False)) and str(a.get("model", "")) == name
                    and int(a.get("step", -1)) == step and abs(float(a.get("conf", -1)) - conf) < 1e-9
                    and int(a.get("imgsz", -1)) == imgsz)
    except Exception:
        return False


def drop_fragments(mask: np.ndarray, min_frac: float = 0.03) -> np.ndarray:
    """A mask without its specks: pieces under ``min_frac`` of its area are dropped."""
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i, a in enumerate(areas) if a >= min_frac * areas.sum()]
    return np.isin(lbl, keep)


def one_per_animal(masks: List[np.ndarray], scores: np.ndarray, max_overlap: float = 0.5):
    """Drop a mask lying mostly (> ``max_overlap`` of its area) inside a higher-scoring one."""
    kept = []
    for i in np.argsort(-scores):
        m = masks[i]
        area = int(m.sum())
        if area and any(np.count_nonzero(m & masks[k]) > max_overlap * area for k in kept):
            continue
        kept.append(int(i))
    return kept


def rle_runs(mask: np.ndarray) -> np.ndarray:
    """COCO uncompressed RLE runs (column-major, starting with a 0-run)."""
    flat = np.asarray(mask, np.uint8).ravel(order="F")
    if flat.size == 0:
        return np.zeros(0, np.uint32)
    change = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    runs = np.diff(np.concatenate(([0], change, [flat.size])))
    if flat[0]:
        runs = np.concatenate(([0], runs))
    return runs.astype(np.uint32)


def runs_to_mask(runs: np.ndarray, height: int, width: int) -> np.ndarray:
    flat = np.zeros(height * width, np.uint8)
    pos, val = 0, 0
    for r in runs:
        if val:
            flat[pos:pos + int(r)] = 1
        pos += int(r)
        val ^= 1
    return flat.reshape((width, height)).T.astype(bool)


def order_coarse_to_fine(videos: Sequence[str]) -> List[str]:
    """Videos in coarse-to-fine order: within each folder + camera (name prefix before the
    first underscore), in time order, positions 0, 1/2, 1/4, 3/4 ...; groups interleaved.
    A run stopped at any point has covered every folder and camera evenly."""
    def rinv(k):
        r, f = 0.0, 0.5
        while k:
            r += f * (k & 1)
            k >>= 1
            f /= 2
        return r
    groups = {}
    for v in videos:
        key = (os.path.dirname(v), os.path.basename(v).split("_")[0])
        groups.setdefault(key, []).append(v)
    keyed = []
    for key, vs in groups.items():
        for k, v in enumerate(sorted(vs)):
            keyed.append((rinv(k), key, v))
    return [v for _, _, v in sorted(keyed)]


def infer_video(model, video: str, name: str, step: int, conf: float, imgsz: int,
                device: str = "", iou: float = 0.5,
                on_frame: Optional[Callable] = None,
                should_stop: Optional[Callable[[], bool]] = None) -> dict:
    """Segment every ``step``-th frame of ``video`` with the loaded YOLO ``model`` and write
    the sidecar. ``on_frame(sample_i, n_samples, frame_bgr, masks)`` is called per sample;
    ``should_stop()`` ends the video early (the sidecar is then left incomplete)."""
    import h5py
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise IOError(f"cannot open {video}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    targets = list(range(0, n_frames, step))
    f_idx, f_n = [], []
    d_frame, d_score, d_bbox, d_area, d_cx, d_cy, d_rle = [], [], [], [], [], [], []
    complete = True
    for si, fi in enumerate(targets):
        if should_stop is not None and should_stop():
            complete = False
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        r = model.predict(frame, imgsz=imgsz, conf=conf, iou=iou, retina_masks=True,
                          device=device or None, verbose=False)[0]
        masks, scores = [], np.zeros(0, np.float32)
        if r.masks is not None and len(r.masks):
            raw = r.masks.data.cpu().numpy() > 0.5
            if raw.shape[1:] != (height, width):
                raw = np.stack([cv2.resize(m.astype(np.uint8), (width, height),
                                           interpolation=cv2.INTER_NEAREST) > 0 for m in raw])
            masks = [drop_fragments(m) for m in raw]
            scores = r.boxes.conf.cpu().numpy().astype(np.float32)
            keep = one_per_animal(masks, scores)
            masks = [masks[k] for k in keep]
            scores = scores[keep]
        f_idx.append(fi)
        f_n.append(len(masks))
        for m, s in zip(masks, scores):
            ys, xs = np.nonzero(m)
            if ys.size == 0:
                continue
            d_frame.append(fi)
            d_score.append(float(s))
            d_bbox.append((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
            d_area.append(int(ys.size))
            d_cx.append(float(xs.mean()))
            d_cy.append(float(ys.mean()))
            d_rle.append(rle_runs(m))
        if on_frame is not None:
            on_frame(si, len(targets), frame, masks)
    cap.release()
    out = sidecar_path(video)
    tmp = out + ".part"
    with h5py.File(tmp, "w") as f:
        a = f.attrs
        a["video"] = os.path.abspath(video)
        a["model"] = name
        a["step"] = step
        a["conf"] = conf
        a["iou"] = iou
        a["imgsz"] = imgsz
        a["fps"] = fps
        a["n_frames"] = n_frames
        a["width"] = width
        a["height"] = height
        a["created"] = datetime.now().isoformat(timespec="seconds")
        a["complete"] = complete
        g = f.create_group("frames")
        g.create_dataset("frame_idx", data=np.asarray(f_idx, np.int32))
        g.create_dataset("time_s", data=np.asarray(f_idx, np.float64) / (fps or 1.0))
        g.create_dataset("n_det", data=np.asarray(f_n, np.int16))
        g = f.create_group("det")
        g.create_dataset("frame_idx", data=np.asarray(d_frame, np.int32))
        g.create_dataset("score", data=np.asarray(d_score, np.float32))
        g.create_dataset("bbox", data=np.asarray(d_bbox, np.float32).reshape(-1, 4))
        g.create_dataset("area", data=np.asarray(d_area, np.int32))
        g.create_dataset("cx", data=np.asarray(d_cx, np.float32))
        g.create_dataset("cy", data=np.asarray(d_cy, np.float32))
        rle = g.create_dataset("rle", (len(d_rle),), dtype=h5py.vlen_dtype(np.uint32))
        for i, runs in enumerate(d_rle):
            rle[i] = runs
    os.replace(tmp, out)
    return {"video": video, "samples": len(f_idx), "detections": len(d_frame),
            "complete": complete, "sidecar": out}


def find_videos(items: Sequence[str]) -> List[str]:
    """Videos from files, folders (recursively) and .txt lists (one path per line)."""
    out = []
    for it in items:
        if os.path.isdir(it):
            for root, _, files in os.walk(it):
                out += [os.path.join(root, f) for f in sorted(files) if f.lower().endswith(VIDEO_EXTS)]
        elif it.lower().endswith(".txt"):
            with open(it) as fh:
                out += [ln.strip() for ln in fh if ln.strip()]
        else:
            out.append(it)
    seen, uniq = set(), []
    for v in out:
        k = os.path.normcase(os.path.abspath(v))
        if k not in seen:
            seen.add(k)
            uniq.append(v)
    return uniq


def run_batch(model_dir: str, videos: Sequence[str], step: int = 250, conf: float = 0.25,
              imgsz: Optional[int] = None, device: str = "",
              on_video: Optional[Callable] = None, on_frame: Optional[Callable] = None,
              should_stop: Optional[Callable[[], bool]] = None,
              is_paused: Optional[Callable[[], bool]] = None, log=print) -> dict:
    """Run the model over the videos (skipping those already done the same way)."""
    from ultralytics import YOLO
    weights = resolve_weights(model_dir)
    name = model_name(weights)
    if imgsz is None:
        imgsz = 1024
        cfg = os.path.join(os.path.dirname(weights), "training_config.json")
        if os.path.exists(cfg):
            import json
            imgsz = int(json.load(open(cfg)).get("imgsz", imgsz))
    model = YOLO(weights)
    todo = [v for v in videos if not sidecar_done(v, name, step, conf, imgsz)]
    log(f"[batch] {name}: {len(videos)} video(s), {len(videos) - len(todo)} already done, "
        f"{len(todo)} to go; every {step} frames, conf {conf}, imgsz {imgsz}")
    t0 = time.time()
    done = 0
    for k, v in enumerate(todo):
        while is_paused is not None and is_paused() and not (should_stop and should_stop()):
            time.sleep(0.2)
        if should_stop is not None and should_stop():
            break
        if on_video is not None:
            on_video(k, len(todo), v)
        try:
            s = infer_video(model, v, name, step, conf, imgsz, device,
                            on_frame=on_frame, should_stop=should_stop)
            done += int(s["complete"])
            log(f"[batch] {k + 1}/{len(todo)} {os.path.basename(v)}: {s['samples']} frames, "
                f"{s['detections']} masks ({time.time() - t0:.0f} s)")
        except Exception as e:      # one bad file must not end the run
            log(f"[batch] {os.path.basename(v)}: FAILED {e}")
    return {"videos": len(todo), "done": done, "seconds": time.time() - t0, "model": name}
