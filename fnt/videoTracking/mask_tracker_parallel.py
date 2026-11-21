"""Track several videos at once: one worker process per video.

Full-video inference is held back by the CPU, not the GPU. On the lab PC
(RTX 4070 Ti SUPER, i9-14900K) one YOLO11l-seg run on 1080p open-field video
reached ~23 fps with the GPU 30% busy: reading, the model's pre- and
post-processing, tracking, drawing and writing run one after another for each
frame. Several videos in separate processes overlap all of that. They are
processes, not threads, because most of the per-frame work is Python, which
threads would serialise.

Each worker runs :func:`infer_one_video`, the same code the one-video-at-a-time
path runs, so a video's outputs do not depend on how many ran beside it.

Nothing here imports Qt; ``mask_tracker_parallel_view`` puts a run on screen.
"""
from __future__ import annotations

import io
import multiprocessing as mp
import os
import queue
import subprocess
import sys
import time
import traceback
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# What one worker needs, for choosing how many run at once. Measured on the lab
# PC (one YOLO11l-seg worker with masks: ~1.3 GB of GPU memory, about 2.5 GB of
# RAM); rounded up so the estimate errs towards fewer workers.
GPU_MB_PER_WORKER = 1500
GPU_MB_RESERVE = 2000          # left for the display and anything else on the GPU
RAM_GB_PER_WORKER = 3.0
CORES_PER_WORKER = 3           # decode, Python, encode
CORES_PER_CPU_WORKER = 6       # a worker running the model on the CPU as well
MAX_WORKERS = 8

PREVIEW_WIDTH = 480            # px; the preview tiles are small
PREVIEW_INTERVAL_S = 0.25      # at most four preview frames a second per video
PROGRESS_INTERVAL_S = 0.5


# ---------------------------------------------------------------------------
# Choosing how many videos run at once
# ---------------------------------------------------------------------------

def _physical_cores() -> int:
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except Exception:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def _available_ram_gb() -> Optional[float]:
    try:
        import psutil
        return psutil.virtual_memory().available / 1024 ** 3
    except Exception:
        return None


def _gpu_free_mb() -> Optional[int]:
    """Free memory on GPU 0, from nvidia-smi (asking torch would put a CUDA
    context into the calling process for good)."""
    try:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, creationflags=flags)
        if out.returncode == 0 and out.stdout.strip():
            return int(out.stdout.strip().splitlines()[0])
    except Exception:
        pass
    return None


def auto_worker_count(n_videos: int, device: str = "auto",
                      cores: Optional[int] = None,
                      ram_gb: Optional[float] = None,
                      gpu_free_mb: Optional[int] = None) -> Tuple[int, str]:
    """How many videos to track at once on this machine, and why.

    The smallest of: physical cores / cores per worker, available RAM / RAM per
    worker, free GPU memory / GPU memory per worker (when the model runs on the
    GPU), MAX_WORKERS and the number of videos. The measured values can be
    passed in (tests); otherwise they are read from the machine.
    """
    if n_videos <= 0:
        return 0, "no videos"
    on_cpu = device == "cpu"
    cores = _physical_cores() if cores is None else cores
    ram_gb = _available_ram_gb() if ram_gb is None else ram_gb
    if not on_cpu and gpu_free_mb is None:
        gpu_free_mb = _gpu_free_mb()

    limits = {"CPU": max(1, cores // (CORES_PER_CPU_WORKER if on_cpu else CORES_PER_WORKER))}
    if ram_gb is not None:
        limits["RAM"] = max(1, int(ram_gb // RAM_GB_PER_WORKER))
    if not on_cpu and gpu_free_mb is not None:
        limits["GPU"] = max(1, (gpu_free_mb - GPU_MB_RESERVE) // GPU_MB_PER_WORKER)
    limits["max"] = MAX_WORKERS
    limits["videos"] = n_videos
    n = int(min(limits.values()))
    why = ", ".join(f"{k} {v}" for k, v in limits.items())
    return n, f"{n} at once (limits: {why})"


# ---------------------------------------------------------------------------
# One video: shared by the one-at-a-time path and the parallel workers
# ---------------------------------------------------------------------------

def load_classifier(classifier_dir: Optional[str]):
    """Load the behavior classifier.

    Returns ``(model, class_names, device, scene_cfg)``, where scene_cfg is
    the stored configuration for a scene-aware checkpoint and None for a
    legacy single-silhouette one. Returns None outright when nothing
    loadable is there.
    """
    if not classifier_dir:
        return None

    import json
    from .behavior_classifier import load_scene_classifier
    scene = load_scene_classifier(classifier_dir)
    if scene is not None:
        model, class_names, device, cfg = scene
        n_soc = cfg.get("n_social_samples")
        print(f"[Classifier Inference] Loaded scene-aware "
              f"{cfg.get('backbone', '?')} ({len(class_names)} classes: "
              f"{', '.join(class_names)}) on {device.upper()}"
              + (f"; trained with {n_soc} multi-animal samples"
                 if n_soc is not None else ""))
        return model, class_names, device, cfg

    import torch
    from torchvision import models
    cfg_path = os.path.join(classifier_dir, "classifier_config.json")
    weights_path = os.path.join(classifier_dir, "best_classifier.pth")
    if not os.path.isfile(cfg_path) or not os.path.isfile(weights_path):
        print(f"[Classifier Inference] Missing config or weights in {classifier_dir}")
        return None

    with open(cfg_path) as f:
        cfg = json.load(f)
    class_names = cfg.get("class_names", [])
    n_classes = cfg.get("n_classes", len(class_names))
    backbone = cfg.get("backbone", "ResNet-18")

    from torch import nn
    if backbone == "ResNet-34":
        model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, n_classes)
    elif backbone == "MobileNetV3":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_classes)
    else:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, n_classes)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()

    print(f"[Classifier Inference] Loaded legacy single-silhouette {backbone} "
          f"({n_classes} classes: {', '.join(class_names)}) on {device.upper()}. "
          f"Social categories cannot be represented by this checkpoint — "
          f"retrain to get a scene-aware model.")
    return model, class_names, device, None


def apply_min_bout_filter(rows, min_bout):
    """Replace behavior bouts shorter than min_bout frames with NC.

    Groups consecutive same-behavior predictions per object. Bouts
    shorter than min_bout are relabeled NC to suppress flickering.
    """
    from itertools import groupby

    per_obj = {}
    for i, r in enumerate(rows):
        per_obj.setdefault(r["object_id"], []).append(i)

    for obj_id, indices in per_obj.items():
        labels = [rows[i]["behavior"] for i in indices]
        for _, grp in groupby(enumerate(labels), key=lambda x: x[1]):
            grp_list = list(grp)
            if len(grp_list) < min_bout and grp_list[0][1] != "NC":
                for local_pos, _ in grp_list:
                    rows[indices[local_pos]]["behavior"] = "NC"


def infer_one_video(video_path: str, model_dir: str, config, cls_info=None, *,
                    cls_window: int = 15, nc_threshold: float = 0.5,
                    uncertain_gap: float = 0.10, min_bout: int = 5,
                    create_tracked_video: bool = True,
                    progress: Optional[Callable] = None,
                    should_stop: Optional[Callable] = None,
                    stop_requested: Optional[Callable] = None,
                    frame_callback: Optional[Callable] = None,
                    run_info: Optional[Dict] = None) -> Dict:
    """Detect, track and (with a classifier) label behavior in one video.

    ``should_stop`` is polled every frame (it may block while paused);
    ``stop_requested`` is the plain, non-blocking question asked once the
    video ends. A stopped video's output folder is removed, since it holds a
    partial run, and the result says ``stopped``. ``run_info`` goes into the
    video's run_settings.json (with the classifier settings, when one runs).
    """
    from collections import deque
    from .mask_tracker_inference import run_inference_on_video

    cls_model = cls_class_names = cls_device = cls_scene_cfg = None
    torch = None
    if cls_info:
        cls_model, cls_class_names, cls_device, cls_scene_cfg = cls_info
        import torch
        from .behavior_classifier import predict_scene
        from .silhouette_extractor import (
            generate_composite, generate_scene_composite, pairwise_features,
            pairwise_vector,
        )

    mask_buffers: dict = {}
    current_labels: dict = {}
    behavior_rows: list = []

    def _track_cb(matched, frame_idx, fps):
        if cls_model is None:
            return None

        # Every known object gets an entry every frame, present or
        # not, so the per-object buffers stay aligned in time and a
        # neighbour's position can be read at the same index as the
        # focal animal's. Appending only for detected objects would
        # silently shear the windows apart whenever one is missed.
        for obj_id in set(mask_buffers) | set(matched):
            if obj_id not in mask_buffers:
                mask_buffers[obj_id] = deque(maxlen=cls_window)
            det = matched.get(obj_id)
            entry = None
            mask = det.get("mask") if det is not None else None
            if mask is not None:
                bbox = det["bbox"]
                h_f, w_f = mask.shape[:2]
                x1 = max(0, int(round(float(bbox[0]))))
                y1 = max(0, int(round(float(bbox[1]))))
                x2 = min(w_f, int(round(float(bbox[2]))))
                y2 = min(h_f, int(round(float(bbox[3]))))
                if x2 > x1 and y2 > y1:
                    entry = {
                        "crop": mask[y1:y2, x1:x2],
                        "bbox": (x1, y1, x2, y2),
                    }
            mask_buffers[obj_id].append(entry)

        # Snapshot once rather than per focal animal: the inner
        # loop below is quadratic in object count otherwise.
        windows = {oid: list(buf) for oid, buf in mask_buffers.items()}
        for obj_id, window in windows.items():
            if all(e is None for e in window):
                # Gone for a whole window. Drop it so a long video
                # does not accumulate dead buffers, and so it stops
                # counting as a neighbour of anyone.
                mask_buffers.pop(obj_id, None)
                current_labels.pop(obj_id, None)

        for obj_id in list(mask_buffers.keys()):
            window = windows.get(obj_id) or []
            if len(window) < cls_window:
                continue

            if cls_scene_cfg is not None:
                neighbors = []
                for i in range(len(window)):
                    neighbors.append([
                        other[i] for oid, other in windows.items()
                        if oid != obj_id and oid in mask_buffers
                        and len(other) == len(window)
                        and other[i] is not None
                    ])
                scene = generate_scene_composite(
                    window, neighbors, output_size=(128, 128),
                )
                vec = pairwise_vector(
                    pairwise_features(window, neighbors, fps=fps)
                )
                probs_np = predict_scene(
                    cls_model, cls_scene_cfg, scene, vec, cls_device,
                )
                probs = torch.from_numpy(probs_np)
            else:
                crops = [w["crop"] if w is not None else None for w in window]
                boxes = [w["bbox"] if w is not None else None for w in window]
                composite = generate_composite(
                    crops, output_size=(128, 128), bboxes=boxes)
                img = composite.astype(np.float32) / 255.0
                inp = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(cls_device)
                with torch.no_grad():
                    probs = torch.softmax(cls_model(inp), dim=1)[0].cpu()

            sorted_probs, sorted_idx = probs.sort(descending=True)
            conf = sorted_probs[0].item()
            pred_idx = sorted_idx[0].item()
            gap = (sorted_probs[0] - sorted_probs[1]).item() if len(sorted_probs) > 1 else conf

            if conf < nc_threshold or gap < uncertain_gap:
                beh_name = "NC"
            else:
                beh_name = cls_class_names[pred_idx]
            current_labels[obj_id] = (beh_name, conf)

            behavior_rows.append({
                "frame": frame_idx,
                "object_id": obj_id,
                "behavior": beh_name,
                "confidence": round(conf, 4),
                "gap": round(gap, 4),
            })

        return current_labels if current_labels else None

    result = run_inference_on_video(
        video_path,
        model_dir,
        config,
        progress=progress,
        should_stop=should_stop,
        frame_callback=frame_callback,
        track_callback=_track_cb if cls_model else None,
        create_tracked_video=create_tracked_video,
        run_info=dict(run_info or {}, **({"classifier_settings": {
            "cls_window": cls_window, "nc_threshold": nc_threshold,
            "uncertain_gap": uncertain_gap, "min_bout": min_bout}} if cls_model else {})),
    )
    result["video_path"] = video_path

    if stop_requested is not None and stop_requested():
        out_dir = result.get("output_dir", "")
        if out_dir and os.path.isdir(out_dir):
            import shutil
            shutil.rmtree(out_dir, ignore_errors=True)
            print(f"[MTT Inference] Removed incomplete output: {out_dir}")
        result["stopped"] = True
        return result

    if behavior_rows:
        per_obj_first = {}
        for row in behavior_rows:
            oid = row["object_id"]
            if oid not in per_obj_first:
                per_obj_first[oid] = row
        backfill = []
        for oid, first_row in per_obj_first.items():
            first_frame = first_row["frame"]
            for fi in range(first_frame):
                backfill.append({
                    "frame": fi,
                    "object_id": oid,
                    "behavior": first_row["behavior"],
                    "confidence": first_row["confidence"],
                    "gap": first_row["gap"],
                })
        if backfill:
            behavior_rows = backfill + behavior_rows

        apply_min_bout_filter(behavior_rows, min_bout)

        import csv
        out_dir = result.get("output_dir", "")
        beh_csv = os.path.join(out_dir, "behavior_predictions.csv")
        fieldnames = ["frame", "object_id", "behavior", "confidence", "gap"]
        behavior_rows.sort(key=lambda r: (r["object_id"], r["frame"]))
        with open(beh_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(behavior_rows)
        result["behavior_csv"] = beh_csv
        n_pred = len(behavior_rows)
        n_nc = sum(1 for r in behavior_rows if r["behavior"] == "NC")
        n_objs = len(set(r["object_id"] for r in behavior_rows))
        print(f"[Classifier Inference] Saved {n_pred} predictions "
              f"({n_nc} NC) for {n_objs} objects → {beh_csv}")

    return result


# ---------------------------------------------------------------------------
# The worker processes
# ---------------------------------------------------------------------------

def shrink_frame(rgb: np.ndarray, width: int = PREVIEW_WIDTH) -> np.ndarray:
    """The frame scaled down to ``width`` px (aspect kept); smaller frames as-is."""
    import cv2
    h, w = rgb.shape[:2]
    if w <= width:
        return np.ascontiguousarray(rgb)
    return cv2.resize(rgb, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)


class _QueueLog(io.TextIOBase):
    """A worker's stdout and stderr: each line becomes a ("log", ...) message."""

    def __init__(self, msgs, wid):
        self._msgs = msgs
        self._wid = wid

    def write(self, s):
        if s and s.strip():
            for line in s.rstrip().splitlines():
                if line.strip():
                    self._msgs.put(("log", self._wid, line))
        return len(s) if s else 0

    def flush(self):
        pass


def _worker_main(wid: int, jobs, msgs, stop_ev, pause_ev, model_dir: str,
                 config_dict: Dict, opts: Dict) -> None:
    """Take videos off ``jobs`` until it is empty or a stop is asked for.

    Messages, all tuples starting with a kind and this worker's id:
    ("started", wid, idx, path), ("progress", wid, idx, frame, total, fps),
    ("frame", wid, idx, rgb, frame), ("finished", wid, idx, result),
    ("stopped", wid, idx, path), ("error", wid, idx, text), ("log", wid, line),
    ("exit", wid).
    """
    sys.stdout = sys.stderr = _QueueLog(msgs, wid)
    try:
        from fnt import tool_host
        tool_host.install_faulthandler(f"Mask Tracker worker {wid + 1}")
    except Exception:
        pass
    threads = max(1, int(opts.get("threads", 2)))
    try:
        import cv2
        cv2.setNumThreads(threads)
    except Exception:
        pass
    try:
        import torch
        torch.set_num_threads(threads)
    except Exception:
        pass

    try:
        from .mask_tracker_inference import MaskInferenceConfig
        config = MaskInferenceConfig(**config_dict)
        cls_info = load_classifier(opts.get("classifier_dir"))
    except Exception as e:
        traceback.print_exc()
        msgs.put(("error", wid, -1, f"worker {wid + 1} could not start: {e}"))
        msgs.put(("exit", wid))
        return

    def should_stop():
        while pause_ev.is_set() and not stop_ev.is_set():
            time.sleep(0.2)
        return stop_ev.is_set()

    show_preview = bool(opts.get("show_preview", True))
    preview_width = int(opts.get("preview_width", PREVIEW_WIDTH))
    preview_interval = float(opts.get("preview_interval", PREVIEW_INTERVAL_S))

    while not stop_ev.is_set():
        try:
            idx, video = jobs.get(timeout=2.0)
        except queue.Empty:
            break
        msgs.put(("started", wid, idx, video))
        st = {"prog": 0.0, "prev": 0.0, "t1": None, "f1": 0}

        def progress(f, t, idx=idx, st=st):
            now = time.time()
            if st["t1"] is None:          # fps from the first frame on, not the model load
                st["t1"], st["f1"] = now, f
            if now - st["prog"] >= PROGRESS_INTERVAL_S or f >= t:
                st["prog"] = now
                dt = now - st["t1"]
                fps = (f - st["f1"]) / dt if dt > 0 else 0.0
                msgs.put(("progress", wid, idx, f, t, fps))

        def on_frame(rgb, f, idx=idx, st=st):
            now = time.time()
            if now - st["prev"] < preview_interval:
                return
            st["prev"] = now
            msgs.put(("frame", wid, idx, shrink_frame(rgb, preview_width), f))

        try:
            result = infer_one_video(
                video, model_dir, config, cls_info,
                cls_window=opts.get("cls_window", 15),
                nc_threshold=opts.get("nc_threshold", 0.5),
                uncertain_gap=opts.get("uncertain_gap", 0.10),
                min_bout=opts.get("min_bout", 5),
                create_tracked_video=opts.get("create_tracked_video", True),
                progress=progress,
                should_stop=should_stop,
                stop_requested=stop_ev.is_set,
                frame_callback=on_frame if show_preview else None,
                run_info={"classifier": opts.get("classifier_dir"),
                          "parallel": {"workers": opts.get("n_workers"), "worker": wid + 1,
                                       "auto": opts.get("auto_reason")}},
            )
        except Exception as e:
            traceback.print_exc()
            msgs.put(("error", wid, idx, f"{os.path.basename(video)}: {e}"))
            continue
        if result.get("stopped"):
            msgs.put(("stopped", wid, idx, video))
            break
        msgs.put(("finished", wid, idx, result))
    msgs.put(("exit", wid))


class ParallelInference:
    """Runs a list of videos through ``n_workers`` worker processes.

    Not a thread: the owner calls :meth:`messages` in a loop (see
    ``mask_tracker_parallel_view.ParallelInferenceWorker``) until
    :meth:`done`. Every worker message must be read, or a worker blocks on a
    full pipe and never exits.
    """

    def __init__(self, video_paths: List[str], model_dir: str, config_dict: Dict,
                 n_workers: int, opts: Optional[Dict] = None):
        self.video_paths = list(video_paths)
        self.model_dir = model_dir
        self.config_dict = dict(config_dict)
        self.n_workers = max(1, min(int(n_workers), len(self.video_paths)))
        self.opts = dict(opts or {})
        self.opts.setdefault("threads", max(1, _physical_cores() // self.n_workers))
        self.opts["n_workers"] = self.n_workers
        self._procs: List = []
        self._exited: set = set()
        self._current: Dict[int, int] = {}      # worker -> index of the video it is on

    def start(self) -> None:
        ctx = mp.get_context("spawn")
        self._jobs = ctx.Queue()
        for i, v in enumerate(self.video_paths):
            self._jobs.put((i, v))
        self._msgs = ctx.Queue()
        self._stop = ctx.Event()
        self._pause = ctx.Event()
        for wid in range(self.n_workers):
            p = ctx.Process(
                target=_worker_main,
                args=(wid, self._jobs, self._msgs, self._stop, self._pause,
                      self.model_dir, self.config_dict, self.opts),
                name=f"mtt-worker-{wid + 1}", daemon=True)
            p.start()
            self._procs.append(p)

    def messages(self, timeout: float = 0.1) -> List[tuple]:
        """Every message waiting (blocking up to ``timeout`` for the first).

        A worker that died without saying so (a native crash) is reported once
        as an error on the video it had, then as exited.
        """
        out = []
        try:
            out.append(self._msgs.get(timeout=timeout))
            while True:
                out.append(self._msgs.get_nowait())
        except queue.Empty:
            pass
        for m in out:
            if m[0] == "started":
                self._current[m[1]] = m[2]
            elif m[0] in ("finished", "stopped", "error"):
                self._current.pop(m[1], None)
            elif m[0] == "exit":
                self._exited.add(m[1])
        if not out:
            for wid, p in enumerate(self._procs):
                if wid not in self._exited and not p.is_alive():
                    self._exited.add(wid)
                    idx = self._current.pop(wid, -1)
                    name = os.path.basename(self.video_paths[idx]) if idx >= 0 else "?"
                    out.append(("error", wid, idx,
                                f"worker {wid + 1} ended unexpectedly (exit code "
                                f"{p.exitcode}) during {name}"))
                    out.append(("exit", wid))
        return out

    def done(self) -> bool:
        return len(self._exited) >= len(self._procs)

    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()

    @property
    def paused(self) -> bool:
        return self._pause.is_set()

    def stop(self) -> None:
        """Ask every worker to stop after its current frame; each removes its
        video's partial output folder, then exits."""
        self._stop.set()
        self._pause.clear()

    def terminate(self, timeout: float = 10.0) -> None:
        """Stop, wait up to ``timeout`` for the workers to clean up, then kill
        any still running (closing the window must not leave them behind)."""
        self.stop()
        t_end = time.time() + timeout
        while time.time() < t_end and not self.done():
            self.messages(timeout=0.1)
        for p in self._procs:
            if p.is_alive():
                p.terminate()
