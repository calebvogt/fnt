"""Mask R-CNN inference and multi-object tracking for the Mask Tracker Pipeline.

Loads a trained Mask R-CNN checkpoint and runs per-frame inference on video
files, then assembles trajectories via centroid-distance matching (Hungarian
or greedy). Produces trajectory CSVs and annotated output video.

Heavy deps (torch, torchvision, scipy) are imported lazily.
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


TRACK_COLORS = [
    (0, 120, 255), (255, 80, 0), (0, 200, 80), (200, 0, 200),
    (255, 255, 0), (0, 255, 255), (255, 0, 100), (128, 200, 0),
    (100, 0, 255), (255, 128, 0), (0, 128, 255), (200, 200, 0),
]


@dataclass
class MaskInferenceConfig:
    model_path: str = ""
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.5
    device: str = "auto"
    max_detections: int = 1
    # Analysed frames a track may go undetected before it ends. 0 = never;
    # -1 = auto: "bytetrack" 5 (it ends at the 6th; never with max_detections 1),
    # "centroid" never ends them.
    max_disappeared_frames: int = -1
    iou_match_threshold: float = 0.3
    matching_algorithm: str = "hungarian"
    # Frame-to-frame linker (how each frame's detections join tracks):
    #   "bytetrack" (default since 2026-11-20): cost = centroid distance + mask
    #     overlap, two passes by confidence (low-confidence detections only
    #     continue tracks, never start one), constant-velocity Kalman motion.
    #   "centroid": FNT's linker before 2026-11-20 (centroid distance within 15%
    #     of the frame diagonal, tracks never end by default, at the object
    #     limit a detection takes over the nearest free track). Runs recorded
    #     without a linker were made with it.
    linker: str = "bytetrack"
    # bytetrack: farthest a centroid may move per video frame, in px (x frame
    # step per analysed frame). 0 = auto: half the median animal size
    # (sqrt mask area), at least 1.2% of the frame diagonal.
    gate_px: float = 0.0
    # bytetrack: weight of (1 - mask IoU) beside distance / gate in the cost.
    iou_weight: float = 1.0
    # bytetrack: detections at or above this confidence are matched first and
    # may start tracks; below it they only continue one (else they keep an id
    # of their own, from 1000001 up). 0 = auto: min(0.5, 0.8 x median
    # confidence); < 0 = one pass.
    two_pass_conf: float = 0.0
    # bytetrack: "kalman" (constant velocity) or "none".
    motion: str = "kalman"
    # At max_detections tracks, an unmatched detection continues the nearest
    # free track (one with no detection on that frame). None = auto: on for
    # "centroid" and for max_detections 1 (one animal: every detection is it).
    takeover: Optional[bool] = None
    inference_size: int = 0
    use_masks: bool = False
    # Persist per-frame silhouettes to <video>_MaskTracker/<video>_tracks.h5.
    # Without this the masks are computed and thrown away, so proofreading an
    # identity swap or re-running behavior classification means re-detecting
    # the whole video.
    save_masks: bool = True
    # Detect and track every Nth frame only (1: every frame). The frames in
    # between are read but not analysed, so frame numbers and times stay those
    # of the video; trajectories hold the analysed frames, and the tracked
    # video plays them at fps / N (same length as the source).
    # max_disappeared_frames counts analysed frames.
    frame_step: int = 1


_SYSTEM_PROFILE_CACHE: Optional[Dict] = None


def detect_system_profile(force: bool = False) -> Dict:
    """Auto-detect hardware and return recommended inference settings.

    Result is cached after the first call — detection spawns subprocesses and
    imports torch (to probe CUDA), which is slow, and the hardware doesn't
    change during a session. Pass force=True to re-detect.
    """
    global _SYSTEM_PROFILE_CACHE
    if _SYSTEM_PROFILE_CACHE is not None and not force:
        return _SYSTEM_PROFILE_CACHE
    profile = _detect_system_profile_uncached()
    _SYSTEM_PROFILE_CACHE = profile
    return profile


def _detect_system_profile_uncached() -> Dict:
    """Auto-detect hardware and return recommended inference settings."""
    import platform
    import subprocess

    profile: Dict = {
        "os": platform.system(),
        "arch": platform.machine(),
        "cpu_cores": os.cpu_count() or 1,
        "ram_gb": 0,
        "chip": "Unknown",
        "has_cuda": False,
        "cuda_name": None,
        "has_mps": False,
        "torch_version": None,
        "torch_cuda": None,        # CUDA version torch was built with (None => CPU-only build)
        "gpu_physical": None,      # NVIDIA GPU physically present (via nvidia-smi)
        "gpu_hint": "",            # actionable message when a GPU is present but unusable
        "recommended_device": "cpu",
        "recommended_resolution": 512,
        "recommended_use_masks": False,
    }

    if profile["os"] == "Darwin":
        try:
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0:
                profile["ram_gb"] = int(result.stdout.strip()) // (1024 ** 3)
        except Exception:
            pass
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0 and result.stdout.strip():
                profile["chip"] = result.stdout.strip()
        except Exception:
            profile["chip"] = platform.processor() or "Apple Silicon"
    elif profile["os"] == "Windows":
        # PowerShell is the reliable approach on modern Windows (wmic is
        # deprecated). Launching PowerShell is slow, so fetch RAM and CPU
        # name in a single invocation instead of two.
        profile["chip"] = platform.processor() or "Unknown"
        try:
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory; "
                 "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
                if lines and lines[0].isdigit():
                    profile["ram_gb"] = int(lines[0]) // (1024 ** 3)
                if len(lines) >= 2 and lines[1]:
                    profile["chip"] = lines[1]
        except Exception:
            pass
    else:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        profile["ram_gb"] = kb // (1024 ** 2)
                        break
        except Exception:
            pass
        profile["chip"] = platform.processor() or "Unknown"

    try:
        import torch
        profile["torch_version"] = torch.__version__
        profile["torch_cuda"] = torch.version.cuda
        profile["has_cuda"] = torch.cuda.is_available()
        if profile["has_cuda"]:
            profile["cuda_name"] = torch.cuda.get_device_name(0)
        profile["has_mps"] = bool(
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        )
    except ImportError:
        pass

    # If torch can't use CUDA, check whether an NVIDIA GPU is physically present
    # (via nvidia-smi) so we can explain *why* it isn't being used, rather than
    # just reporting "CPU only" on a machine that clearly has a GPU.
    if not profile["has_cuda"] and not profile["has_mps"]:
        gpu_name = _detect_nvidia_gpu()
        if gpu_name:
            profile["gpu_physical"] = gpu_name
            if profile["torch_cuda"] is None:
                profile["gpu_hint"] = (
                    f"{gpu_name} found, but PyTorch is a CPU-only build "
                    f"(torch {profile['torch_version']}). Reinstall PyTorch with "
                    f"CUDA support to enable GPU acceleration."
                )
            else:
                profile["gpu_hint"] = (
                    f"{gpu_name} found, but PyTorch (built for CUDA "
                    f"{profile['torch_cuda']}) can't access it — check/update "
                    f"your NVIDIA driver."
                )

    if profile["has_cuda"]:
        profile["recommended_device"] = "cuda"
        profile["recommended_resolution"] = 0
        profile["recommended_use_masks"] = True
    elif profile["os"] == "Darwin" and profile["arch"] == "arm64":
        profile["recommended_device"] = "cpu"
        profile["recommended_resolution"] = 384
        profile["recommended_use_masks"] = False
    else:
        profile["recommended_device"] = "cpu"
        profile["recommended_resolution"] = 512
        profile["recommended_use_masks"] = False

    return profile


def _detect_nvidia_gpu() -> Optional[str]:
    """Return the name of a physically present NVIDIA GPU via nvidia-smi, else None.

    This works even when the installed PyTorch is a CPU-only build (which is the
    common reason a real GPU goes unused), because it queries the driver directly.
    """
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
            if lines:
                return lines[0]
    except Exception:
        pass
    return None


def _resolve_device(pref: str) -> str:
    import torch
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "mps":
        return "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


# Shape descriptors written for every detection, in output column order.
# Defined once here because three places have to agree on them: the CSV
# inference writes, the HDF5 track store, and the CSV the proofreader
# regenerates after editing identities.
SHAPE_FIELDS = (
    "major_axis_px",
    "minor_axis_px",
    "elongation",
    "orientation_deg",
    "eccentricity",
    "perimeter_px",
    "solidity",
)
_EMPTY_SHAPE = {f: None for f in SHAPE_FIELDS}


def mask_shape_features(
    mask: Optional[np.ndarray],
    ys: Optional[np.ndarray] = None,
    xs: Optional[np.ndarray] = None,
) -> Dict:
    """Posture descriptors for one instance mask.

    Segmenting instead of landmarking is this tool's whole bet, and until
    now the trajectory table spent each mask on a centroid and a pixel
    count. These are the classical overhead-rodent posture discriminators:
    a rearing animal is compact and round, a walking one is elongated along
    its heading, and a grooming one is neither.

    Everything comes from the equivalent-ellipse image moments plus one
    contour pass, both taken on the mask's bounding-box crop, so cost scales
    with the animal rather than the frame.

    ``orientation_deg`` is the major axis measured anticlockwise from the
    image x axis and wrapped to [0, 180). A silhouette has no head, so that
    is a genuine 180-degree ambiguity rather than a rounding artefact;
    resolve it downstream against the direction of travel where it matters.

    ``perimeter_px`` traces the pixel boundary, which overestimates a smooth
    outline by a few percent because it walks a staircase — a drawn circle
    of radius 60 measures about 400 rather than 377. That bias is stable, so
    the value compares cleanly across frames and animals, but do not read a
    circularity computed from it as an absolute.

    Every key in :data:`SHAPE_FIELDS` is always present, set to None when
    there is no mask, so box-only runs still produce a uniform table.
    """
    if mask is None:
        return dict(_EMPTY_SHAPE)
    if ys is None or xs is None:
        ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return dict(_EMPTY_SHAPE)

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sub = np.ascontiguousarray(mask[y0:y1, x0:x1]).astype(np.uint8)

    m = cv2.moments(sub, binaryImage=True)
    area = m["m00"]
    if area <= 0:
        return dict(_EMPTY_SHAPE)

    mu20 = m["mu20"] / area
    mu02 = m["mu02"] / area
    mu11 = m["mu11"] / area
    common = math.sqrt(max(0.0, 4.0 * mu11 * mu11 + (mu20 - mu02) ** 2))
    lam1 = max(0.0, (mu20 + mu02 + common) / 2.0)
    lam2 = max(0.0, (mu20 + mu02 - common) / 2.0)
    major = 4.0 * math.sqrt(lam1)
    minor = 4.0 * math.sqrt(lam2)

    # The moment angle is computed in image coordinates, where y grows
    # downward, so it is negated to report the anticlockwise angle a reader
    # of the CSV would assume.
    theta = 0.5 * math.atan2(2.0 * mu11, mu20 - mu02)
    orientation = math.degrees(-theta) % 180.0

    perimeter = None
    solidity = None
    contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        outline = max(contours, key=cv2.contourArea)
        perimeter = float(cv2.arcLength(outline, True))
        hull_area = float(cv2.contourArea(cv2.convexHull(outline)))
        if hull_area > 0:
            # Pixel count over polygon area, so the ratio can drift just
            # past 1 on small convex blobs. Cap it rather than emit 1.03.
            solidity = min(1.0, float(area) / hull_area)

    return {
        "major_axis_px": round(major, 2),
        "minor_axis_px": round(minor, 2),
        "elongation": round(major / minor, 3) if minor > 1e-6 else None,
        "orientation_deg": round(orientation, 1),
        "eccentricity": (round(math.sqrt(max(0.0, 1.0 - lam2 / lam1)), 3)
                         if lam1 > 1e-12 else None),
        "perimeter_px": round(perimeter, 2) if perimeter is not None else None,
        "solidity": round(solidity, 3) if solidity is not None else None,
    }


class MaskRCNNInference:
    """Loads a trained Mask R-CNN and runs per-frame inference."""

    def __init__(self, model_dir: str, device: str = "auto",
                 inference_size: int = 0, use_masks: bool = False):
        self.model_dir = model_dir
        self.device = device
        self.model = None
        self.num_classes = 2
        self.categories: Dict = {}
        self._max_dim: int = 800
        self._inference_size: int = inference_size
        self._use_masks: bool = use_masks

    def load_model(self):
        import torch

        device = _resolve_device(self.device)
        self.device = device

        config_path = os.path.join(self.model_dir, "training_config.json")
        backbone = "resnet50"
        min_size = 480
        max_size = 1333
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            self.num_classes = config.get("num_classes", 2)
            self.categories = config.get("categories", {})
            backbone = config.get("backbone", "resnet50")
            min_size = config.get("min_size", 480)
            max_size = config.get("max_size", 1333)

        self._max_dim = max(min_size, max_size)

        from .mask_tracker_training import _build_model
        self.model = _build_model(self.num_classes, backbone=backbone,
                                  min_size=min_size, max_size=max_size)

        weights_path = os.path.join(self.model_dir, "weights_best.pt")
        if not os.path.exists(weights_path):
            weights_path = os.path.join(self.model_dir, "weights.pt")

        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state_dict)

        if self._inference_size > 0:
            self.model.transform.min_size = (self._inference_size,)
            self.model.transform.max_size = self._inference_size
            self._max_dim = self._inference_size

        if not self._use_masks:
            self.model.roi_heads.mask_roi_pool = None

        self.model.to(device)
        self.model.eval()

        self._use_half = False
        if device in ("mps", "cuda"):
            try:
                self.model.half()
                self._use_half = True
            except Exception:
                pass

        precision = "float16" if self._use_half else "float32"
        mask_str = "masks" if self._use_masks else "boxes-only"
        print(f"[MTT Inference] Model loaded on {device} ({precision}), backbone={backbone}, "
              f"inference={self._max_dim}px, {mask_str}")

    def _pre_resize(self, frame_rgb: np.ndarray) -> Tuple[np.ndarray, float]:
        """Resize frame on CPU before GPU transfer to reduce MPS memory pressure."""
        h, w = frame_rgb.shape[:2]
        if max(h, w) <= self._max_dim:
            return frame_rgb, 1.0
        scale = self._max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale

    def predict(
        self,
        frame_rgb: np.ndarray,
        confidence_threshold: float = 0.5,
        max_detections: int = 0,
    ) -> Dict:
        """Run inference on a single frame.

        Returns dict with keys: boxes, masks, scores, labels (numpy arrays).
        Masks are rescaled back to original frame dimensions.
        """
        import torch

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        orig_h, orig_w = frame_rgb.shape[:2]
        resized, scale = self._pre_resize(frame_rgb)

        dtype = torch.float16 if self._use_half else torch.float32
        img_tensor = torch.as_tensor(resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.to(dtype=dtype, device=self.device)

        with torch.no_grad():
            outputs = self.model([img_tensor])[0]

        if self.device == "mps":
            torch.mps.synchronize()

        scores = outputs["scores"].cpu().numpy()
        keep = scores >= confidence_threshold

        boxes = outputs["boxes"].cpu().numpy()[keep]
        scores = scores[keep]
        labels = outputs["labels"].cpu().numpy()[keep]

        if self._use_masks and "masks" in outputs:
            masks = (outputs["masks"].cpu().numpy()[keep, 0] > 0.5).astype(bool)
        else:
            masks = None

        if max_detections > 0 and len(scores) > max_detections:
            top_idx = np.argsort(scores)[::-1][:max_detections]
            boxes = boxes[top_idx]
            scores = scores[top_idx]
            labels = labels[top_idx]
            if masks is not None:
                masks = masks[top_idx]

        if scale != 1.0:
            boxes = boxes / scale
            if masks is not None and len(masks) > 0:
                full_masks = np.zeros((len(masks), orig_h, orig_w), dtype=bool)
                for i, m in enumerate(masks):
                    full_masks[i] = cv2.resize(
                        m.astype(np.uint8), (orig_w, orig_h),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                masks = full_masks

        return {
            "boxes": boxes,
            "masks": masks,
            "scores": scores,
            "labels": labels,
        }


#: Ids from here on: low-confidence detections that continued no track (bytetrack
#: linker), each its own one-detection track, so no detection is lost.
SINGLETON_ID0 = 1_000_001


def mask_crop(mask: np.ndarray, ys: Optional[np.ndarray] = None,
              xs: Optional[np.ndarray] = None) -> Optional[Tuple[int, int, np.ndarray]]:
    """(x0, y0, the mask cropped to its own box), or None for an empty mask."""
    if ys is None or xs is None:
        ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    x0, y0 = int(xs.min()), int(ys.min())
    return x0, y0, np.asarray(mask[y0:int(ys.max()) + 1, x0:int(xs.max()) + 1], dtype=bool)


def crop_iou(a, b, dx: float = 0.0, dy: float = 0.0) -> float:
    """IoU of two cropped masks (mask_crop), ``a`` shifted by (dx, dy)."""
    if a is None or b is None:
        return 0.0
    ax, ay, am = a[0] + int(round(dx)), a[1] + int(round(dy)), a[2]
    bx, by, bm = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + am.shape[1], bx + bm.shape[1]), min(ay + am.shape[0], by + bm.shape[0])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = int(np.logical_and(am[y0 - ay:y1 - ay, x0 - ax:x1 - ax],
                               bm[y0 - by:y1 - by, x0 - bx:x1 - bx]).sum())
    return inter / (int(am.sum()) + int(bm.sum()) - inter) if inter else 0.0


def box_iou(a, b, dx: float = 0.0, dy: float = 0.0) -> float:
    """IoU of two boxes (x1, y1, x2, y2), ``a`` shifted by (dx, dy)."""
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = float(a[0]) + dx, float(a[1]) + dy, float(a[2]) + dx, float(a[3]) + dy
    iw = min(ax2, float(b[2])) - max(ax1, float(b[0]))
    ih = min(ay2, float(b[3])) - max(ay1, float(b[1]))
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1])) - inter
    return inter / union if union > 0 else 0.0


class _ConstantVelocityKF:
    """Constant-velocity Kalman filter on a centroid; one predict per analysed frame."""
    F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)

    def __init__(self, xy, q: float = 40.0, r: float = 6.0):
        self.x = np.array([xy[0], xy[1], 0.0, 0.0], float)
        self.P = np.diag([r * r, r * r, 100.0 ** 2, 100.0 ** 2])
        g = np.array([[0.5, 0], [0, 0.5], [1, 0], [0, 1]])
        self.Q = g @ g.T * q * q
        self.R = np.eye(2) * r * r

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, xy):
        y = np.asarray(xy, float) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P


class MultiObjectTracker:
    """Frame-to-frame multi-object tracker: joins each analysed frame's
    detections to tracks. ``config.linker`` picks how (see MaskInferenceConfig):
    "bytetrack" (distance + mask overlap, two passes by confidence, Kalman
    motion) or "centroid" (the earlier centroid-distance linker, unchanged)."""

    #: bytetrack's auto track end: a track survives this many missed analysed
    #: frames and ends at the next (0.5 s at 10 analysed frames per second)
    AUTO_MAX_AGE = 5
    _STATS_MAX = 1000          # detections the auto gate / two-pass split are measured on

    def __init__(self, config: MaskInferenceConfig):
        self.config = config
        self.next_object_id = 1
        self.active_tracks: Dict[int, Dict] = {}
        self.disappeared: Dict[int, int] = {}
        self.records: List[Dict] = []
        self._max_match_dist: float = 0.0
        # bytetrack state
        self._kf: Dict[int, _ConstantVelocityKF] = {}
        self._last_k: Dict[int, int] = {}
        self._at_k: Dict[int, int] = {}
        self._next_single = SINGLETON_ID0
        self._sizes: List[float] = []
        self._confs: List[float] = []
        self._body: Optional[float] = None
        self._conf_med: Optional[float] = None
        self._diag: float = 0.0
        self.n_singletons = 0

    # -- settings, resolved -------------------------------------------------
    @property
    def linker(self) -> str:
        return "centroid" if str(getattr(self.config, "linker", "bytetrack")).lower() == "centroid" \
            else "bytetrack"

    def _single_animal(self) -> bool:
        return int(self.config.max_detections or 0) == 1

    def _takeover(self) -> bool:
        t = getattr(self.config, "takeover", None)
        if t is None:
            return self.linker == "centroid" or self._single_animal()
        return bool(t)

    def _max_age(self) -> int:
        """Analysed frames a track may go undetected (0 = never)."""
        md = int(self.config.max_disappeared_frames)
        if md >= 0:
            return md
        if self.linker == "centroid" or self._single_animal():
            return 0
        return self.AUTO_MAX_AGE

    def _step(self) -> int:
        return max(1, int(getattr(self.config, "frame_step", 1) or 1))

    def _gate(self) -> float:
        """bytetrack: farthest a centroid may move between analysed frames, px."""
        g = float(getattr(self.config, "gate_px", 0) or 0)
        if g <= 0:
            g = max(0.5 * self._body if self._body else 0.0, 0.012 * self._diag if self._diag else 0.0)
            if g <= 0:
                g = 25.0
        return g * self._step()

    def _two_pass(self) -> float:
        """The two-pass confidence split (0 = one pass)."""
        t = float(getattr(self.config, "two_pass_conf", 0.0) or 0.0)
        if t < 0:
            return 0.0
        if t > 0:
            return t
        return min(0.5, 0.8 * self._conf_med) if self._conf_med is not None else 0.5

    def resolved_settings(self) -> Dict:
        """The linker settings as used (autos resolved), for the run record."""
        out = {"linker": self.linker, "max_disappeared_frames": self._max_age(),
               "takeover": self._takeover()}
        if self.linker == "bytetrack":
            out.update(gate_px_per_analysed_frame=round(self._gate(), 1),
                       two_pass_conf=round(self._two_pass(), 3),
                       iou_weight=float(getattr(self.config, "iou_weight", 1.0)),
                       motion=str(getattr(self.config, "motion", "kalman")),
                       median_animal_px=round(self._body, 1) if self._body else None,
                       low_confidence_singletons=self.n_singletons)
        return out

    def _get_max_match_dist(self, frame_h: int, frame_w: int) -> float:
        """Auto-compute max matching distance as 15% of frame diagonal."""
        if self._max_match_dist > 0:
            return self._max_match_dist
        self._max_match_dist = 0.15 * np.sqrt(frame_h**2 + frame_w**2)
        return self._max_match_dist

    # -- one frame --------------------------------------------------------
    def update(self, detections: Dict, frame_idx: int, fps: float = 30.0,
               frame_hw: Optional[Tuple[int, int]] = None) -> Dict[int, Dict]:
        """Process detections for one frame and return matched track assignments."""
        n_det = len(detections["scores"]) if len(detections["scores"]) > 0 else 0

        masks = detections["masks"]
        has_masks = masks is not None
        want_crop = self.linker == "bytetrack"

        current_dets = []
        for i in range(n_det):
            box = detections["boxes"][i]
            crop = None
            if has_masks:
                mask = masks[i]
                ys, xs = np.where(mask)
                if len(xs) == 0:
                    cx = float((box[0] + box[2]) / 2)
                    cy = float((box[1] + box[3]) / 2)
                else:
                    cx, cy = float(xs.mean()), float(ys.mean())
                    if want_crop:
                        crop = mask_crop(mask, ys, xs)
                area = int(mask.sum())
                # Reuses the nonzero indices already computed above, so the
                # shape descriptors cost one small crop rather than a second
                # pass over the full frame.
                shape = mask_shape_features(mask, ys, xs)
            else:
                mask = None
                cx = float((box[0] + box[2]) / 2)
                cy = float((box[1] + box[3]) / 2)
                area = int((box[2] - box[0]) * (box[3] - box[1]))
                shape = dict(_EMPTY_SHAPE)
            det = {
                "centroid": (cx, cy),
                "bbox": box,
                "mask": mask,
                "score": float(detections["scores"][i]),
                "label": int(detections["labels"][i]),
                "area": area,
                "shape": shape,
            }
            if want_crop:
                det["crop"] = crop
            current_dets.append(det)

        if frame_hw is None and has_masks and len(masks) > 0:
            frame_hw = tuple(masks[0].shape[:2])
        return self.update_dets(current_dets, frame_idx, fps, frame_hw)

    def update_dets(self, current_dets: List[Dict], frame_idx: int, fps: float = 30.0,
                    frame_hw: Optional[Tuple[int, int]] = None) -> Dict[int, Dict]:
        """``update`` on detections already prepared: dicts with centroid, bbox,
        mask (or None), score, label, area, shape and, for bytetrack, crop
        (mask_crop, or None: boxes overlap instead). Used to link again from a
        saved track store without re-detecting."""
        if self.linker == "centroid":
            return self._update_centroid(current_dets, frame_idx, fps, frame_hw)
        return self._update_bytetrack(current_dets, frame_idx, fps, frame_hw)

    # -- bytetrack --------------------------------------------------------
    def _note_stats(self, dets: List[Dict]):
        if len(self._sizes) >= self._STATS_MAX:
            return
        n0 = len(self._sizes)
        for d in dets:
            if len(self._sizes) >= self._STATS_MAX:
                break
            if d.get("area"):
                self._sizes.append(math.sqrt(float(d["area"])))
                self._confs.append(float(d["score"]))
        n = len(self._sizes)
        if n >= 30 and (self._body is None or n // 50 != n0 // 50 or n == self._STATS_MAX):
            self._body = float(np.median(self._sizes))
            self._conf_med = float(np.median(self._confs))

    def _new_track(self, det: Dict, k: int) -> int:
        tid = self.next_object_id
        self.next_object_id += 1
        self.active_tracks[tid] = det
        self.disappeared[tid] = 0
        self._last_k[tid] = k
        self._at_k[tid] = k
        if str(getattr(self.config, "motion", "kalman")).lower() == "kalman":
            self._kf[tid] = _ConstantVelocityKF(det["centroid"])
        return tid

    def _end_track(self, tid: int):
        for d in (self.active_tracks, self.disappeared, self._kf, self._last_k, self._at_k):
            d.pop(tid, None)

    def _continue_track(self, tid: int, det: Dict, k: int):
        kf = self._kf.get(tid)
        if kf is not None:
            kf.update(det["centroid"])
        self.active_tracks[tid] = det
        self.disappeared[tid] = 0
        self._last_k[tid] = k

    def _assign(self, tids: List[int], dets: List[Dict], pred: Dict[int, np.ndarray], gate: float):
        """Gated Hungarian on distance / gate + iou_weight x (1 - overlap):
        [(track id, det index)]."""
        if not tids or not dets:
            return []
        from scipy.optimize import linear_sum_assignment
        w_iou = float(getattr(self.config, "iou_weight", 1.0))
        C = np.full((len(tids), len(dets)), np.inf)
        for a, t in enumerate(tids):
            tr, p = self.active_tracks[t], pred[t]
            shift = p - np.asarray(tr["centroid"], float)
            for b, d in enumerate(dets):
                dist = float(np.hypot(d["centroid"][0] - p[0], d["centroid"][1] - p[1]))
                if dist > gate:
                    continue
                c = dist / gate
                if w_iou:
                    if tr.get("crop") is not None and d.get("crop") is not None:
                        ov = crop_iou(tr["crop"], d["crop"], *shift)
                    else:
                        ov = box_iou(tr.get("bbox"), d.get("bbox"), *shift)
                    c += w_iou * (1.0 - ov)
                C[a, b] = c
        ri, ci = linear_sum_assignment(np.where(np.isfinite(C), C, 1e9))
        return [(tids[a], int(b)) for a, b in zip(ri, ci) if np.isfinite(C[a, b])]

    def _update_bytetrack(self, dets: List[Dict], frame_idx: int, fps: float,
                          frame_hw: Optional[Tuple[int, int]]) -> Dict[int, Dict]:
        if frame_hw is not None and not self._diag:
            self._diag = float(np.hypot(*frame_hw))
        self._note_stats(dets)
        k = int(frame_idx) // self._step()
        max_age = self._max_age()
        if max_age > 0:
            # missed analysed frames since the track's last detection, as the centroid linker counts them
            for tid in [t for t in self.active_tracks if k - self._last_k.get(t, k) - 1 > max_age]:
                self._end_track(tid)
        pred: Dict[int, np.ndarray] = {}
        for tid in self.active_tracks:
            kf = self._kf.get(tid)
            if kf is not None:
                for _ in range(max(0, k - self._at_k.get(tid, k))):
                    kf.predict()
                self._at_k[tid] = k
                pred[tid] = kf.x[:2].copy()
            else:
                pred[tid] = np.asarray(self.active_tracks[tid]["centroid"], float)
            self.disappeared[tid] = k - self._last_k.get(tid, k)

        gate = self._gate()
        split = self._two_pass()
        hi = [i for i, d in enumerate(dets) if d["score"] >= split]
        lo = [i for i, d in enumerate(dets) if d["score"] < split]
        tids = list(self.active_tracks)
        got: Dict[int, int] = {}                          # det index -> track id
        for t, b in self._assign(tids, [dets[i] for i in hi], pred, gate):
            got[hi[b]] = t
        taken = set(got.values())
        rest = [t for t in tids if t not in taken]
        for t, b in self._assign(rest, [dets[i] for i in lo], pred, gate):
            got[lo[b]] = t
        for i, t in got.items():
            self._continue_track(t, dets[i], k)

        matched: Dict[int, Dict] = {}
        for i in sorted(got, key=lambda i: got[i]):
            matched[got[i]] = dets[i]
            self._record(got[i], dets[i], frame_idx, fps)
        takeover = self._takeover()
        limit = int(self.config.max_detections or 0)
        for i, d in enumerate(dets):
            if i in got:
                continue
            if takeover and limit > 0 and len(self.active_tracks) >= limit:
                free = [t for t in self.active_tracks if t not in matched]
                if not free:
                    continue                                 # an extra: left out, as before
                t = min(free, key=lambda t: float(np.hypot(*(np.asarray(d["centroid"]) - pred.get(
                    t, np.asarray(self.active_tracks[t]["centroid"], float))))))
                self._continue_track(t, d, k)
            elif d["score"] >= split:
                t = self._new_track(d, k)
            else:
                t = self._next_single                        # low confidence: never starts a track
                self._next_single += 1
                self.n_singletons += 1
            matched[t] = d
            self._record(t, d, frame_idx, fps)
        return matched

    # -- centroid (the linker before 2026-11-20; unchanged) ----------------
    def _update_centroid(self, current_dets: List[Dict], frame_idx: int, fps: float,
                         frame_hw: Optional[Tuple[int, int]]) -> Dict[int, Dict]:
        if not self.active_tracks:
            matched = {}
            for det in current_dets:
                obj_id = self.next_object_id
                self.next_object_id += 1
                self.active_tracks[obj_id] = det
                self.disappeared[obj_id] = 0
                matched[obj_id] = det
                self._record(obj_id, det, frame_idx, fps)
            return matched

        max_age = self._max_age()
        if not current_dets:
            for obj_id in list(self.active_tracks.keys()):
                self.disappeared[obj_id] = self.disappeared.get(obj_id, 0) + 1
                if max_age > 0 and self.disappeared[obj_id] > max_age:
                    del self.active_tracks[obj_id]
                    del self.disappeared[obj_id]
            return {}

        track_ids = list(self.active_tracks.keys())
        n_tracks = len(track_ids)
        n_dets = len(current_dets)

        if frame_hw is not None:
            max_dist = self._get_max_match_dist(frame_hw[0], frame_hw[1])
        else:
            max_dist = 200.0

        dist_matrix = np.zeros((n_tracks, n_dets))
        for i, tid in enumerate(track_ids):
            tx, ty = self.active_tracks[tid]["centroid"]
            for j, det in enumerate(current_dets):
                dx, dy = det["centroid"]
                dist_matrix[i, j] = np.sqrt((tx - dx)**2 + (ty - dy)**2)

        if self.config.matching_algorithm == "greedy":
            matched_rows, used_cols = self._match_greedy(
                dist_matrix, track_ids, current_dets, max_dist, frame_idx, fps,
            )
        else:
            matched_rows, used_cols = self._match_hungarian(
                dist_matrix, track_ids, current_dets, max_dist, frame_idx, fps,
            )

        matched = {}
        for row_idx in matched_rows:
            obj_id = track_ids[row_idx]
            matched[obj_id] = self.active_tracks[obj_id]

        for row_idx in range(n_tracks):
            if row_idx not in matched_rows:
                obj_id = track_ids[row_idx]
                self.disappeared[obj_id] = self.disappeared.get(obj_id, 0) + 1
                if max_age > 0 and self.disappeared[obj_id] > max_age:
                    del self.active_tracks[obj_id]
                    del self.disappeared[obj_id]

        takeover = self._takeover()
        for col_idx in range(n_dets):
            if col_idx not in used_cols:
                det = current_dets[col_idx]
                at_limit = (takeover and self.config.max_detections > 0
                            and len(self.active_tracks) >= self.config.max_detections)
                if at_limit:
                    # As many tracks as animals: the detection is one of them seen again elsewhere, so it
                    # continues the nearest track that has no detection on this frame. A track already
                    # given a detection this frame is never taken over (that put two animals under one
                    # ID and dropped one); with every track taken, the detection is an extra and is left out.
                    free = [tid for tid in self.active_tracks if tid not in matched]
                    if not free:
                        continue
                    best_tid = min(
                        free,
                        key=lambda tid: np.sqrt(
                            (self.active_tracks[tid]["centroid"][0] - det["centroid"][0])**2
                            + (self.active_tracks[tid]["centroid"][1] - det["centroid"][1])**2
                        ),
                    )
                    self.active_tracks[best_tid] = det
                    self.disappeared[best_tid] = 0
                    matched[best_tid] = det
                    self._record(best_tid, det, frame_idx, fps)
                else:
                    obj_id = self.next_object_id
                    self.next_object_id += 1
                    self.active_tracks[obj_id] = det
                    self.disappeared[obj_id] = 0
                    matched[obj_id] = det
                    self._record(obj_id, det, frame_idx, fps)

        return matched

    def _match_hungarian(
        self, dist_matrix, track_ids, current_dets, max_dist, frame_idx, fps,
    ) -> Tuple[set, set]:
        from scipy.optimize import linear_sum_assignment
        # Pairs beyond max_dist can never match, so they must not steer the assignment: left at their
        # distance, a far-off pair could make the total smaller by giving a nearby track the wrong detection
        # (a swap). A prohibitive cost leaves them out.
        gated = np.where(dist_matrix <= max_dist, dist_matrix, 1e9)
        row_indices, col_indices = linear_sum_assignment(gated)

        matched_rows = set()
        used_cols = set()
        for row, col in zip(row_indices, col_indices):
            if dist_matrix[row, col] <= max_dist:
                obj_id = track_ids[row]
                det = current_dets[col]
                self.active_tracks[obj_id] = det
                self.disappeared[obj_id] = 0
                self._record(obj_id, det, frame_idx, fps)
                matched_rows.add(row)
                used_cols.add(col)
        return matched_rows, used_cols

    def _match_greedy(
        self, dist_matrix, track_ids, current_dets, max_dist, frame_idx, fps,
    ) -> Tuple[set, set]:
        n_tracks, n_dets = dist_matrix.shape
        flat_indices = np.argsort(dist_matrix, axis=None)

        matched_rows = set()
        used_cols = set()
        for flat_idx in flat_indices:
            i, j = divmod(int(flat_idx), n_dets)
            if i in matched_rows or j in used_cols:
                continue
            if dist_matrix[i, j] > max_dist:
                break
            obj_id = track_ids[i]
            det = current_dets[j]
            self.active_tracks[obj_id] = det
            self.disappeared[obj_id] = 0
            self._record(obj_id, det, frame_idx, fps)
            matched_rows.add(i)
            used_cols.add(j)
        return matched_rows, used_cols

    def _record(self, obj_id: int, det: Dict, frame_idx: int, fps: float):
        cx, cy = det["centroid"]
        bbox = det["bbox"]
        row = {
            "frame": frame_idx,
            "time_s": round(frame_idx / fps, 4),
            "object_id": obj_id,
            "label": det["label"],
            "x": round(cx, 2),
            "y": round(cy, 2),
            "bbox_x1": round(float(bbox[0]), 2),
            "bbox_y1": round(float(bbox[1]), 2),
            "bbox_x2": round(float(bbox[2]), 2),
            "bbox_y2": round(float(bbox[3]), 2),
            "mask_area": det["area"],
            "confidence": round(det["score"], 4),
        }
        row.update(det.get("shape") or _EMPTY_SHAPE)
        self.records.append(row)

    def get_trajectories(self, categories: Optional[Dict] = None) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame()
        df = pd.DataFrame(self.records)
        df = df.sort_values(["object_id", "frame"]).reset_index(drop=True)

        add_names_and_velocities(df, categories)

        col_order = ["frame", "time_s", "object_id", "object_name", "label",
                     "x", "y", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                     "mask_area", *SHAPE_FIELDS,
                     "confidence", "vx", "vy", "speed"]
        col_order = [c for c in col_order if c in df.columns]
        return df[col_order]



def add_names_and_velocities(df: pd.DataFrame, categories: Optional[Dict] = None) -> None:
    """object_name (class_N, numbered per class in id order) and vx, vy, speed
    (per track, from consecutive rows) on a trajectories table sorted by
    object_id then frame. Grouped rather than one pass per track: a linker
    leaving thousands of one-detection tracks made the per-track loop crawl."""
    if categories is not None:
        cat_map = {int(k): v for k, v in categories.items()}
        first_label = df.groupby("object_id", sort=True)["label"].first()
        class_counters: Dict[str, int] = defaultdict(int)
        obj_names: Dict[int, str] = {}
        for oid, label in first_label.items():
            class_name = cat_map.get(int(label), f"class{int(label)}")
            class_counters[class_name] += 1
            obj_names[oid] = f"{class_name}_{class_counters[class_name]}"
        df["object_name"] = df["object_id"].map(obj_names)
    g = df.groupby("object_id", sort=False)
    dx = g["x"].diff()
    dy = g["y"].diff()
    dt = g["time_s"].diff().replace(0, np.nan)
    df["vx"] = (dx / dt).round(2)
    df["vy"] = (dy / dt).round(2)
    df["speed"] = (np.sqrt(dx**2 + dy**2) / dt).round(2)


def _draw_annotations(
    frame_bgr: np.ndarray,
    matched: Dict[int, Dict],
    categories: Dict,
    contour_thickness: int = 2,
    behavior_labels: Optional[Dict[int, tuple]] = None,
) -> np.ndarray:
    """Draw track annotations on a frame (mask contour outlines or bounding boxes).

    Uses contour outlines instead of filled overlays so the animal remains
    fully visible — the outline directly represents the silhouette boundary
    that downstream behavioral classifiers will use.

    Args:
        behavior_labels: Optional {obj_id: (behavior_name, confidence)} dict.
            When provided, appends behavior prediction to the label text.
    """
    overlay = frame_bgr.copy()

    cat_map = {}
    for k, v in categories.items():
        cat_map[int(k)] = v

    for obj_id, det in matched.items():
        color = TRACK_COLORS[(obj_id - 1) % len(TRACK_COLORS)]

        if det["mask"] is not None:
            mask_u8 = det["mask"].astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(overlay, contours, -1, color, contour_thickness,
                             cv2.LINE_AA)
        else:
            box = det["bbox"]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        cx, cy = int(det["centroid"][0]), int(det["centroid"][1])
        class_name = cat_map.get(det["label"], f"class{det['label']}")
        label_text = f"{class_name}_{obj_id}"

        if behavior_labels and obj_id in behavior_labels:
            beh_name, beh_conf = behavior_labels[obj_id]
            label_text += f": {beh_name} {beh_conf:.0%}"

        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tx, ty = cx - tw // 2, cy - 8
        cv2.rectangle(overlay, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2), (0, 0, 0), -1)
        cv2.putText(overlay, label_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.circle(overlay, (cx, cy), 3, color, -1)

    return overlay


def _detect_architecture(model_dir: str) -> str:
    """Read training_config.json and return the architecture string."""
    config_path = os.path.join(model_dir, "training_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        return cfg.get("architecture", "maskrcnn")
    return "maskrcnn"


def run_inference_on_video(
    video_path: str,
    model_dir: str,
    config: MaskInferenceConfig,
    progress: Optional[Callable] = None,
    should_stop: Optional[Callable] = None,
    frame_callback: Optional[Callable] = None,
    track_callback: Optional[Callable] = None,
    create_tracked_video: bool = True,
    run_info: Optional[Dict] = None,
) -> Dict:
    """Run full inference + tracking pipeline on a video.

    The output folder also gets ``run_settings.json`` (model, settings,
    ``run_info``, timing, frames without a detection) and ``inference.log``
    (see ``mtt_runs``).

    Args:
        video_path: Path to video file.
        model_dir: Directory containing weights.pt and training_config.json.
        config: Inference and tracking configuration.
        progress: Callback ``progress(frame_idx, total_frames)``.
        should_stop: Callable returning True to abort.
        frame_callback: Optional ``frame_callback(frame_rgb, frame_idx)``
            called periodically with the annotated frame (RGB numpy array).

    Returns:
        Dict with 'output_dir', 'csv_path', 'num_tracks', 'total_frames'.
    """
    import platform, sys, time
    from dataclasses import asdict
    from datetime import datetime
    from . import mtt_runs

    _saved_fd = None
    _null_fd = None
    if platform.system() == "Darwin":
        try:
            _saved_fd = os.dup(2)
            _null_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(_null_fd, 2)
        except OSError:
            _saved_fd = None

    mask_writer = None
    tee = mtt_runs.LogTee(sys.stdout)
    _real_stdout = sys.stdout
    sys.stdout = tee
    rec: Dict = {
        "video": video_path, "model_dir": model_dir, "settings": asdict(config),
        "create_tracked_video": create_tracked_video,
        "started": datetime.now().isoformat(timespec="seconds"), "complete": False,
    }
    rec.update(run_info or {})
    output_dir = None
    stopped = False
    n_no_det = 0
    frame_idx = 0
    t_start = time.time()
    try:
        architecture = _detect_architecture(model_dir)
        rec["architecture"] = architecture
        if architecture == "yolov11-seg":
            from .yolo_inference import YOLOInference
            inference = YOLOInference(model_dir, device=config.device,
                                      inference_size=config.inference_size,
                                      use_masks=config.use_masks)
        else:
            inference = MaskRCNNInference(model_dir, device=config.device,
                                          inference_size=config.inference_size,
                                          use_masks=config.use_masks)
        inference.load_model()
        if hasattr(inference, "use_source"):      # per-trial ROI, if the model has them
            inference.use_source(video_path)
            r = getattr(inference, "roi", None)
            if r is not None and getattr(getattr(inference, "_roi_all", None), "by_folder", None):
                print(f"[MTT Inference]   ROI for this video: {r.describe()}")
            rec["roi"] = r.to_dict() if r is not None else None

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        step = max(1, int(getattr(config, "frame_step", 1) or 1))
        rec.update(frames=total_frames, fps=fps, width=w, height=h)
        print(f"[MTT Inference] Video: {os.path.basename(video_path)}")
        print(f"[MTT Inference]   {total_frames} frames, {fps:.1f} fps, {w}x{h}"
              + (f", every {step}th frame analysed ({fps / step:.2f} per s)" if step > 1 else ""))
        _md = config.max_disappeared_frames
        print(f"[MTT Inference]   confidence={config.confidence_threshold}, "
              f"max_detections={config.max_detections}, "
              f"max_disappear={'never' if _md == 0 else ('auto' if _md < 0 else _md)}, "
              f"linker={getattr(config, 'linker', 'bytetrack')}"
              + (f", matching={config.matching_algorithm}"
                 if str(getattr(config, 'linker', '')).lower() == 'centroid' else ""))

        video_stem = Path(video_path).stem
        video_dir = str(Path(video_path).parent)
        output_dir = os.path.join(video_dir, f"{video_stem}_MaskTracker")
        os.makedirs(output_dir, exist_ok=True)

        annotated_path = os.path.join(output_dir, f"{video_stem}_annotated.mp4")
        writer = None
        if create_tracked_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(annotated_path, fourcc, fps / step, (w, h))

        if config.save_masks:
            from .track_store import TrackMaskWriter, tracks_path_for
            try:
                mask_writer = TrackMaskWriter(
                    tracks_path_for(video_path, output_dir),
                    video_path=video_path, fps=fps, frame_count=total_frames,
                    width=w, height=h, model_dir=model_dir,
                    has_masks=config.use_masks,
                )
            except Exception as e:
                # Never let mask persistence take the run down — the CSV and
                # annotated video are the primary outputs.
                print(f"[MTT Inference] Mask store disabled ({e})")
                mask_writer = None

        tracker = MultiObjectTracker(config)
        frame_idx = 0
        t_start = time.time()
        t_last_log = t_start
        t_accum = {"read": 0.0, "model": 0.0, "track": 0.0, "draw": 0.0, "write": 0.0}
        t_accum_n = 0

        while True:
            if should_stop and should_stop():
                print(f"[MTT Inference] Stopped by user at frame {frame_idx}/{total_frames}")
                stopped = True
                break

            t0 = time.time()
            if step > 1 and frame_idx % step:
                # Between analysed frames: decoded (the codec needs it) but
                # not converted, detected or drawn.
                if not cap.grab():
                    break
                frame_idx += 1
                if progress:
                    progress(frame_idx, total_frames)
                continue
            ret, frame = cap.read()
            if not ret:
                break
            t1 = time.time()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = inference.predict(
                frame_rgb, config.confidence_threshold, config.max_detections,
            )
            t2 = time.time()
            if len(detections["scores"]) == 0:
                n_no_det += 1

            matched = tracker.update(detections, frame_idx, fps, frame_hw=(h, w))
            if mask_writer is not None and matched:
                mask_writer.add_frame(frame_idx, matched)
            t3 = time.time()

            behavior_labels = None
            if track_callback:
                behavior_labels = track_callback(matched, frame_idx, fps)

            if writer or frame_callback:
                annotated = _draw_annotations(
                    frame, matched, inference.categories,
                    behavior_labels=behavior_labels,
                )
            t4 = time.time()

            if writer:
                writer.write(annotated)
            t5 = time.time()

            if frame_callback and ((frame_idx // step) % 3 == 0 or frame_idx == 0):
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_callback(annotated_rgb, frame_idx)

            t_accum["read"] += t1 - t0
            t_accum["model"] += t2 - t1
            t_accum["track"] += t3 - t2
            t_accum["draw"] += t4 - t3
            t_accum["write"] += t5 - t4
            t_accum_n += 1

            frame_idx += 1
            if progress:
                progress(frame_idx, total_frames)

            now = time.time()
            if now - t_last_log >= 5.0 or frame_idx == 1:
                elapsed = now - t_start
                fps_actual = frame_idx / elapsed if elapsed > 0 else 0
                n_det = len(detections["scores"])
                n_tracks = len(tracker.active_tracks)
                if t_accum_n > 0:
                    avg = {k: (v / t_accum_n) * 1000 for k, v in t_accum.items()}
                    print(f"[MTT Inference]   frame {frame_idx}/{total_frames} "
                          f"({fps_actual:.1f} fps) — {n_det} det, {n_tracks} tracks | "
                          f"read={avg['read']:.0f}ms model={avg['model']:.0f}ms "
                          f"track={avg['track']:.0f}ms draw={avg['draw']:.0f}ms "
                          f"write={avg['write']:.0f}ms")
                else:
                    print(f"[MTT Inference]   frame {frame_idx}/{total_frames} "
                          f"({fps_actual:.1f} fps) — {n_det} det, {n_tracks} tracks")
                t_accum = {"read": 0.0, "model": 0.0, "track": 0.0, "draw": 0.0, "write": 0.0}
                t_accum_n = 0
                t_last_log = now

        cap.release()
        if writer:
            writer.release()
        tracks_h5 = None
        if mask_writer is not None:
            tracks_h5 = mask_writer.path
            n_mask_rows = mask_writer.n_rows
            mask_writer.close()
            from .track_store import long_path
            # long_path: a store path past Windows' 260 characters (NVR camera
            # names) made getsize raise here, after every frame was tracked
            # and before the CSV was written.
            size_mb = os.path.getsize(long_path(tracks_h5)) / (1024 * 1024)
            print(f"[MTT Inference] Masks: {n_mask_rows} detections -> "
                  f"{os.path.basename(tracks_h5)} ({size_mb:.1f} MB)")

        elapsed = time.time() - t_start
        fps_actual = frame_idx / elapsed if elapsed > 0 else 0
        print(f"[MTT Inference] Done: {frame_idx} frames in {elapsed:.1f}s ({fps_actual:.1f} fps)")

        rec["linker_resolved"] = tracker.resolved_settings()
        print(f"[MTT Inference] Linker: {rec['linker_resolved']}")
        df_raw = tracker.get_trajectories(categories=inference.categories)

        csv_path = os.path.join(output_dir, "trajectories.csv")

        n_tracks = 0
        if not df_raw.empty:
            df_raw.to_csv(csv_path, index=False)
            n_tracks = df_raw["object_id"].nunique()
            print(f"[MTT Inference] {n_tracks} unique tracks written to {output_dir}/")
        if writer:
            print(f"[MTT Inference] Annotated video: {annotated_path}")
        rec.update(complete=not stopped, tracks=n_tracks)

        return {
            "output_dir": output_dir,
            "csv_path": csv_path,
            "tracks_h5": tracks_h5,
            "annotated_video": annotated_path if writer else None,
            "num_tracks": n_tracks,
            "total_frames": frame_idx,
        }

    finally:
        # close() is idempotent, so the normal path having already closed the
        # writer is fine; this is here for the crash path.
        if mask_writer is not None:
            try:
                mask_writer.close()
            except Exception:
                pass
        sys.stdout = _real_stdout
        if output_dir and os.path.isdir(output_dir):
            elapsed = time.time() - t_start
            rec.update(finished=datetime.now().isoformat(timespec="seconds"),
                       elapsed_s=round(elapsed, 1), frames_processed=frame_idx,
                       frames_without_detection=n_no_det,
                       fps_processed=round(frame_idx / elapsed, 2) if elapsed > 0 else None)
            if stopped:
                rec["stopped"] = True
            try:
                mtt_runs.write_run_record(output_dir, rec, tee.lines)
            except Exception as e:
                print(f"[MTT Inference] Could not write the run record ({e})")
        if _saved_fd is not None:
            os.dup2(_saved_fd, 2)
            os.close(_saved_fd)
        if _null_fd is not None:
            os.close(_null_fd)
