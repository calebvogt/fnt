"""Silhouette extraction from mask tracker models.

Runs a trained mask model (Mask R-CNN or YOLOv11-seg) on video frames,
tracks objects across frames, and saves per-object silhouette crops to
HDF5 files for downstream behavioral classification.

Also provides composite image generation: a single image summarizing a
short clip by overlaying time-colored silhouettes (blue = oldest frame,
red = newest frame).

Heavy deps (torch, h5py) are imported lazily.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np


def _detect_architecture(model_dir: str) -> str:
    config_path = os.path.join(model_dir, "training_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        return cfg.get("architecture", "maskrcnn")
    return "maskrcnn"


def _load_categories(model_dir: str) -> Dict[int, str]:
    config_path = os.path.join(model_dir, "training_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        cats = cfg.get("categories", {})
        return {int(k): v for k, v in cats.items()}
    return {}


class SilhouetteExtractor:
    """Extracts per-object silhouette crops from video using a trained mask model.

    Uses the same model loading and tracking pipeline as mask_tracker_inference
    but saves binary mask crops to HDF5 instead of trajectory CSVs.
    """

    def __init__(self, model_dir: str, device: str = "auto"):
        self.model_dir = model_dir
        self.device = device
        self.inference = None
        self.categories: Dict[int, str] = {}

    def load_model(self):
        architecture = _detect_architecture(self.model_dir)
        if architecture == "yolov11-seg":
            from .yolo_inference import YOLOInference
            self.inference = YOLOInference(
                self.model_dir, device=self.device,
                inference_size=0, use_masks=True,
            )
        else:
            from .mask_tracker_inference import MaskRCNNInference
            self.inference = MaskRCNNInference(
                self.model_dir, device=self.device,
                inference_size=0, use_masks=True,
            )
        self.inference.load_model()
        self.categories = _load_categories(self.model_dir)

    def extract_video(
        self,
        video_path: str,
        output_dir: str,
        confidence_threshold: float = 0.5,
        max_detections: int = 0,
        max_disappeared_frames: int = 30,
        matching_algorithm: str = "hungarian",
        progress: Optional[Callable] = None,
        should_stop: Optional[Callable] = None,
    ) -> Dict:
        """Run mask model + tracker on a video and save silhouette crops to HDF5.

        Args:
            video_path: Path to input video.
            output_dir: Directory to write HDF5 files into (e.g. behavior_classifier/silhouettes/).
            confidence_threshold: Minimum detection confidence.
            max_detections: Max objects per frame (0 = unlimited).
            max_disappeared_frames: Frames before a lost track is dropped.
            matching_algorithm: "hungarian" or "greedy".
            progress: Optional callback(frame_idx, total_frames).
            should_stop: Optional callable returning True to abort.

        Returns:
            Dict with keys: output_dir, video_stem, num_tracks, total_frames, fps.
        """
        import h5py
        from .mask_tracker_inference import MaskInferenceConfig, MultiObjectTracker

        if self.inference is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_stem = Path(video_path).stem
        video_out_dir = os.path.join(output_dir, video_stem)
        os.makedirs(video_out_dir, exist_ok=True)

        config = MaskInferenceConfig(
            confidence_threshold=confidence_threshold,
            max_detections=max_detections,
            max_disappeared_frames=max_disappeared_frames,
            matching_algorithm=matching_algorithm,
            use_masks=True,
        )
        tracker = MultiObjectTracker(config)

        # Accumulate per-object data in memory, write to HDF5 at the end.
        # Each entry: {frame_idx, mask_crop, centroid, bbox, area, label}
        object_data: Dict[int, List[Dict]] = {}

        frame_idx = 0
        while True:
            if should_stop and should_stop():
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = self.inference.predict(
                frame_rgb, confidence_threshold, max_detections,
            )

            matched = tracker.update(
                detections, frame_idx, fps, frame_hw=(frame_h, frame_w),
            )

            for obj_id, det in matched.items():
                mask = det.get("mask")
                if mask is None:
                    continue

                bbox = det["bbox"]
                x1 = max(0, int(round(float(bbox[0]))))
                y1 = max(0, int(round(float(bbox[1]))))
                x2 = min(frame_w, int(round(float(bbox[2]))))
                y2 = min(frame_h, int(round(float(bbox[3]))))

                if x2 <= x1 or y2 <= y1:
                    continue

                mask_crop = mask[y1:y2, x1:x2]

                if obj_id not in object_data:
                    object_data[obj_id] = []

                object_data[obj_id].append({
                    "frame_idx": frame_idx,
                    "mask_crop": mask_crop,
                    "centroid": det["centroid"],
                    "bbox": (x1, y1, x2, y2),
                    "area": det["area"],
                    "label": det["label"],
                })

            frame_idx += 1
            if progress:
                progress(frame_idx, total_frames)

        cap.release()

        # Write HDF5 files
        for obj_id, frames_data in object_data.items():
            h5_path = os.path.join(video_out_dir, f"object_{obj_id}.h5")
            n = len(frames_data)

            frame_indices = np.array([d["frame_idx"] for d in frames_data], dtype=np.int32)
            centroids = np.array([d["centroid"] for d in frames_data], dtype=np.float32)
            bboxes = np.array([d["bbox"] for d in frames_data], dtype=np.float32)
            areas = np.array([d["area"] for d in frames_data], dtype=np.int32)

            with h5py.File(h5_path, "w") as f:
                f.attrs["object_id"] = obj_id
                f.attrs["label"] = int(frames_data[0]["label"])
                f.attrs["class_name"] = self.categories.get(
                    int(frames_data[0]["label"]), f"class_{frames_data[0]['label']}"
                )
                f.attrs["video_path"] = video_path
                f.attrs["video_stem"] = video_stem
                f.attrs["fps"] = fps
                f.attrs["total_frames"] = total_frames
                f.attrs["frame_width"] = frame_w
                f.attrs["frame_height"] = frame_h
                f.attrs["num_detections"] = n

                f.create_dataset("frame_indices", data=frame_indices)
                f.create_dataset("centroids", data=centroids)
                f.create_dataset("bboxes", data=bboxes)
                f.create_dataset("areas", data=areas)

                # Variable-size mask crops stored as individual datasets per frame
                masks_grp = f.create_group("masks")
                for i, d in enumerate(frames_data):
                    masks_grp.create_dataset(
                        str(d["frame_idx"]),
                        data=d["mask_crop"].astype(np.uint8),
                        compression="gzip",
                        compression_opts=4,
                    )

        return {
            "output_dir": video_out_dir,
            "video_stem": video_stem,
            "num_tracks": len(object_data),
            "total_frames": frame_idx,
            "fps": fps,
        }


def load_silhouette_clip(
    h5_path: str,
    start_frame: int,
    clip_length: int = 15,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Load a clip of mask crops from an HDF5 silhouette file.

    Args:
        h5_path: Path to object_N.h5 file.
        start_frame: First frame index of the clip.
        clip_length: Number of frames in the clip.

    Returns:
        Tuple of (mask_crops, centroids, bboxes) for frames in range
        [start_frame, start_frame + clip_length). mask_crops is a list
        of 2D bool arrays (may vary in size). Missing frames get None entries.
    """
    import h5py

    mask_crops = []
    centroids_list = []
    bboxes_list = []

    with h5py.File(h5_path, "r") as f:
        frame_indices = f["frame_indices"][:]
        all_centroids = f["centroids"][:]
        all_bboxes = f["bboxes"][:]
        masks_grp = f["masks"]

        frame_to_idx = {int(fi): i for i, fi in enumerate(frame_indices)}

        for fi in range(start_frame, start_frame + clip_length):
            if fi in frame_to_idx:
                local_idx = frame_to_idx[fi]
                mask_key = str(fi)
                if mask_key in masks_grp:
                    mask_crops.append(masks_grp[mask_key][:].astype(bool))
                else:
                    mask_crops.append(None)
                centroids_list.append(all_centroids[local_idx])
                bboxes_list.append(all_bboxes[local_idx])
            else:
                mask_crops.append(None)
                centroids_list.append(np.array([np.nan, np.nan], dtype=np.float32))
                bboxes_list.append(np.array([np.nan] * 4, dtype=np.float32))

    return (
        mask_crops,
        np.array(centroids_list, dtype=np.float32),
        np.array(bboxes_list, dtype=np.float32),
    )


def generate_composite(
    mask_crops: List[Optional[np.ndarray]],
    output_size: Tuple[int, int] = (128, 128),
    contour_thickness: int = 1,
    bboxes: Optional[List[Optional[Tuple[int, int, int, int]]]] = None,
) -> np.ndarray:
    """Generate a blue-to-red time-colored contour composite on black background.

    Draws contours at native resolution, pads to square (preserving aspect
    ratio), then resizes to ``output_size``.

    Args:
        mask_crops: List of 2D boolean mask arrays (one per frame).
            None entries are skipped.
        output_size: (height, width) of the final square output image.
        contour_thickness: Pixel thickness of contour lines.
        bboxes: Optional per-frame bounding boxes (x1, y1, x2, y2) used to
            spatially align crops of different sizes onto a common canvas.
            When None, all crops are assumed to be the same size (e.g. from
            a union-bbox crop).

    Returns:
        RGB uint8 numpy array of shape (H, W, 3).
    """
    out_h, out_w = output_size
    valid = [(i, m) for i, m in enumerate(mask_crops) if m is not None]
    if not valid:
        return np.zeros((out_h, out_w, 3), dtype=np.uint8)

    if bboxes is not None:
        valid_bboxes = [bboxes[i] for i, _ in valid if bboxes[i] is not None]
        if valid_bboxes:
            ux1 = min(int(b[0]) for b in valid_bboxes)
            uy1 = min(int(b[1]) for b in valid_bboxes)
            ux2 = max(int(b[2]) for b in valid_bboxes)
            uy2 = max(int(b[3]) for b in valid_bboxes)
            canvas_h = max(1, uy2 - uy1)
            canvas_w = max(1, ux2 - ux1)
        else:
            first_m = valid[0][1]
            canvas_h, canvas_w = first_m.shape[:2]
            ux1, uy1 = 0, 0
    else:
        first_m = valid[0][1]
        canvas_h, canvas_w = first_m.shape[:2]
        ux1, uy1 = 0, 0

    composite = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    n_total = len(mask_crops)

    for i, mask in valid:
        if bboxes is not None and bboxes[i] is not None:
            bx1 = int(bboxes[i][0]) - ux1
            by1 = int(bboxes[i][1]) - uy1
            placed = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
            mh, mw = mask.shape[:2]
            end_y = min(by1 + mh, canvas_h)
            end_x = min(bx1 + mw, canvas_w)
            src_h = end_y - by1
            src_w = end_x - bx1
            if src_h > 0 and src_w > 0:
                placed[by1:end_y, bx1:end_x] = mask[:src_h, :src_w].astype(np.uint8) * 255
        else:
            mh, mw = mask.shape[:2]
            if mh == canvas_h and mw == canvas_w:
                placed = mask.astype(np.uint8) * 255
            else:
                placed = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
                end_y = min(mh, canvas_h)
                end_x = min(mw, canvas_w)
                placed[:end_y, :end_x] = mask[:end_y, :end_x].astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            placed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            continue

        t = i / max(n_total - 1, 1)
        hue = int(120 * (1 - t))
        hsv_pixel = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        rgb_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2RGB)[0, 0]
        color = (int(rgb_pixel[0]), int(rgb_pixel[1]), int(rgb_pixel[2]))

        cv2.drawContours(composite, contours, -1, color, contour_thickness)

    max_dim = max(canvas_h, canvas_w)
    padded = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    y_off = (max_dim - canvas_h) // 2
    x_off = (max_dim - canvas_w) // 2
    padded[y_off:y_off + canvas_h, x_off:x_off + canvas_w] = composite

    if max_dim != out_h or max_dim != out_w:
        padded = cv2.resize(padded, (out_w, out_h), interpolation=cv2.INTER_AREA)

    return padded


def generate_composite_from_h5(
    h5_path: str,
    start_frame: int,
    clip_length: int = 15,
    output_size: Tuple[int, int] = (128, 128),
) -> np.ndarray:
    """Convenience: load a clip from HDF5 and generate its composite image."""
    mask_crops, _, bboxes_arr = load_silhouette_clip(h5_path, start_frame, clip_length)
    bbox_list: List[Optional[Tuple[int, int, int, int]]] = []
    for i, crop in enumerate(mask_crops):
        if crop is not None and not np.any(np.isnan(bboxes_arr[i])):
            b = bboxes_arr[i]
            bbox_list.append((int(b[0]), int(b[1]), int(b[2]), int(b[3])))
        else:
            bbox_list.append(None)
    return generate_composite(mask_crops, output_size, bboxes=bbox_list)


# ===========================================================================
# Scene-aware representation for social behavior
#
# The composite above shows one animal's silhouette on a black background.
# That is enough for solo posture -- locomotion, rearing, grooming -- and
# structurally incapable of representing huddling or attack, which are
# defined by a second animal the crop excludes.
#
# What follows widens the representation to the focal animal *in company*:
#
#   * a six-channel image, the focal animal's time-coded contours in the
#     first three channels and every neighbour's in the last three, drawn on
#     one canvas so relative position and scale survive; and
#   * a short vector of pairwise quantities that a 128-pixel image represents
#     poorly -- separation, closing speed, relative heading.
#
# The focal animal keeps a fixed share of the canvas, so posture stays as
# legible as it was. A neighbour far enough away to fall outside the canvas
# is simply not drawn, and the distance features carry that fact instead.
# ===========================================================================

# Order of the vector returned by :func:`pairwise_features`. Training and
# inference both build their input from it, so it is defined once.
PAIRWISE_FIELDS = (
    "neighbor_present",
    "min_dist_norm",
    "mean_dist_norm",
    "closing_norm",
    "contact_frac",
    "rel_orient_deg",
    "bearing_deg",
    "focal_speed_norm",
    "focal_elong_mean",
    "focal_elong_std",
    "focal_area_cv",
    "focal_solidity_mean",
)

# Canvas is the focal animal's own extent grown by this factor, so roughly
# one body length of context sits on every side.
DEFAULT_CONTEXT_SCALE = 3.0

# Boxes are grown by this fraction of body length before they are tested for
# contact. Bare intersection is far too brittle: two animals huddled flank to
# flank commonly leave a few pixels of background between their masks, so a
# strict test reports no contact for the clearest huddle in the dataset while
# reporting contact for a passing animal whose box happens to clip a whisker.
# Scaling the slack by body length keeps the definition camera-independent.
CONTACT_PAD_FRAC = 0.15


def _time_color(index: int, n_total: int) -> Tuple[int, int, int]:
    """Blue for the oldest frame through red for the newest."""
    t = index / max(n_total - 1, 1)
    hsv = np.array([[[int(120 * (1 - t)), 255, 255]]], dtype=np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def _paste_crop(canvas_h, canvas_w, crop, bbox, ox, oy) -> Optional[np.ndarray]:
    """Lay one mask crop onto a blank canvas at its position in the frame.

    ``bbox`` is in frame coordinates and ``(ox, oy)`` is the canvas origin in
    those same coordinates, so every instance drawn with one origin keeps its
    true spatial relationship to the others.
    """
    if crop is None or bbox is None:
        return None
    bx = int(round(float(bbox[0]))) - ox
    by = int(round(float(bbox[1]))) - oy
    ch, cw = crop.shape[:2]

    sx0 = max(0, -bx)
    sy0 = max(0, -by)
    dx0 = max(0, bx)
    dy0 = max(0, by)
    w = min(cw - sx0, canvas_w - dx0)
    h = min(ch - sy0, canvas_h - dy0)
    if w <= 0 or h <= 0:
        return None

    out = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    out[dy0:dy0 + h, dx0:dx0 + w] = (
        crop[sy0:sy0 + h, sx0:sx0 + w].astype(np.uint8) * 255
    )
    return out


def _draw_instance(target, crop, bbox, ox, oy, color, thickness) -> None:
    placed = _paste_crop(target.shape[0], target.shape[1], crop, bbox, ox, oy)
    if placed is None or not placed.any():
        return
    contours, _ = cv2.findContours(placed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(target, contours, -1, color, thickness)


def _focal_extent(focal: List[Optional[Dict]]) -> Optional[Tuple[int, int, int, int]]:
    """Union of the focal animal's boxes across the window."""
    boxes = [f["bbox"] for f in focal if f is not None and f.get("bbox") is not None]
    if not boxes:
        return None
    return (
        int(min(float(b[0]) for b in boxes)),
        int(min(float(b[1]) for b in boxes)),
        int(max(float(b[2]) for b in boxes)),
        int(max(float(b[3]) for b in boxes)),
    )


def generate_scene_composite(
    focal: List[Optional[Dict]],
    neighbors: Optional[List[List[Dict]]] = None,
    output_size: Tuple[int, int] = (128, 128),
    contour_thickness: int = 1,
    context_scale: float = DEFAULT_CONTEXT_SCALE,
) -> np.ndarray:
    """Six-channel time-coded composite of the focal animal and its company.

    Args:
        focal: One entry per frame, ``{"crop": bool array, "bbox": (x1, y1,
            x2, y2)}`` in frame coordinates, or None where the animal was not
            detected on that frame.
        neighbors: Per frame, the same dicts for every *other* animal present.
            None or empty leaves the second half all zeros, which is exactly
            what a solo clip should look like.
        output_size: (height, width) of the square result.
        context_scale: How far past the focal animal's own extent the canvas
            reaches. At 3.0 the focal animal fills about a third of the frame.

    Returns:
        uint8 array of shape (H, W, 6). Channels 0-2 hold the focal animal
        coloured blue through red by frame index; channels 3-5 hold the
        neighbours on the identical canvas.
    """
    out_h, out_w = output_size
    n_frames = len(focal)
    extent = _focal_extent(focal)
    if extent is None:
        return np.zeros((out_h, out_w, 6), dtype=np.uint8)

    x1, y1, x2, y2 = extent
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = max(max(x2 - x1, y2 - y1) * context_scale / 2.0, 8.0)
    ox, oy = int(round(cx - half)), int(round(cy - half))
    side = max(1, int(round(half * 2)))

    focal_layer = np.zeros((side, side, 3), dtype=np.uint8)
    other_layer = np.zeros((side, side, 3), dtype=np.uint8)

    for i in range(n_frames):
        color = _time_color(i, n_frames)
        inst = focal[i]
        if inst is not None:
            _draw_instance(focal_layer, inst.get("crop"), inst.get("bbox"),
                           ox, oy, color, contour_thickness)
        if neighbors and i < len(neighbors):
            for other in (neighbors[i] or []):
                _draw_instance(other_layer, other.get("crop"), other.get("bbox"),
                               ox, oy, color, contour_thickness)

    if side != out_h or side != out_w:
        focal_layer = cv2.resize(focal_layer, (out_w, out_h), interpolation=cv2.INTER_AREA)
        other_layer = cv2.resize(other_layer, (out_w, out_h), interpolation=cv2.INTER_AREA)

    return np.concatenate([focal_layer, other_layer], axis=2)


def scene_composite_preview(scene: np.ndarray) -> np.ndarray:
    """Collapse a six-channel scene into something a person can look at.

    The focal animal keeps its blue-to-red time ramp; neighbours become a flat
    grey wash. The point of the preview is to show *that* the model can see
    company and roughly where it is, not to encode time twice in one picture.
    """
    focal = scene[:, :, :3].astype(np.uint16)
    other = scene[:, :, 3:].max(axis=2).astype(np.uint16)
    out = focal.copy()
    for c in range(3):
        out[:, :, c] = np.maximum(out[:, :, c], (other * 0.55).astype(np.uint16))
    return np.clip(out, 0, 255).astype(np.uint8)


def _instance_geometry(inst: Optional[Dict]) -> Optional[Dict]:
    """Centroid, body length and shape descriptors for one instance."""
    if inst is None:
        return None
    crop = inst.get("crop")
    bbox = inst.get("bbox")
    if crop is None or bbox is None or not np.any(crop):
        return None
    from .mask_tracker_inference import mask_shape_features

    ys, xs = np.nonzero(crop)
    shape = mask_shape_features(crop, ys, xs)
    return {
        "centroid": (float(bbox[0]) + float(xs.mean()),
                     float(bbox[1]) + float(ys.mean())),
        "bbox": tuple(float(v) for v in bbox),
        "major": float(shape.get("major_axis_px") or 0.0),
        "area": float(crop.sum()),
        "elongation": shape.get("elongation"),
        "solidity": shape.get("solidity"),
        "orientation": shape.get("orientation_deg"),
    }


def _fold_angle(deg: float) -> float:
    """Fold an angle into [0, 90].

    A silhouette's major axis is a line rather than an arrow, so orientation
    is only defined modulo 180 degrees. Folding again at 90 makes parallel and
    perpendicular the two extremes, which is the distinction separating
    animals lying alongside each other from one facing another's flank.
    """
    d = abs(deg) % 180.0
    return 180.0 - d if d > 90.0 else d


def pairwise_features(
    focal: List[Optional[Dict]],
    neighbors: Optional[List[List[Dict]]] = None,
    fps: float = 30.0,
) -> Dict[str, float]:
    """Social and postural quantities for one focal animal over one window.

    Distances are divided by the focal animal's own body length, so a value
    means the same thing on a camera high over a mesocosm and one close over
    an open field. Every key in :data:`PAIRWISE_FIELDS` is always returned;
    with no neighbour the social entries take deliberately far-away values and
    ``neighbor_present`` reports zero, so the model can always distinguish
    alone from merely-not-measured.
    """
    n_frames = len(focal)
    geo = [_instance_geometry(f) for f in focal]
    present = [g for g in geo if g is not None]

    lengths = [g["major"] for g in present if g["major"] > 1e-6]
    body = float(np.median(lengths)) if lengths else 1.0

    dists: List[float] = []
    rel_orients: List[float] = []
    bearings: List[float] = []
    contacts = 0
    frames_with_neighbor = 0

    for i in range(n_frames):
        g = geo[i]
        if g is None:
            continue
        raw = (neighbors[i] if neighbors and i < len(neighbors) else []) or []
        others = [o for o in (_instance_geometry(n) for n in raw) if o is not None]
        if not others:
            continue
        frames_with_neighbor += 1

        fx, fy = g["centroid"]
        nearest, best = None, float("inf")
        for o in others:
            d = float(np.hypot(fx - o["centroid"][0], fy - o["centroid"][1]))
            if d < best:
                nearest, best = o, d
        dists.append(best / body)

        # Near-touching boxes, not overlapping masks: huddled animals often
        # merge or occlude, so a strict test would under-count exactly the
        # behavior this exists to catch. See CONTACT_PAD_FRAC.
        pad = CONTACT_PAD_FRAC * body
        bx, ob = g["bbox"], nearest["bbox"]
        if (bx[0] - pad <= ob[2] + pad and ob[0] - pad <= bx[2] + pad
                and bx[1] - pad <= ob[3] + pad and ob[1] - pad <= bx[3] + pad):
            contacts += 1

        if g["orientation"] is not None and nearest["orientation"] is not None:
            rel_orients.append(_fold_angle(g["orientation"] - nearest["orientation"]))
            to_other = np.degrees(np.arctan2(-(nearest["centroid"][1] - fy),
                                             nearest["centroid"][0] - fx))
            bearings.append(_fold_angle(to_other - g["orientation"]))

    steps = [float(np.hypot(b["centroid"][0] - a["centroid"][0],
                            b["centroid"][1] - a["centroid"][1]))
             for a, b in zip(present, present[1:])]
    speed = (float(np.mean(steps)) / body * fps) if steps else 0.0

    elong = [g["elongation"] for g in present if g["elongation"] is not None]
    solid = [g["solidity"] for g in present if g["solidity"] is not None]
    areas = [g["area"] for g in present if g["area"] > 0]

    # Negative means the gap narrowed across the window.
    closing = float(dists[-1] - dists[0]) if len(dists) >= 2 else 0.0
    seen = max(1, len(present))

    return {
        "neighbor_present": frames_with_neighbor / seen,
        "min_dist_norm": float(min(dists)) if dists else 8.0,
        "mean_dist_norm": float(np.mean(dists)) if dists else 8.0,
        "closing_norm": closing,
        "contact_frac": contacts / seen,
        "rel_orient_deg": float(np.mean(rel_orients)) if rel_orients else 45.0,
        "bearing_deg": float(np.mean(bearings)) if bearings else 45.0,
        "focal_speed_norm": speed,
        "focal_elong_mean": float(np.mean(elong)) if elong else 1.0,
        "focal_elong_std": float(np.std(elong)) if elong else 0.0,
        "focal_area_cv": (float(np.std(areas) / np.mean(areas))
                          if len(areas) > 1 and np.mean(areas) > 0 else 0.0),
        "focal_solidity_mean": float(np.mean(solid)) if solid else 1.0,
    }


def pairwise_vector(features: Dict[str, float]) -> np.ndarray:
    """The feature dict as a fixed-order float32 vector for the network."""
    return np.array([float(features.get(k, 0.0)) for k in PAIRWISE_FIELDS],
                    dtype=np.float32)


def load_clip_masks_npz(clip_dir: str) -> Dict[int, List[Optional[Dict]]]:
    """Per-object, per-frame masks saved beside a behavior clip.

    Returns ``{object_id: [{mask, bbox, centroid} or None, ...]}`` with
    full-frame masks, matching what the Behavior tab works with. Frames where
    an object was not detected come back as None rather than an empty mask, so
    callers can tell absence from a zero-area detection.
    """
    npz_path = os.path.join(clip_dir, "masks.npz")
    if not os.path.exists(npz_path):
        return {}
    data = np.load(npz_path)
    out: Dict[int, List[Optional[Dict]]] = {}
    for key in data.files:
        arr = data[key]
        frames: List[Optional[Dict]] = []
        for i in range(arr.shape[0]):
            mask = arr[i]
            ys, xs = np.where(mask)
            if len(ys) == 0:
                frames.append(None)
                continue
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            frames.append({
                "mask": mask,
                "bbox": (x1, y1, x2, y2),
                "centroid": ((x1 + x2) / 2, (y1 + y2) / 2),
            })
        out[int(key.replace("obj_", ""))] = frames
    return out


def scene_sample_from_clip(
    masks_dict: Dict[int, List[Optional[Dict]]],
    focal_obj_id: int,
    output_size: Tuple[int, int] = (128, 128),
    fps: float = 30.0,
    context_scale: float = DEFAULT_CONTEXT_SCALE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build one training sample from a saved clip's per-object masks.

    ``masks_dict`` is what the Behavior tab already loads out of ``masks.npz``:
    object id to a per-frame list of ``{mask, bbox, centroid}`` holding
    full-frame masks. Clips recorded before any of this existed therefore need
    no re-extraction, because the representation is derived and never stored.
    """
    def to_instances(entries):
        out = []
        for e in entries or []:
            if e is None or e.get("mask") is None:
                out.append(None)
                continue
            x1, y1, x2, y2 = (int(v) for v in e["bbox"])
            crop = np.asarray(e["mask"])[y1:y2 + 1, x1:x2 + 1]
            out.append({"crop": crop, "bbox": (x1, y1, x2 + 1, y2 + 1)}
                       if crop.size and crop.any() else None)
        return out

    focal = to_instances(masks_dict.get(focal_obj_id, []))
    others = [to_instances(v) for k, v in masks_dict.items() if k != focal_obj_id]
    neighbors: List[List[Dict]] = []
    for i in range(len(focal)):
        neighbors.append([seq[i] for seq in others
                          if i < len(seq) and seq[i] is not None])

    scene = generate_scene_composite(
        focal, neighbors, output_size=output_size, context_scale=context_scale,
    )
    return scene, pairwise_vector(pairwise_features(focal, neighbors, fps=fps))
