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
