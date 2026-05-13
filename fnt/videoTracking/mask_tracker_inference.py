"""Mask R-CNN inference and multi-object tracking for the Mask Tracker Pipeline.

Loads a trained Mask R-CNN checkpoint and runs per-frame inference on video
files, then assembles trajectories via centroid-distance matching (Hungarian
or greedy). Produces trajectory CSVs and annotated output video.

Heavy deps (torch, torchvision, scipy) are imported lazily.
"""
from __future__ import annotations

import json
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
    max_disappeared_frames: int = 0
    iou_match_threshold: float = 0.3
    matching_algorithm: str = "hungarian"
    inference_size: int = 0
    use_masks: bool = False


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


class MultiObjectTracker:
    """Frame-to-frame multi-object tracker using centroid distance matching."""

    def __init__(self, config: MaskInferenceConfig):
        self.config = config
        self.next_object_id = 1
        self.active_tracks: Dict[int, Dict] = {}
        self.disappeared: Dict[int, int] = {}
        self.records: List[Dict] = []
        self._max_match_dist: float = 0.0

    def _get_max_match_dist(self, frame_h: int, frame_w: int) -> float:
        """Auto-compute max matching distance as 15% of frame diagonal."""
        if self._max_match_dist > 0:
            return self._max_match_dist
        self._max_match_dist = 0.15 * np.sqrt(frame_h**2 + frame_w**2)
        return self._max_match_dist

    def update(self, detections: Dict, frame_idx: int, fps: float = 30.0,
               frame_hw: Optional[Tuple[int, int]] = None) -> Dict[int, Dict]:
        """Process detections for one frame and return matched track assignments."""
        n_det = len(detections["scores"]) if len(detections["scores"]) > 0 else 0

        masks = detections["masks"]
        has_masks = masks is not None

        current_dets = []
        for i in range(n_det):
            box = detections["boxes"][i]
            if has_masks:
                mask = masks[i]
                ys, xs = np.where(mask)
                if len(xs) == 0:
                    cx = float((box[0] + box[2]) / 2)
                    cy = float((box[1] + box[3]) / 2)
                else:
                    cx, cy = float(xs.mean()), float(ys.mean())
                area = int(mask.sum())
            else:
                mask = None
                cx = float((box[0] + box[2]) / 2)
                cy = float((box[1] + box[3]) / 2)
                area = int((box[2] - box[0]) * (box[3] - box[1]))
            current_dets.append({
                "centroid": (cx, cy),
                "bbox": box,
                "mask": mask,
                "score": float(detections["scores"][i]),
                "label": int(detections["labels"][i]),
                "area": area,
            })

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

        if not current_dets:
            for obj_id in list(self.active_tracks.keys()):
                self.disappeared[obj_id] = self.disappeared.get(obj_id, 0) + 1
                if self.config.max_disappeared_frames > 0 and self.disappeared[obj_id] > self.config.max_disappeared_frames:
                    del self.active_tracks[obj_id]
                    del self.disappeared[obj_id]
            return {}

        track_ids = list(self.active_tracks.keys())
        n_tracks = len(track_ids)
        n_dets = len(current_dets)

        if frame_hw is not None:
            max_dist = self._get_max_match_dist(frame_hw[0], frame_hw[1])
        elif detections["masks"] is not None and len(detections["masks"]) > 0:
            h, w = detections["masks"][0].shape[:2]
            max_dist = self._get_max_match_dist(h, w)
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
                if self.config.max_disappeared_frames > 0 and self.disappeared[obj_id] > self.config.max_disappeared_frames:
                    del self.active_tracks[obj_id]
                    del self.disappeared[obj_id]

        for col_idx in range(n_dets):
            if col_idx not in used_cols:
                det = current_dets[col_idx]
                at_limit = (self.config.max_detections > 0
                            and len(self.active_tracks) >= self.config.max_detections)
                if at_limit:
                    best_tid = min(
                        self.active_tracks,
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
        row_indices, col_indices = linear_sum_assignment(dist_matrix)

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
        self.records.append({
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
        })

    def get_trajectories(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame()
        df = pd.DataFrame(self.records)
        df = df.sort_values(["object_id", "frame"]).reset_index(drop=True)

        for oid in df["object_id"].unique():
            mask = df["object_id"] == oid
            dx = df.loc[mask, "x"].diff()
            dy = df.loc[mask, "y"].diff()
            dt = df.loc[mask, "time_s"].diff()
            dt = dt.replace(0, np.nan)
            df.loc[mask, "vx"] = (dx / dt).round(2)
            df.loc[mask, "vy"] = (dy / dt).round(2)
            df.loc[mask, "speed"] = (np.sqrt(dx**2 + dy**2) / dt).round(2)

        return df


def _pivot_trajectories(
    df: pd.DataFrame,
    categories: Dict,
    total_frames: int,
    fps: float,
) -> pd.DataFrame:
    """Pivot flat trajectory records into LabGym-style wide format.

    Output columns: frame, time, then one column per tracked object named
    ``{class}_{n}`` (e.g. vole_1, vole_2, port_1).  Each cell contains
    ``(x, y)`` as a string, or empty if the object was not detected that frame.
    """
    if df.empty:
        return pd.DataFrame({"frame": range(total_frames),
                             "time": [round(i / fps, 4) for i in range(total_frames)]})

    cat_map = {}
    for k, v in categories.items():
        cat_map[int(k)] = v

    class_counters: Dict[str, int] = defaultdict(int)
    obj_col_names: Dict[int, str] = {}
    for oid in sorted(df["object_id"].unique()):
        label = int(df.loc[df["object_id"] == oid, "label"].iloc[0])
        class_name = cat_map.get(label, f"class{label}")
        class_counters[class_name] += 1
        obj_col_names[oid] = f"{class_name}_{class_counters[class_name]}"

    all_frames = list(range(total_frames))
    result = pd.DataFrame({
        "frame": all_frames,
        "time": [round(i / fps, 4) for i in all_frames],
    })

    for oid, col_name in obj_col_names.items():
        sub = df.loc[df["object_id"] == oid, ["frame", "x", "y"]].copy()
        coord_map = {
            int(row["frame"]): f"({row['x']}, {row['y']})"
            for _, row in sub.iterrows()
        }
        result[col_name] = result["frame"].map(coord_map).fillna("")

    return result


def _draw_annotations(
    frame_bgr: np.ndarray,
    matched: Dict[int, Dict],
    categories: Dict,
    alpha: float = 0.4,
) -> np.ndarray:
    """Draw track annotations on a frame (mask overlays or bounding boxes)."""
    overlay = frame_bgr.copy()

    cat_map = {}
    for k, v in categories.items():
        cat_map[int(k)] = v

    for obj_id, det in matched.items():
        color = TRACK_COLORS[(obj_id - 1) % len(TRACK_COLORS)]

        if det["mask"] is not None:
            mask = det["mask"]
            overlay[mask] = (
                np.array(overlay[mask], dtype=np.float32) * (1 - alpha)
                + np.array(color, dtype=np.float32) * alpha
            ).astype(np.uint8)
        else:
            box = det["bbox"]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        cx, cy = int(det["centroid"][0]), int(det["centroid"][1])
        class_name = cat_map.get(det["label"], f"class{det['label']}")
        label_text = f"{class_name}_{obj_id}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tx, ty = cx - tw // 2, cy - 8
        cv2.rectangle(overlay, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2), (0, 0, 0), -1)
        cv2.putText(overlay, label_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        cv2.circle(overlay, (cx, cy), 3, color, -1)

    return overlay


def run_inference_on_video(
    video_path: str,
    model_dir: str,
    config: MaskInferenceConfig,
    progress: Optional[Callable] = None,
    should_stop: Optional[Callable] = None,
) -> Dict:
    """Run full inference + tracking pipeline on a video.

    Args:
        video_path: Path to video file.
        model_dir: Directory containing weights.pt and training_config.json.
        config: Inference and tracking configuration.
        progress: Callback ``progress(frame_idx, total_frames)``.
        should_stop: Callable returning True to abort.

    Returns:
        Dict with 'output_dir', 'csv_path', 'num_tracks', 'total_frames'.
    """
    import platform, time

    _saved_fd = None
    _null_fd = None
    if platform.system() == "Darwin":
        try:
            _saved_fd = os.dup(2)
            _null_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(_null_fd, 2)
        except OSError:
            _saved_fd = None

    try:
        inference = MaskRCNNInference(model_dir, device=config.device,
                                      inference_size=config.inference_size,
                                      use_masks=config.use_masks)
        inference.load_model()

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        print(f"[MTT Inference] Video: {os.path.basename(video_path)}")
        print(f"[MTT Inference]   {total_frames} frames, {fps:.1f} fps, {w}x{h}")
        print(f"[MTT Inference]   confidence={config.confidence_threshold}, "
              f"max_detections={config.max_detections}, "
              f"max_disappear={'never' if config.max_disappeared_frames == 0 else config.max_disappeared_frames}, "
              f"matching={config.matching_algorithm}")

        video_stem = Path(video_path).stem
        video_dir = str(Path(video_path).parent)
        output_dir = os.path.join(video_dir, f"{video_stem}_MaskTracker")
        os.makedirs(output_dir, exist_ok=True)

        annotated_path = os.path.join(output_dir, f"{video_stem}_annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(annotated_path, fourcc, fps, (w, h))

        tracker = MultiObjectTracker(config)
        frame_idx = 0
        t_start = time.time()
        t_last_log = t_start
        t_accum = {"read": 0.0, "model": 0.0, "track": 0.0, "draw": 0.0, "write": 0.0}
        t_accum_n = 0

        while True:
            if should_stop and should_stop():
                print(f"[MTT Inference] Stopped by user at frame {frame_idx}/{total_frames}")
                break

            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            t1 = time.time()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = inference.predict(
                frame_rgb, config.confidence_threshold, config.max_detections,
            )
            t2 = time.time()

            matched = tracker.update(detections, frame_idx, fps, frame_hw=(h, w))
            t3 = time.time()

            annotated = _draw_annotations(frame, matched, inference.categories)
            t4 = time.time()

            writer.write(annotated)
            t5 = time.time()

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
        writer.release()

        elapsed = time.time() - t_start
        fps_actual = frame_idx / elapsed if elapsed > 0 else 0
        print(f"[MTT Inference] Done: {frame_idx} frames in {elapsed:.1f}s ({fps_actual:.1f} fps)")

        df_raw = tracker.get_trajectories()

        csv_path = os.path.join(output_dir, "trajectories.csv")
        raw_csv_path = os.path.join(output_dir, "trajectories_detailed.csv")

        n_tracks = 0
        if not df_raw.empty:
            df_pivot = _pivot_trajectories(df_raw, inference.categories, frame_idx, fps)
            df_pivot.to_csv(csv_path, index=False)
            df_raw.to_csv(raw_csv_path, index=False)
            n_tracks = df_raw["object_id"].nunique()
            print(f"[MTT Inference] {n_tracks} unique tracks written to {output_dir}/")
        print(f"[MTT Inference] Annotated video: {annotated_path}")

        return {
            "output_dir": output_dir,
            "csv_path": csv_path,
            "annotated_video": annotated_path,
            "num_tracks": n_tracks,
            "total_frames": frame_idx,
        }

    finally:
        if _saved_fd is not None:
            os.dup2(_saved_fd, 2)
            os.close(_saved_fd)
        if _null_fd is not None:
            os.close(_null_fd)
