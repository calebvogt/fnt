"""YOLOv11 inference backend for the Mask Tracker Pipeline.

Drop-in replacement for MaskRCNNInference — produces the same output dict
format so the tracker, video annotator, and CSV export work unchanged.

Heavy deps (ultralytics, torch) are imported lazily.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

import cv2
import numpy as np


class YOLOInference:
    """Loads a trained YOLOv11-seg model and runs per-frame inference."""

    def __init__(self, model_dir: str, device: str = "auto",
                 inference_size: int = 0, use_masks: bool = True):
        self.model_dir = model_dir
        self.device = device
        self.model = None
        self.num_classes = 2
        self.categories: Dict = {}
        self._inference_size: int = inference_size
        self._use_masks: bool = use_masks
        self._frame_count: int = 0
        self._empty_count: int = 0

    def load_model(self):
        from ultralytics import YOLO

        device = self._resolve_device()
        self.device = device

        config_path = os.path.join(self.model_dir, "training_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            self.num_classes = config.get("num_classes", 2)
            self.categories = config.get("categories", {})
            if self._inference_size == 0:
                self._inference_size = config.get("imgsz", 640)

        weights_path = os.path.join(self.model_dir, "weights_best.pt")
        if not os.path.exists(weights_path):
            weights_path = os.path.join(self.model_dir, "weights.pt")

        self.model = YOLO(weights_path)

        imgsz = self._inference_size if self._inference_size > 0 else 640
        mask_str = "masks" if self._use_masks else "boxes-only"
        print(f"[YOLO Inference] Model loaded on {device}, "
              f"inference={imgsz}px, {mask_str}")

    def _resolve_device(self) -> str:
        import torch
        if self.device == "cpu":
            return "cpu"
        if self.device == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "mps":
            return "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def predict(
        self,
        frame_rgb: np.ndarray,
        confidence_threshold: float = 0.5,
        max_detections: int = 0,
    ) -> Dict:
        """Run inference on a single frame.

        Returns dict with keys: boxes, masks, scores, labels (numpy arrays).
        Same format as MaskRCNNInference.predict().
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        orig_h, orig_w = frame_rgb.shape[:2]
        imgsz = self._inference_size if self._inference_size > 0 else 640
        max_det = max_detections if max_detections > 0 else 300

        # Ultralytics expects BGR numpy arrays (OpenCV convention).
        # The pipeline passes RGB, so we flip channels before predict.
        frame_bgr = frame_rgb[:, :, ::-1]

        results = self.model.predict(
            frame_bgr,
            conf=confidence_threshold,
            max_det=max_det,
            imgsz=imgsz,
            device=self.device,
            verbose=False,
            retina_masks=True,
        )

        if not results or len(results) == 0:
            return {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "masks": None,
                "scores": np.zeros(0, dtype=np.float32),
                "labels": np.zeros(0, dtype=np.int32),
            }

        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = r.boxes.conf.cpu().numpy().astype(np.float32)
        raw_cls = r.boxes.cls.cpu().numpy().astype(int)

        labels = np.array([c + 1 for c in raw_cls], dtype=np.int32)

        # Diagnostic logging
        self._frame_count += 1
        n = len(scores)
        if n == 0:
            self._empty_count += 1
            if self._empty_count == 1:
                print(f"[YOLO Inference] WARNING: 0 detections on frame {self._frame_count} "
                      f"(conf>={confidence_threshold}, imgsz={imgsz})")
            elif self._empty_count == 10:
                print(f"[YOLO Inference] WARNING: 0 detections on 10 of first "
                      f"{self._frame_count} frames — check confidence threshold "
                      f"and resolution settings")
        elif self._frame_count == 1:
            top = float(scores[0]) if len(scores) > 0 else 0
            print(f"[YOLO Inference] First frame: {n} detections, "
                  f"top score={top:.3f}, classes={raw_cls.tolist()}")

        masks = None
        if self._use_masks and r.masks is not None:
            mask_data = r.masks.data.cpu().numpy()
            n = mask_data.shape[0]
            if mask_data.shape[1] != orig_h or mask_data.shape[2] != orig_w:
                full_masks = np.zeros((n, orig_h, orig_w), dtype=bool)
                for i in range(n):
                    full_masks[i] = cv2.resize(
                        mask_data[i].astype(np.uint8), (orig_w, orig_h),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                masks = full_masks
            else:
                masks = mask_data.astype(bool)

        return {
            "boxes": boxes,
            "masks": masks,
            "scores": scores,
            "labels": labels,
        }
