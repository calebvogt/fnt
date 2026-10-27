"""Predicting a Mask Tracker project's frames with a trained model, for review.

The model's masks go into the project's annotations as pending predictions
(``inferred``, with a ``score``), where the reviewer accepts, rejects or fixes
them. Used by the window's post-training prediction and Predict, and headless
by ``mask_tracker_cli predict-frames`` (any device: a CPU run leaves the GPU to
another job).

Which frames: those not yet reviewed -- no accepted mask and not marked empty.
Their earlier predictions are replaced; a spot the reviewer rejected on a frame
is not offered again there.
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional

import cv2
import numpy as np

from .mask_tracker_batch import drop_fragments


def drop_overlapping(detections: list, shape, max_overlap: float = 0.5) -> list:
    """One mask per animal: drop a predicted mask that is mostly (more than
    ``max_overlap`` of its area) inside a higher-scoring one. The first zone
    model stacked masks on masks."""
    from .mask_tracker_annotator import segmentation_to_mask
    kept, kept_masks = [], []
    for d in sorted(detections, key=lambda d: -d.get("score", 0.0)):
        m = segmentation_to_mask(d["segmentation"], shape[0], shape[1]).astype(bool)
        area = int(m.sum())
        if area and any(np.count_nonzero(m & k) > max_overlap * area for k in kept_masks):
            continue
        kept.append(d)
        kept_masks.append(m)
    return kept


def load_engine(model_dir: str, device: str = "auto"):
    """The inference engine for a model run folder (YOLO or Mask R-CNN), loaded."""
    arch = "maskrcnn"
    cfg_path = os.path.join(model_dir, "training_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            arch = json.load(f).get("architecture", "maskrcnn")
    if arch == "yolov11-seg":
        from .yolo_inference import YOLOInference
        engine = YOLOInference(model_dir, device=device, use_masks=True)
    else:
        from .mask_tracker_inference import MaskRCNNInference
        engine = MaskRCNNInference(model_dir, device=device)
    engine.load_model()
    return engine


def frame_detections(engine, img_rgb: np.ndarray, confidence: float, max_det: int) -> List[Dict]:
    """The model's masks on one frame, as COCO detections ({segmentation (RLE),
    bbox, area, category_id (1-based class index), score}): specks dropped,
    one mask per animal."""
    from .mask_tracker_annotator import mask_bbox, mask_to_coco_polygons, mask_to_rle
    result = engine.predict(img_rgb, confidence_threshold=confidence, max_detections=max_det)
    detections = []
    for j in range(len(result.get("scores", []))):
        mask = None
        if result.get("masks") is not None and j < len(result["masks"]):
            mask = result["masks"][j]
        if mask is not None:
            # Store the predicted mask itself -- approving one of these
            # promotes it to ground truth, so it must not arrive already
            # coarsened into a polygon.
            if not mask_to_coco_polygons(mask):
                continue
            mask_bool = drop_fragments(np.asarray(mask).astype(bool))
            segmentation = mask_to_rle(mask_bool)
            bbox = list(mask_bbox(mask_bool))
            area = int(mask_bool.sum())
        else:
            # Box-only model: rectangle mask covering the box.
            x1, y1, x2, y2 = (float(v) for v in result["boxes"][j][:4])
            h_img, w_img = img_rgb.shape[:2]
            rect = np.zeros((h_img, w_img), dtype=bool)
            rect[max(0, int(round(y1))):max(0, int(round(y2))),
                 max(0, int(round(x1))):max(0, int(round(x2)))] = True
            if not rect.any():
                continue
            segmentation = mask_to_rle(rect)
            bbox = [x1, y1, x2 - x1, y2 - y1]
            area = int(rect.sum())
        detections.append({"segmentation": segmentation, "bbox": bbox, "area": area,
                           "category_id": int(result["labels"][j]),
                           "score": float(result["scores"][j])})
    return drop_overlapping(detections, img_rgb.shape[:2])


def unreviewed_frames(coco, names: Iterable[str]) -> List[str]:
    """The frames (file names) with no accepted mask and not empty (marked
    empty, or every prediction on it rejected)."""
    out = []
    for name in names:
        img_id = coco._image_id_map.get(name)
        if img_id is not None and (coco.count_annotations_for_image(img_id, exclude_inferred=True) > 0
                                   or coco.is_image_empty(name)):
            continue
        out.append(name)
    return out


def clear_predictions(coco, names: Iterable[str]) -> None:
    """Remove the pending (not rejected) predictions on these frames."""
    for name in names:
        img_id = coco._image_id_map.get(name)
        if img_id is None:
            continue
        for ann in coco.get_annotations_for_image(img_id):
            if ann.get("inferred", False) and not ann.get("rejected"):
                coco.remove_annotation(ann["id"])


def add_predictions(coco, filename: str, width: int, height: int, detections: List[Dict]) -> float:
    """Add a frame's detections as pending predictions; returns the top score.
    A prediction on a spot the reviewer already rejected (mostly the same mask:
    IoU > 0.5) is not offered again."""
    from .mask_tracker_annotator import segmentation_to_mask
    img_id = coco.get_or_add_image(filename, width, height)
    rejected = [segmentation_to_mask(a["segmentation"], height, width).astype(bool)
                for a in coco.get_annotations_for_image(img_id) if a.get("rejected")]
    if rejected:
        kept = []
        for det in detections:
            m = segmentation_to_mask(det["segmentation"], height, width).astype(bool)
            if not any(np.count_nonzero(m & r) > 0.5 * np.count_nonzero(m | r) for r in rejected):
                kept.append(det)
        detections = kept
    cat_ids = [c["id"] for c in coco.categories]
    max_score = 0.0
    for det in detections:
        label = det["category_id"]            # 1-based index into the categories
        cat_id = cat_ids[label - 1] if 1 <= label <= len(cat_ids) else (cat_ids[0] if cat_ids else None)
        if cat_id is None:
            continue
        score = det.get("score", 0.0)
        max_score = max(max_score, score)
        coco.add_annotation_from_segmentation(
            image_id=img_id, category_id=cat_id, segmentation=det["segmentation"],
            bbox=det["bbox"], area=det["area"], inferred=True, score=score)
    return max_score


def predict_frames(project: str, model_dir: str, frames: Optional[Iterable[str]] = None,
                   confidence: float = 0.25, max_det: int = 100, device: str = "auto",
                   log: Callable[[str], None] = print,
                   should_stop: Optional[Callable[[], bool]] = None) -> Dict:
    """Predict a project's unreviewed frames (``frames``: names in its
    training_frames/, default all) and save the predictions into its
    annotations.json (backed up first). Returns counts."""
    from .mask_tracker_annotator import COCOAnnotationManager
    pdir = os.path.dirname(project) if project.endswith(".json") else project
    ann_path = os.path.join(pdir, "annotations", "annotations.json")
    frames_dir = os.path.join(pdir, "training_frames")
    coco = COCOAnnotationManager()
    if os.path.exists(ann_path):
        coco.load(ann_path)
    names = [os.path.basename(f) for f in frames] if frames is not None else sorted(
        f for f in os.listdir(frames_dir) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    todo = [n for n in unreviewed_frames(coco, names) if os.path.exists(os.path.join(frames_dir, n))]
    out = {"frames": len(todo), "predicted": 0, "masks": 0, "backup": None}
    if not todo:
        log("[predict] no unreviewed frames")
        return out
    engine = load_engine(model_dir, device)
    log(f"[predict] {os.path.basename(os.path.normpath(model_dir))}: {len(todo)} unreviewed frame(s), "
        f"confidence >= {confidence:.2f}")
    clear_predictions(coco, todo)
    for i, name in enumerate(todo):
        if should_stop is not None and should_stop():
            break
        img = cv2.imread(os.path.join(frames_dir, name))
        if img is None:
            continue
        dets = frame_detections(engine, cv2.cvtColor(img, cv2.COLOR_BGR2RGB), confidence, max_det)
        if dets:
            add_predictions(coco, name, img.shape[1], img.shape[0], dets)
            out["predicted"] += 1
            out["masks"] += len(dets)
        if (i + 1) % 25 == 0:
            log(f"[predict] {i + 1}/{len(todo)}")
    if os.path.exists(ann_path):
        out["backup"] = ann_path + f".bak-{datetime.now():%Y%m%d_%H%M%S}-predict"
        shutil.copy2(ann_path, out["backup"])
    coco.export(ann_path)
    log(f"[predict] {out['masks']} mask(s) on {out['predicted']} of {len(todo)} frame(s)")
    return out
