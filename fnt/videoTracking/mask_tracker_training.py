"""Mask R-CNN training pipeline for the Mask Tracker Pipeline.

Trains a torchvision Mask R-CNN on COCO-format annotations produced by the
SAM2 annotator. Heavy deps (torch, torchvision) are imported lazily inside
:func:`train_mask_rcnn`.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class MaskRCNNTrainingConfig:
    coco_json_path: str = ""
    images_dir: str = ""
    output_dir: str = ""
    run_name: str = ""
    num_iterations: int = 1000
    learning_rate: float = 0.005
    batch_size: int = 2
    val_fraction: float = 0.2
    device: str = "auto"
    min_size: int = 480
    max_size: int = 800
    num_workers: int = 0
    horizontal_flip: bool = True
    brightness_jitter: float = 0.2
    backbone: str = "resnet50"
    freeze_backbone: bool = True
    optimizer: str = "adamw"
    early_stop_patience: int = 100
    early_stop_min_delta: float = 0.01

    def resolve_run_dir(self) -> str:
        name = self.run_name or datetime.now().strftime("maskrcnn_%Y%m%d_%H%M%S")
        return str(Path(self.output_dir) / name)


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
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_device_description(device: str) -> str:
    """Return a human-readable description of the resolved device."""
    import torch
    if device == "cuda" and torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return f"CUDA ({name})"
    if device == "mps":
        import platform
        chip = platform.processor() or "Apple Silicon"
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0 and result.stdout.strip():
                chip = result.stdout.strip()
        except Exception:
            pass
        return f"MPS ({chip})"
    return "CPU"


def _polygon_to_mask(segmentation: List[List[float]], height: int, width: int) -> np.ndarray:
    """Convert COCO polygon segmentation to a binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in segmentation:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [pts], 1)
    return mask


class COCOInstanceDataset:
    """PyTorch-compatible dataset for COCO instance segmentation annotations.

    Returns (image_tensor, target_dict) in the format expected by
    torchvision detection models.  Images and masks are pre-resized on the
    CPU to ``max_dim`` before conversion to tensors, dramatically reducing
    GPU memory usage and transfer overhead.
    """

    def __init__(self, coco_json: str, images_dir: str, transforms=None,
                 max_dim: int = 0):
        import torch

        with open(coco_json) as f:
            data = json.load(f)

        self.images_dir = images_dir
        self.transforms = transforms
        self.max_dim = max_dim
        self.torch = torch

        self.images = {img["id"]: img for img in data["images"]}
        self.image_ids = sorted(self.images.keys())

        self.annotations_by_image: Dict[int, List[Dict]] = {}
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            self.annotations_by_image.setdefault(img_id, []).append(ann)

        self.categories = {cat["id"]: cat for cat in data["categories"]}
        self.num_classes = len(self.categories) + 1

    def __len__(self):
        return len(self.image_ids)

    def _pre_resize(self, image: np.ndarray, masks: List[np.ndarray],
                    boxes: List[List[float]]) -> Tuple:
        """Resize image and masks so the longest edge <= max_dim."""
        if self.max_dim <= 0:
            return image, masks, boxes
        h, w = image.shape[:2]
        longest = max(h, w)
        if longest <= self.max_dim:
            return image, masks, boxes
        scale = self.max_dim / longest
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        resized_masks = []
        for m in masks:
            resized_masks.append(
                cv2.resize(m, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            )
        scaled_boxes = []
        for bx in boxes:
            scaled_boxes.append([bx[0] * scale, bx[1] * scale,
                                 bx[2] * scale, bx[3] * scale])
        return image, resized_masks, scaled_boxes

    def __getitem__(self, idx):
        torch = self.torch
        img_info = self.images[self.image_ids[idx]]
        img_path = os.path.join(self.images_dir, img_info["file_name"])
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]
        anns = self.annotations_by_image.get(self.image_ids[idx], [])

        boxes = []
        labels = []
        masks = []

        for ann in anns:
            mask = _polygon_to_mask(ann["segmentation"], h, w)
            if mask.sum() == 0:
                continue

            x, y, bw, bh = ann["bbox"]
            if bw <= 0 or bh <= 0:
                continue

            boxes.append([x, y, x + bw, y + bh])
            labels.append(ann["category_id"])
            masks.append(mask)

        image, masks, boxes = self._pre_resize(image, masks, boxes)
        h, w = image.shape[:2]

        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            masks = torch.zeros((0, h, w), dtype=torch.uint8)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([self.image_ids[idx]]),
        }

        image_tensor = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        if self.transforms is not None:
            try:
                from torchvision import tv_tensors
                image_tensor = tv_tensors.Image(image_tensor)
                target["boxes"] = tv_tensors.BoundingBoxes(
                    target["boxes"], format="XYXY", canvas_size=(h, w),
                )
                target["masks"] = tv_tensors.Mask(target["masks"])
                image_tensor, target = self.transforms(image_tensor, target)
            except ImportError:
                pass

        return image_tensor, target


def _build_mobilenet_maskrcnn(backbone_name: str, num_classes: int,
                               min_size: int, max_size: int):
    from torchvision.models.detection import MaskRCNN
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.backbone_utils import mobilenet_backbone

    weights_map = {
        "mobilenet_v3_large": "MobileNet_V3_Large_Weights",
        "mobilenet_v3_small": "MobileNet_V3_Small_Weights",
    }
    import torchvision.models as tv_models
    weights_cls = getattr(tv_models, weights_map[backbone_name])

    fpn_backbone = mobilenet_backbone(
        backbone_name=backbone_name,
        weights=weights_cls.DEFAULT,
        fpn=True,
        trainable_layers=3,
    )
    n_feature_maps = len(fpn_backbone(
        __import__("torch").randn(1, 3, 64, 64)
    ))
    anchor_sizes = ((32, 64, 128, 256, 512),) * n_feature_maps
    aspect_ratios = ((0.5, 1.0, 2.0),) * n_feature_maps
    return MaskRCNN(
        fpn_backbone,
        num_classes=num_classes,
        rpn_anchor_generator=AnchorGenerator(anchor_sizes, aspect_ratios),
        min_size=min_size,
        max_size=max_size,
        rpn_pre_nms_top_n_train=400,
        rpn_post_nms_top_n_train=200,
        rpn_pre_nms_top_n_test=200,
        rpn_post_nms_top_n_test=100,
        box_batch_size_per_image=128,
    )


def _build_model(num_classes: int, backbone: str = "resnet50",
                  min_size: int = 480, max_size: int = 800):
    if backbone in ("mobilenet_v3_large", "mobilenet_v3_small"):
        return _build_mobilenet_maskrcnn(backbone, num_classes, min_size, max_size)

    from torchvision.models.detection import (
        maskrcnn_resnet50_fpn,
        MaskRCNN_ResNet50_FPN_Weights,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    model = maskrcnn_resnet50_fpn(
        weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT,
        min_size=min_size,
        max_size=max_size,
        rpn_pre_nms_top_n_train=400,
        rpn_post_nms_top_n_train=200,
        rpn_pre_nms_top_n_test=200,
        rpn_post_nms_top_n_test=100,
        box_batch_size_per_image=128,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    return model


def _collate_fn(batch):
    return tuple(zip(*batch))


def train_mask_rcnn(
    cfg: MaskRCNNTrainingConfig,
    progress: Optional[Callable] = None,
    should_stop: Optional[Callable] = None,
) -> Dict:
    """Train Mask R-CNN on COCO annotations.

    Args:
        cfg: Training configuration.
        progress: Callback ``progress(iteration, total_iterations, metrics_dict)``.
        should_stop: Callable returning True if training should abort.

    Returns:
        Summary dict with model_path, final metrics, etc.
    """
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

    import sys
    import time as _time

    import torch
    from torch.utils.data import DataLoader, random_split

    print("[MTT Train] PyTorch version:", torch.__version__)

    device = _resolve_device(cfg.device)
    device_desc = get_device_description(device)
    print(f"[MTT Train] Device: {device} — {device_desc}")
    print(f"[MTT Train] Config: iterations={cfg.num_iterations}, lr={cfg.learning_rate}, "
          f"batch={cfg.batch_size}, val_frac={cfg.val_fraction}, min_size={cfg.min_size}")

    run_dir = cfg.resolve_run_dir()
    os.makedirs(run_dir, exist_ok=True)
    print(f"[MTT Train] Run directory: {run_dir}")

    with open(os.path.join(run_dir, "training_config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    from .mask_tracker_augmentation import build_training_transforms
    train_transforms = build_training_transforms(cfg)

    max_dim = int(cfg.max_size)
    print(f"[MTT Train] Loading dataset from: {cfg.coco_json_path}")
    print(f"[MTT Train] Images directory: {cfg.images_dir}")
    print(f"[MTT Train] Pre-resizing images to max_dim={max_dim} before GPU transfer")
    dataset = COCOInstanceDataset(cfg.coco_json_path, cfg.images_dir,
                                  transforms=train_transforms, max_dim=max_dim)
    num_classes = dataset.num_classes
    print(f"[MTT Train] Dataset: {len(dataset)} images, {num_classes} classes "
          f"(including background)")

    n_val = max(1, int(len(dataset) * cfg.val_fraction))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    print(f"[MTT Train] Split: {n_train} train, {n_val} val")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=_collate_fn,
    )

    backbone_name = cfg.backbone
    backbone_labels = {
        "mobilenet_v3_large": "MobileNetV3-Large-FPN",
        "mobilenet_v3_small": "MobileNetV3-Small-FPN",
        "resnet50": "ResNet-50-FPN",
    }
    backbone_label = backbone_labels.get(backbone_name, backbone_name)
    print(f"[MTT Train] Building Mask R-CNN ({backbone_label})...")
    print(f"[MTT Train] Image transform: min_size={cfg.min_size}, max_size={cfg.max_size}")
    model = _build_model(num_classes, backbone=backbone_name,
                         min_size=cfg.min_size, max_size=cfg.max_size)

    if cfg.freeze_backbone:
        frozen = 0
        for name, p in model.named_parameters():
            if "backbone" in name:
                p.requires_grad = False
                frozen += p.numel()
        print(f"[MTT Train] Backbone frozen ({frozen:,} params) — training heads only")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MTT Train] Parameters: {total_params:,} total, {trainable_params:,} trainable")

    print(f"[MTT Train] Moving model to {device}...")
    model.to(device)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=1e-4)
        print(f"[MTT Train] Optimizer: AdamW (lr={cfg.learning_rate})")
    else:
        optimizer = torch.optim.SGD(params, lr=cfg.learning_rate, momentum=0.9, weight_decay=5e-4)
        print(f"[MTT Train] Optimizer: SGD (lr={cfg.learning_rate}, momentum=0.9)")

    warmup_iters = min(100, cfg.num_iterations // 10)
    warmup_factor = 1.0 / max(warmup_iters, 1)

    iteration = 0
    loss_history = []
    best_loss = float("inf")
    early_stopped = False
    _es_window = 20
    _es_best_smoothed = float("inf")
    _es_plateau_since = 0
    data_iter = iter(train_loader)

    category_info = {
        "num_classes": num_classes,
        "categories": dataset.categories,
    }
    with open(os.path.join(run_dir, "training_config.json"), "w") as f:
        json.dump({**asdict(cfg), **category_info}, f, indent=2)

    es_msg = (f", early stop patience={cfg.early_stop_patience}"
              if cfg.early_stop_patience > 0 else "")
    print(f"[MTT Train] Starting training loop ({cfg.num_iterations} iterations, "
          f"warmup={warmup_iters}{es_msg})...")
    sys.stdout.flush()

    while iteration < cfg.num_iterations:
        if should_stop and should_stop():
            print("[MTT Train] Stop requested by user.")
            break

        t0 = _time.time()
        is_first = (iteration == 0)

        if is_first:
            print("[MTT Train] Loading first batch...", flush=True)

        try:
            images, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            images, targets = next(data_iter)

        if is_first:
            print(f"[MTT Train]   Batch loaded: {len(images)} images, "
                  f"shapes: {[tuple(img.shape) for img in images]}", flush=True)
            for ti, t in enumerate(targets):
                print(f"[MTT Train]   Target[{ti}]: boxes={t['boxes'].shape}, "
                      f"masks={t['masks'].shape}, labels={t['labels'].tolist()}", flush=True)

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if is_first:
            print(f"[MTT Train]   Data moved to {device}", flush=True)

        if iteration < warmup_iters:
            lr_scale = warmup_factor + (1.0 - warmup_factor) * (iteration / warmup_iters)
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.learning_rate * lr_scale

        if is_first:
            print("[MTT Train]   Running forward pass...", flush=True)

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        if is_first:
            print(f"[MTT Train]   Forward OK — loss={losses.item():.4f}", flush=True)
            print("[MTT Train]   Running backward pass...", flush=True)

        optimizer.zero_grad()
        losses.backward()

        if is_first:
            print("[MTT Train]   Backward OK — clipping gradients...", flush=True)

        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()

        loss_val = losses.item()
        loss_history.append(loss_val)
        iteration += 1

        if device == "mps" and (is_first or iteration % 10 == 0):
            if is_first:
                print("[MTT Train]   Synchronizing MPS...", flush=True)
            torch.mps.synchronize()
            torch.mps.empty_cache()

        elapsed = _time.time() - t0
        if iteration == 1:
            print(f"[MTT Train]   First iteration complete ({elapsed:.1f}s)", flush=True)
        if True:
            lr_now = optimizer.param_groups[0]["lr"]
            parts = " | ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
            print(f"[MTT Train] Iter {iteration}/{cfg.num_iterations}  "
                  f"loss={loss_val:.4f}  lr={lr_now:.6f}  ({elapsed:.2f}s)  [{parts}]")
            sys.stdout.flush()

        if loss_val < best_loss:
            best_loss = loss_val
            torch.save(model.state_dict(), os.path.join(run_dir, "weights_best.pt"))

        # Early stopping: compare smoothed loss against best smoothed loss
        if iteration >= _es_window and cfg.early_stop_patience > 0:
            smoothed = sum(loss_history[-_es_window:]) / _es_window
            if smoothed < _es_best_smoothed * (1.0 - cfg.early_stop_min_delta):
                _es_best_smoothed = smoothed
                _es_plateau_since = iteration
            elif iteration - _es_plateau_since >= cfg.early_stop_patience:
                print(f"[MTT Train] Early stopping: loss plateaued for "
                      f"{cfg.early_stop_patience} iterations "
                      f"(smoothed loss: {smoothed:.4f})", flush=True)
                early_stopped = True

        if progress:
            metrics = {
                "loss": loss_val,
                "lr": optimizer.param_groups[0]["lr"],
                "best_loss": best_loss,
            }
            for k, v in loss_dict.items():
                metrics[k] = v.item()
            progress(iteration, cfg.num_iterations, metrics)

        if early_stopped:
            break

    stop_reason = "early stop (plateau)" if early_stopped else "complete"
    print(f"[MTT Train] Training {stop_reason}. {iteration} iterations, "
          f"final_loss={loss_history[-1]:.4f}, best_loss={best_loss:.4f}")

    torch.save(model.state_dict(), os.path.join(run_dir, "weights.pt"))
    print(f"[MTT Train] Weights saved to: {os.path.join(run_dir, 'weights.pt')}")

    summary = {
        "model_path": os.path.join(run_dir, "weights.pt"),
        "best_model_path": os.path.join(run_dir, "weights_best.pt"),
        "run_dir": run_dir,
        "num_classes": num_classes,
        "iterations_completed": iteration,
        "final_loss": loss_history[-1] if loss_history else None,
        "best_loss": best_loss,
        "device": device,
        "device_description": device_desc,
        "backbone": backbone_label,
        "early_stopped": early_stopped,
    }
    with open(os.path.join(run_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary
