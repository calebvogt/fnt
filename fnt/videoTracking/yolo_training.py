"""YOLOv11 instance segmentation training for the Mask Tracker Pipeline.

Converts COCO-format annotations to YOLO format and trains a YOLOv11n-seg
model using the Ultralytics library.  Heavy deps (ultralytics, torch) are
imported lazily inside functions.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import cv2
import numpy as np


@dataclass
class YOLOTrainingConfig:
    coco_json_path: str = ""
    images_dir: str = ""
    output_dir: str = ""
    run_name: str = ""
    num_iterations: int = 100
    learning_rate: float = 0.01
    batch_size: int = 8
    val_fraction: float = 0.2
    device: str = "auto"
    model_variant: str = "yolo11n-seg"
    imgsz: int = 640
    early_stop_patience: int = 50
    freeze_backbone: bool = True
    optimizer: str = "AdamW"
    # Online augmentation (mapped to Ultralytics params)
    aug_fliplr: float = 0.0
    aug_flipud: float = 0.0
    aug_degrees: float = 0.0
    aug_scale: float = 0.0
    aug_hsv_v: float = 0.0
    aug_hsv_s: float = 0.0
    aug_mosaic: float = 0.0

    def resolve_run_dir(self) -> str:
        name = self.run_name or datetime.now().strftime("yolo_%Y%m%d_%H%M%S")
        return str(Path(self.output_dir) / name)


def convert_coco_to_yolo(
    coco_json: str,
    images_dir: str,
    output_dir: str,
    val_fraction: float = 0.2,
) -> str:
    """Convert COCO JSON annotations to YOLO segmentation format.

    Creates the directory structure:
        output_dir/
            images/train/  images/val/
            labels/train/  labels/val/
            data.yaml

    Returns the path to data.yaml.
    """
    with open(coco_json) as f:
        coco = json.load(f)

    cat_id_to_idx: Dict[int, int] = {}
    cat_names: List[str] = []
    for i, cat in enumerate(coco.get("categories", [])):
        cat_id_to_idx[cat["id"]] = i
        cat_names.append(cat["name"])

    img_lookup = {img["id"]: img for img in coco["images"]}
    anns_by_image: Dict[int, list] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    image_ids = sorted(img_lookup.keys())
    n_val = max(1, int(len(image_ids) * val_fraction))
    rng = np.random.RandomState(42)
    rng.shuffle(image_ids)
    val_ids = set(image_ids[:n_val])
    train_ids = set(image_ids[n_val:])

    for split in ("train", "val"):
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    for img_id in image_ids:
        img_info = img_lookup[img_id]
        split = "val" if img_id in val_ids else "train"
        fname = img_info["file_name"]
        w, h = img_info["width"], img_info["height"]

        src = os.path.join(images_dir, fname)
        dst = os.path.join(output_dir, "images", split, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

        label_name = Path(fname).stem + ".txt"
        label_path = os.path.join(output_dir, "labels", split, label_name)

        lines = []
        for ann in anns_by_image.get(img_id, []):
            cls_idx = cat_id_to_idx.get(ann["category_id"])
            if cls_idx is None:
                continue
            for seg in ann["segmentation"]:
                if len(seg) < 6:
                    continue
                coords = []
                for j in range(0, len(seg), 2):
                    nx = seg[j] / w
                    ny = seg[j + 1] / h
                    coords.append(f"{nx:.6f}")
                    coords.append(f"{ny:.6f}")
                lines.append(f"{cls_idx} " + " ".join(coords))

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

    data_yaml_path = os.path.join(output_dir, "data.yaml")
    yaml_content = (
        f"path: {output_dir}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"nc: {len(cat_names)}\n"
        f"names: {cat_names}\n"
    )
    with open(data_yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"[YOLO Convert] {len(train_ids)} train, {len(val_ids)} val images")
    print(f"[YOLO Convert] {len(cat_names)} classes: {cat_names}")
    print(f"[YOLO Convert] data.yaml: {data_yaml_path}")
    return data_yaml_path


def _resolve_device(pref: str) -> str:
    """Resolve training device.  MPS is excluded — YOLO training on Apple
    Metal produces corrupt weights (loss stays ~10x higher than CPU and the
    resulting model cannot detect anything).  MPS inference is fine, but
    training must use CPU or CUDA."""
    import torch
    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if pref == "mps":
        print("[YOLO Train] MPS requested but disabled for training "
              "(known convergence issues) — using CPU instead")
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    # Skip MPS even on "auto" — only use CUDA or CPU for training
    return "cpu"


def train_yolo_seg(
    cfg: YOLOTrainingConfig,
    progress: Optional[Callable] = None,
    should_stop: Optional[Callable] = None,
) -> Dict:
    """Train a YOLOv11 segmentation model on COCO annotations.

    Converts annotations to YOLO format, then runs Ultralytics training.
    Returns a summary dict compatible with the Mask R-CNN training output.
    """
    import logging
    import platform
    import sys
    import time as _time

    import torch

    # Suppress Ultralytics' per-epoch tables and progress bars
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    run_dir = cfg.resolve_run_dir()
    os.makedirs(run_dir, exist_ok=True)

    print(f"[YOLO Train] Run directory: {run_dir}")
    print(f"[YOLO Train] Config: epochs={cfg.num_iterations}, lr={cfg.learning_rate}, "
          f"batch={cfg.batch_size}, imgsz={cfg.imgsz}, model={cfg.model_variant}")

    yolo_data_dir = os.path.join(run_dir, "yolo_dataset")
    data_yaml = convert_coco_to_yolo(
        cfg.coco_json_path, cfg.images_dir, yolo_data_dir, cfg.val_fraction,
    )

    device = _resolve_device(cfg.device)

    # --- Diagnostic info ---
    print(f"[YOLO Train] Python: {platform.python_version()}  |  "
          f"PyTorch: {torch.__version__}  |  OS: {platform.system()} {platform.release()}")
    if "cuda" in device and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        cuda_ver = torch.version.cuda or "N/A"
        print(f"[YOLO Train] GPU: {gpu_name} ({gpu_mem:.1f} GB)  |  CUDA: {cuda_ver}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print(f"[YOLO Train] MPS available (not used for training)")
    print(f"[YOLO Train] Device: {device}")

    if "cuda" in device:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    from ultralytics import YOLO
    try:
        import ultralytics
        print(f"[YOLO Train] Ultralytics: {ultralytics.__version__}")
    except Exception:
        pass

    from .sam2_checkpoint_manager import ensure_yolo_checkpoint
    pretrained_path = ensure_yolo_checkpoint(f"{cfg.model_variant}.pt")
    print(f"[YOLO Train] Pretrained weights: {pretrained_path}")
    model = YOLO(str(pretrained_path))

    # Log model parameter count
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"[YOLO Train] Parameters: {total_params:,} total, {trainable:,} trainable")

    freeze_layers = 10 if cfg.freeze_backbone else 0

    iteration_count = [0]
    best_loss = [float("inf")]
    last_loss = [0.0]
    stopped_early = [False]
    training_log_rows = []

    epoch_t0 = [_time.time()]

    def on_train_epoch_end(trainer):
        """Capture training loss and sub-losses (available before validation)."""
        # Use tloss (epoch-averaged) instead of loss (last batch only).
        # With small batch sizes, last-batch loss is extremely noisy.
        if hasattr(trainer, "tloss") and trainer.tloss is not None:
            loss_val = float(trainer.tloss.mean()) if hasattr(trainer.tloss, "mean") else float(trainer.tloss)
        elif hasattr(trainer, "loss"):
            loss_val = float(trainer.loss)
        else:
            loss_val = 0.0
        if loss_val < best_loss[0]:
            best_loss[0] = loss_val
        last_loss[0] = loss_val
        iteration_count[0] = trainer.epoch + 1
        _epoch_cache["loss"] = loss_val
        _epoch_cache["lr"] = trainer.optimizer.param_groups[0]["lr"]
        sub = {}
        if hasattr(trainer, "tloss") and trainer.tloss is not None:
            names = trainer.loss_names if hasattr(trainer, "loss_names") else []
            items = trainer.tloss.cpu().numpy() if hasattr(trainer.tloss, "cpu") else trainer.tloss
            for i, v in enumerate(items):
                key = names[i] if i < len(names) else f"loss_{i}"
                sub[key] = float(v)
        elif hasattr(trainer, "loss_items") and trainer.loss_items is not None:
            names = trainer.loss_names if hasattr(trainer, "loss_names") else []
            items = trainer.loss_items.cpu().numpy() if hasattr(trainer.loss_items, "cpu") else trainer.loss_items
            for i, v in enumerate(items):
                key = names[i] if i < len(names) else f"loss_{i}"
                sub[key] = float(v)
        _epoch_cache["sub"] = sub

    def on_fit_epoch_end(trainer):
        """Fire after both training and validation — mAP is now available."""
        epoch = trainer.epoch + 1
        total = trainer.epochs
        loss_val = _epoch_cache.get("loss", 0.0)
        sub_losses = _epoch_cache.get("sub", {})
        lr = _epoch_cache.get("lr", 0.0)

        map_vals = {}
        if hasattr(trainer, "metrics") and trainer.metrics:
            m = trainer.metrics
            for key in ("metrics/mAP50(B)", "metrics/mAP50(M)",
                        "metrics/mAP50-95(B)", "metrics/mAP50-95(M)"):
                if key in m:
                    short = key.split("/")[-1]
                    map_vals[short] = float(m[key])

        epoch_sec = _time.time() - epoch_t0[0]
        epoch_t0[0] = _time.time()

        # Single updating line in terminal
        parts = [f"Epoch {epoch}/{total}"]
        parts.append(f"loss={loss_val:.3f}")
        for k in ("box_loss", "seg_loss", "cls_loss"):
            if k in sub_losses:
                short = k.replace("_loss", "")
                parts.append(f"{short}={sub_losses[k]:.3f}")
        if "mAP50(M)" in map_vals:
            parts.append(f"mAP50={map_vals['mAP50(M)']:.3f}")
        elif "mAP50(B)" in map_vals:
            parts.append(f"mAP50={map_vals['mAP50(B)']:.3f}")
        gpu_mem_str = ""
        if "cuda" in device and torch.cuda.is_available():
            gpu_mb = torch.cuda.max_memory_allocated() / 1024**2
            gpu_mem_str = f"  GPU mem: {gpu_mb:.0f}MB"
        elapsed = _time.time() - t0
        parts.append(f"{epoch_sec:.1f}s/epoch")
        parts.append(f"[{elapsed:.0f}s total]")
        print(f"\r[YOLO Train] {' | '.join(parts)}{gpu_mem_str}", end="", flush=True)

        log_row = {"epoch": epoch, "loss": loss_val, "lr": lr, **sub_losses, **map_vals}
        training_log_rows.append(log_row)

        if progress:
            metrics = {
                "loss": loss_val,
                "lr": lr,
                "best_loss": best_loss[0],
                "epoch_time": epoch_sec,
                **sub_losses,
                **map_vals,
            }
            progress(epoch, total, metrics)
        if should_stop and should_stop():
            trainer.stop = True

    _epoch_cache: dict = {}
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    aug_active = (cfg.aug_fliplr > 0 or cfg.aug_flipud > 0 or cfg.aug_degrees > 0
                  or cfg.aug_scale > 0 or cfg.aug_hsv_v > 0)
    if aug_active:
        parts = []
        if cfg.aug_fliplr > 0:
            parts.append(f"fliplr={cfg.aug_fliplr}")
        if cfg.aug_flipud > 0:
            parts.append(f"flipud={cfg.aug_flipud}")
        if cfg.aug_degrees > 0:
            parts.append(f"degrees=±{cfg.aug_degrees}°")
        if cfg.aug_scale > 0:
            parts.append(f"scale=±{cfg.aug_scale}")
        if cfg.aug_hsv_v > 0:
            parts.append(f"hsv_v={cfg.aug_hsv_v}")
        if cfg.aug_hsv_s > 0:
            parts.append(f"hsv_s={cfg.aug_hsv_s}")
        print(f"[YOLO Train] Online augmentation: {', '.join(parts)}")
    else:
        print("[YOLO Train] Augmentation: disabled")

    t0 = _time.time()
    results = model.train(
        data=data_yaml,
        epochs=cfg.num_iterations,
        imgsz=cfg.imgsz,
        batch=cfg.batch_size,
        lr0=cfg.learning_rate,
        patience=cfg.early_stop_patience,
        device=device,
        workers=0,
        project=run_dir,
        name="train",
        exist_ok=True,
        verbose=False,
        freeze=freeze_layers,
        optimizer=cfg.optimizer,
        plots=True,
        save=True,
        # Online augmentation
        fliplr=cfg.aug_fliplr,
        flipud=cfg.aug_flipud,
        degrees=cfg.aug_degrees,
        scale=cfg.aug_scale,
        hsv_v=cfg.aug_hsv_v,
        hsv_s=cfg.aug_hsv_s,
        mosaic=cfg.aug_mosaic,
    )
    elapsed = _time.time() - t0
    print()  # newline after the \r-updating status line

    train_out = os.path.join(run_dir, "train")
    best_pt = os.path.join(train_out, "weights", "best.pt")
    last_pt = os.path.join(train_out, "weights", "last.pt")

    dest_best = os.path.join(run_dir, "weights_best.pt")
    dest_last = os.path.join(run_dir, "weights.pt")
    if os.path.exists(best_pt):
        shutil.copy2(best_pt, dest_best)
    if os.path.exists(last_pt):
        shutil.copy2(last_pt, dest_last)

    # Remove the temporary YOLO dataset conversion directory
    if os.path.isdir(yolo_data_dir):
        shutil.rmtree(yolo_data_dir, ignore_errors=True)

    with open(cfg.coco_json_path) as f:
        coco_data = json.load(f)
    cat_info = {}
    for cat in coco_data.get("categories", []):
        cat_info[str(cat["id"])] = cat["name"]
    num_classes = len(cat_info) + 1

    training_config = {
        **asdict(cfg),
        "architecture": "yolov11-seg",
        "num_classes": num_classes,
        "categories": cat_info,
        "min_size": cfg.imgsz,
        "max_size": cfg.imgsz,
    }
    with open(os.path.join(run_dir, "training_config.json"), "w") as f:
        json.dump(training_config, f, indent=2)

    epochs_completed = iteration_count[0] or cfg.num_iterations
    early_stopped = epochs_completed < cfg.num_iterations

    # Save per-epoch training log
    if training_log_rows:
        import csv as _csv
        log_path = os.path.join(run_dir, "training_log.csv")
        all_keys = list(training_log_rows[0].keys())
        for row in training_log_rows[1:]:
            for k in row:
                if k not in all_keys:
                    all_keys.append(k)
        with open(log_path, "w", newline="") as lf:
            writer = _csv.DictWriter(lf, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(training_log_rows)

    print(f"[YOLO Train] Done: {epochs_completed} epochs in {elapsed:.1f}s")
    print(f"[YOLO Train] Best weights: {dest_best}")

    summary = {
        "model_path": dest_last if os.path.exists(dest_last) else dest_best,
        "best_model_path": dest_best,
        "run_dir": run_dir,
        "num_classes": num_classes,
        "iterations_completed": epochs_completed,
        "final_loss": last_loss[0],
        "best_loss": best_loss[0] if best_loss[0] < float("inf") else None,
        "device": device,
        "device_description": device,
        "backbone": cfg.model_variant,
        "early_stopped": early_stopped,
    }
    with open(os.path.join(run_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary
