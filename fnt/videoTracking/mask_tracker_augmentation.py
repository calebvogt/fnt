"""Offline data augmentation and online training transforms for the Mask Tracker Pipeline.

Offline: reads a COCO JSON + images, applies geometric and photometric transforms
independently, writes augmented images + combined COCO JSON (~8x expansion).

Online: returns torchvision.transforms.v2 pipeline for use during training.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Polygon / annotation helpers
# ---------------------------------------------------------------------------

def _transform_polygon(poly: List[float], point_fn: Callable) -> List[float]:
    """Apply a point transform to a flat COCO polygon [x1,y1,x2,y2,...]."""
    out = []
    for i in range(0, len(poly), 2):
        nx, ny = point_fn(poly[i], poly[i + 1])
        out.extend([nx, ny])
    return out


def _bbox_from_polygon(segmentation: List[List[float]]) -> List[float]:
    """Recompute [x, y, w, h] bbox from COCO segmentation polygons."""
    xs, ys = [], []
    for poly in segmentation:
        for i in range(0, len(poly), 2):
            xs.append(poly[i])
            ys.append(poly[i + 1])
    if not xs:
        return [0, 0, 0, 0]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [x_min, y_min, x_max - x_min, y_max - y_min]


def _transform_anns(
    anns: List[Dict],
    point_fn: Callable,
    new_w: int,
    new_h: int,
) -> List[Dict]:
    """Deep-copy and transform all annotations with the given point function."""
    out = []
    for ann in anns:
        a = copy.deepcopy(ann)
        a["segmentation"] = [_transform_polygon(p, point_fn) for p in a["segmentation"]]
        a["bbox"] = _bbox_from_polygon(a["segmentation"])
        out.append(a)
    return out


# ---------------------------------------------------------------------------
# Transform definitions
# ---------------------------------------------------------------------------

def _build_transforms():
    """Build the list of (suffix, img_fn, point_fn_factory) tuples."""
    transforms = []

    transforms.append((
        "hflip",
        lambda img: cv2.flip(img, 1),
        lambda W, H: (lambda x, y: (W - x, y), W, H),
    ))

    transforms.append((
        "vflip",
        lambda img: cv2.flip(img, 0),
        lambda W, H: (lambda x, y: (x, H - y), W, H),
    ))

    transforms.append((
        "rot90",
        lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
        lambda W, H: (lambda x, y: (H - y, x), H, W),
    ))

    transforms.append((
        "rot180",
        lambda img: cv2.rotate(img, cv2.ROTATE_180),
        lambda W, H: (lambda x, y: (W - x, H - y), W, H),
    ))

    transforms.append((
        "rot270",
        lambda img: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
        lambda W, H: (lambda x, y: (y, W - x), H, W),
    ))

    return transforms


_GEOMETRIC = _build_transforms()


def _photometric_transforms():
    """Return list of (suffix, image_fn) for photometric-only transforms."""
    return [
        ("bright_up", lambda img: cv2.convertScaleAbs(img, alpha=1.0, beta=40)),
        ("bright_dn", lambda img: cv2.convertScaleAbs(img, alpha=1.0, beta=-40)),
        ("blur", lambda img: cv2.GaussianBlur(img, (5, 5), 0)),
        ("contrast_up", lambda img: cv2.convertScaleAbs(img, alpha=1.3, beta=0)),
        ("contrast_dn", lambda img: cv2.convertScaleAbs(img, alpha=0.7, beta=0)),
        ("gnoise", _add_gaussian_noise),
    ]


def _add_gaussian_noise(img: np.ndarray) -> np.ndarray:
    noise = np.random.normal(0, 15, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Stochastic geometric transforms (continuous rotation, random scale)
# ---------------------------------------------------------------------------

def _apply_affine_augmentation(
    img: np.ndarray,
    anns: List[Dict],
    M: np.ndarray,
    W: int,
    H: int,
) -> Tuple[np.ndarray, List[Dict]]:
    """Apply a 2x3 affine matrix to image and polygon annotations."""
    aug_img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT_101)

    def point_fn(x, y):
        nx = M[0, 0] * x + M[0, 1] * y + M[0, 2]
        ny = M[1, 0] * x + M[1, 1] * y + M[1, 2]
        return (max(0.0, min(float(W), nx)), max(0.0, min(float(H), ny)))

    transformed = _transform_anns(anns, point_fn, W, H)
    return aug_img, transformed


def _generate_rotation_variants(
    img: np.ndarray,
    anns: List[Dict],
    W: int,
    H: int,
    max_angle: float,
    n_samples: int,
    rng: np.random.RandomState,
) -> List[Tuple[str, np.ndarray, List[Dict]]]:
    """Generate n_samples random rotation variants within ±max_angle."""
    results = []
    cx, cy = W / 2.0, H / 2.0
    for i in range(n_samples):
        angle = rng.uniform(-max_angle, max_angle)
        while abs(angle) < 2.0:
            angle = rng.uniform(-max_angle, max_angle)
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        aug_img, aug_anns = _apply_affine_augmentation(img, anns, M, W, H)
        suffix = f"rot{angle:+.0f}"
        results.append((suffix, aug_img, aug_anns))
    return results


def _generate_scale_variants(
    img: np.ndarray,
    anns: List[Dict],
    W: int,
    H: int,
    scale_range: Tuple[float, float],
    n_samples: int,
    rng: np.random.RandomState,
) -> List[Tuple[str, np.ndarray, List[Dict]]]:
    """Generate n_samples random scale variants within scale_range."""
    results = []
    cx, cy = W / 2.0, H / 2.0
    for i in range(n_samples):
        scale = rng.uniform(scale_range[0], scale_range[1])
        while abs(scale - 1.0) < 0.05:
            scale = rng.uniform(scale_range[0], scale_range[1])
        M = cv2.getRotationMatrix2D((cx, cy), 0, scale)
        aug_img, aug_anns = _apply_affine_augmentation(img, anns, M, W, H)
        suffix = f"scale{scale:.2f}"
        results.append((suffix, aug_img, aug_anns))
    return results


# ---------------------------------------------------------------------------
# Augmentation config
# ---------------------------------------------------------------------------

class AugmentationConfig:
    """Controls which augmentations to apply."""

    def __init__(self):
        # Deterministic geometric
        self.horizontal_flip: bool = True
        self.vertical_flip: bool = True
        self.rotate_90: bool = True
        self.rotate_180: bool = True
        self.rotate_270: bool = True
        # Stochastic geometric
        self.continuous_rotation: bool = False
        self.rotation_max_angle: float = 30.0
        self.rotation_samples: int = 2
        self.random_scale: bool = False
        self.scale_range: Tuple[float, float] = (0.80, 1.20)
        self.scale_samples: int = 2
        # Photometric
        self.brightness: bool = True
        self.contrast: bool = False
        self.gaussian_noise: bool = False
        self.gaussian_blur: bool = True

    @property
    def enabled_geometric(self) -> List[Tuple]:
        flags = [
            self.horizontal_flip,
            self.vertical_flip,
            self.rotate_90,
            self.rotate_180,
            self.rotate_270,
        ]
        return [t for t, flag in zip(_GEOMETRIC, flags) if flag]

    @property
    def enabled_photometric(self) -> List[Tuple]:
        photo = _photometric_transforms()
        flags = [
            self.brightness, self.brightness, self.gaussian_blur,
            self.contrast, self.contrast, self.gaussian_noise,
        ]
        return [t for t, flag in zip(photo, flags) if flag]

    @property
    def expansion_factor(self) -> int:
        n = 1 + len(self.enabled_geometric) + len(self.enabled_photometric)
        if self.continuous_rotation:
            n += self.rotation_samples
        if self.random_scale:
            n += self.scale_samples
        return n


# ---------------------------------------------------------------------------
# Main offline augmentation
# ---------------------------------------------------------------------------

def augment_coco_dataset(
    coco_json_path: str,
    images_dir: str,
    output_dir: str,
    config: Optional[AugmentationConfig] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> str:
    """Augment a COCO dataset on disk.

    Args:
        coco_json_path: path to the input COCO annotations JSON
        images_dir: directory containing the source images
        output_dir: where to write augmented images + annotations.json
        config: which augmentations to enable (default: all)
        progress_callback: called with (current, total) after each image

    Returns:
        Path to the output annotations.json
    """
    if config is None:
        config = AugmentationConfig()

    with open(coco_json_path) as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    ann_by_image: Dict[int, List[Dict]] = {}
    for ann in annotations:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    os.makedirs(output_dir, exist_ok=True)

    new_images = []
    new_annotations = []
    next_img_id = max((img["id"] for img in images), default=0) + 1
    next_ann_id = max((a["id"] for a in annotations), default=0) + 1

    geo_transforms = config.enabled_geometric
    photo_transforms = config.enabled_photometric
    total = len(images)
    rng = np.random.RandomState(42)

    for idx, img_info in enumerate(images):
        fname = img_info["file_name"]
        src_path = os.path.join(images_dir, fname)
        if not os.path.exists(src_path):
            if progress_callback:
                progress_callback(idx + 1, total)
            continue

        img = cv2.imread(src_path)
        if img is None:
            if progress_callback:
                progress_callback(idx + 1, total)
            continue

        H, W = img.shape[:2]
        stem = Path(fname).stem
        ext = Path(fname).suffix

        dst_orig = os.path.join(output_dir, fname)
        if not os.path.exists(dst_orig):
            shutil.copy2(src_path, dst_orig)

        new_images.append(copy.deepcopy(img_info))
        orig_anns = ann_by_image.get(img_info["id"], [])
        new_annotations.extend(copy.deepcopy(orig_anns))

        # --- Deterministic geometric transforms ---
        for suffix, img_fn, point_fn_factory in geo_transforms:
            aug_img = img_fn(img)
            point_fn, new_W, new_H = point_fn_factory(W, H)

            aug_fname = f"{stem}_{suffix}{ext}"
            cv2.imwrite(os.path.join(output_dir, aug_fname), aug_img)

            aug_img_info = {
                "id": next_img_id,
                "file_name": aug_fname,
                "width": new_W,
                "height": new_H,
            }
            new_images.append(aug_img_info)

            transformed = _transform_anns(orig_anns, point_fn, new_W, new_H)
            for a in transformed:
                a["id"] = next_ann_id
                a["image_id"] = next_img_id
                next_ann_id += 1
            new_annotations.extend(transformed)
            next_img_id += 1

        # --- Stochastic geometric: continuous rotation ---
        if config.continuous_rotation:
            for suffix, aug_img, aug_anns in _generate_rotation_variants(
                img, orig_anns, W, H,
                config.rotation_max_angle, config.rotation_samples, rng,
            ):
                aug_fname = f"{stem}_{suffix}{ext}"
                cv2.imwrite(os.path.join(output_dir, aug_fname), aug_img)
                aug_img_info = {
                    "id": next_img_id,
                    "file_name": aug_fname,
                    "width": W,
                    "height": H,
                }
                new_images.append(aug_img_info)
                for a in aug_anns:
                    a["id"] = next_ann_id
                    a["image_id"] = next_img_id
                    next_ann_id += 1
                new_annotations.extend(aug_anns)
                next_img_id += 1

        # --- Stochastic geometric: random scale ---
        if config.random_scale:
            for suffix, aug_img, aug_anns in _generate_scale_variants(
                img, orig_anns, W, H,
                config.scale_range, config.scale_samples, rng,
            ):
                aug_fname = f"{stem}_{suffix}{ext}"
                cv2.imwrite(os.path.join(output_dir, aug_fname), aug_img)
                aug_img_info = {
                    "id": next_img_id,
                    "file_name": aug_fname,
                    "width": W,
                    "height": H,
                }
                new_images.append(aug_img_info)
                for a in aug_anns:
                    a["id"] = next_ann_id
                    a["image_id"] = next_img_id
                    next_ann_id += 1
                new_annotations.extend(aug_anns)
                next_img_id += 1

        # --- Photometric transforms ---
        for suffix, img_fn in photo_transforms:
            aug_img = img_fn(img)
            aug_fname = f"{stem}_{suffix}{ext}"
            cv2.imwrite(os.path.join(output_dir, aug_fname), aug_img)

            aug_img_info = {
                "id": next_img_id,
                "file_name": aug_fname,
                "width": W,
                "height": H,
            }
            new_images.append(aug_img_info)

            for a in copy.deepcopy(orig_anns):
                a["id"] = next_ann_id
                a["image_id"] = next_img_id
                next_ann_id += 1
                new_annotations.append(a)
            next_img_id += 1

        if progress_callback:
            progress_callback(idx + 1, total)

    out_json = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": categories,
    }
    out_path = os.path.join(output_dir, "annotations.json")
    with open(out_path, "w") as f:
        json.dump(out_json, f, indent=2)

    return out_path


# ---------------------------------------------------------------------------
# Online training transforms (torchvision v2)
# ---------------------------------------------------------------------------

def build_training_transforms(cfg):
    """Build torchvision.transforms.v2 pipeline from a MaskRCNNTrainingConfig.

    Returns None if torchvision.transforms.v2 is not available.
    """
    try:
        from torchvision.transforms import v2
    except ImportError:
        return None

    transforms_list = []
    if getattr(cfg, "horizontal_flip", False):
        transforms_list.append(v2.RandomHorizontalFlip(p=0.5))
    brightness = getattr(cfg, "brightness_jitter", 0.0)
    if brightness > 0:
        transforms_list.append(v2.ColorJitter(brightness=brightness))

    if not transforms_list:
        return None
    return v2.Compose(transforms_list)
