"""SAM2 image annotation backend for the Mask Tracker Pipeline.

Wraps SAM2ImagePredictor for interactive single-image segmentation and
manages COCO-format annotation export.

Heavy deps (torch, sam2) are imported lazily inside class methods so the
module is safe to import from the GUI even when those packages aren't installed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def _resolve_sam2_config(config_name: str) -> str:
    """Resolve a short config filename to the Hydra config name SAM2 expects.

    SAM2's __init__.py registers the sam2 package as a Hydra config module,
    so build_sam2() needs a path relative to that package root, e.g.
    "configs/sam2.1/sam2.1_hiera_t.yaml".
    """
    if config_name.startswith("configs/"):
        return config_name

    import sam2
    sam2_pkg_dir = Path(sam2.__file__).parent
    for subdir in ("sam2.1", "sam2"):
        candidate = sam2_pkg_dir / "configs" / subdir / config_name
        if candidate.exists():
            return f"configs/{subdir}/{config_name}"

    candidate = sam2_pkg_dir / "configs" / config_name
    if candidate.exists():
        return f"configs/{config_name}"

    raise FileNotFoundError(
        f"Could not find SAM2 config '{config_name}' in sam2 package configs."
    )


class SAM2ImageSegmenter:
    """Wraps SAM2ImagePredictor for interactive single-image segmentation."""

    def __init__(self, checkpoint_path: str, config_name: str, device: str = "auto"):
        self.checkpoint_path = checkpoint_path
        self.config_name = config_name
        self.device = device
        self.predictor = None

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
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_model(self):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        device = self._resolve_device()
        config_path = _resolve_sam2_config(self.config_name)
        sam2_model = build_sam2(config_path, self.checkpoint_path, device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        self.device = device

    def set_image(self, image_rgb: np.ndarray):
        if self.predictor is None:
            self.load_model()
        self.predictor.set_image(image_rgb)

    def predict_mask(
        self,
        positive_points: List[Tuple[int, int]],
        negative_points: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[np.ndarray, float]:
        """Run SAM2 prediction from point prompts.

        Returns (best_mask, score) where best_mask is a boolean H×W array.
        """
        if self.predictor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        import torch

        points = list(positive_points)
        labels = [1] * len(positive_points)
        if negative_points:
            points.extend(negative_points)
            labels.extend([0] * len(negative_points))

        point_coords = np.array(points, dtype=np.float32)
        point_labels = np.array(labels, dtype=np.int32)

        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

        best_idx = int(np.argmax(scores))
        return masks[best_idx].astype(bool), float(scores[best_idx])


# Polygon simplification tolerance (pixels) for CONTOURS DERIVED FROM MASKS.
# Ground truth is stored as RLE (see mask_to_rle) and is pixel-exact; this
# tolerance only affects the editable outline drawn in the GUI and the polygon
# form exported for tools that cannot read RLE (e.g. YOLO).
MASK_SIMPLIFY_EPS = 0.6


# ---------------------------------------------------------------------------
# COCO RLE (run-length encoding)
#
# Masks are MTT's whole premise, so ground truth is stored losslessly rather
# than as a simplified polygon. This is COCO's *uncompressed* RLE — counts as
# a list of ints — deliberately, not the compressed-string variant: encoding
# and decoding it needs only numpy, so a project never becomes unreadable
# because the optional pycocotools extra isn't installed. It is still valid
# COCO; pycocotools reads it via frPyObjects.
#
# Runs are column-major (Fortran order) and always start with a background
# run, per the COCO spec — hence the leading 0 when a mask starts on a
# foreground pixel.
# ---------------------------------------------------------------------------

def mask_to_rle(mask: np.ndarray) -> Dict:
    """Encode a binary mask as COCO uncompressed RLE.

    Run boundaries come from an elementwise ``!=`` rather than ``np.diff``:
    it is about twice as fast on a 1080p mask (this runs once per detection
    per frame during inference) and does not depend on uint8 subtraction
    wrapping around to signal a 1→0 transition.
    """
    m = np.asarray(mask, dtype=np.uint8)
    h, w = m.shape[:2]
    flat = m.ravel(order="F")
    if flat.size == 0:
        return {"counts": [], "size": [int(h), int(w)]}

    change_idx = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate(([0], change_idx, [flat.size]))
    counts = np.diff(bounds)
    if flat[0]:
        counts = np.concatenate(([0], counts))
    return {"counts": counts.astype(np.int64).tolist(),
            "size": [int(h), int(w)]}


def rle_to_mask(rle: Dict) -> np.ndarray:
    """Decode COCO RLE (uncompressed list, or compressed if pycocotools is up)."""
    h, w = int(rle["size"][0]), int(rle["size"][1])
    counts = rle["counts"]

    if isinstance(counts, (str, bytes)):
        # Compressed RLE only shows up in COCO files from other tools.
        from pycocotools import mask as mask_utils
        return mask_utils.decode(rle).astype(bool)

    flat = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    value = 0
    for count in counts:
        count = int(count)
        if count < 0:
            break
        end = min(pos + count, flat.size)
        if value and end > pos:
            flat[pos:end] = 1
        pos = end
        value ^= 1
        if pos >= flat.size:
            break
    return flat.reshape((h, w), order="F").astype(bool)


def is_rle(segmentation) -> bool:
    """True if a COCO segmentation field holds RLE rather than polygons."""
    return isinstance(segmentation, dict) and "counts" in segmentation


def segmentation_to_mask(segmentation, height: int, width: int) -> np.ndarray:
    """Decode any COCO segmentation to a binary mask.

    Accepts RLE (current format) and polygon lists (projects annotated before
    RLE storage), so old and new annotations coexist without a migration.
    """
    if is_rle(segmentation):
        return rle_to_mask(segmentation)
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in segmentation or []:
        if len(poly) < 6:
            continue
        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def segmentation_to_polygons(
    segmentation, height: int, width: int,
    simplify_epsilon: float = MASK_SIMPLIFY_EPS,
) -> List[List[float]]:
    """Get polygon outlines for any COCO segmentation (display / YOLO export)."""
    if is_rle(segmentation):
        return mask_to_coco_polygons(rle_to_mask(segmentation), simplify_epsilon)
    return [list(p) for p in (segmentation or []) if len(p) >= 6]


def mask_to_coco_polygons(mask: np.ndarray, simplify_epsilon: float = MASK_SIMPLIFY_EPS) -> List[List[float]]:
    """Convert a binary mask to COCO polygon format.

    Returns a list of polygons, each a flat list [x1, y1, x2, y2, ...].
    Only returns polygons with >= 6 coordinates (3 points).
    """
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue
        simplified = cv2.approxPolyDP(contour, simplify_epsilon, True)
        if len(simplified) < 3:
            continue
        poly = simplified.flatten().tolist()
        if len(poly) >= 6:
            polygons.append(poly)
    return polygons


def mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Compute COCO-format bbox [x, y, width, height] from a binary mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return (0, 0, 0, 0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return (int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1))


class COCOAnnotationManager:
    """Builds, loads, and saves COCO instance segmentation JSON."""

    def __init__(self):
        self.images: List[Dict] = []
        self.annotations: List[Dict] = []
        self.categories: List[Dict] = []
        self._next_ann_id = 1
        self._next_img_id = 1
        self._next_cat_id = 1
        self._image_id_map: Dict[str, int] = {}
        self._category_name_map: Dict[str, int] = {}
        self.auto_save_path: Optional[str] = None

    def _auto_save(self):
        if self.auto_save_path:
            self.export(self.auto_save_path)

    def add_category(self, name: str) -> int:
        if name in self._category_name_map:
            return self._category_name_map[name]
        cat_id = self._next_cat_id
        self._next_cat_id += 1
        self.categories.append({"id": cat_id, "name": name, "supercategory": ""})
        self._category_name_map[name] = cat_id
        return cat_id

    def remove_category(self, name: str):
        if name not in self._category_name_map:
            return
        cat_id = self._category_name_map.pop(name)
        self.categories = [c for c in self.categories if c["id"] != cat_id]
        self.annotations = [a for a in self.annotations if a["category_id"] != cat_id]
        self._auto_save()

    def get_or_add_image(self, filename: str, width: int, height: int) -> int:
        if filename in self._image_id_map:
            return self._image_id_map[filename]
        img_id = self._next_img_id
        self._next_img_id += 1
        self.images.append({
            "id": img_id,
            "file_name": filename,
            "width": width,
            "height": height,
        })
        self._image_id_map[filename] = img_id
        return img_id

    def add_annotation(
        self,
        image_id: int,
        category_id: int,
        mask: np.ndarray,
        inferred: bool = False,
    ) -> int:
        mask = np.asarray(mask).astype(bool)
        area = int(mask.sum())
        if area == 0:
            return -1
        # Reject masks too small to have a usable outline, matching the old
        # polygon-based guard, but store the mask itself losslessly.
        if not mask_to_coco_polygons(mask):
            return -1
        bbox = mask_bbox(mask)

        ann_id = self._next_ann_id
        self._next_ann_id += 1
        ann = {
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": mask_to_rle(mask),
            "bbox": list(bbox),
            "area": area,
            "iscrowd": 0,
        }
        if inferred:
            ann["inferred"] = True
        self.annotations.append(ann)
        self._auto_save()
        return ann_id

    def add_annotation_from_segmentation(
        self,
        image_id: int,
        category_id: int,
        segmentation,
        bbox: List[float],
        area: int,
        inferred: bool = False,
        score: float = None,
    ) -> int:
        """Add an annotation from a pre-built segmentation (RLE or polygons)."""
        ann_id = self._next_ann_id
        self._next_ann_id += 1
        ann = {
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": segmentation,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0,
        }
        if inferred:
            ann["inferred"] = True
        if score is not None:
            ann["score"] = score
        self.annotations.append(ann)
        self._auto_save()
        return ann_id

    def add_annotation_from_mask(
        self,
        image_id: int,
        category_id: int,
        mask: np.ndarray,
        inferred: bool = False,
        score: float = None,
    ) -> int:
        """Add an annotation from a binary mask, stored losslessly as RLE."""
        mask = np.asarray(mask).astype(bool)
        if not mask.any():
            return -1
        ann_id = self._next_ann_id
        self._next_ann_id += 1
        ann = {
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": mask_to_rle(mask),
            "bbox": list(mask_bbox(mask)),
            "area": int(mask.sum()),
            "iscrowd": 0,
        }
        if inferred:
            ann["inferred"] = True
        if score is not None:
            ann["score"] = score
        self.annotations.append(ann)
        self._auto_save()
        return ann_id

    def approve_annotation(self, ann_id: int):
        """Remove inferred flag and confidence score from an annotation (approve it)."""
        for ann in self.annotations:
            if ann["id"] == ann_id:
                ann.pop("inferred", None)
                ann.pop("score", None)
                self._auto_save()
                return

    def is_inferred(self, ann_id: int) -> bool:
        for ann in self.annotations:
            if ann["id"] == ann_id:
                return ann.get("inferred", False)
        return False

    def count_annotations_for_image(self, image_id: int, exclude_inferred: bool = False) -> int:
        count = 0
        for a in self.annotations:
            if a["image_id"] == image_id:
                if exclude_inferred and a.get("inferred", False):
                    continue
                count += 1
        return count

    def has_inferred_for_image(self, image_id: int) -> bool:
        return any(
            a["image_id"] == image_id and a.get("inferred", False)
            for a in self.annotations
        )

    def image_dims(self, image_id: int) -> Optional[Tuple[int, int]]:
        """(height, width) for an image id, or None if unknown."""
        for img in self.images:
            if img["id"] == image_id:
                return int(img["height"]), int(img["width"])
        return None

    def update_annotation_polygon(self, ann_id: int, points: list):
        """Replace an annotation's shape with a hand-edited polygon.

        The edited outline is rasterised and stored as RLE like every other
        mask, so the storage format stays uniform: readers never need to
        handle a mix of shapes within one project's new annotations.
        """
        if len(points) < 3:
            return
        for ann in self.annotations:
            if ann["id"] != ann_id:
                continue
            dims = self.image_dims(ann["image_id"])
            if dims is None:
                return
            h, w = dims
            mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], 1)
            mask = mask.astype(bool)
            ann["segmentation"] = mask_to_rle(mask)
            ann["bbox"] = list(mask_bbox(mask))
            ann["area"] = int(mask.sum())
            self._auto_save()
            return

    def update_annotation_mask(self, ann_id: int, mask: np.ndarray):
        """Replace an annotation's shape with a binary mask (lossless)."""
        mask = np.asarray(mask).astype(bool)
        for ann in self.annotations:
            if ann["id"] == ann_id:
                ann["segmentation"] = mask_to_rle(mask)
                ann["bbox"] = list(mask_bbox(mask))
                ann["area"] = int(mask.sum())
                self._auto_save()
                return

    def apply_annotation_fields(self, ann_id: int, fields: Dict):
        """Overwrite selected fields on an annotation (undo/redo support)."""
        for ann in self.annotations:
            if ann["id"] == ann_id:
                for k, v in fields.items():
                    ann[k] = v
                self._auto_save()
                return

    def update_annotation_category(self, ann_id: int, category_id: int):
        for ann in self.annotations:
            if ann["id"] == ann_id:
                ann["category_id"] = category_id
                self._auto_save()
                return

    def remove_annotation(self, ann_id: int):
        self.annotations = [a for a in self.annotations if a["id"] != ann_id]
        self._auto_save()

    def restore_annotation(self, ann_dict: Dict):
        """Re-insert a previously removed annotation (undo support).

        The original id is preserved; _next_ann_id is bumped if needed so
        future annotations never collide with restored ones.
        """
        if any(a["id"] == ann_dict["id"] for a in self.annotations):
            return
        self.annotations.append(ann_dict)
        self._next_ann_id = max(self._next_ann_id, ann_dict["id"] + 1)
        self._auto_save()

    def get_annotation(self, ann_id: int) -> Optional[Dict]:
        for a in self.annotations:
            if a["id"] == ann_id:
                return a
        return None

    def set_inferred(self, ann_id: int, inferred: bool, score: float = None):
        """Set or clear the inferred flag (undo support for approve)."""
        for ann in self.annotations:
            if ann["id"] == ann_id:
                if inferred:
                    ann["inferred"] = True
                    if score is not None:
                        ann["score"] = score
                else:
                    ann.pop("inferred", None)
                    ann.pop("score", None)
                self._auto_save()
                return

    def get_annotations_for_image(self, image_id: int) -> List[Dict]:
        return [a for a in self.annotations if a["image_id"] == image_id]

    def get_category_name(self, category_id: int) -> str:
        for c in self.categories:
            if c["id"] == category_id:
                return c["name"]
        return "unknown"

    def export(self, output_path: str, exclude_inferred: bool = False):
        if exclude_inferred:
            anns = [a for a in self.annotations if not a.get("inferred", False)]
        else:
            anns = self.annotations
        data = {
            "images": self.images,
            "annotations": anns,
            "categories": self.categories,
        }
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, json_path: str):
        with open(json_path) as f:
            data = json.load(f)

        self.images = data.get("images", [])
        self.annotations = data.get("annotations", [])
        self.categories = data.get("categories", [])

        self._image_id_map = {img["file_name"]: img["id"] for img in self.images}
        self._category_name_map = {cat["name"]: cat["id"] for cat in self.categories}

        self._next_img_id = max((img["id"] for img in self.images), default=0) + 1
        self._next_ann_id = max((a["id"] for a in self.annotations), default=0) + 1
        self._next_cat_id = max((c["id"] for c in self.categories), default=0) + 1

    def get_stats(self) -> Dict:
        return {
            "num_images": len(self.images),
            "num_annotations": len(self.annotations),
            "num_categories": len(self.categories),
            "categories": [c["name"] for c in self.categories],
            "images_with_annotations": len(set(a["image_id"] for a in self.annotations)),
        }

    def suggest_min_size(self, target_object_px: int = 24) -> Tuple[Optional[int], Dict]:
        """Suggest an optimal image min_size based on annotation and image sizes.

        Finds the smallest mask dimension across all annotations, then
        calculates the resize needed to keep that object at least
        ``target_object_px`` pixels after the shortest-edge resize.
        Returns (suggested_size, details_dict) or (None, {}).
        """
        if not self.annotations or not self.images:
            return None, {}

        img_dims = {img["id"]: (img["width"], img["height"]) for img in self.images}

        smallest_mask_dim = float("inf")
        shortest_edge = None

        for ann in self.annotations:
            _, _, bw, bh = ann["bbox"]
            if bw <= 0 or bh <= 0:
                continue
            smallest_mask_dim = min(smallest_mask_dim, bw, bh)

            img_id = ann["image_id"]
            if img_id in img_dims:
                w, h = img_dims[img_id]
                edge = min(w, h)
                if shortest_edge is None or edge < shortest_edge:
                    shortest_edge = edge

        if smallest_mask_dim == float("inf") or shortest_edge is None:
            return None, {}

        scale = target_object_px / smallest_mask_dim
        raw = shortest_edge * scale
        rounded = max(256, min(1024, int(round(raw / 32) * 32)))
        details = {
            "smallest_mask_dim": smallest_mask_dim,
            "shortest_edge": shortest_edge,
            "scale": scale,
            "raw": raw,
        }
        return rounded, details
