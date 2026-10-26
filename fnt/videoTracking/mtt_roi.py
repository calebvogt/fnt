"""A camera's region of interest for Mask Tracker models.

A stationary camera sees more than the animals' space: the room around an
arena, the lids of boxes the animals sit under. The ROI, drawn once per
project, says where the model may look:

    crop     [x0, y0, x1, y1]: frames are cut to this box before the model
             sees them (more pixels per mouse at the same input size)
    exclude  polygons [[x, y], ...]: areas filled with flat grey, so nothing
             there is learned or detected (a mouse seen through a box lid)

It lives in ``project_config.json`` under ``"roi"`` with the frame size it
was drawn on (``"frame_size": [w, h]``; other sizes are scaled to it), and is
copied into each model's ``training_config.json``: training exports frames
cut and filled, and every way of running that model (post-training
prediction, Predict, batch sidecars, behaviour and Fusion QC detectors)
prepares frames the same way, maps the masks back to full-frame coordinates
and drops detections centred in excluded or cut-away areas.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

ROI_KEY = "roi"
#: Grey of excluded areas: Ultralytics' letterbox padding, so filled areas look
#: like the padding the model already ignores.
FILL = 114


@dataclass
class ROI:
    crop: Optional[Tuple[int, int, int, int]] = None
    exclude: List[List[Tuple[float, float]]] = field(default_factory=list)
    frame_size: Optional[Tuple[int, int]] = None      # (w, h) it was drawn on
    fill: int = FILL

    # ------------------------------------------------------------- storage
    @classmethod
    def from_dict(cls, d: Optional[Dict]) -> Optional["ROI"]:
        if not d:
            return None
        crop = d.get("crop")
        roi = cls(crop=tuple(int(round(v)) for v in crop) if crop else None,
                  exclude=[[(float(x), float(y)) for x, y in poly]
                           for poly in d.get("exclude", []) if len(poly) >= 3],
                  frame_size=tuple(int(v) for v in d["frame_size"]) if d.get("frame_size") else None,
                  fill=int(d.get("fill", FILL)))
        return None if roi.is_empty() else roi

    def to_dict(self) -> Dict:
        d: Dict = {}
        if self.crop:
            d["crop"] = [int(v) for v in self.crop]
        if self.exclude:
            d["exclude"] = [[[round(x, 1), round(y, 1)] for x, y in poly] for poly in self.exclude]
        if self.frame_size:
            d["frame_size"] = [int(v) for v in self.frame_size]
        if self.fill != FILL:
            d["fill"] = int(self.fill)
        return d

    def is_empty(self) -> bool:
        return not self.crop and not self.exclude

    def describe(self) -> str:
        parts = []
        if self.crop:
            x0, y0, x1, y1 = self.crop
            parts.append(f"crop {x1 - x0}x{y1 - y0} at ({x0}, {y0})")
        if self.exclude:
            parts.append(f"{len(self.exclude)} excluded area(s)")
        return ", ".join(parts) or "none"

    # -------------------------------------------------------------- geometry
    def for_size(self, w: int, h: int) -> "ROI":
        """This ROI on a ``w`` x ``h`` frame (scaled from the size it was drawn
        on, and the crop kept inside the frame)."""
        sx = sy = 1.0
        if self.frame_size and tuple(self.frame_size) != (w, h):
            sx, sy = w / self.frame_size[0], h / self.frame_size[1]
        crop = None
        if self.crop:
            x0, y0, x1, y1 = self.crop
            x0, x1 = sorted((int(round(x0 * sx)), int(round(x1 * sx))))
            y0, y1 = sorted((int(round(y0 * sy)), int(round(y1 * sy))))
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            if x1 - x0 >= 8 and y1 - y0 >= 8:
                crop = (x0, y0, x1, y1)
        exclude = [[(x * sx, y * sy) for x, y in poly] for poly in self.exclude]
        return ROI(crop=crop, exclude=exclude, frame_size=(w, h), fill=self.fill)

    def box(self, w: int, h: int) -> Tuple[int, int, int, int]:
        """The crop on a ``w`` x ``h`` frame (the whole frame without one)."""
        return self.for_size(w, h).crop or (0, 0, w, h)

    def keep_mask(self, h: int, w: int) -> np.ndarray:
        """True where the model may look on an ``h`` x ``w`` frame."""
        key = (h, w)
        cache = self.__dict__.setdefault("_keep_cache", {})
        if key not in cache:
            r = self.for_size(w, h)
            m = np.zeros((h, w), np.uint8)
            x0, y0, x1, y1 = r.crop or (0, 0, w, h)
            m[y0:y1, x0:x1] = 1
            for poly in r.exclude:
                cv2.fillPoly(m, [np.round(np.asarray(poly)).astype(np.int32)], 0)
            cache[key] = m.astype(bool)
        return cache[key]

    def allows(self, x: float, y: float, h: int, w: int) -> bool:
        keep = self.keep_mask(h, w)
        xi, yi = int(x), int(y)
        return 0 <= xi < w and 0 <= yi < h and bool(keep[yi, xi])

    # ------------------------------------------------------------ the frames
    def apply(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """The frame the model sees: cut to the crop, excluded areas filled
        grey. Returns it and its origin (x0, y0) in the full frame."""
        h, w = img.shape[:2]
        x0, y0, x1, y1 = self.box(w, h)
        keep = self.keep_mask(h, w)[y0:y1, x0:x1]
        out = img[y0:y1, x0:x1].copy()
        out[~keep] = self.fill
        return out, (x0, y0)

    def to_frame(self, masks: np.ndarray, h: int, w: int) -> Tuple[np.ndarray, List[int]]:
        """Masks predicted on the prepared frame ([N, crop h, crop w]) back on
        the full ``h`` x ``w`` frame, cut to the area the model may look at.
        Returns them and the indices of those kept: a mask centred in an
        excluded or cut-away area, or with nothing left, is dropped."""
        x0, y0, x1, y1 = self.box(w, h)
        n = len(masks)
        full = np.zeros((n, h, w), bool)
        kept = []
        for i in range(n):
            m = np.asarray(masks[i], bool)
            if m.shape != (y1 - y0, x1 - x0):
                m = cv2.resize(m.astype(np.uint8), (x1 - x0, y1 - y0),
                               interpolation=cv2.INTER_NEAREST) > 0
            full[i, y0:y1, x0:x1] = m
            cut = self.clip(full[i])
            if cut is not None:
                full[i] = cut
                kept.append(i)
        return full, kept

    def clip(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """A full-frame mask cut to the area the model may look at; None when
        it is centred outside that area (or nothing is left)."""
        keep = self.keep_mask(*mask.shape)
        ys, xs = np.nonzero(mask)
        if not ys.size or not keep[int(ys.mean()), int(xs.mean())]:
            return None
        out = mask & keep
        return out if out.any() else None


# ------------------------------------------------------------------ loading
def project_roi(project: str) -> Optional[ROI]:
    """A project's ROI (``project`` is its folder or project_config.json)."""
    path = project if project.endswith(".json") else os.path.join(project, "project_config.json")
    try:
        with open(path, encoding="utf-8") as fh:
            return ROI.from_dict(json.load(fh).get(ROI_KEY))
    except (OSError, ValueError):
        return None


def model_roi(model: str) -> Optional[ROI]:
    """The ROI a model was trained with (``model`` is its run folder or a
    weights file in it); None for models trained on whole frames."""
    d = model if os.path.isdir(model) else os.path.dirname(model)
    try:
        with open(os.path.join(d, "training_config.json"), encoding="utf-8") as fh:
            return ROI.from_dict(json.load(fh).get(ROI_KEY))
    except (OSError, ValueError):
        return None
