"""A camera's region of interest for Mask Tracker models.

A stationary camera sees more than the animals' space: the room around an
arena, the lids of boxes the animals sit under. The ROI, drawn once per
project, says where the model may look:

    include  polygons [[x, y], ...]: the area the model sees (none: the whole
             frame). Frames are cut to the box around it before the model
             sees them (more pixels per mouse at the same input size), and
             everything outside it is filled flat grey.
    exclude  polygons: holes in that area, filled grey too, so nothing there
             is learned or detected (a mouse seen through a box lid)

It lives in ``project_config.json`` under ``"roi"`` with the frame size it
was drawn on (``"frame_size": [w, h]``; other sizes are scaled to it), and is
copied into each model's ``training_config.json``: training exports frames
cut and filled, and every way of running that model (post-training
prediction, Predict, batch sidecars, behaviour and Fusion QC detectors)
prepares frames the same way, maps the masks back to full-frame coordinates
and drops detections centred outside the area.
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

Polygon = List[Tuple[float, float]]


def _polys(raw) -> List[Polygon]:
    return [[(float(x), float(y)) for x, y in p] for p in raw or [] if len(p) >= 3]


@dataclass
class ROI:
    include: List[Polygon] = field(default_factory=list)
    exclude: List[Polygon] = field(default_factory=list)
    frame_size: Optional[Tuple[int, int]] = None      # (w, h) it was drawn on
    fill: int = FILL

    # ------------------------------------------------------------- storage
    @classmethod
    def from_dict(cls, d: Optional[Dict]) -> Optional["ROI"]:
        if not d:
            return None
        include = _polys(d.get("include"))
        if d.get("crop"):                   # the first version's box (x1, y1 exclusive)
            x0, y0, x1, y1 = (float(v) for v in d["crop"])
            include.append([(x0, y0), (x1 - 1, y0), (x1 - 1, y1 - 1), (x0, y1 - 1)])
        roi = cls(include=include, exclude=_polys(d.get("exclude")),
                  frame_size=tuple(int(v) for v in d["frame_size"]) if d.get("frame_size") else None,
                  fill=int(d.get("fill", FILL)))
        return None if roi.is_empty() else roi

    def to_dict(self) -> Dict:
        d: Dict = {}
        for key in ("include", "exclude"):
            polys = getattr(self, key)
            if polys:
                d[key] = [[[round(x, 1), round(y, 1)] for x, y in p] for p in polys]
        if self.frame_size:
            d["frame_size"] = [int(v) for v in self.frame_size]
        if self.fill != FILL:
            d["fill"] = int(self.fill)
        return d

    def is_empty(self) -> bool:
        return not self.include and not self.exclude

    def describe(self) -> str:
        parts = []
        if self.include:
            w, h = self.frame_size or (0, 0)
            if w and h:
                x0, y0, x1, y1 = self.box(w, h)
                parts.append(f"area {x1 - x0}x{y1 - y0} at ({x0}, {y0})")
            else:
                parts.append(f"{len(self.include)} included area(s)")
        if self.exclude:
            parts.append(f"{len(self.exclude)} excluded area(s)")
        return ", ".join(parts) or "none"

    # -------------------------------------------------------------- geometry
    def for_size(self, w: int, h: int) -> "ROI":
        """This ROI on a ``w`` x ``h`` frame (scaled from the size it was drawn on)."""
        sx = sy = 1.0
        if self.frame_size and tuple(self.frame_size) != (w, h):
            sx, sy = w / self.frame_size[0], h / self.frame_size[1]

        def scale(polys):
            return [[(x * sx, y * sy) for x, y in p] for p in polys]
        return ROI(include=scale(self.include), exclude=scale(self.exclude),
                   frame_size=(w, h), fill=self.fill)

    def keep_mask(self, h: int, w: int) -> np.ndarray:
        """True where the model may look on an ``h`` x ``w`` frame."""
        key = (h, w)
        cache = self.__dict__.setdefault("_keep_cache", {})
        if key not in cache:
            r = self.for_size(w, h)
            if r.include:
                m = np.zeros((h, w), np.uint8)
                for p in r.include:
                    cv2.fillPoly(m, [np.round(np.asarray(p)).astype(np.int32)], 1)
            else:
                m = np.ones((h, w), np.uint8)
            for p in r.exclude:
                cv2.fillPoly(m, [np.round(np.asarray(p)).astype(np.int32)], 0)
            cache[key] = m.astype(bool)
        return cache[key]

    def box(self, w: int, h: int) -> Tuple[int, int, int, int]:
        """The box frames are cut to on a ``w`` x ``h`` frame: around the area
        the model may look at (the whole frame if that is empty)."""
        cache = self.__dict__.setdefault("_box_cache", {})
        if (w, h) not in cache:
            ys, xs = np.nonzero(self.keep_mask(h, w))
            cache[(w, h)] = ((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
                             if ys.size else (0, 0, w, h))
        return cache[(w, h)]

    def allows(self, x: float, y: float, h: int, w: int) -> bool:
        keep = self.keep_mask(h, w)
        xi, yi = int(x), int(y)
        return 0 <= xi < w and 0 <= yi < h and bool(keep[yi, xi])

    # ------------------------------------------------------------ the frames
    def apply(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """The frame the model sees: cut to the box, everything outside the
        area filled grey. Returns it and its origin (x0, y0) in the full frame."""
        h, w = img.shape[:2]
        x0, y0, x1, y1 = self.box(w, h)
        keep = self.keep_mask(h, w)[y0:y1, x0:x1]
        out = img[y0:y1, x0:x1].copy()
        out[~keep] = self.fill
        return out, (x0, y0)

    def to_frame(self, masks: np.ndarray, h: int, w: int) -> Tuple[np.ndarray, List[int]]:
        """Masks predicted on the prepared frame ([N, box h, box w]) back on
        the full ``h`` x ``w`` frame, cut to the area the model may look at.
        Returns them and the indices of those kept: a mask centred outside the
        area, or with nothing left, is dropped."""
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
