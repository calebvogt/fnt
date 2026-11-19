"""Overhead camera pixels -> floor metres: lens, walls, and (optionally) the UWB frame.

A wide overhead lens bends straight walls outward (barrel distortion), so a plain perspective mapping is off
near the walls. The floor outline drawn for a camera (its ROI, which follows the arena's walls) fixes this
without a calibration target:

1. **Lens**: radial distortion (k1, k2) and its centre are fitted so that the outline's points along each wall
   become a straight line.
2. **Walls**: the four straightened walls are intersected for the floor's corners, and a homography maps them
   onto the arena rectangle in floor-plan metres. ``flip_x`` / ``flip_y`` say which way round the camera looks.
3. **UWB frame** (optional): an affine fitted to matched (floor, UWB) positions, robustly (RANSAC), moves the
   floor plan onto the tracking system's coordinates (a small shift, scale and rotation).

``FloorMap.to_floor(xy)`` applies all three; ``FloorMap.to_pixels(xy)`` goes back (for drawing floor-plan
shapes onto the video). Saved as JSON.

    python -m fnt.videoTracking.mtt_floor fit --outline <run_settings.json | roi json> \
        --arena X0 X1 Y0 Y1 [--flip-x] [--flip-y] --out floor_map.json [--check-frame frame.png]

VoleCosm (2026-11-18): the overhead camera (cam5, VC-T005) - outline bow 1.6% -> 0.6% of a wall's length; after
the UWB affine 71% of overhead positions within 0.2 m of UWB (67% without the lens step).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

F_NOMINAL = 1000.0           # focal length the distortion is expressed against (only k1, k2 scale with it)


def _cv2():
    import cv2
    return cv2


def wall_points(outline: np.ndarray, band: float = 0.08, corner: float = 0.15,
                max_off: float = 0.006) -> Dict[str, np.ndarray]:
    """The outline's vertices along each wall: within ``band`` (share of the outline's span) of a side of its
    bounding box, more than ``corner`` from the ends of that side (the corners are where boxes or notches sit),
    and on the wall's smooth curve: a point more than ``max_off`` of the span off a parabola through the others
    (a notch, a cut round an object) is left out. Returns {"top", "bottom", "left", "right": (n, 2) arrays};
    image y grows downwards."""
    p = np.asarray(outline, float)
    x0, y0 = p.min(0)
    x1, y1 = p.max(0)
    w, h = x1 - x0, y1 - y0
    x, y = p[:, 0], p[:, 1]
    mid_x = (x > x0 + corner * w) & (x < x1 - corner * w)
    mid_y = (y > y0 + corner * h) & (y < y1 - corner * h)
    walls = {"top": p[(y < y0 + band * h) & mid_x], "bottom": p[(y > y1 - band * h) & mid_x],
             "left": p[(x < x0 + band * w) & mid_y], "right": p[(x > x1 - band * w) & mid_y]}
    for k, q in walls.items():
        along, off = (0, 1) if k in ("top", "bottom") else (1, 0)
        span = w if along == 0 else h
        while len(q) > 4:                            # drop the worst point while one sits off the curve
            c = np.polyfit(q[:, along], q[:, off], 2)
            r = np.abs(np.polyval(c, q[:, along]) - q[:, off])
            if r.max() <= max_off * span:
                break
            q = np.delete(q, np.argmax(r), axis=0)
        walls[k] = q
    return walls


def _K(cx, cy):
    return np.array([[F_NOMINAL, 0, cx], [0, F_NOMINAL, cy], [0, 0, 1]], float)


def undistort(xy, k1, k2, cx, cy) -> np.ndarray:
    """Pixels -> undistorted normalised image coordinates."""
    cv2 = _cv2()
    q = np.asarray(xy, np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(q, _K(cx, cy), np.array([k1, k2, 0, 0], float)).reshape(-1, 2)


def distort(q, k1, k2, cx, cy) -> np.ndarray:
    """Undistorted normalised coordinates -> pixels."""
    cv2 = _cv2()
    pts = np.column_stack([np.asarray(q, float).reshape(-1, 2), np.ones(len(np.asarray(q).reshape(-1, 2)))])
    im, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), _K(cx, cy), np.array([k1, k2, 0, 0], float))
    return im.reshape(-1, 2)


def _line(q):
    c = q.mean(0)
    _, _, vt = np.linalg.svd(q - c)
    return c, vt[0], vt[1]


def _bow(q):
    c, d, n = _line(q)
    return (q - c) @ n / max(np.ptp(q @ d), 1e-9)


def fit_lens(walls: Dict[str, np.ndarray], size: Tuple[int, int], fit_centre: bool = True) -> Tuple[np.ndarray, float, float]:
    """(k1, k2, cx, cy), the walls' bow before and after (RMS, share of a wall's length)."""
    from scipy.optimize import least_squares
    w, h = size
    p0 = np.array([0.0, 0.0, w / 2, h / 2])
    free = [0, 1, 2, 3] if fit_centre else [0, 1]

    def resid(v):
        p = p0.copy()
        p[free] = v
        return np.concatenate([_bow(undistort(q, *p)) for q in walls.values() if len(q) >= 3])

    before = float(np.sqrt(np.mean(resid(p0[free]) ** 2)))
    lo = [-1.5, -1.5, w * 0.3, h * 0.3][:len(free)]
    hi = [1.5, 1.5, w * 0.7, h * 0.7][:len(free)]
    r = least_squares(resid, p0[free], bounds=(lo, hi))
    p = p0.copy()
    p[free] = r.x
    return p, before, float(np.sqrt(np.mean(r.fun ** 2)))


@dataclass
class FloorMap:
    lens: List[float]                                  # k1, k2, cx, cy
    H: List[List[float]]                               # undistorted -> floor-plan metres
    A: List[List[float]] = field(default_factory=lambda: [[1, 0, 0], [0, 1, 0]])   # floor plan -> UWB frame
    arena: Optional[List[float]] = None                # x0, x1, y0, y1 (m)
    notes: Dict = field(default_factory=dict)

    @classmethod
    def from_outline(cls, outline, arena: Sequence[float], size=(1920, 1080), flip_x=False, flip_y=True,
                     fit_centre=True) -> "FloorMap":
        """Lens and walls from a floor outline. ``flip_y`` (default): the image's top is the floor plan's high y
        (a camera looking down with the plan's y axis pointing up the image)."""
        cv2 = _cv2()
        walls = wall_points(outline)
        missing = [k for k, v in walls.items() if len(v) < 3]
        if missing:
            raise ValueError(f"the outline has fewer than 3 points along: {', '.join(missing)}")
        lens, before, after = fit_lens(walls, size, fit_centre)
        L = {k: _line(undistort(v, *lens)) for k, v in walls.items()}

        def meet(a, b):
            (c1, d1, _), (c2, d2, _) = a, b
            t = np.linalg.solve(np.column_stack([d1, -d2]), c2 - c1)
            return c1 + t[0] * d1

        img = np.float32([meet(L["top"], L["left"]), meet(L["top"], L["right"]),
                          meet(L["bottom"], L["right"]), meet(L["bottom"], L["left"])])
        x0, x1, y0, y1 = arena
        xs = (x1, x0) if flip_x else (x0, x1)
        ys = (y1, y0) if flip_y else (y0, y1)                                  # image top first
        flo = np.float32([[xs[0], ys[0]], [xs[1], ys[0]], [xs[1], ys[1]], [xs[0], ys[1]]])
        H = cv2.getPerspectiveTransform(img, flo)
        return cls(lens=[float(v) for v in lens], H=H.tolist(), arena=[float(v) for v in arena],
                   notes={"wall_bow_before": before, "wall_bow_after": after, "flip_x": flip_x, "flip_y": flip_y})

    def _plan(self, xy) -> np.ndarray:
        cv2 = _cv2()
        q = undistort(xy, *self.lens).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(q, np.array(self.H, float)).reshape(-1, 2)

    def to_floor(self, xy) -> np.ndarray:
        A = np.array(self.A, float)
        return self._plan(xy) @ A[:, :2].T + A[:, 2]

    def to_pixels(self, floor_xy) -> np.ndarray:
        cv2 = _cv2()
        A = np.vstack([np.array(self.A, float), [0, 0, 1]])
        f = np.column_stack([np.asarray(floor_xy, float).reshape(-1, 2), np.ones(len(np.asarray(floor_xy).reshape(-1, 2)))])
        plan = (np.linalg.inv(A) @ f.T).T[:, :2]
        q = cv2.perspectiveTransform(plan.reshape(-1, 1, 2), np.linalg.inv(np.array(self.H, float))).reshape(-1, 2)
        return distort(q, *self.lens)

    def align(self, pixels, targets, threshold: float = 0.25) -> Dict:
        """Fit the affine onto ``targets`` (floor metres, e.g. UWB) from matched ``pixels``; robust to wrong
        pairs. Returns the median distance before and after and the share of pairs kept."""
        cv2 = _cv2()
        plan = self._plan(pixels)
        t = np.asarray(targets, float)
        before = float(np.median(np.linalg.norm(self.to_floor(pixels) - t, axis=1)))
        A, inl = cv2.estimateAffine2D(plan.astype(np.float32), t.astype(np.float32), method=cv2.RANSAC,
                                      ransacReprojThreshold=threshold)
        if A is None:
            raise ValueError("no affine fits these pairs")
        self.A = A.tolist()
        after = float(np.median(np.linalg.norm(self.to_floor(pixels) - t, axis=1)))
        out = {"median_before_m": before, "median_after_m": after, "kept": float(inl.mean()), "pairs": len(t)}
        self.notes["align"] = out
        return out

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=1)

    @classmethod
    def load(cls, path: str) -> "FloorMap":
        with open(path, encoding="utf-8") as fh:
            return cls(**json.load(fh))


def _outline_from(path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    """A floor outline from a Mask Tracker run_settings.json (its roi) or an ROI json (include polygons)."""
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    roi = d.get("roi", d)
    poly = max(roi["include"], key=len)
    size = tuple(roi.get("frame_size") or (1920, 1080))
    return np.asarray(poly, float), size


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="lens and walls from a floor outline")
    f.add_argument("--outline", required=True, help="a Mask Tracker run_settings.json or an ROI json")
    f.add_argument("--arena", nargs=4, type=float, required=True, metavar=("X0", "X1", "Y0", "Y1"),
                   help="the arena's walls in floor-plan metres")
    f.add_argument("--flip-x", action="store_true", help="the image's left is the plan's high x")
    f.add_argument("--no-flip-y", dest="flip_y", action="store_false",
                   help="the image's top is the plan's LOW y (default: high y)")
    f.add_argument("--out", required=True)
    f.add_argument("--check-frame", help="draw the floor grid on this frame next to the output (a .jpg)")
    a = ap.parse_args(argv)
    outline, size = _outline_from(a.outline)
    fm = FloorMap.from_outline(outline, a.arena, size=size, flip_x=a.flip_x, flip_y=a.flip_y)
    fm.save(a.out)
    print(f"lens k1 {fm.lens[0]:+.3f} k2 {fm.lens[1]:+.3f} centre ({fm.lens[2]:.0f}, {fm.lens[3]:.0f}); wall bow "
          f"{fm.notes['wall_bow_before']:.4f} -> {fm.notes['wall_bow_after']:.4f} of a wall's length; -> {a.out}")
    if a.check_frame:
        cv2 = _cv2()
        im = cv2.imread(a.check_frame)
        x0, x1, y0, y1 = a.arena
        for gx in np.linspace(x0, x1, 12):
            cv2.polylines(im, [np.int32(fm.to_pixels(np.column_stack([np.full(60, gx), np.linspace(y0, y1, 60)])))],
                          False, (0, 255, 255), 1, cv2.LINE_AA)
        for gy in np.linspace(y0, y1, 12):
            cv2.polylines(im, [np.int32(fm.to_pixels(np.column_stack([np.linspace(x0, x1, 60), np.full(60, gy)])))],
                          False, (0, 255, 255), 1, cv2.LINE_AA)
        for q in outline:
            cv2.circle(im, tuple(np.int32(q)), 4, (0, 0, 255), -1)
        out = a.out.rsplit(".", 1)[0] + "_check.jpg"
        cv2.imwrite(out, im)
        print("check:", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
