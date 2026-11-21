"""Overhead floor mapping: a synthetic barrel lens over a known floor is undone from the floor outline alone."""
import numpy as np

from fnt.videoTracking import mtt_floor as F

ARENA = (0.2, 3.0, 0.2, 2.8)
TRUE = (-0.30, 0.08, 960.0, 540.0)                     # the camera's lens


def _camera(floor_xy):
    """Floor metres -> pixels through a perspective view and the barrel lens (top of image = high y)."""
    import cv2
    x0, x1, y0, y1 = ARENA
    flo = np.float32([[x0, y1], [x1, y1], [x1, y0], [x0, y0]])
    und = np.float32([[-0.55, -0.48], [0.52, -0.5], [0.5, 0.47], [-0.53, 0.49]])     # undistorted corners
    Hinv = cv2.getPerspectiveTransform(flo, und)
    q = cv2.perspectiveTransform(np.asarray(floor_xy, np.float64).reshape(-1, 1, 2), Hinv).reshape(-1, 2)
    return F.distort(q, *TRUE)


def _outline():
    x0, x1, y0, y1 = ARENA
    s = np.linspace(0, 1, 9)[1:-1]
    sides = [np.column_stack([x0 + (x1 - x0) * s, np.full(len(s), y1)]),        # top
             np.column_stack([np.full(len(s), x1), y1 - (y1 - y0) * s]),        # right
             np.column_stack([x1 - (x1 - x0) * s, np.full(len(s), y0)]),        # bottom
             np.column_stack([np.full(len(s), x0), y0 + (y1 - y0) * s])]        # left
    return _camera(np.vstack(sides))


def test_the_lens_and_walls_are_recovered_from_the_outline():
    fm = F.FloorMap.from_outline(_outline(), ARENA, size=(1920, 1080), flip_y=True)
    assert fm.notes["wall_bow_after"] < fm.notes["wall_bow_before"] / 5
    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(0.3, 2.9, 200), rng.uniform(0.3, 2.7, 200)])
    err = np.linalg.norm(fm.to_floor(_camera(pts)) - pts, axis=1)
    assert np.median(err) < 0.02 and err.max() < 0.06            # within a few cm everywhere, walls included
    back = fm.to_pixels(pts)
    assert np.max(np.linalg.norm(back - _camera(pts), axis=1)) < 3     # and back to pixels


def test_aligning_to_a_shifted_tracking_frame_ignores_wrong_pairs(tmp_path):
    fm = F.FloorMap.from_outline(_outline(), ARENA, size=(1920, 1080))
    rng = np.random.default_rng(1)
    pts = np.column_stack([rng.uniform(0.3, 2.9, 300), rng.uniform(0.3, 2.7, 300)])
    target = pts + [0.12, -0.08]                                   # the tracking system's frame is offset
    target[:60] = rng.uniform(0.2, 3.0, (60, 2))                  # a fifth of the pairs are wrong matches
    out = fm.align(_camera(pts), target)
    assert out["median_before_m"] > 0.1 and out["median_after_m"] < 0.03
    fm.save(tmp_path / "fm.json")
    again = F.FloorMap.load(tmp_path / "fm.json")
    assert np.allclose(again.to_floor(_camera(pts[60:])), target[60:], atol=0.05)
