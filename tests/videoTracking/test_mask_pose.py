"""Mask-first pose: keypoints and pixel features from masks, on synthetic mice.

A synthetic mouse is an ellipse (or an egg, narrow at the head) with an optional thin tail
leaving the rump. The tail decides which end is the head; without one, the taper and the
direction of travel do, and the head stays on one end along a track.

Runs under pytest, or directly.
"""
import cv2
import numpy as np
import pytest

from fnt.videoTracking import mask_pose as mp


def mouse(cx=100, cy=100, angle=0.0, length=60, width=24, tail=0, egg=False, size=(220, 260)):
    """Mask of a mouse whose head points along ``angle`` (degrees, image coordinates)."""
    m = np.zeros(size, np.uint8)
    u = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
    v = np.array([-u[1], u[0]])
    c = np.array([cx, cy], float)
    if egg:
        # Narrow toward the head: width falls linearly from rump to nose.
        pts = []
        for s in np.linspace(-1, 1, 60):
            half = (width / 2) * np.sqrt(max(0.0, 1 - s * s)) * (1.15 - 0.55 * (s + 1) / 2)
            pts.append(c + u * s * length / 2 + v * half)
        for s in np.linspace(1, -1, 60):
            half = (width / 2) * np.sqrt(max(0.0, 1 - s * s)) * (1.15 - 0.55 * (s + 1) / 2)
            pts.append(c + u * s * length / 2 - v * half)
        cv2.fillPoly(m, [np.int32(pts)], 1)
    else:
        cv2.ellipse(m, (int(cx), int(cy)), (length // 2, width // 2), angle, 0, 360, 1, -1)
    if tail:
        a = c - u * (length / 2 - 2)
        b = c - u * (length / 2 + tail)
        cv2.line(m, tuple(np.int32(a)), tuple(np.int32(b)), 1, 3)
    return m.astype(bool)


def head_tip(cx, cy, angle, length):
    u = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
    return np.array([cx, cy]) + u * length / 2


def test_tail_marks_the_rear():
    m = mouse(angle=30, tail=50)
    g = mp.mask_geometry(m)
    assert g is not None and g["tail"] is not None
    oriented, _ = mp.orient_track([g], [None], fps=15)
    kp = mp.keypoints(oriented[0])
    assert np.linalg.norm(kp["nose"] - head_tip(100, 100, 30, 60)) < 8
    # The tail base sits at the rump, not out at the tail tip.
    rump = np.array([100, 100]) - np.array([np.cos(np.radians(30)), np.sin(np.radians(30))]) * 30
    assert np.linalg.norm(kp["tail_base"] - rump) < 10
    assert 45 < kp["body_length"] < 75


def test_small_holes_fill_but_a_tail_loop_stays_open():
    m = mouse(tail=0).astype(np.uint8)
    m[100, 100] = 0                                   # a speck: filled
    cv2.circle(m, (40, 100), 22, 1, 3)                # a curled tail touching the rump
    filled = mp.fill_small_holes(m.astype(bool))
    assert filled[100, 100]
    assert not filled[100, 40]                        # inside the loop stays background


def test_motion_picks_the_head_without_a_tail():
    # A plain ellipse moving right: no tail, no taper; motion decides.
    geoms, cents = [], []
    rng = np.random.default_rng(0)
    for i in range(30):
        cx = 60 + 3 * i
        g = mp.mask_geometry(mouse(cx=cx, angle=0, size=(220, 300)))
        if rng.random() < 0.5:
            g = mp.flip_geometry(g)                   # arbitrary end order per frame
        geoms.append(g)
        cents.append(np.array([cx, 100.0]))
    oriented, conf = mp.orient_track(geoms, cents, fps=15)
    for i, g in enumerate(oriented):
        assert g["spine"][0][0] > g["spine"][-1][0], f"frame {i}: head not at the leading end"


def test_taper_picks_the_narrow_end():
    g = mp.mask_geometry(mouse(angle=180, egg=True))
    oriented, _ = mp.orient_track([g, mp.flip_geometry(g)], [None, None], fps=15)
    for o in oriented:
        assert o["spine"][0][0] < o["spine"][-1][0]  # head toward -x (angle 180)


def test_head_stays_put_through_a_weak_contrary_frame():
    # Tail visible on every frame but one, where the evidence is neutral.
    geoms = [mp.mask_geometry(mouse(angle=90, tail=50)) for _ in range(10)]
    geoms[5] = dict(geoms[5], tail=None)
    oriented, _ = mp.orient_track(geoms, [None] * 10, fps=15)
    noses = [mp.keypoints(o)["nose"] for o in oriented]
    assert all(n[1] > 100 for n in noses)             # head toward +y (angle 90)


def test_motion_by_thirds_finds_the_moving_head():
    m = mouse(angle=0, tail=40)
    g = mp.orient_track([mp.mask_geometry(m)], [None], fps=15)[0][0]
    kp = mp.keypoints(g)
    diff = np.zeros(m.shape, np.uint8)
    diff[:, 115:140] = 50                             # only around the head (x 100..130)
    head, mid, rear = mp.motion_by_thirds(diff, m, (0, 0), kp["body_spine"])
    assert head > 5 * max(mid, 1e-3) and rear == 0


def test_body_aligned_crop_points_the_nose_up():
    m = mouse(angle=45, tail=40)
    g = mp.orient_track([mp.mask_geometry(m)], [None], fps=15)[0][0]
    kp = mp.keypoints(g)
    grey = np.full(m.shape, 60, np.uint8)
    nose = np.int32(kp["nose"])
    cv2.circle(grey, tuple(nose), 6, 250, -1)         # a bright head
    crop = mp.body_aligned_crop(grey, m, (0, 0), kp, size=96)
    ys, xs = np.nonzero(crop > 200)
    assert len(ys) and ys.mean() < 40                 # the bright head is in the top part


def test_pair_distances_in_body_lengths():
    a = {"nose": np.array([0.0, 0]), "tail_base": np.array([-50.0, 0]), "mid": np.array([-25.0, 0])}
    b = {"nose": np.array([10.0, 0]), "tail_base": np.array([60.0, 0]), "mid": np.array([35.0, 0])}
    d = mp.pair_distances(a, b, 50)
    assert d["nose_nose"] == pytest.approx(0.2)
    assert d["nose_tailbase"] == pytest.approx(1.2)
    assert d["centroid"] == pytest.approx(1.2)


def test_runs_bridge_short_gaps():
    f = [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1]
    assert mp.runs(f, min_len=3, gap=2) == [(1, 7)]
    assert mp.runs(f, min_len=2, gap=0) == [(1, 2), (5, 7), (13, 14)]
    assert mp.runs([], 1) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
