"""The activity gate (frame_activity): frame sizes read from the MP4 index match the video,
moving seconds stand out from still ones, the frame plan (moving / uncertain / keyframes),
and a gated batch sidecar -- with a fake model, so no weights or GPU are needed.

Runs under pytest, or directly.
"""
import os

import cv2
import h5py
import numpy as np
import pytest

import fnt.videoTracking.frame_activity as A
import fnt.videoTracking.mask_tracker_batch as B
from test_batch_inference import FakeModel

FPS = 15


def _video(path, seconds=30, moving=(10, 15), size=(160, 120)):
    """A grey scene with a textured patch that changes during the ``moving`` seconds (an
    encoder predicts a flat shape sliding along too well to stand for an animal)."""
    w, h = size
    rng = np.random.default_rng(0)
    still = rng.integers(0, 255, (30, 30, 3), dtype=np.uint8)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for i in range(seconds * FPS):
        img = np.full((h, w, 3), 90, np.uint8)
        s = i / FPS
        if moving[0] <= s < moving[1]:
            x = 20 + int((s - moving[0]) * 40) % 100
            img[40:70, x:x + 30] = rng.integers(0, 255, (30, 30, 3), dtype=np.uint8)
        else:
            img[40:70, 20:50] = still
        vw.write(img)
    vw.release()
    return path


def test_sizes_come_from_the_index(tmp_path):
    v = _video(str(tmp_path / "CAM13_2021-10-25_02_46_13_295.mp4"))
    fs = A.mp4_frame_sizes(v)
    cap = cv2.VideoCapture(v)
    assert len(fs.sizes) == int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 30 * FPS
    cap.release()
    assert fs.fps == pytest.approx(FPS)
    assert fs.keys[0] == 0 and np.all(np.diff(fs.keys) > 0)
    assert fs.sizes.sum() <= os.path.getsize(v)


def test_no_index_is_none(tmp_path):
    v = _video(str(tmp_path / "CAM13_a.mp4"))
    cut = str(tmp_path / "CAM13_cut.mp4")
    with open(v, "rb") as src, open(cut, "wb") as dst:       # a recording cut off before its index
        dst.write(src.read(os.path.getsize(v) // 2))
    assert A.mp4_frame_sizes(cut) is None
    assert A.mp4_frame_sizes(str(tmp_path / "missing.mp4")) is None


def test_moving_seconds_stand_out_and_are_planned(tmp_path):
    v = _video(str(tmp_path / "CAM13_2021-10-25_02_46_13_295.mp4"))
    act = A.activity_ratios([v], baseline_s=60)
    fs, ratio = act[v]
    assert len(ratio) == 30
    assert np.all(ratio[10:15] > 2)                  # the square moves
    assert np.nanmax(ratio[:9]) < 1.3 and np.nanmax(ratio[16:]) < 1.3
    frames, tiers = A.plan_frames(fs, ratio, pad_s=1, low_fps=2)
    busy = frames[tiers == A.TIER_BUSY]
    assert set(range(9 * FPS, 16 * FPS)) <= set(busy)        # moving seconds, padded by 1 s
    assert busy.min() == 9 * FPS and busy.max() == 16 * FPS - 1
    rest = frames[tiers == A.TIER_KEY]
    assert set(rest) <= set(fs.keys) and 0 in rest
    assert len(frames) < 0.4 * len(fs.sizes)


def test_plan_tiers():
    fs = A.FrameSizes(np.ones(10 * FPS, np.uint32), np.arange(0, 10 * FPS, 50), float(FPS))
    ratio = np.array([1, 1, 1.5, 1, 1, 3, 1, 1, 1, np.nan])
    frames, tiers = A.plan_frames(fs, ratio, pad_s=0, low_fps=2)
    t = dict(zip(frames, tiers))
    assert all(t[f] == A.TIER_BUSY for f in range(5 * FPS, 6 * FPS))
    assert all(t[f] == A.TIER_BUSY for f in range(9 * FPS, 10 * FPS))     # unknown: busy
    assert sorted(f for f, k in t.items() if k == A.TIER_LOW) == [2 * FPS + 4, 2 * FPS + 12]     # every 8th frame, from the 5th
    assert [f for f, k in t.items() if k == A.TIER_KEY] == [0, 50, 100]
    frames, tiers = A.plan_frames(fs, ratio, pad_s=1, low_fps=0, busy_fps=5)
    busy = frames[tiers == A.TIER_BUSY]
    assert len(busy) == 5 * 3 + 5 * 2                  # seconds 4-6 and 8-9, five a second
    assert not np.any(tiers == A.TIER_LOW)


def test_gated_sidecar(tmp_path):
    v = _video(str(tmp_path / "CAM13_2021-10-25_02_46_13_295.mp4"))
    gate = dict(A.DEFAULT_GATE, baseline_s=60)
    fs, ratio = A.activity_ratios([v], baseline_s=60)[v]
    frames, tiers = A.plan_frames(fs, ratio, **gate)
    res = B.infer_video(FakeModel(), v, "fake", step=250, conf=0.25, imgsz=64,
                        frames=frames, tiers=tiers, gate=gate)
    assert res["complete"] and res["samples"] == len(frames)
    with h5py.File(B.sidecar_path(v)) as f:
        assert list(f["frames/frame_idx"][:]) == list(frames)
        assert list(f["frames/tier"][:]) == list(tiers)
        assert int(f.attrs["step"]) == 0
    assert B.sidecar_done(v, "fake", 250, 0.25, 64, gate)
    assert not B.sidecar_done(v, "fake", 250, 0.25, 64)              # not the ungated run
    assert not B.sidecar_done(v, "fake", 250, 0.25, 64, dict(gate, hi=3.0))


def test_read_ahead_reads_the_right_frames(tmp_path):
    """Frames read forward (grab) or by a seek are the frames asked for."""
    v = str(tmp_path / "CAM13_count.mp4")
    vw = cv2.VideoWriter(v, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 48))
    rng = np.random.default_rng(1)
    for i in range(120):
        vw.write(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8))
    vw.release()
    cap = cv2.VideoCapture(v)
    truth = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        truth.append(fr)
    cap.release()
    seen = {}

    def on_frame(si, n, frame, masks):
        seen[si] = frame.copy()

    want = [0, 3, 4, 5, 40, 41, 100, 7]                  # forward, a jump, and back
    B.infer_video(FakeModel(), v, "fake", 250, 0.25, 64, frames=want, on_frame=on_frame)
    for si, fi in enumerate(want):
        assert np.array_equal(seen[si], truth[fi]), fi


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
