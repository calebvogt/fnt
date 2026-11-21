"""Batch video inference (mask_tracker_batch): sidecar masks round-trip, the coarse-to-fine
video order, a run that skips videos already done, and the sidecar layout -- with a fake
model, so no weights or GPU are needed.

Runs under pytest, or directly.
"""
import os

import cv2
import h5py
import numpy as np
import pytest

import fnt.videoTracking.mask_tracker_batch as B


def test_rle_runs_round_trip():
    m = np.zeros((30, 40), bool)
    m[0, 0] = True                                 # starts with a 1: leading 0-run
    m[5:20, 10:25] = True
    m[29, 39] = True
    assert np.array_equal(B.runs_to_mask(B.rle_runs(m), 30, 40), m)


def test_coarse_to_fine_spreads_over_folders_and_cameras():
    vids = [os.path.join("T1", f"CAM13_{i:02d}.mp4") for i in range(8)] + \
           [os.path.join("T1", f"CAM14_{i:02d}.mp4") for i in range(8)] + \
           [os.path.join("T2", f"CAM13_{i:02d}.mp4") for i in range(8)]
    order = B.order_coarse_to_fine(vids)
    assert sorted(order) == sorted(vids)
    first = order[:3]                              # one from each folder/camera first
    assert {(os.path.dirname(v), os.path.basename(v)[:5]) for v in first} == \
           {("T1", "CAM13"), ("T1", "CAM14"), ("T2", "CAM13")}
    assert os.path.basename(order[3])[-6:-4] == "04"   # then the middles


def test_one_per_animal_and_fragments():
    a = np.zeros((40, 40), bool)
    a[5:25, 5:25] = True
    b = np.zeros((40, 40), bool)
    b[6:24, 6:24] = True                           # inside a
    assert B.one_per_animal([a, b], np.array([0.5, 0.9])) == [1]
    s = a.copy()
    s[38, 38] = True
    assert int(B.drop_fragments(s).sum()) == 400


class _T:
    """Stands in for a torch tensor (importing torch before Qt crashes the test
    process on Windows, so the fake model avoids it)."""
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class FakeModel:
    """Predicts one 20 x 30 mask per frame."""
    class _R:
        def __init__(self, h, w):
            m = np.zeros((1, h, w), np.float32)
            m[0, 10:30, 10:40] = 1
            self.masks = type("M", (), {"data": _T(m), "__len__": lambda s: 1})()
            self.boxes = type("Bx", (), {"conf": _T([0.9])})()

    def predict(self, frame, **kw):
        return [self._R(*frame.shape[:2])]


def test_infer_video_writes_a_complete_sidecar(tmp_path):
    video = str(tmp_path / "CAM13_test.mp4")
    vw = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 15, (64, 48))
    for _ in range(30):
        vw.write(np.zeros((48, 64, 3), np.uint8))
    vw.release()
    res = B.infer_video(FakeModel(), video, "fake", step=10, conf=0.25, imgsz=64)
    assert res["complete"] and res["samples"] == 3 and res["detections"] == 3
    with h5py.File(B.sidecar_path(video)) as f:
        assert list(f["frames/frame_idx"][:]) == [0, 10, 20]
        assert int(f["det/area"][0]) == 600
        m = B.runs_to_mask(f["det/rle"][0], 48, 64)
        assert int(m.sum()) == 600 and m[10:30, 10:40].all()
    assert B.sidecar_done(video, "fake", 10, 0.25, 64)
    assert not B.sidecar_done(video, "other", 10, 0.25, 64)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
