"""Mask Tracker's centroid linker (the one before 2026-11-20) at its object limit (no model or video
needed). The default linker is tested in test_mtt_linker.py."""
import numpy as np

from fnt.videoTracking.mask_tracker_inference import MaskInferenceConfig, MultiObjectTracker


def _dets(*centres):
    boxes = np.array([[x - 5, y - 5, x + 5, y + 5] for x, y in centres], float)
    return {"boxes": boxes, "scores": np.full(len(centres), 0.9), "labels": np.ones(len(centres), int),
            "masks": None}


def _ids_at(tr, frame):
    return [r["object_id"] for r in tr.records if r["frame"] == frame]


def test_an_extra_detection_never_takes_an_id_already_used_on_that_frame():
    for matching in ("hungarian", "greedy"):
        tr = MultiObjectTracker(MaskInferenceConfig(max_detections=2, matching_algorithm=matching, linker="centroid"))
        tr.update(_dets((10, 10), (100, 100)), 0)
        out = tr.update(_dets((12, 10), (102, 100), (600, 600)), 1)     # a third, far away: the limit is 2
        ids = _ids_at(tr, 1)
        assert sorted(ids) == [1, 2] and len(ids) == len(set(ids))       # each animal once, the extra left out
        assert out[1]["centroid"] == (12.0, 10.0) and out[2]["centroid"] == (102.0, 100.0)


def test_at_the_limit_a_lost_animal_is_picked_up_where_it_reappears():
    tr = MultiObjectTracker(MaskInferenceConfig(max_detections=2, linker="centroid"))
    tr.update(_dets((10, 10), (100, 100)), 0)
    tr.update(_dets((102, 100), (700, 650)), 1)          # animal 1 is now far off (beyond the match distance)
    by_id = {r["object_id"]: (r["x"], r["y"]) for r in tr.records if r["frame"] == 1}
    assert by_id == {2: (102.0, 100.0), 1: (700.0, 650.0)}
