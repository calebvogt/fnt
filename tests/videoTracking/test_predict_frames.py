"""Predicting a project's frames for review (mtt_predict; mask_tracker_cli
predict-frames, and the window's post-training prediction): only unreviewed
frames are predicted, their old predictions are replaced, a rejected spot is not
offered again, and reviewed frames are left alone.

Runs under pytest, or directly.
"""
import json
import os

import cv2
import numpy as np
import pytest

from fnt.videoTracking import mtt_predict as P
from fnt.videoTracking.mask_tracker_annotator import mask_to_rle

H, W = 48, 64


def _box(x0, y0, x1, y1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


class _Engine:
    """Finds the same two mice on every frame (a speck rides on the first)."""

    def predict(self, img_rgb, confidence_threshold=0.5, max_detections=0):
        a = _box(5, 5, 25, 25)
        a[40, 60] = True
        b = _box(35, 20, 55, 40)
        return {"masks": np.stack([a, b]), "scores": np.array([0.9, 0.4], np.float32),
                "labels": np.array([1, 1]), "boxes": np.array([[5, 5, 25, 25], [35, 20, 55, 40]], np.float32)}


def _ann(aid, img, m, **kw):
    return {"id": aid, "image_id": img, "category_id": 7, "segmentation": mask_to_rle(m),
            "bbox": [0, 0, 1, 1], "area": int(m.sum()), "iscrowd": 0, **kw}


@pytest.fixture
def project(tmp_path):
    d = tmp_path / "p"
    (d / "training_frames").mkdir(parents=True)
    (d / "annotations").mkdir()
    for k in range(5):
        cv2.imwrite(str(d / "training_frames" / f"f{k}.png"), np.zeros((H, W, 3), np.uint8))
    (d / "project_config.json").write_text(json.dumps({"project_dir": str(d), "categories": ["mouse"]}))
    coco = {"images": [{"id": i + 1, "file_name": f"f{i}.png", "width": W, "height": H} for i in range(4)],
            "annotations": [
                _ann(1, 1, _box(0, 0, 10, 10)),                                # f0: labelled
                _ann(2, 3, _box(1, 1, 9, 9), inferred=True, score=0.3),        # f2: an old prediction
                _ann(3, 4, _box(35, 20, 55, 40), inferred=True, rejected=True, score=0.5)],  # f3: rejected
            "categories": [{"id": 7, "name": "mouse", "supercategory": ""}]}
    coco["images"][1]["empty"] = True                                         # f1: empty
    (d / "annotations" / "annotations.json").write_text(json.dumps(coco))
    return d


def test_predicts_only_unreviewed_frames(project, monkeypatch):
    monkeypatch.setattr(P, "load_engine", lambda model_dir, device="auto": _Engine())
    out = P.predict_frames(str(project), "unused", log=lambda m: None)
    assert out["frames"] == 3 and out["predicted"] == 3 and out["backup"]
    c = json.loads((project / "annotations" / "annotations.json").read_text())
    names = {i["id"]: i["file_name"] for i in c["images"]}
    by = {}
    for a in c["annotations"]:
        by.setdefault(names[a["image_id"]], []).append(a)
    assert [a.get("inferred", False) for a in by["f0.png"]] == [False]       # labelled: untouched
    assert "f1.png" not in by                                                 # empty: untouched
    f2 = [a for a in by["f2.png"]]
    assert len(f2) == 2 and all(a["inferred"] and a["id"] != 2 for a in f2)   # old prediction replaced
    f3 = [a for a in by["f3.png"] if not a.get("rejected")]
    assert len(f3) == 1 and f3[0]["score"] == pytest.approx(0.9)              # rejected spot not re-offered
    assert len(by["f4.png"]) == 2                                             # a new frame
    assert all(a["category_id"] == 7 for a in c["annotations"])
    assert max(a["area"] for a in by["f4.png"]) == 400                        # the speck dropped


def test_focus_limits_the_frames(project, monkeypatch):
    monkeypatch.setattr(P, "load_engine", lambda model_dir, device="auto": _Engine())
    out = P.predict_frames(str(project), "unused", frames=["f0.png", "f4.png"], log=lambda m: None)
    assert out["frames"] == 1 and out["masks"] == 2


def test_overlapping_masks_keep_the_higher_score():
    m = _box(5, 5, 25, 25)
    dets = [{"segmentation": mask_to_rle(m), "score": 0.5},
            {"segmentation": mask_to_rle(_box(6, 6, 24, 24)), "score": 0.4}]
    assert P.drop_overlapping(dets, (H, W)) == dets[:1]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
