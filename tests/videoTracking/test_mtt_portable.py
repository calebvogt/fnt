"""Portable Mask Tracker sessions: pack a focus list into a self-contained folder,
label it elsewhere, merge the labels back (mtt_portable, mask_tracker_cli).

Runs under pytest, or directly.
"""
import json
import os
import tempfile

import numpy as np
import pytest

from fnt.videoTracking import mtt_portable as P
from fnt.videoTracking.mask_tracker_cli import main as cli


def _project():
    """A project with three frames: a.png labelled (one mouse), b.png with a
    pending prediction, c.png untouched; categories mouse."""
    d = tempfile.mkdtemp(prefix="mtt_portable_")
    os.makedirs(os.path.join(d, "training_frames"))
    os.makedirs(os.path.join(d, "annotations"))
    import cv2
    for f in ("a.png", "b.png", "c.png"):
        cv2.imwrite(os.path.join(d, "training_frames", f), np.zeros((20, 30, 3), np.uint8))
    with open(os.path.join(d, "project_config.json"), "w") as fh:
        json.dump({"project_dir": d, "video_paths": ["Z:/v.mp4"], "categories": ["mouse"],
                   "sam2_model_path": "C:/x", "source_videos": ["Z:/v.mp4"]}, fh)
    coco = {"images": [{"id": 1, "file_name": "a.png", "width": 30, "height": 20},
                       {"id": 2, "file_name": "b.png", "width": 30, "height": 20},
                       {"id": 5, "file_name": "z.png", "width": 30, "height": 20}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "segmentation": [[1, 1, 5, 1, 5, 5]],
                             "bbox": [1, 1, 4, 4], "area": 8, "iscrowd": 0},
                            {"id": 2, "image_id": 2, "category_id": 1, "segmentation": [[2, 2, 6, 2, 6, 6]],
                             "bbox": [2, 2, 4, 4], "area": 8, "iscrowd": 0, "inferred": True, "score": 0.7},
                            {"id": 3, "image_id": 5, "category_id": 1, "segmentation": [[0, 0, 1, 0, 1, 1]],
                             "bbox": [0, 0, 1, 1], "area": 1, "iscrowd": 0}],
            "categories": [{"id": 1, "name": "mouse", "supercategory": ""}]}
    with open(os.path.join(d, "annotations", "annotations.json"), "w") as fh:
        json.dump(coco, fh)
    with open(os.path.join(d, "focus.txt"), "w") as fh:
        fh.write("b.png\na.png\nc.png\n")
    return d


def _ann(d):
    with open(os.path.join(d, "annotations", "annotations.json")) as fh:
        return json.load(fh)


def test_a_session_holds_the_frames_their_labels_and_opens_as_a_project():
    proj = _project()
    out = tempfile.mkdtemp()
    s = P.pack_session(proj, P.read_focus(os.path.join(proj, "focus.txt")), out, "s1", "label the mice")
    assert sorted(os.listdir(os.path.join(s, "training_frames"))) == ["a.png", "b.png", "c.png"]
    sub = _ann(s)
    assert sorted(i["file_name"] for i in sub["images"]) == ["a.png", "b.png"]   # z.png not in the list
    assert len(sub["annotations"]) == 2
    cfg = json.load(open(os.path.join(s, "project_config.json")))
    assert cfg["video_paths"] == [] and cfg["sam2_model_path"] is None and cfg["categories"] == ["mouse"]
    assert P.read_focus(os.path.join(s, "focus.txt")) == ["b.png", "a.png", "c.png"]
    meta = json.load(open(os.path.join(s, P.SESSION_FILE)))
    assert meta["status"] == "packed" and meta["origin_project"] == os.path.abspath(proj)


def test_labels_made_in_the_session_merge_back_frame_by_frame():
    proj = _project()
    s = P.pack_session(proj, ["b.png", "a.png", "c.png"], tempfile.mkdtemp(), "s2")
    sub = _ann(s)
    # in the session: b's prediction accepted (inferred dropped), c marked empty, a untouched
    for a in sub["annotations"]:
        if a["image_id"] == 2:
            a.pop("inferred")
            a.pop("score")
    sub["images"].append({"id": 9, "file_name": "c.png", "width": 30, "height": 20, "empty": True})
    with open(os.path.join(s, "annotations", "annotations.json"), "w") as fh:
        json.dump(sub, fh)
    res = P.unpack_session(s)
    assert res["frames"] == 3 and not res["conflicts"] and os.path.exists(res["backup"])
    after = _ann(proj)
    by = {i["file_name"]: i for i in after["images"]}
    b_anns = [a for a in after["annotations"] if a["image_id"] == by["b.png"]["id"]]
    assert len(b_anns) == 1 and "inferred" not in b_anns[0]
    assert by["c.png"].get("empty") is True
    assert len([a for a in after["annotations"] if a["image_id"] == by["z.png"]["id"]]) == 1   # untouched
    assert len({a["id"] for a in after["annotations"]}) == len(after["annotations"])           # ids unique
    assert json.load(open(os.path.join(s, P.SESSION_FILE)))["status"] == "merged"


def test_a_frame_changed_in_the_project_since_packing_stops_the_merge():
    proj = _project()
    s = P.pack_session(proj, ["a.png", "b.png"], tempfile.mkdtemp(), "s3")
    coco = _ann(proj)
    coco["annotations"] = [a for a in coco["annotations"] if a["image_id"] != 1]   # a.png edited at home
    with open(os.path.join(proj, "annotations", "annotations.json"), "w") as fh:
        json.dump(coco, fh)
    with pytest.raises(RuntimeError, match="changed in the project since packing"):
        P.unpack_session(s)
    assert P.unpack_session(s, force=True)["conflicts"] == ["a.png"]


def test_the_cli_packs_and_unpacks():
    proj = _project()
    out = tempfile.mkdtemp()
    assert cli(["pack-session", "--project", proj, "--focus", os.path.join(proj, "focus.txt"),
                "--out", out, "--name", "s4"]) == 0
    s = os.path.join(out, "s4")
    assert cli(["unpack-session", "--session", s, "--dry-run"]) == 0
    assert json.load(open(os.path.join(s, P.SESSION_FILE)))["status"] == "packed"   # dry run wrote nothing
    assert cli(["unpack-session", "--session", s]) == 0


def test_a_session_opens_in_mask_tracker_with_its_labels_and_order():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication
    import fnt.videoTracking.mask_tracker_gui as G
    app = QApplication.instance() or QApplication([])
    proj = _project()
    s = P.pack_session(proj, ["b.png", "a.png", "c.png"], tempfile.mkdtemp(), "s5")
    w = G.MaskTrackerWindow()
    w._launched = True
    w._load_project(os.path.join(s, "project_config.json"))
    assert os.path.normcase(w._project_dir) == os.path.normcase(s)
    shown, missing = w.focus_frames(P.read_focus(os.path.join(s, "focus.txt")))
    assert shown == 3 and not missing
    assert [os.path.basename(p) for _, _, p in w._extracted_frames] == ["b.png", "a.png", "c.png"]
    assert len(w._coco.annotations) == 2                     # a's label, b's prediction
    w.close()
    app.processEvents()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
