"""Empty frames and Space in the Mask Tracker.

A frame the reviewer marks empty (N, or the Empty Frame button) is reviewed and
holds no object: it trains as a background image, Enter and post-training
inference skip it, and marking it drops its pending masks. The training export
holds only reviewed images -- an image whose masks are all still pending used
to go in with no annotation and train as "no mouse here".

Space walks the current video's queued frames and, past the last one, carries
on to the next frame in the queue instead of wrapping back within the video.

Runs under pytest, or directly.
"""
import json
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

import fnt.videoTracking.mask_tracker_gui as G  # noqa: E402
from fnt.videoTracking.mask_tracker_annotator import COCOAnnotationManager  # noqa: E402
from fnt.videoTracking.yolo_training import convert_coco_to_yolo  # noqa: E402


def square_mask(h=48, w=64):
    m = np.zeros((h, w), bool)
    m[10:30, 10:40] = True
    return m


def manager():
    c = COCOAnnotationManager()
    cat = c.add_category("mouse")
    return c, cat


def test_marking_empty_drops_pending_and_refuses_accepted():
    c, cat = manager()
    pend = c.get_or_add_image("pend.png", 64, 48)
    c.add_annotation_from_mask(pend, cat, square_mask(), inferred=True, score=0.4)
    assert c.set_image_empty("pend.png", 64, 48) == pend
    assert c.is_image_empty("pend.png")
    assert c.get_annotations_for_image(pend) == []          # pending mask dropped

    acc = c.get_or_add_image("acc.png", 64, 48)
    c.add_annotation_from_mask(acc, cat, square_mask())
    assert c.set_image_empty("acc.png", 64, 48) == -1       # has an accepted mask
    assert not c.is_image_empty("acc.png")

    c.set_image_empty("pend.png", 64, 48, empty=False)
    assert not c.is_image_empty("pend.png")


def test_a_mask_drawn_later_wins_over_the_empty_flag():
    c, cat = manager()
    img = c.set_image_empty("f.png", 64, 48)
    c.add_annotation_from_mask(img, cat, square_mask())
    assert not c.is_image_empty("f.png")
    assert c.count_empty_images() == 0


def test_training_export_holds_only_reviewed_images(tmp_path):
    c, cat = manager()
    acc = c.get_or_add_image("acc.png", 64, 48)
    c.add_annotation_from_mask(acc, cat, square_mask())
    pend = c.get_or_add_image("pend.png", 64, 48)
    c.add_annotation_from_mask(pend, cat, square_mask(), inferred=True, score=0.9)
    c.set_image_empty("empty.png", 64, 48)
    out = tmp_path / "train.json"
    c.export(str(out), exclude_inferred=True)
    d = json.load(open(out))
    assert sorted(i["file_name"] for i in d["images"]) == ["acc.png", "empty.png"]
    assert all(not a.get("inferred") for a in d["annotations"])
    # The full (auto-save) export still keeps every image and pending mask.
    full = tmp_path / "all.json"
    c.export(str(full))
    assert len(json.load(open(full))["images"]) == 3

    # YOLO: the empty frame becomes a background image with an empty label file.
    imgs = tmp_path / "frames"
    imgs.mkdir()
    for name in ("acc.png", "pend.png", "empty.png"):
        cv2.imwrite(str(imgs / name), np.zeros((48, 64, 3), np.uint8))
    yolo = tmp_path / "yolo"
    convert_coco_to_yolo(str(out), str(imgs), str(yolo), val_fraction=0.5)
    labels = {p.name: p.read_text() for p in yolo.rglob("labels/*/*.txt")}
    assert set(labels) == {"acc.txt", "empty.txt"}
    assert labels["empty.txt"] == "" and labels["acc.txt"].startswith("0 ")


# ----------------------------------------------------------------------
# the window
# ----------------------------------------------------------------------
FRAMES = ["CAM13_a_frame_000001.png", "CAM13_a_frame_000002.png",
          "CAM14_b_frame_000003.png", "CAM14_b_frame_000004.png"]
_APP = None


def window():
    global _APP
    _APP = QApplication.instance() or QApplication([])
    root = tempfile.mkdtemp(prefix="mtt_empty_")
    proj = os.path.join(root, "proj")
    for sub in ("training_frames", "annotations", "models"):
        os.makedirs(os.path.join(proj, sub))
    json.dump({"video_paths": [], "source_videos": [], "categories": ["mouse"],
               "sam2_model_path": None}, open(os.path.join(proj, "project_config.json"), "w"))
    for name in FRAMES:
        cv2.imwrite(os.path.join(proj, "training_frames", name), np.zeros((48, 64, 3), np.uint8))
    w = G.MaskTrackerWindow()
    w._launched = True
    w._load_project(os.path.join(proj, "project_config.json"))
    if "mouse" not in w._categories:
        w._coco.add_category("mouse")
        w._categories.append("mouse")
    return w


def row_text(w, name):
    for i in range(w.frame_list.topLevelItemCount()):
        it = w.frame_list.topLevelItem(i)
        if it.text(0) == name:
            return it.text(1)
    raise KeyError(name)


def test_n_marks_empty_advances_and_unmarks():
    w = window()
    assert w.windowTitle().startswith("Mask Tracker")
    w._select_frame_row(0)
    assert w.current_frame_idx == 0
    w._toggle_empty_frame()                       # what N and the button call
    assert w._coco.is_image_empty(FRAMES[0])
    assert row_text(w, FRAMES[0]) == "empty ✓"
    assert w.current_frame_idx == 1               # moved on to the next unreviewed frame

    w._select_frame_row(0)
    assert w.btn_empty_frame.isChecked()
    w._toggle_empty_frame()                       # unmark
    assert not w._coco.is_image_empty(FRAMES[0])
    assert not w.btn_empty_frame.isChecked()
    assert w.current_frame_idx == 0               # unmarking does not move
    w.close()


def test_enter_and_post_inference_skip_empty_frames(monkeypatch):
    w = window()
    w._select_frame_row(1)
    w._toggle_empty_frame()                       # frame 1 empty
    w._select_frame_row(0)
    w._next_frame()                               # Enter
    assert w.current_frame_idx == 2               # skips the empty frame 1

    captured = {}

    class FakeWorker:
        class _Sig:
            def connect(self, *_):
                pass

        def __init__(self, model_dir, frame_paths, **kw):
            captured["paths"] = [os.path.basename(p) for p in frame_paths]
            self.progress = self.frame_result = self.finished = self.error = self._Sig()

        def start(self):
            pass
    monkeypatch.setattr(G, "PostTrainInferenceWorker", FakeWorker)
    w._run_post_train_inference("model_dir")
    assert FRAMES[1] not in captured["paths"] and len(captured["paths"]) == 3
    w.close()


def test_space_continues_through_the_queue_instead_of_wrapping(monkeypatch):
    w = window()
    w._select_frame_row(1)                        # second (last) queued frame of video a
    monkeypatch.setattr(w, "_scrubber_source", lambda: ("a.mp4", 2))
    monkeypatch.setattr(w._scrubber, "next_mark", lambda pos: 1)   # would wrap to frame 1
    monkeypatch.setattr(w._scrubber, "prev_mark", lambda pos: 1)
    w._scrubber_rows = {1: 0, 2: 1}
    w._on_timeline_advance(1)
    assert w.current_frame_idx == 2               # first frame of the next video, not frame 0

    w._select_frame_row(3)
    w._step_queue(1)                              # end of the queue wraps to the start
    assert w.current_frame_idx == 0
    w._step_queue(-1)
    assert w.current_frame_idx == 3
    w.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
