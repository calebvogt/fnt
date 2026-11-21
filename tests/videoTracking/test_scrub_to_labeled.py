"""Scrubbing the source video with the arrow keys onto a frame that is in the
Training Frames Queue opens it as the queue frame, with its masks: stepping
back and forth around a labeled frame used to bring it back bare. The timeline's
playhead takes the state of the frame on screen.

Runs under pytest, or directly.
"""
import json
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PyQt5.QtCore import QEvent, Qt  # noqa: E402
from PyQt5.QtGui import QKeyEvent  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

import fnt.videoTracking.mask_tracker_gui as G  # noqa: E402
from fnt.videoTracking.mask_tracker_cli import extract_frames, frame_name  # noqa: E402

_APP = None


def _project():
    root = tempfile.mkdtemp(prefix="mtt_scrub_")
    video = os.path.join(root, "CAM12_2021-10-25_02_46_13_295.mp4")
    vw = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 15, (64, 48))
    for k in range(20):
        vw.write(np.full((48, 64, 3), 10 * k, np.uint8))
    vw.release()
    proj = os.path.join(root, "p")
    os.makedirs(os.path.join(proj, "annotations"))
    with open(os.path.join(proj, "project_config.json"), "w") as fh:
        json.dump({"project_dir": proj, "video_paths": [video], "categories": ["mouse"],
                   "source_videos": []}, fh)
    extract_frames(proj, [(video, 5), (video, 10)])
    labeled = frame_name(video, 10)
    coco = {"images": [{"id": 1, "file_name": labeled, "width": 64, "height": 48}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "iscrowd": 0, "area": 100,
                             "segmentation": [[10, 10, 30, 10, 30, 30, 10, 30]], "bbox": [10, 10, 20, 20]}],
            "categories": [{"id": 1, "name": "mouse", "supercategory": ""}]}
    with open(os.path.join(proj, "annotations", "annotations.json"), "w") as fh:
        json.dump(coco, fh)
    return proj, video, labeled


def _key(w, key):
    w.keyPressEvent(QKeyEvent(QEvent.KeyPress, key, Qt.NoModifier))
    QApplication.processEvents()


def test_stepping_back_onto_a_labeled_frame_shows_its_masks():
    global _APP
    _APP = QApplication.instance() or QApplication([])
    proj, video, labeled = _project()
    w = G.MaskTrackerWindow()
    w._launched = True
    w._load_project(os.path.join(proj, "project_config.json"))
    row = next(i for i, (_, f, _) in enumerate(w._extracted_frames) if f == 10)
    w._select_frame_row(row)
    QApplication.processEvents()
    assert len(w.preview.annotations) == 1
    assert w._scrubber.playhead_color() == G.FrameScrubber.STATE_COLORS["labeled"]
    _key(w, Qt.Key_Left)
    _key(w, Qt.Key_Left)                       # frame 8: plain video
    assert w._annot_in_video_mode and w._video_frame_idx == 8
    assert w._scrubber.playhead_color() == G.FrameScrubber.PLAYHEAD_COLOR
    _key(w, Qt.Key_Right)
    _key(w, Qt.Key_Right)                      # back on frame 10: the queue frame, with its mask
    assert not w._annot_in_video_mode
    assert w.current_frame_idx == row
    assert len(w.preview.annotations) == 1
    w.close()


def test_the_playhead_takes_the_state_of_its_frame():
    _APP2 = QApplication.instance() or QApplication([])
    s = G.FrameScrubber()
    s.set_range(100, 15)
    s.set_marks({10: "labeled", 20: "predicted", 30: "queued"})
    for pos, want in ((10, s.STATE_COLORS["labeled"]), (20, s.STATE_COLORS["predicted"]),
                      (30, G.QColor("#ffffff")), (40, s.PLAYHEAD_COLOR)):
        s.set_position(pos)
        assert s.playhead_color() == want
    s.resize(400, 22)
    s.grab()                                   # paints without error


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
