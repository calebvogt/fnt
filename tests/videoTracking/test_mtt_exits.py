"""Exits: exits.json per video folder, the cameras found in folders, and the editor saving
what is drawn or confirmed."""
import os

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fnt.videoTracking.mtt_exits import cameras_in, load_exits, median_frame, save_camera_exits


def _clip(path, value, n=3, size=(64, 48)):
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 15, size)
    for _ in range(n):
        w.write(np.full((size[1], size[0], 3), value, np.uint8))
    w.release()


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "T001"
    d.mkdir()
    for i, v in enumerate((50, 60, 250)):
        _clip(str(d / f"CAM13_2022-01-01_0{i}_00_00_000.mp4"), v)
    _clip(str(d / "CAM14_2022-01-01_00_00_00_000.mp4"), 100)
    return str(d)


def test_save_keeps_other_cameras(folder):
    assert load_exits(folder) == {}
    save_camera_exits(folder, "CAM13", [[(1, 2), (3, 2), (3, 4)]], (64, 48), "auto", score=0.7)
    save_camera_exits(folder, "CAM14", [[(5, 5), (9, 5), (9, 9), (5, 9)], [(1, 1), (2, 2)]], (64, 48))
    ex = load_exits(folder)
    assert ex["CAM13"] == {"exits": [[[1, 2], [3, 2], [3, 4]]], "frame_size": [64, 48],
                           "source": "auto", "score": 0.7}
    assert ex["CAM14"]["source"] == "checked"
    assert len(ex["CAM14"]["exits"]) == 1          # a two-corner outline is not an exit


def test_cameras_and_median(folder):
    cams = cameras_in([os.path.dirname(folder)])
    key13 = (os.path.abspath(folder), "CAM13")
    assert set(cams) == {key13, (os.path.abspath(folder), "CAM14")}
    assert [os.path.basename(v)[:8] for v in cams[key13]] == ["CAM13_20"] * 3
    med = median_frame(cams[key13])
    assert med.shape == (48, 64, 3)
    assert abs(int(med.mean()) - 60) <= 3           # the bright clip is outvoted


def test_editor_saves_drawn_and_confirmed(folder):
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QKeyEvent
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])  # noqa: F841
    from fnt.videoTracking.mtt_exits_editor import ExitsWindow
    save_camera_exits(folder, "CAM14", [[(5, 5), (9, 5), (9, 9)]], (64, 48), "auto", score=0.4)
    cams = cameras_in([folder])
    w = ExitsWindow(cams, focus=["T001/CAM14"])
    assert [k[1] for k in w.keys] == ["CAM14", "CAM13"]         # focus first
    for k in w.keys:                                              # load frames synchronously
        w._on_loaded((k, median_frame(cams[k])))
    w.list.setCurrentRow(0)
    w.confirm_next()                                              # CAM14 right as it is
    assert load_exits(folder)["CAM14"] == {"exits": [[[5, 5], [9, 5], [9, 9]]],
                                           "frame_size": [64, 48], "source": "checked",
                                           "score": 0.4}
    assert w.cur[1] == "CAM13"
    w.canvas._points = [(10, 10), (20, 10), (20, 20), (10, 20)]
    w.canvas.close_outline()                                      # drawn on CAM13
    w.keyPressEvent(QKeyEvent(QKeyEvent.KeyPress, Qt.Key_PageUp, Qt.NoModifier))
    assert load_exits(folder)["CAM13"]["exits"] == [[[10, 10], [20, 10], [20, 20], [10, 20]]]
    assert w.cur[1] == "CAM14"
    w.close()
