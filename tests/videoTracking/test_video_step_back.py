"""Holding Left in the source video steps back as fast as Right steps forward:
each Left press used to seek, and a seek decodes forward from the previous
keyframe (~1 s per press on the Synology), so Left crawled at about one frame
a second. Decoded frames are now kept, and stepping back fetches a block with
one seek. The frames shown must still be the right ones.

Runs under pytest, or directly.
"""
import json
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PyQt5.QtCore import QEvent, Qt  # noqa: E402
from PyQt5.QtGui import QKeyEvent  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

import fnt.videoTracking.mask_tracker_gui as G  # noqa: E402

_APP = None


class _CountingCap:
    """A capture that counts its seeks."""

    def __init__(self, cap):
        self.cap = cap
        self.seeks = 0

    def set(self, *a):
        self.seeks += 1
        return self.cap.set(*a)

    def read(self):
        return self.cap.read()

    def get(self, *a):
        return self.cap.get(*a)

    def release(self):
        self.cap.release()


def _project():
    root = tempfile.mkdtemp(prefix="mtt_stepback_")
    video = os.path.join(root, "CAM12_2021-10-25_02_46_13_295.mp4")
    vw = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"mp4v"), 15, (64, 48))
    for k in range(40):
        vw.write(np.full((48, 64, 3), 6 * k, np.uint8))
    vw.release()
    proj = os.path.join(root, "p")
    os.makedirs(os.path.join(proj, "annotations"))
    with open(os.path.join(proj, "project_config.json"), "w") as fh:
        json.dump({"project_dir": proj, "video_paths": [video], "categories": ["mouse"],
                   "source_videos": []}, fh)
    return proj, video


def _key(w, key, mod=Qt.NoModifier):
    w.keyPressEvent(QKeyEvent(QEvent.KeyPress, key, mod))
    QApplication.processEvents()


def test_holding_left_does_not_seek_every_frame():
    global _APP
    _APP = QApplication.instance() or QApplication([])
    proj, video = _project()
    w = G.MaskTrackerWindow()
    w._launched = True
    w._load_project(os.path.join(proj, "project_config.json"))
    assert w._enter_video_mode_at(video, 30)
    cap = w._video_cap = _CountingCap(w._video_cap)
    ref = cv2.VideoCapture(video)
    for k in range(29, 4, -1):                 # 25 presses of Left
        _key(w, Qt.Key_Left)
        assert w._video_frame_idx == k
        ref.set(cv2.CAP_PROP_POS_FRAMES, k)
        _, want = ref.read()
        assert np.array_equal(w._video_frame_cache[k], want), k
    assert cap.seeks <= 2, cap.seeks           # was one per press
    seeks = cap.seeks
    for _ in range(10):                        # forward again over frames already decoded
        _key(w, Qt.Key_Right)
    assert w._video_frame_idx == 15 and cap.seeks == seeks
    _key(w, Qt.Key_Left, Qt.ShiftModifier)     # Shift+Left: 10 back, still the right frame
    ref.set(cv2.CAP_PROP_POS_FRAMES, 5)
    assert w._video_frame_idx == 5 and np.array_equal(w._video_frame_cache[5], ref.read()[1])
    ref.release()
    w.close()


def test_the_cache_is_dropped_with_the_video():
    _APP2 = QApplication.instance() or QApplication([])
    proj, video = _project()
    w = G.MaskTrackerWindow()
    w._launched = True
    w._load_project(os.path.join(proj, "project_config.json"))
    assert w._enter_video_mode_at(video, 10)
    assert w._video_frame_cache
    w._close_video()
    assert not w._video_frame_cache
    w.close()


def test_arrows_scrub_whatever_has_focus():
    # The tab strip used to get focus at startup, so the first Right press opened Behavior;
    # then the frame list took focus and swallowed the arrows. They must scrub from both.
    _APP3 = QApplication.instance() or QApplication([])
    proj, video = _project()
    w = G.MaskTrackerWindow()
    w._launched = True
    w._load_project(os.path.join(proj, "project_config.json"))
    assert w.tab_widget.tabBar().focusPolicy() == Qt.NoFocus
    assert w._enter_video_mode_at(video, 20)
    for target in (w.frame_list, w.tab_widget.tabBar()):
        QApplication.sendEvent(target, QKeyEvent(QEvent.KeyPress, Qt.Key_Left, Qt.NoModifier))
        QApplication.processEvents()
    assert w._video_frame_idx == 18 and w.tab_widget.currentIndex() == 0
    QApplication.sendEvent(w.frame_list, QKeyEvent(QEvent.KeyPress, Qt.Key_Right, Qt.ShiftModifier))
    QApplication.processEvents()
    assert w._video_frame_idx == 28
    w.close()


if __name__ == "__main__":
    test_holding_left_does_not_seek_every_frame()
    test_the_cache_is_dropped_with_the_video()
    test_arrows_scrub_whatever_has_focus()
    print("ok")
