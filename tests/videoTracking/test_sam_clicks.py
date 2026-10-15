"""SAM labeling in the Mask Tracker works as in MAD.

In SAM mode a click places a positive point (more clicks refine the object), a
drag pans, and a double-click does not add a duplicate point. A right-click
excludes an area while an object is being prompted, opens the menu on a
finished mask, and otherwise does nothing (it used to open "Add Manual Mask",
which read as if SAM mode had switched itself off). M toggles SAM mode.

Runs under pytest, or directly.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PyQt5.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PyQt5.QtGui import QKeyEvent, QMouseEvent  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

import fnt.videoTracking.mask_tracker_gui as G  # noqa: E402

_APP = QApplication.instance() or QApplication([])


def preview():
    w = G.AnnotationPreviewWidget()
    w.resize(400, 300)
    w.set_frame(np.zeros((150, 200, 3), np.uint8))
    w.annotation_keys_enabled = True
    w.drawing_mode = "ai"
    asked = []
    w.ai_prediction_requested.connect(lambda: asked.append(1))
    return w, asked


def send(w, kind, x, y, button=Qt.LeftButton):
    buttons = Qt.NoButton if kind == QEvent.MouseButtonRelease else button
    ev = QMouseEvent(kind, QPointF(x, y), button, buttons, Qt.NoModifier)
    QApplication.sendEvent(w, ev)


def click(w, x, y, button=Qt.LeftButton):
    send(w, QEvent.MouseButtonPress, x, y, button)
    send(w, QEvent.MouseButtonRelease, x, y, button)


def test_a_click_places_a_point_and_more_clicks_refine():
    w, asked = preview()
    click(w, 200, 150)
    assert len(w._ai_positive_points) == 1 and w._drawing_active and len(asked) == 1
    click(w, 210, 160)
    assert len(w._ai_positive_points) == 2 and len(asked) == 2


def test_a_drag_pans_without_placing_a_point():
    w, _ = preview()
    send(w, QEvent.MouseButtonPress, 100, 100)
    send(w, QEvent.MouseMove, 160, 130)
    send(w, QEvent.MouseButtonRelease, 160, 130)
    assert w._ai_positive_points == []
    assert (w._pan_x, w._pan_y) == (60, 30)


def test_a_double_click_places_one_point():
    w, _ = preview()
    send(w, QEvent.MouseButtonPress, 200, 150)
    send(w, QEvent.MouseButtonRelease, 200, 150)
    send(w, QEvent.MouseButtonDblClick, 200, 150)
    send(w, QEvent.MouseButtonRelease, 200, 150)
    assert len(w._ai_positive_points) == 1


def test_right_click_excludes_during_a_prompt_and_is_quiet_otherwise(monkeypatch):
    w, _ = preview()
    menus = []
    monkeypatch.setattr(w, "_show_context_menu", lambda *a: menus.append(a))
    click(w, 200, 150, Qt.RightButton)          # SAM on, no prompt, no mask here
    assert menus == [] and w._ai_negative_points == []
    click(w, 200, 150)                          # start an object
    click(w, 220, 150, Qt.RightButton)          # exclude
    assert len(w._ai_negative_points) == 1 and menus == []
    w.drawing_mode = "navigate"
    w._clear_drawing()
    click(w, 200, 150, Qt.RightButton)          # outside SAM mode the menu opens
    assert len(menus) == 1


def test_m_toggles_sam_mode():
    w, _ = preview()
    hits = []
    w.sam_toggle_requested.connect(lambda: hits.append(1))
    QApplication.sendEvent(w, QKeyEvent(QEvent.KeyPress, Qt.Key_M, Qt.NoModifier, "m"))
    assert hits == [1]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
