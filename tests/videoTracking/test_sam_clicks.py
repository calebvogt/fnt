"""SAM labeling in the Mask Tracker works as in MAD.

In SAM mode every click is its own one-point prompt, queued for prediction; the
object under each click joins a pending mask, and Enter saves each separate
pending object as its own mask (mouse_1, mouse_2, ...) while clicks whose
objects touch merge into one. A drag pans and a double-click adds one click.
U undoes the last click; right-click drops a pending object, opens the menu on
a finished mask, and otherwise does nothing (it used to open "Add Manual Mask",
which read as if SAM mode had switched itself off). M toggles SAM mode, and
clicks are ignored while the model loads.

Runs under pytest, or directly.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PyQt5.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PyQt5.QtGui import QKeyEvent, QMouseEvent  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

import fnt.videoTracking.mask_tracker_gui as G  # noqa: E402

_APP = QApplication.instance() or QApplication([])
H, W = 150, 200


def preview():
    w = G.AnnotationPreviewWidget()
    w.resize(400, 300)
    w.set_frame(np.zeros((H, W, 3), np.uint8))
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


def key(w, k):
    QApplication.sendEvent(w, QKeyEvent(QEvent.KeyPress, k, Qt.NoModifier, ""))


def blob(x0, y0, x1, y1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_each_click_is_queued_as_its_own_prompt():
    w, asked = preview()
    click(w, 200, 150)
    assert len(w._ai_clicks) == 1 and w._drawing_active and len(asked) == 1
    click(w, 250, 160)
    assert len(w._ai_clicks) == 2 and len(asked) == 2


def test_separate_objects_become_separate_masks_and_touching_ones_merge():
    w, _ = preview()
    saved = []
    w.annotation_accepted.connect(lambda: saved.append(int(w.get_pending_ai_mask().sum())))
    assert w.add_sam_component(blob(10, 10, 30, 30), (15, 15))      # mouse 1
    assert w.add_sam_component(blob(100, 80, 140, 120), (110, 90))  # mouse 2, far away
    assert w.add_sam_component(blob(28, 10, 50, 30), (40, 20))      # touches mouse 1: same object
    assert len(w.pending_components()) == 2
    key(w, Qt.Key_Return)
    assert sorted(saved) == [800, 1600]                             # 20x20 + 22x20 overlapping; 40x40
    assert not w.has_ai_pending() and w.drawing_mode == "ai"        # SAM stays on


def test_the_component_under_the_click_is_taken():
    w, _ = preview()
    two = blob(10, 10, 30, 30) | blob(100, 80, 140, 120)            # SAM proposed two pieces
    w.add_sam_component(two, (110, 90))
    assert int(w._ai_pending.sum()) == 1600                          # only the clicked piece


def test_u_undoes_the_last_click_and_right_click_drops_an_object():
    w, _ = preview()
    w.add_sam_component(blob(10, 10, 30, 30), (15, 15))
    w.add_sam_component(blob(100, 80, 140, 120), (110, 90))
    key(w, Qt.Key_U)
    assert len(w.pending_components()) == 1
    wx, wy = w._img_to_widget(15, 15)
    click(w, wx, wy, Qt.RightButton)
    assert not w.has_ai_pending()


def test_a_drag_pans_without_a_click():
    w, _ = preview()
    send(w, QEvent.MouseButtonPress, 100, 100)
    send(w, QEvent.MouseMove, 160, 130)
    send(w, QEvent.MouseButtonRelease, 160, 130)
    assert w._ai_clicks == []
    assert (w._pan_x, w._pan_y) == (60, 30)


def test_a_double_click_is_one_click():
    w, _ = preview()
    send(w, QEvent.MouseButtonPress, 200, 150)
    send(w, QEvent.MouseButtonRelease, 200, 150)
    send(w, QEvent.MouseButtonDblClick, 200, 150)
    send(w, QEvent.MouseButtonRelease, 200, 150)
    assert len(w._ai_clicks) == 1


def test_right_click_is_quiet_off_masks_in_sam_mode(monkeypatch):
    w, _ = preview()
    menus = []
    monkeypatch.setattr(w, "_show_context_menu", lambda *a: menus.append(a))
    click(w, 200, 150, Qt.RightButton)          # SAM on, nothing here
    assert menus == []
    w.drawing_mode = "navigate"
    click(w, 200, 150, Qt.RightButton)          # outside SAM mode the menu opens
    assert len(menus) == 1


def test_clicks_are_ignored_while_the_model_loads():
    w, asked = preview()
    w.set_busy("Loading SAM model…")
    click(w, 200, 150)
    click(w, 200, 150, Qt.RightButton)
    assert w._ai_clicks == [] and asked == []
    w.grab()                                   # the overlay paints without error
    w.set_busy(None)
    click(w, 200, 150)
    assert len(w._ai_clicks) == 1


def test_m_toggles_sam_mode():
    w, _ = preview()
    hits = []
    w.sam_toggle_requested.connect(lambda: hits.append(1))
    QApplication.sendEvent(w, QKeyEvent(QEvent.KeyPress, Qt.Key_M, Qt.NoModifier, "m"))
    assert hits == [1]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
