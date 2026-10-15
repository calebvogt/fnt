"""SAM, Paint and Eraser in the Mask Tracker work on pending objects, one per mouse.

In SAM mode a click starts a pending object, or refines the pending object it
lands on or near (Shift+click always starts a new one); right-click inside an
object excludes that spot from it, Shift+right-click drops it. Each object is
predicted from its own points, so touching mice stay apart; SAM's candidate is
chosen by the clicks, not its score alone, and holes are filled. Enter saves
every object as its own mask. Paint adds pixels to an object (a new one off any object),
Eraser removes them; the wheel sizes the brush. U undoes clicks and strokes. A
drag pans in SAM mode, a double-click is one click, clicks are ignored while the
model loads, and M toggles SAM.

Runs under pytest, or directly.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PyQt5.QtCore import QEvent, QPoint, QPointF, Qt  # noqa: E402
from PyQt5.QtGui import QKeyEvent, QMouseEvent, QWheelEvent  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

import fnt.videoTracking.mask_tracker_gui as G  # noqa: E402

_APP = QApplication.instance() or QApplication([])
H, W = 150, 200


def preview(mode="ai"):
    w = G.AnnotationPreviewWidget()
    w.resize(400, 300)
    w.set_frame(np.zeros((H, W, 3), np.uint8))
    w.annotation_keys_enabled = True
    w.drawing_mode = mode
    asked = []
    w.ai_prediction_requested.connect(lambda: asked.append(1))
    return w, asked


def send(w, kind, x, y, button=Qt.LeftButton, mods=Qt.NoModifier):
    buttons = Qt.NoButton if kind == QEvent.MouseButtonRelease else button
    QApplication.sendEvent(w, QMouseEvent(kind, QPointF(x, y), button, buttons, mods))


def click_img(w, ix, iy, button=Qt.LeftButton, mods=Qt.NoModifier):
    x, y = w._img_to_widget(ix, iy)
    send(w, QEvent.MouseButtonPress, x, y, button, mods)
    send(w, QEvent.MouseButtonRelease, x, y, button, mods)


def drag_img(w, p0, p1, button=Qt.LeftButton):
    x0, y0 = w._img_to_widget(*p0)
    x1, y1 = w._img_to_widget(*p1)
    send(w, QEvent.MouseButtonPress, x0, y0, button)
    send(w, QEvent.MouseMove, (x0 + x1) / 2, (y0 + y1) / 2, button)
    send(w, QEvent.MouseMove, x1, y1, button)
    send(w, QEvent.MouseButtonRelease, x1, y1, button)


def key(w, k):
    QApplication.sendEvent(w, QKeyEvent(QEvent.KeyPress, k, Qt.NoModifier, ""))


def blob(x0, y0, x1, y1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return m


def predict_all(w, sam):
    """Serve the preview's SAM queue with a fake predictor ``sam(pos, neg)``."""
    while True:
        job = w.next_ai_job()
        if job is None:
            return
        oid, ver, pos, neg = job
        w.set_object_mask(oid, ver, sam(pos, neg))


def test_clicks_start_objects_and_a_click_inside_refines_it():
    w, asked = preview()
    click_img(w, 20, 20)
    click_img(w, 120, 100)
    assert len(w._ai_objects) == 2 and len(asked) == 2
    predict_all(w, lambda pos, neg: blob(10, 10, 40, 40) if pos[0][0] < 60 else blob(100, 80, 150, 130))
    click_img(w, 30, 30)                                   # inside object 1: refines it
    assert len(w._ai_objects) == 2 and len(w._ai_objects[0]["pos"]) == 2
    click_img(w, 30, 30, mods=Qt.ShiftModifier)            # Shift: a new object even here
    assert len(w._ai_objects) == 3


def test_a_click_near_a_mouse_adds_to_it():
    """A foot or tail SAM left out: a click just outside the mask refines that
    mouse (as in MAD) instead of starting a new one."""
    w, _ = preview()
    click_img(w, 20, 20)
    predict_all(w, lambda pos, neg: blob(10, 10, 40, 40))
    click_img(w, 44, 25)                          # 4 px right of the mask (near = 6 px here)
    assert len(w._ai_objects) == 1 and w._ai_objects[0]["pos"] == [(20, 20), (44, 25)]
    click_img(w, 120, 100)                        # far away: a new mouse
    assert len(w._ai_objects) == 2


def test_the_candidate_is_chosen_by_the_clicks_not_only_the_score():
    """SAM's top score was the dish for a tail click, and both overlapping mice
    after one was excluded; the chosen candidate covers the clicks, none of the
    exclusions, and grows at most MAX_GROWTH times."""
    w, _ = preview()
    click_img(w, 20, 20)
    one = blob(10, 10, 40, 40)                                      # 900 px
    w.set_object_mask(*w.next_ai_job()[:2], np.stack([one]), np.array([0.9]))
    click_img(w, 42, 25)                                            # near: refine
    tail = one | blob(40, 24, 50, 27)                               # 930 px, low score
    dish = blob(0, 0, 120, 100)                                     # 12000 px, top score
    w.set_object_mask(*w.next_ai_job()[:2], np.stack([tail, dish]), np.array([0.35, 0.40]))
    assert int(w._ai_objects[0]["sam"].sum()) == 930
    click_img(w, 30, 30, Qt.RightButton)                            # exclude a spot
    both = blob(10, 10, 50, 40)
    w.set_object_mask(*w.next_ai_job()[:2], np.stack([both, blob(10, 10, 25, 40)]),
                      np.array([0.92, 0.55]))
    assert not w._ai_objects[0]["sam"][30, 30]                      # the exclusion holds


def test_holes_in_a_mouse_are_filled():
    w, _ = preview()
    click_img(w, 12, 12)
    ring = blob(10, 10, 40, 40) & ~blob(20, 20, 30, 30)
    predict_all(w, lambda pos, neg: ring)
    assert w._ai_objects[0]["sam"][25, 25] and int(w._ai_objects[0]["sam"].sum()) == 900


def test_touching_mice_stay_separate_masks():
    w, _ = preview()
    saved = []
    w.annotation_accepted.connect(lambda: saved.append(int(w.get_pending_ai_mask().sum())))
    click_img(w, 15, 15)
    click_img(w, 45, 15)
    # SAM gives each click its own mouse; the two touch along x = 30.
    predict_all(w, lambda pos, neg: blob(0, 0, 30, 30) if pos[0][0] < 30 else blob(30, 0, 60, 30))
    assert len(w.pending_components()) == 2
    key(w, Qt.Key_Return)
    assert saved == [900, 900]                              # two masks, not one of 1800
    assert not w._ai_objects and w.drawing_mode == "navigate"   # Enter turns the tool off


def test_right_click_excludes_from_the_object_and_shift_right_click_drops_it():
    w, asked = preview()
    click_img(w, 20, 20)
    predict_all(w, lambda pos, neg: blob(10, 10, 60, 40))
    click_img(w, 50, 30, Qt.RightButton)                    # exclude the tail end
    obj = w._ai_objects[0]
    assert obj["neg"] == [(50, 30)] and w.has_ai_jobs()
    predict_all(w, lambda pos, neg: blob(10, 10, 40, 40) if neg else blob(10, 10, 60, 40))
    assert int(obj["mask"].sum()) == 900
    click_img(w, 20, 20, Qt.RightButton, Qt.ShiftModifier)  # drop it
    assert not w._ai_objects


def test_the_component_under_the_click_is_taken():
    w, _ = preview()
    click_img(w, 110, 90)
    predict_all(w, lambda pos, neg: blob(10, 10, 30, 30) | blob(100, 80, 140, 120))
    assert int(w._ai_objects[0]["mask"].sum()) == 1600       # only the clicked piece


def test_paint_adds_eraser_removes_wheel_sizes_and_u_undoes():
    w, _ = preview("paint")
    w.brush_radius = 3
    drag_img(w, (20, 20), (60, 20))                          # paint a new object
    assert len(w._ai_objects) == 1 and w._ai_objects[0]["mask"].sum() > 0
    painted = int(w._ai_objects[0]["mask"].sum())
    w.drawing_mode = "erase"
    drag_img(w, (40, 15), (40, 25))                          # erase across the middle
    assert int(w._ai_objects[0]["mask"].sum()) < painted
    key(w, Qt.Key_U)                                         # undo the erase
    assert int(w._ai_objects[0]["mask"].sum()) == painted
    key(w, Qt.Key_U)                                         # undo the paint stroke
    assert not w._ai_objects
    w.drawing_mode = "paint"
    r0 = w.brush_radius
    x, y = w._img_to_widget(50, 50)
    QApplication.sendEvent(w, QWheelEvent(QPointF(x, y), QPointF(x, y), QPoint(0, 0), QPoint(0, 120),
                                          Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False))
    assert w.brush_radius > r0


def test_paint_on_a_sam_object_extends_it_and_enter_saves_it():
    w, _ = preview()
    click_img(w, 20, 20)
    predict_all(w, lambda pos, neg: blob(10, 10, 30, 30))
    w.drawing_mode = "paint"
    w.brush_radius = 2
    drag_img(w, (25, 25), (45, 25))                          # starts on the object: extends it
    assert len(w._ai_objects) == 1 and int(w._ai_objects[0]["mask"].sum()) > 400
    saved = []
    w.annotation_accepted.connect(lambda: saved.append(1))
    key(w, Qt.Key_Return)
    assert saved == [1]


def test_a_drag_pans_in_sam_mode_and_a_double_click_is_one_click():
    w, _ = preview()
    send(w, QEvent.MouseButtonPress, 100, 100)
    send(w, QEvent.MouseMove, 160, 130)
    send(w, QEvent.MouseButtonRelease, 160, 130)
    assert w._ai_objects == [] and (w._pan_x, w._pan_y) == (60, 30)
    w2, _ = preview()
    for kind in (QEvent.MouseButtonPress, QEvent.MouseButtonRelease,
                 QEvent.MouseButtonDblClick, QEvent.MouseButtonRelease):
        send(w2, kind, 200, 150)
    assert len(w2._ai_objects) == 1


def test_right_click_is_quiet_off_objects_in_sam_mode(monkeypatch):
    w, _ = preview()
    menus = []
    monkeypatch.setattr(w, "_show_context_menu", lambda *a: menus.append(a))
    send(w, QEvent.MouseButtonPress, 200, 150, Qt.RightButton)
    assert menus == []
    w.drawing_mode = "navigate"
    send(w, QEvent.MouseButtonPress, 200, 150, Qt.RightButton)
    assert len(menus) == 1


def test_clicks_are_ignored_while_the_model_loads():
    w, asked = preview()
    w.set_busy("Loading SAM model…")
    click_img(w, 20, 20)
    assert w._ai_objects == [] and asked == []
    w.grab()
    w.set_busy(None)
    click_img(w, 20, 20)
    assert len(w._ai_objects) == 1


def test_m_p_x_request_tools():
    w, _ = preview()
    got = []
    w.sam_toggle_requested.connect(lambda: got.append("m"))
    w.tool_requested.connect(got.append)
    for k in (Qt.Key_M, Qt.Key_P, Qt.Key_X):
        key(w, k)
    assert got == ["m", "paint", "erase"]


def test_the_label_sits_on_the_body_not_pulled_along_the_tail():
    """The label dot was the mean of the outline's vertices; a long tail's
    outline holds most of them and dragged it toward the tail."""
    body = blob(40, 40, 80, 70)                       # 40 x 30 body
    tail = blob(80, 54, 190, 56)                      # long thin tail
    outline = G.AnnotationPreviewWidget._mask_outline(body | tail)
    ann = G.AnnotationObject(outline, "mouse")
    x, y = ann.label_anchor()
    assert 50 <= x <= 70 and 48 <= y <= 62          # middle of the body
    mean_x = sum(p[0] for p in outline) / len(outline)
    assert mean_x > 70                                # where the old dot went


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
