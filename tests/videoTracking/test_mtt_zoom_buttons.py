"""Mask Tracker's zoom buttons (- 100% + Fit, beside the zoom %), and zooming
that keeps the point under the cursor (or the middle of the view) still.

Runs under pytest, or directly.
"""
import json
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402


def _window():
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])  # noqa: F841
    import fnt.videoTracking.mask_tracker_gui as G
    d = tempfile.mkdtemp(prefix="mtt_zoom_")
    os.makedirs(os.path.join(d, "training_frames"))
    with open(os.path.join(d, "project_config.json"), "w") as fh:
        json.dump({"project_dir": d, "video_paths": [], "categories": ["mouse"],
                   "source_videos": []}, fh)
    w = G.MaskTrackerWindow()
    w._launched = True
    w._load_project(os.path.join(d, "project_config.json"))
    w.preview.set_frame(np.zeros((100, 200, 3), np.uint8))
    w.preview.resize(400, 400)          # letterboxed: the frame fills the width only
    return w


def _image_point_at(p, wx, wy):
    s = p._effective_scale()
    ox, oy = p._img_to_widget(0, 0)
    return (wx - ox) / s, (wy - oy) / s


def test_buttons_zoom_on_the_middle_and_fit_returns_to_100():
    w = _window()
    p = w.preview
    mid = _image_point_at(p, 200, 200)
    w.btn_zoom_in.click()
    w.btn_zoom_in.click()
    assert p._zoom == pytest.approx(1.25 ** 2)
    assert w.lbl_zoom_info.text() == f"{int(1.25 ** 2 * 100)}%"
    assert np.allclose(_image_point_at(p, 200, 200), mid)       # the middle stays put
    w.btn_zoom_out.click()
    assert p._zoom == pytest.approx(1.25)
    p._pan_x, p._pan_y = 37.0, -12.0                           # panned away
    w.btn_zoom_fit.click()
    assert (p._zoom, p._pan_x, p._pan_y) == (1.0, 0.0, 0.0)
    assert w.lbl_zoom_info.text() == "100%"
    for b in (w.btn_zoom_in, w.btn_zoom_out, w.btn_zoom_fit):
        assert b.isVisibleTo(w)                                 # on the labelling tab
    w.close()


def test_wheel_keeps_the_point_under_the_cursor_and_ignores_sideways_swipes():
    from PyQt5.QtCore import QPoint, QPointF, Qt
    from PyQt5.QtGui import QWheelEvent
    w = _window()
    p = w.preview

    def wheel(x, y, dx, dy):
        pos = QPointF(x, y)
        p.wheelEvent(QWheelEvent(pos, pos, QPoint(0, 0), QPoint(dx, dy),
                                 Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False))
    before = _image_point_at(p, 300, 250)      # below the middle: inside the letterbox offset
    wheel(300, 250, 0, 120)
    wheel(300, 250, 0, 120)
    assert p._zoom > 1.3
    assert np.allclose(_image_point_at(p, 300, 250), before)
    z = p._zoom
    wheel(300, 250, 120, 0)                    # sideways: no zoom
    assert p._zoom == z
    w.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
