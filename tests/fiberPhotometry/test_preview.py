"""The Fiber Photometry window and its scrubbable preview, offscreen."""
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
cv2 = pytest.importorskip("cv2")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from fnt.fiberPhotometry.processing import process  # noqa: E402

_APP = []


def _app():
    if not _APP:
        _APP.append(QApplication.instance() or QApplication([]))
    return _APP[0]


def _dispose(w):
    """Destroy the widget now, while the QApplication still exists: pyqtgraph widgets left for
    interpreter shutdown abort the process."""
    w.close()
    w.deleteLater()
    _app().processEvents()


def _video(path, n=90, fps=30):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    for k in range(n):
        w.write(np.full((120, 160, 3), (k * 3) % 255, np.uint8))
    w.release()


def test_scrubbing_moves_the_traces_with_the_video(tmp_path):
    _app()
    from fnt.fiberPhotometry.preview_view import PreviewPane
    t = np.arange(0, 3, 0.01)
    r = process(0.064 + 0.001 * np.sin(t * 5), 0.055 + 0.0008 * np.sin(t * 5), t, 100.0)
    v = tmp_path / "v.mp4"
    _video(v)
    p = PreviewPane()
    p.set_recording("rec", r, str(v))
    assert p.n_frames == 90 and p.slider.maximum() == 89
    p.window.setValue(2)
    p.slider.setValue(45)                     # frame 45 at 30 fps = 1.5 s
    lo, hi = p.p_sig.getViewBox().viewRange()[0]
    assert (lo + hi) / 2 == pytest.approx(1.5, abs=1e-6) and hi - lo == pytest.approx(2.0)
    assert p.cursors[0].value() == pytest.approx(1.5)
    assert "frame 46/90" in p.lbl_time.text()
    p.toggle_play()
    assert p.timer.isActive()
    p.stop()
    _dispose(p)


def test_without_video_the_preview_still_scrubs():
    _app()
    from fnt.fiberPhotometry.preview_view import PreviewPane
    t = np.arange(0, 10, 0.01)
    r = process(0.064 + 0.001 * np.sin(t), 0.055 + 0.0008 * np.sin(t), t, 100.0)
    p = PreviewPane()
    p.set_recording("rec", r, None)
    assert p.cap is None and p.n_frames == int(9.99 * 30)
    p.slider.setValue(150)
    assert p.cursors[0].value() == pytest.approx(5.0)
    _dispose(p)


def test_the_window_builds_settings_for_each_pipeline():
    _app()
    from fnt.fiberPhotometry.fiber_photometry_pyqt import FiberPhotometryWindow
    w = FiberPhotometryWindow()
    assert w.settings().pipeline == "guppy"
    for k, key in enumerate(("guppy", "matlab", "phat")):
        w.pipeline.setCurrentIndex(k)
        assert w.settings().pipeline == key and w.stack.currentIndex() == k
    w.notch.setText("3.32, 9.94")
    w.phat_bleaching.setCurrentIndex(1)
    s = w.settings()
    assert s.notch_hz == (3.32, 9.94) and s.phat_bleaching == "control"
    _dispose(w)
