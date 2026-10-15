"""SAM predictions run one at a time, and Qt's own messages reach the crash log.

Each SAM click used to start a new prediction thread and drop the one still
running; Qt aborts the process when a running QThread is destroyed, which is
how a labeling session crashed (0xC0000409 in Qt5Core.dll). Clicks now queue:
one prediction thread at a time, each click predicted with its own point, its
object added to the pending mask. That abort is a Qt fatal message, which
neither faulthandler nor the slot exception hook sees, so the Qt message
handler writes it to the crash log.

Runs under pytest, or directly.
"""
import io
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

import fnt.videoTracking.mask_tracker_gui as G  # noqa: E402

_APP = QApplication.instance() or QApplication([])
H, W = 120, 160


class SlowSegmenter:
    """Stands in for SAM2: a 10 x 10 blob at the clicked point, after a moment."""
    predictor = object()

    def __init__(self):
        self.calls = []
        self.running = 0
        self.max_running = 0

    def predict_mask(self, pos, neg):
        self.running += 1
        self.max_running = max(self.max_running, self.running)
        self.calls.append(list(pos))
        time.sleep(0.15)
        x, y = pos[0]
        m = np.zeros((H, W), np.uint8)
        m[max(0, y - 5):y + 5, max(0, x - 5):x + 5] = 1
        self.running -= 1
        return m, 0.9

    def set_image(self, img):
        pass


def pump(until, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        _APP.processEvents()
        if until():
            return True
        time.sleep(0.01)
    return False


def test_quick_clicks_queue_and_each_object_joins_the_pending_mask():
    w = G.MaskTrackerWindow()
    seg = SlowSegmenter()
    w._segmenter = seg
    w._image_set = True
    w.preview.set_frame(np.zeros((H, W, 3), np.uint8))
    w.preview.drawing_mode = "ai"
    clicks = [(20, 20), (60, 30), (100, 60), (140, 100)]   # four separate mice
    for c in clicks:
        w.preview._ai_clicks.append(c)
        w.preview._drawing_active = True
        w._request_ai_prediction()
    assert pump(lambda: not w.preview._ai_clicks and not w._predict_worker.isRunning())
    assert seg.max_running == 1                     # never two at once
    assert seg.calls == [[c] for c in clicks]       # every click predicted, in order, alone
    assert len(w.preview.pending_components()) == 4

    # Esc while a prediction runs: its result is dropped, not added.
    w.preview._ai_clicks.append((80, 80))
    w._request_ai_prediction()
    w.preview._clear_drawing()
    assert pump(lambda: not w._predict_worker.isRunning())
    pump(lambda: False, timeout=0.2)
    assert not w.preview.has_ai_pending()
    w._wait_for_prediction()
    w.close()


def test_qt_messages_reach_the_crash_log(monkeypatch):
    import fnt.tool_host as th
    from PyQt5.QtCore import qInstallMessageHandler, qWarning
    log = io.StringIO()
    monkeypatch.setattr(th, "_FAULT_LOG", log)
    th.install_qt_message_handler()
    try:
        qWarning(b"QThread: something worth knowing")
    finally:
        qInstallMessageHandler(None)
    assert "Qt warning" in log.getvalue()
    assert "QThread: something worth knowing" in log.getvalue()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
