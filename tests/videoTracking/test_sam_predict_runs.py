"""SAM predictions run one at a time, and Qt's own messages reach the crash log.

Each SAM click used to start a new prediction thread and drop the one still
running; Qt aborts the process when a running QThread is destroyed, which is
how a labeling session crashed (0xC0000409 in Qt5Core.dll). Clicks during a
run now ask for one more run with the latest points. That abort is a Qt fatal
message, which neither faulthandler nor the slot exception hook sees, so the
Qt message handler writes it to the crash log.

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


class SlowSegmenter:
    """Stands in for SAM2: each prediction takes a moment and is recorded."""
    predictor = object()

    def __init__(self):
        self.calls = []
        self.running = 0
        self.max_running = 0

    def predict_mask(self, pos, neg):
        self.running += 1
        self.max_running = max(self.max_running, self.running)
        self.calls.append((list(pos), list(neg or [])))
        time.sleep(0.2)
        self.running -= 1
        return np.zeros((10, 10), np.uint8), 0.9

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


def test_quick_clicks_run_one_prediction_at_a_time_and_end_with_the_latest_points():
    w = G.MaskTrackerWindow()
    seg = SlowSegmenter()
    w._segmenter = seg
    w._image_set = True
    w.preview.set_frame(np.zeros((10, 10, 3), np.uint8))
    shown = []
    w.preview.set_ai_mask = lambda mask, score: shown.append(score)

    for i in range(4):                       # four clicks in quick succession
        w.preview._ai_positive_points.append((i, i))
        w._request_ai_prediction()
    assert pump(lambda: not w._predict_worker.isRunning() and not w._predict_again and len(seg.calls) >= 2)
    pump(lambda: False, timeout=0.3)
    assert seg.max_running == 1                # never two at once
    assert len(seg.calls) == 2                 # the first, then one with the latest points
    assert seg.calls[-1][0] == [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert shown                               # the mask reached the preview
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
