"""Three things noticed during the 2026-10-12 QC session.

1. A merge-off re-run skipped every recording a merge-on run had done: the
   resume check compared model, threshold and min-blob but not merging, so 9
   of the first 10 mad_v5_VC files were "already analysed" and nothing split.
2. An ffmpeg terminal flashed up on every file switch: MAD started without a
   console (pythonw, a shortcut) gives each console subprocess its own window.
3. Clicking the speed slider's groove sent the handle to an end: Qt moves one
   page per click, and with nine speeds one page is the whole range.

Runs under pytest, or directly.
"""
import os
import subprocess
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from fnt.usv.usv_detector import fnt_mask_store as ms  # noqa: E402
from fnt.usv.usv_detector.mad_batch import (  # noqa: E402
    RunSettings, file_already_done)


# ---------------------------------------------------------------- resume
def _store(d, merged):
    wav = os.path.join(d, "r.wav")
    open(wav, "wb").close()
    h5 = ms.masks_sibling_path(wav)
    ms.write_pred_masks(h5, [{"blob_id": 0, "mask": np.ones((4, 4), bool),
                              "f_off": 10, "t_off": 10, "model_name": "m1",
                              "threshold": 0.5, "min_blob_pixels": 50}])
    if merged is not None:
        ms.set_infer_run_attrs(h5, last_infer_model="m1",
                               last_infer_threshold=0.5,
                               last_infer_min_blob=50,
                               last_infer_merge=1.0 if merged else 0.0)
    return wav


def _settings(merge):
    return RunSettings(model_name="m1", threshold=0.5, min_blob_pixels=50,
                       merge_consecutive=merge)


def test_a_merge_on_result_is_not_reused_for_a_merge_off_run():
    with tempfile.TemporaryDirectory() as d:
        wav = _store(d, merged=True)
        assert not file_already_done(wav, _settings(False))
        assert file_already_done(wav, _settings(True))


def test_a_merge_off_result_is_not_reused_for_a_merge_on_run():
    with tempfile.TemporaryDirectory() as d:
        wav = _store(d, merged=False)
        assert not file_already_done(wav, _settings(True))
        assert file_already_done(wav, _settings(False))


def test_a_store_that_cannot_say_how_it_merged_is_redone():
    with tempfile.TemporaryDirectory() as d:
        wav = _store(d, merged=None)
        assert not file_already_done(wav, _settings(False))


# ---------------------------------------------------------------- no window
def test_subprocesses_open_no_console_window_on_windows():
    from fnt.usv.usv_detector.spectrogram import NO_WINDOW
    if os.name == "nt":
        assert NO_WINDOW == {"creationflags": subprocess.CREATE_NO_WINDOW}
    else:
        assert NO_WINDOW == {}


def test_every_subprocess_call_in_mad_passes_it():
    """A new call without the flag brings the flashing window back."""
    import re
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    for rel in ("fnt/usv/mad_pyqt.py", "fnt/usv/usv_detector/spectrogram.py"):
        src = open(os.path.join(root, rel), encoding="utf-8").read()
        for m in re.finditer(r"subprocess\.run\(", src):
            depth, i = 0, src.index("(", m.start())
            while True:
                depth += {"(": 1, ")": -1}.get(src[i], 0)
                if depth == 0:
                    break
                i += 1
            call = src[m.start():i + 1]
            assert "NO_WINDOW" in call, f"{rel}: {call[:80]}"


# ---------------------------------------------------------------- slider
_APP = []


def _slider(n):
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt
    import fnt.usv.mad_pyqt as M
    if not _APP:
        _APP.append(QApplication.instance() or QApplication([]))
    s = M.ClickJumpSlider(Qt.Horizontal)
    s.setRange(0, n - 1)
    s.setFixedWidth(300)
    s.resize(300, 24)
    s.setValue(n - 1)
    return s


def _click(s, x):
    from PyQt5.QtCore import QPoint, Qt
    from PyQt5.QtTest import QTest
    QTest.mouseClick(s, Qt.LeftButton, Qt.NoModifier, QPoint(x, 12))


def test_a_click_in_the_middle_lands_in_the_middle():
    s = _slider(9)                  # the nine playback speeds
    _click(s, 150)
    assert 3 <= s.value() <= 5, s.value()


def test_a_click_near_an_end_lands_near_that_end():
    s = _slider(9)
    _click(s, 8)
    assert s.value() == 0
    s.setValue(0)
    _click(s, 292)
    assert s.value() == 8


def test_the_speed_and_score_sliders_jump_to_the_click():
    import fnt.usv.mad_pyqt as M
    _slider(2)                      # makes sure the QApplication exists
    w = M.MADMainWindow()
    try:
        assert isinstance(w.slider_speed, M.ClickJumpSlider)
        assert isinstance(w.slider_min_score, M.ClickJumpSlider)
    finally:
        w.close()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
