"""The first recording after opening a project honours the Window setting.

Seen in a screenshot: "Window: 2.00 s" under a spectrogram spanning ten
seconds. Opening a project clears the spectrogram, and clearing parks the view
on a 10 s placeholder; the first file then loaded "keeping the current view
width", which was the placeholder's. Every file after the first kept that 10 s
too, while the control went on saying 2 s.

The Window control is the user's setting, so it decides the width whenever
there is no real view to keep.

Runs under pytest, or directly.
"""
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402
from scipy.io import wavfile  # noqa: E402

import fnt.usv.mad_pyqt as M  # noqa: E402
from fnt.usv.usv_detector.mad_project import (  # noqa: E402
    MADProjectConfig, create_mad_project)

SR = 250_000
_KEEP = []


def window_with_files(n=2, seconds=30):
    root = tempfile.mkdtemp(prefix="mad_view_")
    wavs = []
    for i in range(n):
        p = os.path.join(root, f"r{i}.wav")
        wavfile.write(p, SR, (np.random.default_rng(i).normal(0, .02,
                                                              SR * seconds)
                              * 32767).astype(np.int16))
        wavs.append(p)
    proj = os.path.join(root, "p")
    create_mad_project(proj)
    app = QApplication.instance() or QApplication([])
    M.MADMainWindow._apply_dark_theme()
    w = M.MADMainWindow()
    _KEEP.append((app, w))
    w.spin_view_window.setValue(2.0)
    w._activate_project(MADProjectConfig.load(proj))
    w._register_audio_files(wavs)
    w._append_audio_paths(wavs)
    w._wait_for_audio_load()
    for _ in range(20):
        app.processEvents()
    return w, wavs


def width(w):
    return w.spectrogram.view_end - w.spectrogram.view_start


def test_the_first_file_after_opening_uses_the_window_setting():
    w, _ = window_with_files()
    assert abs(width(w) - w.spin_view_window.value()) < 1e-6, \
        f"view {width(w):.2f} s under a {w.spin_view_window.value():.2f} s " \
        "Window setting"


def test_the_next_file_keeps_the_same_width():
    w, wavs = window_with_files()
    w._on_file_selected(1)
    w._wait_for_audio_load()
    assert abs(width(w) - 2.0) < 1e-6, width(w)


def test_a_zoom_the_user_made_survives_a_file_switch():
    """Keeping the view across files is the point of the rule; only the
    placeholder must not be kept."""
    w, wavs = window_with_files()
    w.spin_view_window.setValue(5.0)
    assert abs(width(w) - 5.0) < 1e-6
    w._on_file_selected(1)
    w._wait_for_audio_load()
    assert abs(width(w) - 5.0) < 1e-6, width(w)


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
