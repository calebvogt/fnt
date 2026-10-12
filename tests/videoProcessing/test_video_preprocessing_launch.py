"""Video PreProcessing can be started from the command line, on screen.

``--input-dir DIR --start`` preselects the folders and presses Start with the
window's default settings, so a dictated or queued run still shows its
progress bar.

Runs under pytest, or directly.
"""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

import fnt.videoProcessing.videoProcessing as V  # noqa: E402

_APP = []


def _window(monkeypatch):
    if not _APP:
        _APP.append(QApplication.instance() or QApplication([]))
    w = V.VideoProcessingGUI()
    started = []
    monkeypatch.setattr(w, "start_processing", lambda: started.append(1))
    return w, started


def _settle():
    for _ in range(3):
        _APP[0].processEvents()
        time.sleep(0.25)
    _APP[0].processEvents()


def test_input_dir_and_start(monkeypatch, tmp_path):
    w, started = _window(monkeypatch)
    try:
        V.apply_launch_args(w, V._launch_parser().parse_args(
            ["--input-dir", str(tmp_path), "--start"]))
        _settle()
        assert w.selected_dirs == [os.path.abspath(str(tmp_path))]
        assert w.start_btn.isEnabled() and started == [1]
    finally:
        w.close()


def test_without_start_it_only_preselects(monkeypatch, tmp_path):
    w, started = _window(monkeypatch)
    try:
        V.apply_launch_args(w, V._launch_parser().parse_args(
            ["--input-dir", str(tmp_path)]))
        _settle()
        assert w.selected_dirs and started == []
    finally:
        w.close()


def test_a_missing_folder_is_ignored_and_nothing_starts(monkeypatch, tmp_path):
    w, started = _window(monkeypatch)
    try:
        V.apply_launch_args(w, V._launch_parser().parse_args(
            ["--input-dir", str(tmp_path / "nope"), "--start"]))
        _settle()
        assert w.selected_dirs == [] and started == []
    finally:
        w.close()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
