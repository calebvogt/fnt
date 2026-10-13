"""Compress WAV Files can be started from the command line, on screen.

``--input-dir DIR --start`` preselects the folders and presses Start
Compression with the window's default settings, so a dictated or queued run
still shows its progress bar.

Runs under pytest, or directly.
"""
import os
import shutil
import struct
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

import fnt.usv.compress_wavs_pyqt as C  # noqa: E402

_APP = []


def _window(monkeypatch):
    if not _APP:
        _APP.append(QApplication.instance() or QApplication([]))
    w = C.CompressWavsWindow()
    started = []
    monkeypatch.setattr(w, "start_compression", lambda: started.append(1))
    return w, started


def _settle():
    for _ in range(3):
        _APP[0].processEvents()
        time.sleep(0.25)
    _APP[0].processEvents()


def test_input_dir_and_start(monkeypatch, tmp_path):
    w, started = _window(monkeypatch)
    try:
        C.apply_launch_args(w, C._launch_parser().parse_args(
            ["--input-dir", str(tmp_path), "--start"]))
        _settle()
        assert w.folder_paths == [os.path.abspath(str(tmp_path))]
        assert w.btn_process.isEnabled() and started == [1]
    finally:
        w.close()


def test_without_start_it_only_preselects(monkeypatch, tmp_path):
    w, started = _window(monkeypatch)
    try:
        C.apply_launch_args(w, C._launch_parser().parse_args(
            ["--input-dir", str(tmp_path)]))
        _settle()
        assert w.folder_paths and started == []
    finally:
        w.close()


def test_a_missing_folder_is_ignored_and_nothing_starts(monkeypatch, tmp_path):
    w, started = _window(monkeypatch)
    try:
        C.apply_launch_args(w, C._launch_parser().parse_args(
            ["--input-dir", str(tmp_path / "nope"), "--start"]))
        _settle()
        assert w.folder_paths == [] and started == []
    finally:
        w.close()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_worker_writes_250k_mono_adpcm(tmp_path):
    sr = 250000
    pcm = (np.random.default_rng(0).standard_normal(sr // 10) * 1000).astype("<i2").tobytes()
    with open(tmp_path / "a.wav", "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
                + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
                + b"data" + struct.pack("<I", len(pcm)) + pcm)
    out = tmp_path / "proc"
    worker = C.CompressWorker(str(tmp_path), str(out), "adpcm_ima_wav")
    done = []
    worker.finished.connect(lambda ok, msg: done.append(ok))
    worker.run()  # synchronously, in this thread
    assert done == [True]
    with open(out / "a.wav", "rb") as f:
        head = f.read(36)
    channels, rate = struct.unpack_from("<HI", head, 22)
    bits = struct.unpack_from("<H", head, 34)[0]
    assert (channels, rate, bits) == (1, sr, 4)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
