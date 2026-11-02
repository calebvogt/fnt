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


def _write_wav(path, seconds=0.1, sr=250000, seed=0):
    pcm = (np.random.default_rng(seed).standard_normal(int(sr * seconds))
           * 1000).astype("<i2").tobytes()
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
                + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
                + b"data" + struct.pack("<I", len(pcm)) + pcm)


def _run(folder, out, skip_done=False):
    worker = C.CompressWorker(str(folder), str(out), "adpcm_ima_wav",
                              skip_done=skip_done)
    done, log = [], []
    worker.finished.connect(lambda ok, msg: done.append(ok))
    worker.progress.connect(log.append)
    worker.run()  # synchronously, in this thread
    return done, log


needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH")


@needs_ffmpeg
def test_worker_writes_250k_mono_adpcm(tmp_path):
    sr = 250000
    _write_wav(tmp_path / "a.wav", sr=sr)
    out = tmp_path / "proc"
    done, _ = _run(tmp_path, out)
    assert done == [True]
    with open(out / "a.wav", "rb") as f:
        head = f.read(36)
    channels, rate = struct.unpack_from("<HI", head, 22)
    bits = struct.unpack_from("<H", head, 34)[0]
    assert (channels, rate, bits) == (1, sr, 4)
    assert not list(out.glob("*.part"))


@needs_ffmpeg
def test_skip_done_keeps_finished_files_and_redoes_a_cut_short_one(tmp_path):
    """Resuming after a crash: a complete output is kept as it is; one cut
    short (a different length from its original) is compressed again."""
    _write_wav(tmp_path / "a.wav", seconds=0.5, seed=1)
    _write_wav(tmp_path / "b.wav", seconds=0.5, seed=2)
    out = tmp_path / "proc"
    assert _run(tmp_path, out)[0] == [True]
    whole = (out / "a.wav").stat().st_mtime_ns
    _write_wav(out / "b.wav", seconds=0.1, seed=3)   # a short, stale output
    assert not C.already_compressed(str(tmp_path / "b.wav"), str(out / "b.wav"))
    time.sleep(0.05)

    done, log = _run(tmp_path, out, skip_done=True)
    assert done == [True]
    assert (out / "a.wav").stat().st_mtime_ns == whole
    assert any("Already compressed: a.wav" in m for m in log)
    assert any("Processing: b.wav" in m for m in log)
    assert C.already_compressed(str(tmp_path / "b.wav"), str(out / "b.wav"))


@needs_ffmpeg
def test_without_skip_done_everything_is_redone(tmp_path):
    _write_wav(tmp_path / "a.wav")
    out = tmp_path / "proc"
    _run(tmp_path, out)
    done, log = _run(tmp_path, out)
    assert done == [True] and any("Processing: a.wav" in m for m in log)


@needs_ffmpeg
def test_a_failed_file_leaves_nothing_under_its_name(tmp_path):
    (tmp_path / "bad.wav").write_bytes(b"not a wav file at all")
    out = tmp_path / "proc"
    _run(tmp_path, out)
    assert list(out.iterdir()) == []


def test_skip_done_flag_ticks_the_box(monkeypatch, tmp_path):
    w, started = _window(monkeypatch)
    try:
        assert not w.chk_skip_done.isChecked()
        C.apply_launch_args(w, C._launch_parser().parse_args(
            ["--input-dir", str(tmp_path), "--skip-done"]))
        assert w.chk_skip_done.isChecked() and started == []
    finally:
        w.close()


def test_missing_output_is_not_already_compressed(tmp_path):
    assert not C.already_compressed(str(tmp_path / "a.wav"),
                                    str(tmp_path / "proc" / "a.wav"))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
