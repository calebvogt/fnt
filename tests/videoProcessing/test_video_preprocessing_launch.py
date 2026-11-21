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


def test_shard_sets_the_window_part_and_title(monkeypatch, tmp_path):
    w, started = _window(monkeypatch)
    try:
        V.apply_launch_args(w, V._launch_parser().parse_args(
            ["--input-dir", str(tmp_path), "--shard", "2/4", "--start"]))
        _settle()
        assert w._shard == (2, 4) and "(shard 2/4)" in w.windowTitle()
        assert started == [1]
    finally:
        w.close()


def test_a_bad_shard_stops_the_launch(monkeypatch, tmp_path):
    w, _ = _window(monkeypatch)
    try:
        for bad in ("5/4", "0/4", "2", "a/b"):
            with pytest.raises(SystemExit):
                V.apply_launch_args(w, V._launch_parser().parse_args(
                    ["--input-dir", str(tmp_path), "--shard", bad]))
    finally:
        w.close()


def test_shards_split_the_files_with_no_overlap():
    names = [f"VoleCosm_Camera{c}_2026100{d}000000.avi" for c in range(1, 9) for d in range(1, 10)]
    parts = [[f for f in names if V.in_shard(f, (k, 4))] for k in range(1, 5)]
    assert sorted(sum(parts, [])) == sorted(names)
    assert all(parts)  # every window gets work
    assert all(V.in_shard(f, None) for f in names)


def test_the_worker_processes_only_its_shard(monkeypatch, tmp_path):
    if not _APP:
        _APP.append(QApplication.instance() or QApplication([]))
    for i in range(20):
        (tmp_path / f"clip_{i:02d}.avi").write_bytes(b"x")
    done = {}
    for k in (1, 2, 3):
        w = V.VideoProcessorWorker([str(tmp_path)], [], 30, False, False, True, "mp4", 20,
                                   "1080p", "libx265", "ultrafast", shard=(k, 3))
        seen = []
        monkeypatch.setattr(w, "process_single_file",
                            lambda f, d, i, attempt=1, seen=seen: seen.append(os.path.basename(f)) or True)
        w.run()
        done[k] = seen
    assert sorted(sum(done.values(), [])) == [f"clip_{i:02d}.avi" for i in range(20)]
    assert not set(done[1]) & set(done[2]) and not set(done[2]) & set(done[3])


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
