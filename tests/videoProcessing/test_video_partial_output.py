"""Video PreProcessing writes each encode to <output>.partial and renames it only
when complete: a stop, failure or crash never leaves a half-encoded video under
the real name, which a --skip-existing run would count as done."""
import os

from PyQt5.QtWidgets import QApplication

from fnt.videoProcessing import videoProcessing as V

_APP = []


def _worker(tmp_path):
    if not _APP:
        _APP.append(QApplication.instance() or QApplication([]))
    return V.VideoProcessorWorker([str(tmp_path)], [], 30, False, False, True, "mp4", 20,
                                  "1080p", "libx265", "ultrafast")


class _FakeFfmpeg:
    """Stands in for subprocess.Popen: writes some bytes to the output path (the
    command's last argument), prints a few lines, and exits with ``code``."""
    code = 0

    def __init__(self, cmd, **kwargs):
        self.out = cmd[-1]
        with open(self.out, "wb") as f:
            f.write(b"\0" * 1000)
        self.stdout = iter(["frame=1\n", "frame=2\n"])
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = type(self).code
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


def _video(tmp_path):
    src = tmp_path / "cam1.avi"
    src.write_bytes(b"\0" * 100)
    out_dir = tmp_path / "proc"
    out_dir.mkdir()
    return str(src), str(out_dir), str(out_dir / "cam1.mp4")


def test_the_command_names_the_container_and_writes_the_partial(tmp_path):
    w = _worker(tmp_path)
    cmd = w.build_ffmpeg_command("in.avi", "out.mp4.partial")
    assert cmd[-3:] == ["-f", "mp4", "out.mp4.partial"]


def test_a_complete_encode_gets_the_real_name(monkeypatch, tmp_path):
    w = _worker(tmp_path)
    src, out_dir, out = _video(tmp_path)
    _FakeFfmpeg.code = 0
    monkeypatch.setattr(V.subprocess, "Popen", _FakeFfmpeg)
    monkeypatch.setattr(w, "_get_video_resolution", lambda p: (1920, 1080))
    assert w.process_single_file(src, out_dir, 1) is True
    assert os.path.getsize(out) == 1000
    assert not os.path.exists(out + ".partial")


def test_a_stop_throws_the_half_encoded_video_away(monkeypatch, tmp_path):
    w = _worker(tmp_path)
    src, out_dir, out = _video(tmp_path)
    monkeypatch.setattr(V.subprocess, "Popen", _FakeFfmpeg)
    w.should_stop = True
    assert w.process_single_file(src, out_dir, 1) is False
    assert not os.path.exists(out)
    assert not os.path.exists(out + ".partial")


def test_a_failed_encode_keeps_the_older_output(monkeypatch, tmp_path):
    w = _worker(tmp_path)
    src, out_dir, out = _video(tmp_path)
    with open(out, "wb") as f:
        f.write(b"old")
    _FakeFfmpeg.code = 1
    monkeypatch.setattr(V.subprocess, "Popen", _FakeFfmpeg)
    try:
        assert w.process_single_file(src, out_dir, 1) is False
    finally:
        _FakeFfmpeg.code = 0
    assert open(out, "rb").read() == b"old"
    assert not os.path.exists(out + ".partial")


def test_a_wrong_resolution_is_not_renamed(monkeypatch, tmp_path):
    w = _worker(tmp_path)
    src, out_dir, out = _video(tmp_path)
    monkeypatch.setattr(V.subprocess, "Popen", _FakeFfmpeg)
    monkeypatch.setattr(w, "_get_video_resolution", lambda p: (1280, 720))
    assert w.process_single_file(src, out_dir, 1) is False
    assert not os.path.exists(out)
    assert not os.path.exists(out + ".partial")


def test_skip_existing_redoes_an_unreadable_output(monkeypatch, tmp_path):
    w = _worker(tmp_path)
    src, out_dir, out = _video(tmp_path)
    with open(out, "wb") as f:
        f.write(b"truncated")  # a half-written file from an older version
    w.skip_already_processed = True
    monkeypatch.setattr(w, "_get_video_resolution", lambda p: None)  # ffprobe: moov atom not found
    work, skipped = w._plan_work([(src, out_dir)])
    assert work == [(src, out_dir)] and skipped == 0
