"""Tests for the Camera Grid window's trial/queue workspace handling.

These drive the real QMainWindow offscreen rather than a stand-in, because the
bug they pin was in the interaction between the widgets and the job snapshot:
queueing a trial left every camera loaded, so the next trial merged into it.

Qt is created once per session and the window never shown, so nothing here
needs a display, ffmpeg, or the network share.
"""

import datetime
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fnt.videoProcessing.camera_timeline import (
    CameraTrack, Segment, build_timeline,
)

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication          # noqa: E402

from fnt.videoProcessing.camera_grid_encode import GridLayout   # noqa: E402
from fnt.videoProcessing.camera_grid_pyqt import (              # noqa: E402
    CameraGridWindow,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def win(qapp):
    w = CameraGridWindow()
    yield w
    w.deleteLater()


def make_track(name, folder, day="20260224000000", n=4, hours=24.0):
    """A track of `n` equal segments covering `hours`, with no gaps."""
    each = hours * 3600.0 / n
    segs = [Segment(os.path.join(folder, f"{name}_{day}({i}).mp4"),
                    day, i, each) for i in range(n)]
    placed, gaps = build_timeline(segs)
    track = CameraTrack(name, folder, placed, gaps)
    track.trial_root = folder
    return track


def load_trial(win, trial_root, camera_names):
    """Put a trial into the window the way a finished scan would."""
    for cam in camera_names:
        track = make_track(cam, trial_root)
        win.on_scan_done(track)
    win._on_discovery_done(len(camera_names))


# --- queueing clears the workspace -----------------------------------------

def test_queue_clears_the_loaded_trial(win, tmp_path):
    t006 = str(tmp_path / "T006")
    load_trial(win, t006, ["Camera1", "Camera2"])
    assert len(win.tracks) == 2

    win.add_to_queue()

    assert len(win.queue) == 1
    assert win.tracks == {}
    assert win.camera_list.count() == 0
    assert win.designer.layout_model.assignments == {}
    assert win.out_dir_edit.text() == ""
    assert win.prefix_edit.text() == ""


def test_queued_job_survives_the_clear(win, tmp_path):
    """The snapshot must not be a live view of the widgets it came from."""
    t006 = str(tmp_path / "T006")
    load_trial(win, t006, ["Camera1", "Camera2"])
    win.add_to_queue()

    job = win.queue[0]
    assert job.label == "T006"
    assert sorted(job.tracks) == ["Camera1", "Camera2"]
    assert sorted(job.cameras) == ["Camera1", "Camera2"]
    assert job.chunks                      # the window it will encode
    assert job.out_dir == os.path.join(t006, "grid")
    assert job.prefix == "T006"


def test_second_trial_does_not_merge_into_the_first(win, tmp_path):
    """The reported bug: T007 loaded on top of a queued T006.

    Symptom was cameras named Camera1_2/Camera2_2 and a job whose chunks
    spanned both trials at once.
    """
    t006 = str(tmp_path / "T006")
    t007 = str(tmp_path / "T007")
    load_trial(win, t006, ["Camera1", "Camera2"])
    win.add_to_queue()

    load_trial(win, t007, ["Camera1", "Camera2"])
    win.add_to_queue()

    assert len(win.queue) == 2
    assert [j.label for j in win.queue] == ["T006", "T007"]
    for job in win.queue:
        assert sorted(job.tracks) == ["Camera1", "Camera2"], "dedupe suffix leaked"
    assert win.queue[0].out_dir != win.queue[1].out_dir
    # Each job covers one trial's footage, not both.
    assert win.queue[0].total_seconds == win.queue[1].total_seconds


def test_clear_keeps_output_settings(win, tmp_path):
    """Resolution/fps/codec carry across trials; the trial's own fields don't."""
    load_trial(win, str(tmp_path / "T006"), ["Camera1"])
    before = win.current_settings()

    win.add_to_queue()
    after = win.current_settings()

    assert (after.width, after.height) == (before.width, before.height)
    assert after.fps == before.fps
    assert after.codec == before.codec
    assert after.chunk_mode == before.chunk_mode


def test_clear_resets_the_grid_to_the_default(win, tmp_path):
    """A grid grown to fit five cameras must not size the next trial."""
    load_trial(win, str(tmp_path / "T006"),
               [f"Camera{i}" for i in range(1, 6)])
    grown = win.designer.layout_model
    assert grown.rows * grown.cols >= 5

    win.add_to_queue()

    assert (win.designer.layout_model.rows,
            win.designer.layout_model.cols) == (2, 2)
    assert win.grid_combo.currentText() == "2 x 2"


def test_recalibrating_after_queueing_cannot_rewrite_a_queued_job(win, tmp_path):
    """Tracks are copied into the job, so later nudges only affect new work."""
    load_trial(win, str(tmp_path / "T006"), ["Camera1"])
    live = win.tracks["Camera1"]
    win.add_to_queue()

    live.clock_offset = 5.0
    assert win.queue[0].tracks["Camera1"].clock_offset == 0.0
