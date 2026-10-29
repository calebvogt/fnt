"""Tracking several videos at once (mask_tracker_parallel).

How many run at once comes from the machine's limits; a worker takes videos
off the queue, reports progress, small preview frames and results, and stops
when asked; a stopped video's partial output folder is removed; the tile grid
has one tile per worker and takes the preview's place on the Inference tab.

Runs under pytest, or directly.
"""
import os
import queue
import sys
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

import fnt.videoTracking.mask_tracker_parallel as P  # noqa: E402

_APP = []


def _app():
    if not _APP:
        _APP.append(QApplication.instance() or QApplication([]))
    return _APP[0]


# --- how many at once -------------------------------------------------------

def test_auto_count_is_the_tightest_limit():
    n, why = P.auto_worker_count(10, "cuda", cores=24, ram_gb=120, gpu_free_mb=14000)
    assert n == 8 and "CPU 8" in why and "GPU 8" in why
    assert P.auto_worker_count(3, "cuda", cores=24, ram_gb=120, gpu_free_mb=14000)[0] == 3
    assert P.auto_worker_count(10, "cuda", cores=24, ram_gb=120, gpu_free_mb=5000)[0] == 2
    assert P.auto_worker_count(10, "cuda", cores=24, ram_gb=7, gpu_free_mb=14000)[0] == 2


def test_auto_count_on_the_cpu_ignores_the_gpu():
    n, why = P.auto_worker_count(10, "cpu", cores=24, ram_gb=120, gpu_free_mb=0)
    assert n == 4 and "GPU" not in why


def test_auto_count_never_below_one():
    assert P.auto_worker_count(5, "cuda", cores=2, ram_gb=1, gpu_free_mb=100)[0] == 1
    assert P.auto_worker_count(0, "cuda")[0] == 0


# --- one worker, in this process --------------------------------------------

def _run_worker(monkeypatch, videos, fake, stop_after=None, opts=None):
    """Run _worker_main on a thread with plain queues; return its messages."""
    monkeypatch.setattr(P, "infer_one_video", fake)
    monkeypatch.setattr(P, "load_classifier", lambda d: None)
    jobs, msgs = queue.Queue(), queue.Queue()
    for i, v in enumerate(videos):
        jobs.put((i, v))
    stop_ev, pause_ev = threading.Event(), threading.Event()
    if stop_after is not None:
        fake.stop_ev = stop_ev
    out, err = sys.stdout, sys.stderr
    try:
        P._worker_main(0, jobs, msgs, stop_ev, pause_ev, "model", {},
                       dict({"preview_interval": 0.0, "preview_width": 32}, **(opts or {})))
    finally:
        sys.stdout, sys.stderr = out, err
    got = []
    while not msgs.empty():
        got.append(msgs.get())
    return got


class _Fake:
    """Stands in for infer_one_video: 10 frames, a preview frame each, a result."""

    def __init__(self, stop_on=None):
        self.stop_on = stop_on
        self.stop_ev = None
        self.calls = []

    def __call__(self, video, model_dir, config, cls_info, **kw):
        self.calls.append(video)
        for f in range(1, 11):
            kw["progress"](f, 10)
            if kw["frame_callback"] is not None:
                kw["frame_callback"](np.zeros((90, 160, 3), np.uint8), f)
        if self.stop_on == video:
            self.stop_ev.set()
            return {"video_path": video, "stopped": True}
        print(f"done {video}")
        return {"video_path": video, "num_tracks": 1, "total_frames": 10}


def test_a_worker_takes_every_video_and_reports(monkeypatch):
    fake = _Fake()
    got = _run_worker(monkeypatch, ["a.mp4", "b.mp4"], fake)
    kinds = [m[0] for m in got]
    assert fake.calls == ["a.mp4", "b.mp4"]
    assert kinds.count("started") == 2 and kinds.count("finished") == 2
    assert kinds[-1] == "exit"
    assert [m[2] for m in got if m[0] == "finished"] == [0, 1]
    prog = [m for m in got if m[0] == "progress"]
    assert prog and prog[-1][3:5] == (10, 10)          # the last frame is always reported
    frames = [m for m in got if m[0] == "frame"]
    assert frames and frames[0][3].shape == (18, 32, 3)  # shrunk to the preview width
    assert ("log", 0, "done a.mp4") in got               # stdout comes back as log lines


def test_no_preview_frames_when_the_preview_is_off(monkeypatch):
    got = _run_worker(monkeypatch, ["a.mp4"], _Fake(), opts={"show_preview": False})
    assert not [m for m in got if m[0] == "frame"]


def test_a_stop_ends_the_worker_after_its_video(monkeypatch):
    fake = _Fake(stop_on="a.mp4")
    got = _run_worker(monkeypatch, ["a.mp4", "b.mp4"], fake, stop_after=True)
    kinds = [m[0] for m in got]
    assert fake.calls == ["a.mp4"]                     # b.mp4 never started
    assert "stopped" in kinds and "finished" not in kinds and kinds[-1] == "exit"


def test_a_failed_video_does_not_end_the_worker(monkeypatch):
    def fake(video, *a, **kw):
        if video == "bad.mp4":
            raise RuntimeError("cannot read")
        return {"video_path": video, "num_tracks": 1, "total_frames": 1}
    got = _run_worker(monkeypatch, ["bad.mp4", "good.mp4"], fake)
    assert [m[3] for m in got if m[0] == "error"] == ["bad.mp4: cannot read"]
    assert [m[2] for m in got if m[0] == "finished"] == [1]


# --- the shared per-video code ----------------------------------------------

def test_a_stopped_video_loses_its_partial_output(monkeypatch, tmp_path):
    import fnt.videoTracking.mask_tracker_inference as MI
    out = tmp_path / "v_MaskTracker"

    def fake_run(video_path, model_dir, config, **kw):
        out.mkdir()
        (out / "v_annotated.mp4").write_bytes(b"half")
        return {"output_dir": str(out), "num_tracks": 1, "total_frames": 5}
    monkeypatch.setattr(MI, "run_inference_on_video", fake_run)
    r = P.infer_one_video("v.mp4", "m", object(), stop_requested=lambda: True)
    assert r["stopped"] and not out.exists()
    out_kept = tmp_path / "v_MaskTracker"
    r = P.infer_one_video("v.mp4", "m", object(), stop_requested=lambda: False)
    assert not r.get("stopped") and out_kept.exists()


def test_min_bout_filter():
    rows = [{"object_id": 1, "behavior": b} for b in "AAABBAAAAA"]
    P.apply_min_bout_filter(rows, 3)
    assert "".join(r["behavior"][0] for r in rows) == "AAANNAAAAA"


# --- the tiles ----------------------------------------------------------------

def test_the_grid_has_a_tile_per_worker():
    _app()
    from fnt.videoTracking.mask_tracker_parallel_view import InferenceGrid
    g = InferenceGrid()
    g.resize(800, 600)
    g.set_slots(5)
    assert len(g.tiles) == 5
    g.video_started(2, 6, r"C:\v\T014_F9953_PreOFT.mp4")
    assert g.tiles[2].lbl_name.text() == "7. T014_F9953_PreOFT.mp4"
    g.video_progress(2, 900, 18000, 30.0)
    assert "frame 900/18000" in g.tiles[2].lbl_stats.text()
    g.video_frame(2, np.full((90, 160, 3), 200, np.uint8), 900)
    assert g.tiles[2]._image is not None
    g.set_slots(2)
    assert len(g.tiles) == 2


def test_the_grid_takes_the_previews_place_on_the_inference_tab(monkeypatch):
    _app()
    import fnt.videoTracking.mask_tracker_gui as G
    from PyQt5.QtWidgets import QMessageBox
    # No project is open: the Inference tab would ask to open one.
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.No)
    win = G.MaskTrackerWindow()
    win.show()
    win.tab_widget.setCurrentIndex(win._inference_tab_idx)
    assert win.preview.isVisible() and not win._infer_grid.isVisible()
    win._show_infer_grid(True)
    assert win._infer_grid.isVisible() and not win.preview.isVisible()
    win.tab_widget.setCurrentIndex(0)                 # another tab: the preview again
    assert not win._infer_grid.isVisible()
    win.tab_widget.setCurrentIndex(win._inference_tab_idx)
    assert win._infer_grid.isVisible() and not win.preview.isVisible()
    win._show_infer_grid(False)
    assert win.preview.isVisible() and not win._infer_grid.isVisible()
    assert win.chk_parallel_videos.isChecked() is False
    assert win.spin_parallel_videos.text() == "Auto"
    win.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
