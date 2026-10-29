"""Whole-video tracking from the command line, and what a run leaves behind.

Each <video>_MaskTracker folder gets run_settings.json and inference.log; a
video folder's MaskTracker_summary.csv has a row per tracked video;
``mask_tracker_cli track`` takes every Inference setting as a flag (or an
earlier run's, with --settings-from) and opens the window on them; --redo moves
old output aside instead of deleting it.

Runs under pytest, or directly.
"""
import argparse
import csv
import json
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

import fnt.videoTracking.mask_tracker_cli as C  # noqa: E402
import fnt.videoTracking.mask_tracker_inference as MI  # noqa: E402
import fnt.videoTracking.mtt_runs as R  # noqa: E402

N_FRAMES = 12
_APP = []


def _app():
    if not _APP:
        _APP.append(QApplication.instance() or QApplication([]))
    return _APP[0]


def make_video(path, n=N_FRAMES):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 15, (64, 48))
    for i in range(n):
        vw.write(np.full((48, 64, 3), 40, np.uint8))
    vw.release()


class _FakeModel:
    """One 10x10 animal per frame, none on frames 3 and 7."""
    categories = {"1": "vole"}

    def __init__(self, model_dir, device="auto", inference_size=0, use_masks=True):
        self.i = 0

    def load_model(self):
        print("[Fake] model loaded")

    def predict(self, frame, conf, max_det):
        i, self.i = self.i, self.i + 1
        if i in (3, 7):
            return {"boxes": np.zeros((0, 4), np.float32), "masks": None,
                    "scores": np.zeros(0, np.float32), "labels": np.zeros(0, np.int32)}
        m = np.zeros(frame.shape[:2], bool)
        m[10:20, 10 + i:20 + i] = True
        return {"boxes": np.array([[10 + i, 10, 20 + i, 20]], np.float32), "masks": m[None],
                "scores": np.array([0.9], np.float32), "labels": np.array([1], np.int32)}


@pytest.fixture
def tracked(monkeypatch, tmp_path):
    monkeypatch.setattr(MI, "MaskRCNNInference", _FakeModel)
    monkeypatch.setattr(MI, "_detect_architecture", lambda d: "maskrcnn")
    video = str(tmp_path / "cam1.avi")
    make_video(video)
    cfg = MI.MaskInferenceConfig(confidence_threshold=0.4, max_detections=1, use_masks=True,
                                 save_masks=False)
    res = MI.run_inference_on_video(video, str(tmp_path / "model"), cfg,
                                    create_tracked_video=False,
                                    run_info={"parallel": {"workers": 3, "worker": 2}})
    return video, res


# --- the run record -----------------------------------------------------------

def test_a_run_records_its_settings_and_log(tracked):
    video, res = tracked
    rec = R.read_run_settings(res["output_dir"])
    assert rec["complete"] is True and rec["frames_processed"] == N_FRAMES
    assert rec["frames_without_detection"] == 2
    assert rec["settings"]["confidence_threshold"] == 0.4
    assert rec["settings"]["max_detections"] == 1
    assert rec["create_tracked_video"] is False
    assert rec["parallel"] == {"workers": 3, "worker": 2}
    assert rec["tracks"] >= 1 and rec["fnt_version"] and rec["finished"]
    log = open(os.path.join(res["output_dir"], R.LOG_FILE), encoding="utf-8").read()
    assert "[Fake] model loaded" in log and "[MTT Inference] Done" in log


def test_the_folder_summary_has_a_row_per_tracked_video(tracked, tmp_path):
    video, res = tracked
    # An older run without run_settings.json: read from its trajectories.
    old = tmp_path / "cam2_MaskTracker"
    old.mkdir()
    (tmp_path / "cam2.avi").write_bytes(b"")
    (old / "trajectories.csv").write_text("frame,object_id\n0,1\n1,1\n")
    path = R.write_folder_summary(str(tmp_path))
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    by = {r["video"]: r for r in rows}
    assert by["cam1.avi"]["status"] == "done"
    assert by["cam1.avi"]["frames_without_detection"] == "2"
    assert by["cam1.avi"]["workers"] == "3"
    assert by["cam2.avi"]["status"] == "no record" and by["cam2.avi"]["tracks"] == "1"


def test_tracking_folders_never_count_as_videos(tracked, tmp_path):
    video, res = tracked
    (tmp_path / "cam1_MaskTracker" / "cam1_annotated.mp4").write_bytes(b"x")
    assert R.find_track_videos([str(tmp_path)]) == [video]


# --- settings from the command line ---------------------------------------------

def _track_args(argv):
    return C._parser().parse_args(["track", "--videos", "v.mp4"] + argv)


def test_every_setting_has_a_flag():
    a = _track_args(["--conf", "0.6", "--max-objects", "2", "--max-disappear", "never",
                     "--iou", "0.4", "--matching", "greedy", "--resolution", "512",
                     "--infer-device", "cuda", "--no-masks", "--no-tracked-video",
                     "--save-masks", "--no-preview", "--classifier", "none",
                     "--parallel", "auto"])
    d = C.inference_settings(a)
    assert d == {"confidence_threshold": 0.6, "max_detections": 2, "max_disappeared_frames": 0,
                 "iou_match_threshold": 0.4, "matching_algorithm": "greedy", "inference_size": 512,
                 "device": "cuda", "use_masks": False, "create_tracked_video": False,
                 "save_masks": True, "show_preview": False, "classifier_dir": None,
                 "parallel": "auto"}
    assert C.inference_settings(_track_args([])) == {}       # nothing given: the tab's values


def test_flags_survive_the_trip_to_the_window():
    a = _track_args(["--conf", "0.6", "--no-masks", "--parallel", "3", "--quit-when-done"])
    again = argparse.ArgumentParser()
    C.add_inference_settings(again)
    b = again.parse_args(C.settings_argv(a))
    assert C.inference_settings(b) == C.inference_settings(a) and b.quit_when_done


def test_settings_from_an_earlier_run_and_a_flag_wins(tracked):
    video, res = tracked
    d = C.inference_settings(_track_args(["--settings-from", res["output_dir"], "--conf", "0.7"]))
    assert d["max_detections"] == 1 and d["use_masks"] is True
    assert d["confidence_threshold"] == 0.7
    assert d["create_tracked_video"] is False
    assert d["model_dir"].endswith("model")


def test_bad_parallel_is_refused():
    with pytest.raises(ValueError):
        C.inference_settings(_track_args(["--parallel", "lots"]))


# --- the track command ----------------------------------------------------------

def _project(tmp_path):
    proj = tmp_path / "proj"
    run = proj / "models" / "run1"
    run.mkdir(parents=True)
    (proj / "project_config.json").write_text(json.dumps({"project_dir": str(proj)}))
    (run / "weights.pt").write_bytes(b"")
    return str(proj), str(run)


def test_track_opens_the_models_project_on_the_videos(monkeypatch, tmp_path):
    import fnt.videoTracking.mask_tracker_gui as G
    proj, run = _project(tmp_path)
    vids = tmp_path / "videos"
    vids.mkdir()
    make_video(str(vids / "a.avi"))
    seen = {}
    monkeypatch.setattr(G, "main", lambda argv: seen.setdefault("argv", argv) and 0)
    C.main(["track", "--videos", str(vids), "--model", run, "--parallel", "auto", "--conf", "0.5"])
    argv = seen["argv"]
    assert argv[:2] == ["--project", proj]
    assert argv[argv.index("--infer") + 1] == str(vids / "a.avi")
    assert argv[argv.index("--infer-model") + 1] == run
    assert "--parallel" in argv and "--conf" in argv


def test_redo_moves_the_old_output_aside(monkeypatch, tmp_path):
    import fnt.videoTracking.mask_tracker_gui as G
    proj, run = _project(tmp_path)
    vids = tmp_path / "videos"
    vids.mkdir()
    make_video(str(vids / "a.avi"))
    (vids / "a_MaskTracker").mkdir()
    (vids / "a_MaskTracker" / "trajectories.csv").write_text("frame\n0\n")
    monkeypatch.setattr(G, "main", lambda argv: 0)
    C.main(["track", "--videos", str(vids), "--model", run, "--redo"])
    assert not (vids / "a_MaskTracker").exists()
    assert (vids / "_TO_DELETE_tracking" / "a_MaskTracker" / "trajectories.csv").exists()


def test_track_needs_a_project_for_a_loose_model(tmp_path, capsys):
    loose = tmp_path / "loose"
    loose.mkdir()
    assert C.main(["track", "--videos", str(tmp_path), "--model", str(loose)]) == 2


# --- the tab shows what the run uses ----------------------------------------------

def test_settings_go_into_the_inference_tab(monkeypatch):
    _app()
    import fnt.videoTracking.mask_tracker_gui as G
    from PyQt5.QtWidgets import QMessageBox
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.No)
    win = G.MaskTrackerWindow()
    win._apply_inference_settings({
        "confidence_threshold": 0.65, "max_detections": 1, "max_disappeared_frames": 45,
        "iou_match_threshold": 0.25, "matching_algorithm": "greedy", "inference_size": 512,
        "device": "cpu", "use_masks": True, "save_masks": False, "create_tracked_video": False,
        "show_preview": False, "classifier_dir": None, "parallel": "3"})
    assert win.spin_track_confidence.value() == pytest.approx(0.65)
    assert win.spin_track_max_det.value() == 1
    assert win.combo_track_disappear.currentText() == "45"
    assert win.combo_track_matching.currentText().startswith("Greedy")
    assert win.combo_track_resolution.currentText() == "512px"
    assert win.combo_track_device.currentText() == "CPU"
    assert win.chk_track_masks.isChecked() and not win.chk_save_masks.isChecked()
    assert not win.chk_create_tracked_video.isChecked() and not win.chk_inference_preview.isChecked()
    assert win.chk_parallel_videos.isChecked() and win.spin_parallel_videos.value() == 3
    win.close()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
