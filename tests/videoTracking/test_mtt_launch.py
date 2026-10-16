"""The Mask Tracker from the command line.

``python -m fnt.videoTracking.mask_tracker_cli extract-frames`` cuts listed
frames into a project under the GUI's names and registers their videos;
``python -m fnt.videoTracking.mask_tracker_gui --project P --focus LIST``
shows only those frames; ``--train`` with training flags puts the flags in the
Train panel (others keep the panel's value) and starts the run. A project
opened this way must never stop at a dialog.

Runs under pytest, or directly.
"""
import json
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PyQt5.QtWidgets import QApplication, QMessageBox  # noqa: E402

import fnt.videoTracking.mask_tracker_cli as C  # noqa: E402
import fnt.videoTracking.mask_tracker_gui as G  # noqa: E402

N_FRAMES = 30


def make_video(path):
    """30 frames; frame i is a flat grey of level 8*i (readable back after codec loss)."""
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 15, (64, 48))
    for i in range(N_FRAMES):
        vw.write(np.full((48, 64, 3), 8 * i, np.uint8))
    vw.release()


def make_project(root, frames=()):
    proj = os.path.join(root, "proj")
    os.makedirs(os.path.join(proj, "training_frames"))
    os.makedirs(os.path.join(proj, "annotations"))
    os.makedirs(os.path.join(proj, "models"))
    cfg = {"project_dir": "Z:/somewhere/else", "video_paths": [], "source_videos": [],
           "categories": ["mouse"], "sam2_model_path": None}
    with open(os.path.join(proj, "project_config.json"), "w") as fh:
        json.dump(cfg, fh)
    for name in frames:
        cv2.imwrite(os.path.join(proj, "training_frames", name), np.zeros((48, 64, 3), np.uint8))
    return proj


# ----------------------------------------------------------------------
# focus items
# ----------------------------------------------------------------------
def test_focus_items_from_names_paths_and_lists(tmp_path):
    lst = tmp_path / "focus.txt"
    lst.write_text("# round 1\nCAM13_a_frame_000010.png\n\nCAM14_b_frame_000020\n", encoding="utf-8")
    names = G.expand_focus_items([
        r"Z:\proj\training_frames\CAM13_a_frame_000010.png",   # duplicate of the list's first
        str(lst),
        "/x/y/CAM15_c_frame_000030.jpg",
    ])
    assert names == ["CAM13_a_frame_000010.png", "CAM14_b_frame_000020.png",
                     "CAM15_c_frame_000030.jpg"]


# ----------------------------------------------------------------------
# extract-frames
# ----------------------------------------------------------------------
def test_extract_frames_names_content_registry_and_focus_list(tmp_path):
    video = str(tmp_path / "CAM13_2021-10-22_12_14_14_870.avi")
    make_video(video)
    proj = make_project(str(tmp_path))
    spec = tmp_path / "frames.csv"
    spec.write_text(f"video,frame\n{video},3\n{video},10\n{video},10\n{video},29\n{video},99\n",
                    encoding="utf-8")
    focus = tmp_path / "focus.txt"
    rc = C.main(["extract-frames", "--project", proj, "--frames", str(spec),
                 "--focus-out", str(focus)])
    assert rc == 0
    tf = os.path.join(proj, "training_frames")
    want = [f"CAM13_2021-10-22_12_14_14_870_frame_{i:06d}.png" for i in (3, 10, 29)]
    assert sorted(os.listdir(tf)) == sorted(want)
    for i, name in zip((3, 10, 29), want):
        img = cv2.imread(os.path.join(tf, name), cv2.IMREAD_GRAYSCALE)
        assert abs(float(img.mean()) - 8 * i) < 4, name     # the right frame, not a neighbour
    assert focus.read_text(encoding="utf-8").split() == want
    cfg = json.load(open(os.path.join(proj, "project_config.json")))
    assert cfg["source_videos"] == [video]
    # A second run keeps the frames and registers nothing twice.
    res = C.extract_frames(proj, C.read_frame_list(str(spec)))
    assert res["registered"] == 0 and res["frames"] == want
    assert any("frame 99" in p for p in res["problems"])


def test_frame_list_needs_video_and_frame_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("path,idx\na,1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        C.read_frame_list(str(bad))


# ----------------------------------------------------------------------
# the window
# ----------------------------------------------------------------------
_APP = None


def window():
    global _APP
    _APP = QApplication.instance() or QApplication([])
    w = G.MaskTrackerWindow()
    w._launched = True
    return w


FRAMES = ["CAM13_a_frame_000001.png", "CAM13_a_frame_000002.png", "CAM14_b_frame_000003.png"]


def test_launched_open_never_asks_and_focus_limits_the_queue(monkeypatch):
    root = tempfile.mkdtemp(prefix="mtt_launch_")
    proj = make_project(root, FRAMES)
    cfg_path = os.path.join(proj, "project_config.json")
    cfg = json.load(open(cfg_path))
    cfg["video_paths"] = [os.path.join(root, "gone.mp4")]          # a missing video
    json.dump(cfg, open(cfg_path, "w"))

    def no_dialog(*a, **k):
        raise AssertionError("a dialog was raised on a command-line open")
    monkeypatch.setattr(QMessageBox, "question", no_dialog)
    monkeypatch.setattr(QMessageBox, "warning", no_dialog)
    w = window()
    w._load_project(cfg_path)
    assert w._project_dir == proj            # the file's folder, not the stale stored path
    assert len(w._extracted_frames) == 3

    shown, missing = w.focus_frames([FRAMES[2], "CAM99_z_frame_000009"])
    assert shown == 1 and missing == ["CAM99_z_frame_000009.png"]
    assert [os.path.basename(p) for _, _, p in w._extracted_frames] == [FRAMES[2]]
    assert "focus: 1 frames" in w.windowTitle()
    assert w.act_show_all_frames.isEnabled()

    w.show_all_frames()
    assert len(w._extracted_frames) == 3
    assert "focus" not in w.windowTitle()
    assert not w.act_show_all_frames.isEnabled()
    w.close()


def test_training_defaults_and_predict_flag():
    """The first 8x8 zone model trained with the backbone frozen, AdamW at 0.01
    and no mosaic; the defaults are now unfrozen, 0.002 and mosaic + copy-paste."""
    w = window()
    assert not w.chk_freeze_backbone.isChecked()
    w.combo_optimizer.setCurrentText("SGD")
    assert abs(w.spin_lr.value() - 0.01) < 1e-9
    w.combo_optimizer.setCurrentText("AdamW")
    assert abs(w.spin_lr.value() - 0.002) < 1e-9
    assert w._mosaic_override is None and w._copy_paste_override is None
    args = G._launch_parser().parse_args(["--project", "p", "--predict", "m", "--mosaic", "0"])
    assert args.predict == "m"
    G.apply_training_args(w, args)
    assert w._mosaic_override == 0.0
    w.close()


def test_augment_turns_the_transforms_on():
    """--augment used to tick only the master checkbox; every transform under it
    stayed off, so the run trained with no augmentation."""
    w = window()
    args = G._launch_parser().parse_args(["--project", "p", "--augment"])
    G.apply_training_args(w, args)
    assert w.chk_augment.isChecked()
    assert w.combo_aug_rotation.currentText() == "±180°"
    assert w.chk_aug_scale.isChecked() and w.chk_aug_brightness.isChecked()
    assert w.chk_aug_contrast.isChecked()
    args = G._launch_parser().parse_args(["--project", "p", "--augment", "--aug-rotation", "15"])
    G.apply_training_args(w, args)
    assert w.combo_aug_rotation.currentText() == "±15°"
    w.close()


def test_training_flags_fill_the_panel_and_leave_the_rest(monkeypatch):
    w = window()
    lr_before = w.spin_lr.value()
    patience_before = w.spin_patience.value()
    args = G._launch_parser().parse_args([
        "--project", "p", "--train", "--arch", "yolo11s-seg", "--epochs", "50",
        "--batch", "4", "--imgsz", "1280", "--device", "cuda", "--no-augment",
        "--post-infer", "--post-infer-conf", "0.4"])
    applied = G.apply_training_args(w, args)
    assert w.combo_architecture.currentText() == "YOLOv11-small-seg"
    assert w.combo_device.currentText() == "CUDA (NVIDIA)"
    assert w.spin_iterations.value() == 50 and w.spin_batch.value() == 4
    assert not w.chk_augment.isChecked() and w.chk_post_infer.isChecked()
    assert abs(w.spin_post_infer_conf.value() - 0.4) < 1e-9
    assert w._imgsz_override == 1280
    assert w.spin_lr.value() == lr_before and w.spin_patience.value() == patience_before
    assert "imgsz=1280" in applied

    # The run is handed the override; without one it keeps 640.
    captured = []

    class FakeWorker:
        class _Sig:
            def connect(self, *_):
                pass

        def __init__(self, cfg):
            captured.append(cfg)
            self.progress = self.finished = self.error = self._Sig()

        def start(self):
            pass
    monkeypatch.setattr(G, "TrainWorker", FakeWorker)
    w._project_dir = tempfile.mkdtemp(prefix="mtt_train_")
    w._launch_training("ann.json", "frames", 3)
    assert captured[-1]["imgsz"] == 1280 and captured[-1]["model_variant"] == "yolo11s-seg"
    w._imgsz_override = None
    w._launch_training("ann.json", "frames", 3)
    assert captured[-1]["imgsz"] == 640
    w.close()


def test_train_and_focus_need_a_project():
    with pytest.raises(SystemExit):
        G.main(["--train"])
    with pytest.raises(SystemExit):
        G.main(["--focus", "a.png"])
    with pytest.raises(SystemExit):
        G.main(["--project", os.path.join(tempfile.gettempdir(), "no_such_mtt_project")])


def test_training_window_only_needs_train():
    with pytest.raises(SystemExit):
        G.main(["--project", "x", "--training-window-only"])


def test_the_training_window_is_its_own_window():
    """Owned by the main window it stayed stacked above it: the main window
    could not be brought to the front during a run."""
    w = window()
    assert w._seg_train_window.parent() is None
    w.close()


def test_window_only_closing_quits_once_no_run_is_going(monkeypatch):
    """--training-window-only: the training window is all of Mask Tracker, so
    closing it quits; mid-run it asks first, and Cancel keeps it open."""
    from PyQt5.QtGui import QCloseEvent
    quit_calls = []
    monkeypatch.setattr(G.QTimer, "singleShot", lambda ms, fn: quit_calls.append(fn))
    w = window()
    w._window_only = True
    ev = QCloseEvent()
    w._seg_train_window.closeEvent(ev)                       # nothing running
    assert ev.isAccepted() and len(quit_calls) == 1

    class Busy:
        def isRunning(self):
            return True
    w._train_worker = Busy()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Cancel)
    ev = QCloseEvent()
    w._seg_train_window.closeEvent(ev)
    assert not ev.isAccepted() and len(quit_calls) == 1      # still open, no quit
    del w._train_worker
    w._window_only = False
    w.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
