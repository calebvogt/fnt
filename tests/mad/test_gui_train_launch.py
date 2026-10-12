"""A training run started from outside the GUI, shown in the GUI.

``python -m fnt.usv.mad_pyqt --project P --train --split-mode file`` (or an
assistant asked to "train a new model") must:

* put every setting that has a control into the panel, so the window shows
  what the run trains with;
* carry the settings without a control (LR schedule, faint augmentation,
  per-recording cap) into this run's config;
* NOT run post-training inference unless asked -- a dictated "train a new
  model" must not re-detect every recording in the project, and the overwrite
  confirmation it would raise would sit unanswered on an unattended machine;
* leave the next press of Run Training using the panel alone.

Runs under pytest, or directly.
"""
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

import fnt.usv.mad_pyqt as M  # noqa: E402
from fnt.usv.usv_detector.mad_project import (  # noqa: E402
    MADProjectConfig, create_mad_project)

_GUI = None


def gui():
    global _GUI
    if _GUI is None:
        root = tempfile.mkdtemp(prefix="mad_launch_")
        proj = os.path.join(root, "p")
        create_mad_project(proj)
        app = QApplication.instance() or QApplication([])
        M.MADMainWindow._apply_dark_theme()
        w = M.MADMainWindow()
        w._activate_project(MADProjectConfig.load(proj))
        for _ in range(20):
            app.processEvents()
        _GUI = (w, app)
    return _GUI[0]


class Captured:
    def __init__(self):
        self.cfg = None
        self.post = None
        self.confirm_asked = False


@pytest.fixture
def win(monkeypatch):
    """The window with every expensive or blocking step of a run stubbed, so
    _on_inline_train runs for real up to the hand-off to the worker."""
    import fnt.usv.usv_detector.mad_examples as ex
    w = gui()
    cap = Captured()
    monkeypatch.setattr(w, "_consolidate_sibling_examples", lambda: True)
    monkeypatch.setattr(ex, "count_examples", lambda *a, **k: 5)
    monkeypatch.setattr(w, "_show_gpu_setup_dialog", lambda **k: None)
    monkeypatch.setattr(w, "_post_training_targets", lambda: ["x.wav"])

    def confirm(_):
        cap.confirm_asked = True
        return True
    monkeypatch.setattr(w, "_confirm_overwrite_predictions", confirm)
    monkeypatch.setattr(w, "_show_training_dialog", lambda: None)
    monkeypatch.setattr(w, "_show_preview_dialog", lambda **k: None)
    monkeypatch.setattr(w.train_panel, "start_run", lambda: None)

    def start(cfg, post_inference_wavs=None, reporter=None):
        cap.cfg = cfg
        cap.post = list(post_inference_wavs or [])
    monkeypatch.setattr(w, "_start_training", start)
    w._training_active = False
    w._cap = cap
    yield w
    w._training_active = False
    w._update_run_button()
    w._set_train_config_enabled(True)


def test_panel_settings_are_shown_in_the_panel(win):
    win.launch_training({"split_mode": "file", "n_epochs": 321,
                         "early_stop_patience": 17})
    assert win.combo_train_split.currentData() == "file"
    assert win.spin_train_epochs.value() == 321
    assert win.spin_train_patience.value() == 17
    assert win._cap.cfg.split_mode == "file"
    assert win._cap.cfg.n_epochs == 321


def test_settings_without_a_control_reach_the_run(win):
    win.launch_training({"max_examples_per_recording": 77,
                         "augment_faint": False, "lr_schedule": "none"})
    cfg = win._cap.cfg
    assert cfg.max_examples_per_recording == 77
    assert cfg.augment_faint is False
    assert cfg.lr_schedule == "none"


def test_a_launched_run_does_not_infer_unless_asked(win):
    win.launch_training({})
    assert win._cap.post == []
    assert not win._cap.confirm_asked, "an unattended run must not block"


def test_infer_after_runs_inference_without_a_blocking_prompt(win):
    win.launch_training({}, infer_after=True)
    assert win._cap.post == ["x.wav"]
    assert not win._cap.confirm_asked, "the flag is the confirmation"


def test_the_next_button_press_uses_the_panel_alone(win):
    win.launch_training({"max_examples_per_recording": 77})
    win._training_active = False
    win._on_train_clicked()
    default = M.UNetTrainingConfig if hasattr(M, "UNetTrainingConfig") else \
        __import__("fnt.usv.usv_detector.mad_training",
                   fromlist=["x"]).UNetTrainingConfig
    assert win._cap.cfg.max_examples_per_recording == \
        default(project_dir="x").max_examples_per_recording
    assert win._cap.confirm_asked, "a normal press still confirms overwrite"


def test_a_value_the_panel_cannot_hold_still_reaches_the_run(win):
    """An encoder the dropdown does not list must not be silently dropped."""
    win.launch_training({"encoder_name": "resnet34_not_in_list"})
    assert win._cap.cfg.encoder_name == "resnet34_not_in_list"


def test_the_launch_parser_needs_a_project_to_train():
    with pytest.raises(SystemExit):
        M.main(["--train"])


def test_post_inference_settings_can_be_dictated():
    """A launched run inherited whatever the panel last held -- min blob 100
    and merge off, against the 50 / merge-on runs it was meant to repeat."""
    a = M._launch_parser().parse_args(
        ["--project", "p", "--train", "--infer-after", "--infer-threshold",
         "0.45", "--infer-min-blob", "50", "--infer-merge",
         "--infer-merge-gap", "0.02", "--no-infer-harmonics"])
    assert (a.infer_threshold, a.infer_min_blob, a.infer_merge,
            a.infer_merge_gap, a.infer_harmonics) == (0.45, 50, True, 0.02,
                                                      False)
    w = gui()
    w.set_post_inference_settings(
        threshold=a.infer_threshold, min_blob=a.infer_min_blob,
        merge=a.infer_merge, merge_gap=a.infer_merge_gap,
        harmonics=a.infer_harmonics)
    assert abs(w.spin_post_threshold.value() - 0.45) < 1e-9
    assert w.spin_post_min_blob.value() == 50
    assert w.chk_post_merge.isChecked()
    assert abs(w.spin_post_merge_gap.value() - 0.02) < 1e-9
    assert not w.chk_post_harmonics.isChecked()
    # The Run Inference copies are linked: one setting, two widgets.
    assert w.spin_infer_min_blob.value() == 50
    assert w.chk_infer_merge.isChecked()


def test_undictated_post_inference_settings_keep_the_panel():
    a = M._launch_parser().parse_args(["--project", "p", "--train"])
    assert (a.infer_threshold, a.infer_min_blob, a.infer_merge,
            a.infer_merge_gap, a.infer_harmonics) == (None,) * 5
    w = gui()
    w.spin_post_min_blob.setValue(123)
    w.chk_post_merge.setChecked(False)
    w.set_post_inference_settings()
    assert w.spin_post_min_blob.value() == 123
    assert not w.chk_post_merge.isChecked()



# ---------------------------------------------------------------------------
# --training-window-only: a dictated run shows the training window and nothing
# else ("what I wanted was just the segmentation training window").
# ---------------------------------------------------------------------------
def test_window_only_needs_train_and_refuses_hidden_inference():
    for argv in (["--project", "p", "--training-window-only"],
                 ["--project", "p", "--train", "--training-window-only",
                  "--infer-after"]):
        with pytest.raises(SystemExit):
            M.main(argv)


def test_window_only_keeps_the_main_window_hidden():
    w = gui()
    app = QApplication.instance()
    try:
        M.apply_launch_window_mode(w, app, True)
        assert not w.isVisible() and w._window_only
        assert not app.quitOnLastWindowClosed()
    finally:
        app.setQuitOnLastWindowClosed(True)
        w._window_only = False


class _Quit:
    def __init__(self, monkeypatch):
        self.n = 0
        monkeypatch.setattr(QApplication.instance(), "quit", self)

    def __call__(self):
        self.n += 1


def _close(w):
    from PyQt5.QtGui import QCloseEvent
    ev = QCloseEvent()
    w._on_train_dialog_close(ev)
    for _ in range(3):
        QApplication.instance().processEvents()
    return ev


def test_closing_the_only_window_after_the_run_quits(monkeypatch):
    w = gui()
    quit_ = _Quit(monkeypatch)
    monkeypatch.setattr(w, "_window_only", True, raising=False)
    w._training_active = False
    assert _close(w).isAccepted()
    assert quit_.n == 1


def test_closing_mid_run_offers_stop_or_cancel_not_keep_running(monkeypatch):
    w = gui()
    quit_ = _Quit(monkeypatch)
    monkeypatch.setattr(w, "_window_only", True, raising=False)
    w._training_active = True
    stopped = []
    monkeypatch.setattr(w.train_panel, "_on_stop", lambda: stopped.append(1))
    try:
        monkeypatch.setattr(w, "_window_only_close_prompt", lambda: "cancel")
        assert not _close(w).isAccepted()
        assert not stopped and quit_.n == 0

        monkeypatch.setattr(w, "_window_only_close_prompt", lambda: "stop")
        assert _close(w).isAccepted()
        assert stopped and w._quit_when_idle
        assert quit_.n == 0, "must not quit before the run winds down"

        w._after_training_run()                 # the worker reports it is done
        QApplication.instance().processEvents()
        assert quit_.n == 1
    finally:
        w._training_active = False
        w._quit_when_idle = False


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
