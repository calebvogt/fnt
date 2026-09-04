"""The model panel is two sections, each owning one action.

It began as a single QGroupBox titled "Model Training & Inference" holding ~40
widgets, and the confusion had two distinct causes:

* **Asymmetry.** The training config was collapsed behind a checkbox while
  ~15 inference controls sat permanently open, so a box promising both opened
  showing almost nothing but inference. Both halves now fold away.
* **The coupling ran between boxes.** To train-and-then-infer you ticked
  "Train a new model" in one place and a *scope checkbox somewhere else*, and a
  shared button silently relabelled itself "Run Training + Inference". You had
  to read two controls in two sections to know what one button would do.

So: **Run Training** owns the training config and states its own follow-on
action in a dropdown (Nothing / Current file / All files / Select files…, the
SLEAP shape). **Run Inference** owns the trained-model picker — training never
reads that dropdown, it writes into it, so a neutral "Model" box above training
implied a relationship that does not exist. Each section has one button whose
label states one action.

Runs under pytest, or directly.
"""
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication, QGroupBox
from scipy.io import wavfile

import fnt.usv.mad_pyqt as M
from fnt.usv.usv_detector.mad_project import MADProjectConfig, create_mad_project

SR = 250_000
_GUI = None


def gui():
    """One window with three recordings, so scope choices have something to
    count."""
    global _GUI
    if _GUI is None:
        root = tempfile.mkdtemp(prefix="mad_sect_")
        wavs = []
        for i in range(3):
            p = os.path.join(root, f"r{i}.wav")
            wavfile.write(p, SR, (np.random.default_rng(i).normal(0, .05, SR)
                                  * 32767).astype(np.int16))
            wavs.append(p)
        proj = os.path.join(root, "p")
        create_mad_project(proj)
        app = QApplication.instance() or QApplication([])
        M.MADMainWindow._apply_dark_theme()
        w = M.MADMainWindow()
        w._activate_project(MADProjectConfig.load(proj))
        w._register_audio_files(wavs)
        w._append_audio_paths(wavs)
        w._wait_for_audio_load()
        _GUI = (w, wavs, app)
    return _GUI[0], _GUI[1]


def _titles(w):
    return [g.title() for g in w.findChildren(QGroupBox) if g.title()]


def _box(w, title):
    return [g for g in w.findChildren(QGroupBox) if g.title() == title][0]


def _set_after(w, index):
    """Change the post-training scope without firing the file-picker."""
    w.combo_train_after.blockSignals(True)
    w.combo_train_after.setCurrentIndex(index)
    w.combo_train_after.blockSignals(False)


# ----------------------------------------------------------------------
# Structure
# ----------------------------------------------------------------------
def test_the_one_combined_box_is_gone():
    t = _titles(gui()[0])
    assert not any("Training" in x and "Inference" in x for x in t), t


def test_two_sections_named_for_their_action():
    w, _ = gui()
    t = _titles(w)
    assert "Run Training" in t and "Run Inference" in t, t
    assert "Model" not in t, "the neutral Model box should be dissolved"


def test_training_comes_before_inference():
    t = _titles(gui()[0])
    assert t.index("Run Training") < t.index("Run Inference"), t


def test_the_model_picker_belongs_to_inference():
    """Every reference to combo_deploy_model either populates it or reads it to
    run inference. Training writes into it; it is not a training input."""
    w, _ = gui()
    infer = _box(w, "Run Inference")
    for name in ("combo_deploy_model", "btn_deploy_refresh",
                 "btn_deploy_load_project", "btn_quick_infer",
                 "btn_eval_model", "lbl_deploy_model_info"):
        assert infer.isAncestorOf(getattr(w, name)), name


# ----------------------------------------------------------------------
# One button per section
# ----------------------------------------------------------------------
def test_each_section_owns_one_button():
    w, _ = gui()
    assert _box(w, "Run Training").isAncestorOf(w.btn_train_run)
    assert _box(w, "Run Inference").isAncestorOf(w.btn_infer_run)


def test_the_inference_button_states_one_action():
    w, _ = gui()
    w._update_run_button()
    assert w.btn_infer_run.text() == "Run Inference", w.btn_infer_run.text()


def test_the_training_button_names_its_follow_on():
    w, _ = gui()
    _set_after(w, 0)
    w._update_run_button()
    plain = w.btn_train_run.text()
    assert plain.startswith("Run Training (") and "Inference" not in plain, plain

    _set_after(w, 2)                       # all files
    w._update_run_button()
    assert "+ Inference on 3" in w.btn_train_run.text(), w.btn_train_run.text()


def test_the_label_count_stays_on_the_training_button():
    """It is how you know when retraining is worth it, and it reaches you
    while your eyes are on the spectrogram."""
    w, _ = gui()
    _set_after(w, 0)
    w._update_run_button()
    assert "label" in w.btn_train_run.text()


# ----------------------------------------------------------------------
# The post-training scope, and its independence from the inference scope
# ----------------------------------------------------------------------
def test_the_post_training_choices_match_the_sleap_shape():
    w, _ = gui()
    items = [w.combo_train_after.itemText(i)
             for i in range(w.combo_train_after.count())]
    assert len(items) == 4, items
    assert items[0] == "Nothing"
    assert "Current file" in items[1]
    assert "All files" in items[2]
    assert "Select" in items[3]


def test_each_choice_resolves_to_the_right_recordings():
    w, wavs = gui()
    _set_after(w, 0)
    assert w._post_training_targets() == []
    _set_after(w, 1)
    assert len(w._post_training_targets()) == 1
    _set_after(w, 2)
    assert len(w._post_training_targets()) == len(wavs)


def test_select_files_uses_the_stored_selection():
    w, wavs = gui()
    w._post_train_selection = [wavs[0], wavs[2]]
    _set_after(w, 3)
    assert w._post_training_targets() == [wavs[0], wavs[2]]


def test_the_inference_scope_no_longer_drives_training():
    """THE bug this split exists to fix: ticking a checkbox in the inference
    box used to decide what training did afterwards."""
    w, _ = gui()
    w.chk_scope_audio.setChecked(True)
    _set_after(w, 0)
    assert w._post_training_targets() == [], \
        "the inference scope still leaks into post-training"


# ----------------------------------------------------------------------
# Symmetry
# ----------------------------------------------------------------------
def test_each_section_folds_entirely_behind_one_toggle():
    """One checkbox per section, and it folds EVERYTHING under it — the run
    button included — so an unopened section costs one line of column."""
    w, _ = gui()
    w.chk_train_settings.setChecked(False)
    w.chk_infer_settings.setChecked(False)
    QApplication.processEvents()
    assert w._train_body.isHidden()
    assert w._infer_body.isHidden()


def test_the_run_buttons_fold_away_too():
    w, _ = gui()
    w.chk_train_settings.setChecked(False)
    w.chk_infer_settings.setChecked(False)
    QApplication.processEvents()
    for widget in (w.btn_train_run, w.combo_train_after,
                   w.btn_infer_run, w.combo_deploy_model,
                   w.spin_infer_threshold):
        assert not widget.isVisibleTo(w), widget


def test_one_toggle_opens_the_whole_section():
    w, _ = gui()
    w.chk_train_settings.setChecked(True)
    QApplication.processEvents()
    for widget in (w.combo_arch, w.combo_train_after, w.btn_train_run):
        assert widget.isVisibleTo(w), widget
    assert not w.btn_infer_run.isVisibleTo(w), "the other section opened too"

    w.chk_infer_settings.setChecked(True)
    QApplication.processEvents()
    for widget in (w.combo_deploy_model, w.chk_scope_audio,
                   w.spin_infer_threshold, w.chk_infer_redetect,
                   w.btn_infer_run):
        assert widget.isVisibleTo(w), widget
    w.chk_train_settings.setChecked(False)
    w.chk_infer_settings.setChecked(False)
    QApplication.processEvents()


def test_there_is_no_nested_second_toggle():
    """One fold per section — the old inner "Detection settings" checkbox is
    subsumed rather than nested inside the new one."""
    w, _ = gui()
    assert not hasattr(w, "_infer_config_widget")


def test_starting_a_run_reveals_its_section():
    """Otherwise a job's progress hides behind the toggle that started it."""
    w, _ = gui()
    w.chk_infer_settings.setChecked(False)
    QApplication.processEvents()
    w._reveal_section('infer')
    QApplication.processEvents()
    assert w.chk_infer_settings.isChecked()
    assert w.btn_infer_run.isVisibleTo(w)
    w.chk_infer_settings.setChecked(False)
    QApplication.processEvents()


# ----------------------------------------------------------------------
# Nothing lost
# ----------------------------------------------------------------------
WIDGETS = [
    "combo_deploy_model", "btn_deploy_refresh", "btn_deploy_load_project",
    "btn_quick_infer", "btn_eval_model", "lbl_deploy_model_info",
    "combo_arch", "combo_train_encoder", "spin_train_epochs",
    "spin_train_patience", "spin_train_batch", "spin_train_lr",
    "spin_train_val", "combo_train_loss", "chk_train_augment",
    "combo_train_device", "chk_scope_audio", "combo_scope_audio",
    "chk_scope_folder", "btn_pick_infer_folder", "lbl_infer_folder",
    "spin_infer_threshold", "spin_infer_min_blob", "combo_infer_device",
    "spin_infer_batch", "chk_infer_amp", "chk_infer_redetect",
    "btn_infer_run", "btn_train_run", "btn_infer_pause", "infer_panel",
    "combo_train_after",
]


def test_every_control_survived():
    w, _ = gui()
    assert [n for n in WIDGETS if not hasattr(w, n)] == []


def test_the_evaluate_button_is_in_a_layout():
    """It was added to a row that had already been handed to the layout — it
    rendered anyway, but read as a bug to anyone tracing the layout."""
    w, _ = gui()
    assert w.btn_eval_model.parent() is not None
    w.chk_infer_settings.setChecked(True)      # it lives inside the fold now
    QApplication.processEvents()
    assert w.btn_eval_model.isVisibleTo(w)
    w.chk_infer_settings.setChecked(False)
    QApplication.processEvents()


def test_the_dead_training_dialog_is_gone():
    """RunTrainingDialog had no call sites and its build_config omitted
    training_data_dir — the only field train_unet reads — so wiring it up
    would have trained on nothing."""
    assert not hasattr(M, "RunTrainingDialog")


def test_the_window_still_paints():
    w, _ = gui()
    w.resize(1400, 900)
    assert w.grab().toImage().width() > 0


if __name__ == "__main__":
    import sys
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print("  OK   " + name, flush=True)
        except Exception:
            fails += 1
            print("  FAIL " + name, flush=True)
            traceback.print_exc()
    print("")
    print("ALL OK" if not fails else str(fails) + " FAILURE(S)", flush=True)
    sys.stdout.flush()
    os._exit(1 if fails else 0)
