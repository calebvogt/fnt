"""Where the probability threshold lives, and who reads it.

It is a *detection* parameter, not a training one — training learns a
probability map, and the threshold is where you slice that map into calls, so
it can be changed and re-run without retraining.

The original design put it only in **Run Inference**, and made the Run Training
section quote it, because a post-training run borrowed those settings: the
number that decided how many calls came back was set in the other box. That is
now solved properly. Both sections carry the detection settings, and
``_link_infer_settings`` keeps the two copies showing one value, so the chained
run reads the controls the user was actually looking at when they pressed the
button.

This suite pins that contract, plus one thing that has to stay true either way:
a run records itself in the ``.mad`` and does not write a CSV. The CSV is an
export you ask for.

Runs under pytest, or directly.
"""
import inspect
import os
import re
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
    global _GUI
    if _GUI is None:
        root = tempfile.mkdtemp(prefix="mad_thr_")
        wavs = []
        for i in range(2):
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


def _box(w, title):
    return [g for g in w.findChildren(QGroupBox) if g.title() == title][0]


def _set_after(w, index):
    w.combo_train_after.blockSignals(True)
    w.combo_train_after.setCurrentIndex(index)
    w.combo_train_after.blockSignals(False)
    w._update_train_after_label()


# ----------------------------------------------------------------------
# Two copies of one setting
# ----------------------------------------------------------------------
def test_both_sections_carry_a_threshold():
    """The training box no longer borrows the inference box's number; a
    post-training run is configured where it is started."""
    w, _ = gui()
    assert _box(w, "Run Inference").isAncestorOf(w.spin_infer_threshold)
    assert _box(w, "Run Training").isAncestorOf(w.spin_post_threshold)


def test_the_two_copies_stay_in_step_both_ways():
    """One setting, two widgets. Letting them drift would mean setting a
    threshold in one box and having the other silently override it."""
    w, _ = gui()
    keep = w.spin_infer_threshold.value()
    try:
        w.spin_infer_threshold.setValue(0.42)
        assert abs(w.spin_post_threshold.value() - 0.42) < 1e-9
        w.spin_post_threshold.setValue(0.66)
        assert abs(w.spin_infer_threshold.value() - 0.66) < 1e-9
    finally:
        w.spin_infer_threshold.setValue(keep)


def test_the_link_does_not_loop():
    """Two widgets connected both ways need a re-entrancy guard or the first
    edit recurses until the stack gives out."""
    w, _ = gui()
    keep = w.spin_infer_threshold.value()
    try:
        for v in (0.2, 0.8, 0.35):
            w.spin_infer_threshold.setValue(v)
            assert abs(w.spin_post_threshold.value() - v) < 1e-9
    finally:
        w.spin_infer_threshold.setValue(keep)


def test_min_blob_is_linked_too():
    w, _ = gui()
    keep = w.spin_infer_min_blob.value()
    try:
        w.spin_infer_min_blob.setValue(77)
        assert w.spin_post_min_blob.value() == 77
    finally:
        w.spin_infer_min_blob.setValue(keep)


def test_the_widget_is_built_with_a_sane_default():
    """The live value is a saved preference (``mad/infer/threshold``), so it is
    whatever this machine last used — assert the constructed default and the
    range instead of a number that depends on who ran the tests."""
    src = inspect.getsource(M.MADMainWindow)
    assert "self.spin_infer_threshold.setValue(0.5)" in src
    w, _ = gui()
    assert w.spin_infer_threshold.minimum() >= 0.01
    assert w.spin_infer_threshold.maximum() <= 0.99
    assert 0.01 <= w.spin_infer_threshold.value() <= 0.99


# ----------------------------------------------------------------------
# The chained run reads the controls it is started from
# ----------------------------------------------------------------------
def test_the_chained_run_reads_the_training_sections_copy():
    src = inspect.getsource(M.MADMainWindow._run_post_training_inference)
    assert "self.spin_post_threshold.value()" in src
    assert "self.spin_post_min_blob.value()" in src


def test_the_manual_run_reads_the_inference_sections_copy():
    src = inspect.getsource(M.MADMainWindow._on_deploy_infer)
    assert "self.spin_infer_threshold.value()" in src


def test_the_post_training_settings_appear_only_when_something_will_run():
    """Detection settings under 'then run inference on: None' would be
    controls for a run that is not going to happen."""
    w, _ = gui()
    # The whole training section folds behind one checkbox, so isVisibleTo(w)
    # is False either way while it is shut. Open it and ask about this widget.
    w.chk_train_settings.setChecked(True)
    QApplication.processEvents()
    try:
        _set_after(w, 0)                   # None
        assert not w._post_infer_settings.isVisibleTo(w.chk_train_settings.parentWidget())
        _set_after(w, 2)                   # all files
        assert w._post_infer_settings.isVisibleTo(w.chk_train_settings.parentWidget())
    finally:
        _set_after(w, 0)
        w.chk_train_settings.setChecked(False)
        QApplication.processEvents()


def test_the_follow_on_line_warns_that_the_cut_is_destructive():
    """These read like view options and are not: the threshold decides what is
    written to disk at all, and lowering it later means re-running."""
    w, _ = gui()
    _set_after(w, 2)
    text = w.lbl_train_after.text()
    assert "will be analyzed" in text, text
    assert "cannot be lowered afterwards" in text, text
    _set_after(w, 0)


def test_no_recordings_matching_the_choice_is_said_plainly():
    w, _ = gui()
    w._post_train_selection = []
    _set_after(w, 3)                       # Select files… with nothing picked
    assert "No recordings match" in w.lbl_train_after.text()
    _set_after(w, 0)


# ----------------------------------------------------------------------
# A run records itself in the .mad, not in a CSV
# ----------------------------------------------------------------------
def test_neither_live_run_writes_a_csv():
    """Both were left at save_blob_csv=True from before the store became the
    source of truth, so every run silently produced a CSV alongside it."""
    for fn in (M.MADMainWindow._run_post_training_inference,
               M.MADMainWindow._on_deploy_infer):
        assert "save_blob_csv=False" in inspect.getsource(fn), fn
    hard_true = re.findall(r"save_blob_csv=True", inspect.getsource(M))
    assert hard_true == [], hard_true


def test_the_csv_is_still_offered_deliberately():
    """Export did not go away — it moved to a menu item you choose, which is
    the point: a run records itself in the .mad, and the CSV is produced on
    request from that record."""
    w, _ = gui()
    assert "CSV" in w.act_export_csv.text()
    assert w.act_export_csv.isEnabled()


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
