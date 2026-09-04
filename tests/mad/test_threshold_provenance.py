"""Where the probability threshold lives, and who reads it.

It is a *detection* parameter, not a training one, so it sits in Run Inference.
But the Run Training section can chain a run onto the end of training, and that
run borrows Run Inference's detection settings — making the threshold a hidden
input to a button in the other section. The training section says so, and
quotes the current value.

Also pins that neither live run writes a CSV: the .mad store is the record of a
run, and the CSV is an export the user asks for.

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
# It is an inference control
# ----------------------------------------------------------------------
def test_the_threshold_is_an_inference_control():
    w, _ = gui()
    assert _box(w, "Run Inference").isAncestorOf(w.spin_infer_threshold)
    assert not _box(w, "Run Training").isAncestorOf(w.spin_infer_threshold)


def test_it_is_reachable_by_opening_one_toggle():
    """Folded by default, so it has to be one tick away — not two."""
    w, _ = gui()
    w.chk_infer_settings.setChecked(False)
    QApplication.processEvents()
    assert not w.spin_infer_threshold.isVisibleTo(w)
    w.chk_infer_settings.setChecked(True)
    QApplication.processEvents()
    assert w.spin_infer_threshold.isVisibleTo(w)
    w.chk_infer_settings.setChecked(False)
    QApplication.processEvents()


def test_its_default_is_a_half():
    assert gui()[0].spin_infer_threshold.value() == 0.5


# ----------------------------------------------------------------------
# The training section quotes it, because the chained run uses it
# ----------------------------------------------------------------------
def test_the_chained_run_reads_the_inference_threshold():
    src = inspect.getsource(M.MADMainWindow._run_post_training_inference)
    assert "self.spin_infer_threshold.value()" in src


def test_the_training_section_names_the_threshold_it_will_use():
    w, _ = gui()
    _set_after(w, 0)
    assert not w.lbl_train_after.isVisibleTo(w) or not w.lbl_train_after.text()

    w.spin_infer_threshold.setValue(0.30)
    _set_after(w, 2)                       # all files
    text = w.lbl_train_after.text()
    assert "0.30" in text, text
    assert "Run Inference" in text, text


def test_the_quote_follows_the_spinbox():
    """A stale number would be worse than none."""
    w, _ = gui()
    _set_after(w, 2)
    w.spin_infer_threshold.setValue(0.65)
    QApplication.processEvents()
    assert "0.65" in w.lbl_train_after.text(), w.lbl_train_after.text()
    w.spin_infer_threshold.setValue(0.5)


def test_no_threshold_is_quoted_when_nothing_will_run():
    w, _ = gui()
    _set_after(w, 0)
    assert not w.lbl_train_after.isVisibleTo(w)


def test_the_dropdown_tooltip_points_at_the_other_section():
    w, _ = gui()
    tip = w.combo_train_after.toolTip()
    assert "Run Inference" in tip and "threshold" in tip, tip


# ----------------------------------------------------------------------
# A run records itself in the .mad, not in a CSV
# ----------------------------------------------------------------------
def test_neither_live_run_writes_a_csv():
    """Both were left at save_blob_csv=True from before the store became the
    source of truth, so every run silently produced a CSV alongside it."""
    src = inspect.getsource(M)
    for fn in (M.MADMainWindow._run_post_training_inference,):
        assert "save_blob_csv=False" in inspect.getsource(fn), fn
    # and no run config anywhere still hard-codes True
    hard_true = re.findall(r"save_blob_csv=True", src)
    assert hard_true == [], hard_true


def test_the_csv_is_still_offered_deliberately():
    """Export did not go away — it moved to a menu item the user chooses,
    which is the whole point: a run records itself in the .mad, and the CSV is
    produced on request from that record."""
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
