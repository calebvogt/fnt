"""Pruning and emptying are two different actions, in two different places.

They used to be one "Clear All" button opening one dialog that offered both,
which meant the everyday action (drop the silent recordings) sat one mis-click
from the irreversible one (drop everything) and the safe answer depended on
reading two similar sentences. On a partially-scanned list it cleared 443
recordings down to 6.

Now:

* "Remove Empty…" in the Audio panel  -- can ONLY remove recordings holding
  nothing, refuses to run before the scan has finished, and has no destructive
  sibling in its dialog.
* File -> Empty the Audio List…       -- the once-a-project action, behind a
  menu, defaulting to Cancel.

Both unregister only; File -> Restore Cleared Recordings… puts them back.

Runs under pytest, or directly.
"""
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication, QMessageBox
from scipy.io import wavfile

import fnt.usv.mad_pyqt as M
from fnt.usv.usv_detector.mad_project import MADProjectConfig, create_mad_project

SR = 250_000
_GUI = None


def gui():
    global _GUI
    if _GUI is None:
        root = tempfile.mkdtemp(prefix="mad_split_")
        wavs = []
        for i in range(5):
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
        for _ in range(30):
            app.processEvents()
        _GUI = (w, wavs, app, root)
    return _GUI[0], _GUI[1]


class Spy:
    """Answer every dialog with the button carrying ``role``, and record what
    was asked. Nothing is allowed to block."""

    def __init__(self, role=QMessageBox.AcceptRole):
        self.role = role
        self.texts = []
        self.removed = []
        self.info = []

    def __enter__(self):
        w, _ = gui()
        self._exec = QMessageBox.exec_
        self._info = QMessageBox.information
        self._rm = M.MADMainWindow._remove_files_by_path
        spy = self

        def fake_exec(box):
            spy.texts.append(box.text() + " | " + box.informativeText())
            for b in box.buttons():
                if box.buttonRole(b) == spy.role:
                    box._picked = b
                    return 0
            box._picked = None
            return 0

        QMessageBox.exec_ = fake_exec
        QMessageBox.clickedButton = lambda box: getattr(box, '_picked', None)
        QMessageBox.information = staticmethod(
            lambda *a, **k: spy.info.append(a[2] if len(a) > 2 else ""))
        M.MADMainWindow._remove_files_by_path = (
            lambda self, paths, delete_embedded=False: spy.removed.extend(paths))
        return spy

    def __exit__(self, *a):
        QMessageBox.exec_ = self._exec
        QMessageBox.information = self._info
        del QMessageBox.clickedButton
        M.MADMainWindow._remove_files_by_path = self._rm


def scanned(w, paths, counts=None):
    w._counts_scanned = {os.path.basename(p) for p in paths}
    w._file_count_cache = dict(counts or {})
    w._counts_complete = True


# ------------------------------------------------------- Remove Empty…
def test_remove_empty_drops_only_the_empty_ones():
    w, wavs = gui()
    scanned(w, wavs, {os.path.basename(wavs[0]): (4, 0, 0),
                      os.path.basename(wavs[1]): (0, 0, 7)})
    with Spy() as spy:
        w._remove_empty_files()
    assert sorted(spy.removed) == sorted(wavs[2:])


def test_remove_empty_has_no_destructive_button_at_all():
    """The invariant the split buys: whatever gets clicked in this dialog,
    a reviewed recording cannot leave the list."""
    w, wavs = gui()
    scanned(w, wavs, {os.path.basename(wavs[0]): (4, 0, 0)})
    with Spy(role=QMessageBox.DestructiveRole) as spy:
        w._remove_empty_files()
    assert spy.removed == [], "a destructive option was reachable from here"


def test_remove_empty_refuses_while_the_scan_is_running():
    w, wavs = gui()
    scanned(w, wavs)
    w._counts_complete = False
    with Spy() as spy:
        w._remove_empty_files()
    assert spy.removed == []
    assert spy.info and "still checking" in spy.info[0].lower()


def test_remove_empty_says_so_when_there_is_nothing_to_do():
    w, wavs = gui()
    scanned(w, wavs, {os.path.basename(p): (1, 0, 0) for p in wavs})
    with Spy() as spy:
        w._remove_empty_files()
    assert spy.removed == []
    assert spy.info


def test_remove_empty_cancels_cleanly():
    w, wavs = gui()
    scanned(w, wavs)
    with Spy(role=QMessageBox.RejectRole) as spy:
        w._remove_empty_files()
    assert spy.removed == []


# ------------------------------------------------- File -> Empty the list
def test_emptying_takes_everything():
    w, wavs = gui()
    scanned(w, wavs, {os.path.basename(wavs[0]): (4, 0, 0)})
    with Spy(role=QMessageBox.DestructiveRole) as spy:
        w._empty_audio_list()
    assert sorted(spy.removed) == sorted(wavs)


def test_emptying_names_the_review_it_would_cost():
    """The one number that makes someone reconsider."""
    w, wavs = gui()
    scanned(w, wavs, {os.path.basename(wavs[0]): (4, 0, 0),
                      os.path.basename(wavs[1]): (0, 0, 9)})
    with Spy(role=QMessageBox.RejectRole) as spy:
        w._empty_audio_list()
    assert spy.texts and "2 of them carry accepted or rejected" in spy.texts[0]


def test_emptying_admits_when_it_cannot_count_the_review_yet():
    w, wavs = gui()
    scanned(w, wavs)
    w._counts_complete = False
    with Spy(role=QMessageBox.RejectRole) as spy:
        w._empty_audio_list()
    assert spy.texts and "still checking" in spy.texts[0].lower()


def test_emptying_defaults_to_cancel():
    """Enter must never be the destructive answer — the reflex that dismisses
    a dialog used to empty the list."""
    w, wavs = gui()
    scanned(w, wavs)
    seen = {}
    real = QMessageBox.exec_

    def fake(box):
        seen['role'] = box.buttonRole(box.defaultButton())
        box._picked = None
        return 0
    QMessageBox.exec_ = fake
    QMessageBox.clickedButton = lambda box: getattr(box, '_picked', None)
    try:
        w._empty_audio_list()
    finally:
        QMessageBox.exec_ = real
        del QMessageBox.clickedButton
    assert seen['role'] == QMessageBox.RejectRole


# --------------------------------------------------------------- placement
def test_the_audio_panel_button_is_the_pruning_one():
    w, _ = gui()
    assert "Empty" in w.btn_remove_empty.text()
    assert not hasattr(w, 'btn_clear_files'), "the combined button is gone"


def test_emptying_the_list_is_reachable_only_from_the_menu():
    w, _ = gui()
    act = w.act_empty_list
    assert act.shortcut().isEmpty(), "no shortcut on a once-a-project action"
    assert "Empty" in act.text()


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
