"""A recording queued for a running job cannot be reviewed until it lands.

Inference rewrites a recording's predictions wholesale. Reviewing one before
its turn means deciding on detections that are about to be replaced — and
accepting one mints a training example for a detection that will not exist.
Observed live: a file reviewed while the batch was still running showed 680
pending from the previous model while its store already held 1.

Locking is scoped to the run's own queue, not to "a run is happening":

* queued, not yet written  -> locked
* already finished/failed  -> free (reviewing finished files while the rest
  scan is the whole point of marking them done as they land)
* not in this run at all   -> free

Runs under pytest, or directly.
"""
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication
from scipy.io import wavfile

import fnt.usv.mad_pyqt as M
from fnt.usv.usv_detector.mad_project import MADProjectConfig, create_mad_project

SR = 250_000
_GUI = None


def gui():
    """One window, four recordings."""
    global _GUI
    if _GUI is None:
        root = tempfile.mkdtemp(prefix="mad_lock_")
        wavs = []
        for i in range(4):
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
        _GUI = (w, wavs, app)
    return _GUI[0], _GUI[1]


def lock(w, paths):
    w._infer_locked = {w._path_key(p) for p in paths}
    w._refresh_file_list_locks()


def row_enabled(w, i):
    return bool(w.file_list.item(i).flags() & Qt.ItemIsEnabled)


def teardown():
    w, _ = gui()
    w._unlock_all_infer_files()


# ----------------------------------------------------------------------
def test_nothing_is_locked_when_no_run_is_going():
    w, wavs = gui()
    teardown()
    assert not any(w._is_infer_locked(p) for p in wavs)
    assert all(row_enabled(w, i) for i in range(len(wavs)))


def test_queued_recordings_are_locked_and_disabled():
    w, wavs = gui()
    try:
        lock(w, wavs[1:])
        assert not w._is_infer_locked(wavs[0])
        assert all(w._is_infer_locked(p) for p in wavs[1:])
        assert row_enabled(w, 0)
        assert not any(row_enabled(w, i) for i in (1, 2, 3))
    finally:
        teardown()


def test_a_file_outside_the_run_is_never_locked():
    """Running on a subset must not lock the rest of the Audio list."""
    w, wavs = gui()
    try:
        lock(w, [wavs[2]])
        assert [w._is_infer_locked(p) for p in wavs] == [False, False, True, False]
        assert [row_enabled(w, i) for i in range(4)] == [True, True, False, True]
    finally:
        teardown()


def test_finishing_a_file_releases_it():
    w, wavs = gui()
    try:
        lock(w, wavs)
        w._set_file_item_state(wavs[1], 'done', 3)
        assert not w._is_infer_locked(wavs[1])
        assert w._is_infer_locked(wavs[2]), "the rest stay locked"
        assert row_enabled(w, 1)
    finally:
        teardown()


def test_a_failed_file_is_released_too():
    """Nothing more will be written to it, so holding it locked strands it."""
    w, wavs = gui()
    try:
        lock(w, wavs)
        w._set_file_item_state(wavs[3], 'error')
        assert not w._is_infer_locked(wavs[3])
    finally:
        teardown()


def test_selecting_a_locked_row_is_refused_and_explained():
    """Disabled rows are unreachable by mouse, but Prev/Next and the
    completion prompt set the row directly."""
    w, wavs = gui()
    try:
        w.current_file_idx = 0
        lock(w, wavs[1:])
        w._on_file_selected(2)
        assert w.current_file_idx == 0, "a locked file was opened"
        msg = w.status_bar.currentMessage()
        assert "queued" in msg and os.path.basename(wavs[2]) in msg, msg
    finally:
        teardown()


def test_selecting_a_released_row_works():
    w, wavs = gui()
    try:
        w.current_file_idx = 0
        lock(w, wavs[1:])
        w._set_file_item_state(wavs[1], 'done', 1)
        w._on_file_selected(1)
        assert w.current_file_idx == 1
    finally:
        teardown()


def test_the_end_of_a_run_releases_everything():
    w, wavs = gui()
    lock(w, wavs)
    w._unlock_all_infer_files()
    assert not any(w._is_infer_locked(p) for p in wavs)
    assert all(row_enabled(w, i) for i in range(len(wavs)))


def test_a_locked_row_says_why_on_hover():
    w, wavs = gui()
    try:
        lock(w, [wavs[1]])
        tip = w.file_list.item(1).toolTip()
        assert "Queued" in tip and "inference" in tip, tip
    finally:
        teardown()


def test_rebuilding_the_list_keeps_the_locks():
    """_refresh_file_list recreates every row, and a lock that survived only
    in the old QListWidgetItem would silently open the file."""
    w, wavs = gui()
    try:
        lock(w, wavs[2:])
        w._refresh_file_list()
        assert [row_enabled(w, i) for i in range(4)] == [True, True, False, False]
    finally:
        teardown()


def test_the_lock_key_is_case_and_form_insensitive():
    """Paths reach this from the run's list and from the Audio list, which do
    not always agree on case or separators on Windows."""
    w, wavs = gui()
    try:
        lock(w, [wavs[0]])
        assert w._is_infer_locked(wavs[0].upper())
        assert w._is_infer_locked(os.path.join(os.path.dirname(wavs[0]), ".",
                                               os.path.basename(wavs[0])))
    finally:
        teardown()


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
