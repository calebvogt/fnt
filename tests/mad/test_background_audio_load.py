"""Reading a recording happens off the UI thread, and stale reads are dropped.

Decoding a 10-minute 250 kHz recording is ~3 s the first time it is touched, and
doing it inline froze the whole window on every click in the Audio list -- the
list included, so you could not even change your mind while waiting.

Moving it to a worker introduces the failure that has to be tested: a read that
lands *after* the user has moved on. Without the request token, clicking A then
B would paint A's samples over B, and the annotations already loaded for B would
sit on the wrong recording. That is silent and wrong, which is worse than slow.

Runs under pytest, or directly.
"""
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtWidgets import QApplication
from scipy.io import wavfile

import fnt.usv.mad_pyqt as M
from fnt.usv.usv_detector.mad_project import MADProjectConfig, create_mad_project

SR = 250_000
_GUI = None


def gui():
    """Three recordings of distinguishable length, so which one is loaded is
    visible in total_duration alone."""
    global _GUI
    if _GUI is None:
        root = tempfile.mkdtemp(prefix="mad_bg_")
        paths = []
        for i, secs in enumerate((1.0, 2.0, 3.0)):
            p = os.path.join(root, f"rec{i}.wav")
            rng = np.random.default_rng(i)
            wavfile.write(p, SR, (rng.normal(0, .05, int(SR * secs))
                                  * 32767).astype(np.int16))
            paths.append(p)
        proj = os.path.join(root, "p")
        create_mad_project(proj)
        app = QApplication.instance() or QApplication([])
        M.MADMainWindow._apply_dark_theme()
        w = M.MADMainWindow()
        w._activate_project(MADProjectConfig.load(proj))
        w._register_audio_files(paths)
        w._append_audio_paths(paths)
        w._wait_for_audio_load()
        _GUI = (w, paths, app)
    return _GUI[:2]


def _select(w, row):
    w.current_file_idx = row
    w._load_current_file()


# ----------------------------------------------------------------------
def test_the_read_is_handed_off_rather_than_done_inline():
    """_load_current_file returns with the work still outstanding."""
    w, _paths = gui()
    w.audio_data = None
    _select(w, 2)
    assert w._audio_workers, "the read was done on the UI thread"
    assert w._loading_path is not None
    assert w._wait_for_audio_load()


def test_the_file_actually_opens():
    w, paths = gui()
    _select(w, 1)
    assert w._wait_for_audio_load()
    assert abs(w.spectrogram.total_duration - 2.0) < 0.05
    assert os.path.normcase(w._loaded_wav_path) == os.path.normcase(paths[1])
    assert w._loading_path is None


def test_a_superseded_read_never_lands():
    """Click A then B: A's samples must not arrive and overwrite B."""
    w, paths = gui()
    _select(w, 0)                 # 1 s recording, still in flight
    stale_token = w._audio_token
    _select(w, 2)                 # 3 s recording — supersedes it
    assert w._wait_for_audio_load()
    assert abs(w.spectrogram.total_duration - 3.0) < 0.05, \
        "the superseded read painted over the current file"
    assert os.path.normcase(w._loaded_wav_path) == os.path.normcase(paths[2])
    assert stale_token != w._audio_token


def test_a_late_stale_result_is_discarded_outright():
    """Deliver a stale result by hand — the guard, not the timing, is the test."""
    w, _paths = gui()
    _select(w, 1)
    assert w._wait_for_audio_load()
    before = w.spectrogram.total_duration
    w._on_audio_loaded(w._audio_token - 5,
                       np.zeros(SR * 9, dtype=np.float32), SR, "")
    assert w.spectrogram.total_duration == before


def test_rapid_switching_leaves_the_last_file_open():
    w, paths = gui()
    for row in (0, 1, 2, 0, 2, 1):
        _select(w, row)
    assert w._wait_for_audio_load()
    assert abs(w.spectrogram.total_duration - 2.0) < 0.05
    assert os.path.normcase(w._loaded_wav_path) == os.path.normcase(paths[1])


def test_no_worker_is_left_behind():
    w, _paths = gui()
    for row in (0, 2, 1):
        _select(w, row)
    assert w._wait_for_audio_load()
    assert w._audio_workers == {}, "a worker outlived its load"
    assert w._audio_then == {}, "a completion callback was never consumed"


def test_reselecting_the_row_being_loaded_does_not_restart_it():
    w, paths = gui()
    w.audio_data = None
    _select(w, 2)
    token = w._audio_token
    w._on_file_selected(2)        # same row, load still in flight
    assert w._audio_token == token, "the in-flight read was restarted"
    assert w._wait_for_audio_load()


def test_a_completion_callback_runs_after_the_file_is_open():
    """Callers with follow-up work (painting a file's detections) need the
    samples in place first."""
    w, paths = gui()
    seen = []
    _select(w, 0)
    assert w._wait_for_audio_load()
    w.current_file_idx = 2
    w._load_current_file(
        then=lambda: seen.append(w.spectrogram.total_duration))
    assert seen == [], "the callback ran before the read finished"
    assert w._wait_for_audio_load()
    assert len(seen) == 1
    assert abs(seen[0] - 3.0) < 0.05, "it ran before the samples were in place"


def test_a_superseded_callback_does_not_run():
    """Its recording was abandoned; running it would act on the wrong file."""
    w, _paths = gui()
    seen = []
    _select(w, 0)
    w.current_file_idx = 1
    w._load_current_file(then=lambda: seen.append('stale'))
    w.current_file_idx = 2
    w._load_current_file()
    assert w._wait_for_audio_load()
    assert seen == [], "a callback for an abandoned read fired"


# ----------------------------------------------------------------------
# What the user sees while a read is in flight
# ----------------------------------------------------------------------
def test_the_previous_file_stays_on_screen_until_the_new_one_is_ready():
    """Moving the read off the UI thread made the intermediate state visible.

    A window that blanks its detections for three seconds and then refills them
    looks broken in a way the old three-second freeze did not, so the old view
    has to survive until the replacement is in hand.
    """
    w, _paths = gui()
    _select(w, 1)
    assert w._wait_for_audio_load()
    w.spectrogram.annotations = [
        {'id': M._new_ann_id('p'), 'blob_id': None, 'category': 'USV',
         'status': 'accepted', 'score': 1.0, 'f0': 100, 'f1': 110,
         't0': 200, 't1': 230, 'mask': np.ones((10, 30), bool)}]
    w._refresh_annotation_list()
    before = len(w.spectrogram.annotations)

    _select(w, 2)                              # read now in flight
    assert w._audio_workers, "expected an outstanding read"
    assert len(w.spectrogram.annotations) == before, \
        "the previous file's detections were cleared before the new ones existed"

    assert w._wait_for_audio_load()


def test_the_count_badge_survives_a_load():
    """current_file_idx moves to the new file the moment it is clicked, while
    the annotations still describe the old one. Reconciling the row's counts
    from memory in that window read as "no calls here" and wiped the badge."""
    w, paths = gui()
    _select(w, 1)
    assert w._wait_for_audio_load()
    target = os.path.basename(paths[2])
    w._file_count_cache[target] = (913, 0, 0)

    _select(w, 2)                              # read in flight
    w._update_file_list_counts()               # what a UI refresh would do
    assert w._file_count_cache.get(target) == (913, 0, 0), \
        "the badge was cleared while the file was still loading"

    assert w._wait_for_audio_load()


def test_counts_reconcile_once_the_file_is_actually_open():
    """The guard must not be so strict that real edits stop showing."""
    w, paths = gui()
    _select(w, 2)
    assert w._wait_for_audio_load()
    w.spectrogram.annotations = [
        {'id': M._new_ann_id('p'), 'blob_id': None, 'category': 'USV',
         'status': 'accepted', 'score': 1.0, 'f0': 100, 'f1': 110,
         't0': 200, 't1': 230, 'mask': np.ones((10, 30), bool)}]
    w._update_file_list_counts()
    assert w._file_count_cache.get(os.path.basename(paths[2])) == (1, 0, 0)


def test_a_failed_read_is_reported_and_does_not_wedge_the_loader():
    w, paths = gui()
    bad = os.path.join(os.path.dirname(paths[0]), "broken.wav")
    with open(bad, 'wb') as f:
        f.write(b"this is not a wav file")
    w.audio_files.append(bad)
    warned = []
    orig = M.QMessageBox.warning
    M.QMessageBox.warning = staticmethod(
        lambda *a, **k: warned.append(a[-1] if a else ''))
    try:
        _select(w, len(w.audio_files) - 1)
        assert w._wait_for_audio_load()
        assert warned, "a broken recording failed silently"
    finally:
        M.QMessageBox.warning = orig
        w.audio_files.pop()
    _select(w, 1)                       # loader still works afterwards
    assert w._wait_for_audio_load()
    assert abs(w.spectrogram.total_duration - 2.0) < 0.05


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
