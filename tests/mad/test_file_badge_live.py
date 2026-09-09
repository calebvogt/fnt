"""The Audio-list badge tracks review, not just the last inference run.

Each row carries ``(accepted, pending, rejected)``. Inference writes it as it
finishes each file — that is what lets you QC finished recordings while later
ones are still scanning — but it was then frozen: accept fifty calls and the
row still read what the run produced.

The bulk paths were fine; they call ``_refresh_annotation_list``, which ends in
``_update_file_list_counts()`` and reconciles the on-screen file from memory.
The gap was the FAST path. ``_after_review_decision`` restyles just the one row
that changed and skips the rebuild entirely when it can — which is every
ordinary A/R keystroke, the exact case where the badge matters.

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


def settle(n=50):
    """Run queued Qt work to completion."""
    for _ in range(n):
        QApplication.processEvents()


def gui():
    """One window, two recordings, with predictions on the first."""
    global _GUI
    if _GUI is None:
        root = tempfile.mkdtemp(prefix="mad_badge_")
        wavs = []
        for i in range(2):
            p = os.path.join(root, f"r{i}.wav")
            wavfile.write(p, SR, (np.random.default_rng(i).normal(0, .05, SR * 2)
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
        w.resize(1400, 900)
        w.show()
        app.processEvents()
        # Auto-advance is off for these tests. It is not what is under test,
        # and its selection change queues a spectrogram seek that segfaults the
        # offscreen platform when the event loop next runs — taking the process
        # down with no traceback, several tests later.
        w._auto_advance = False
        if hasattr(w, 'chk_auto_advance'):
            w.chk_auto_advance.setChecked(False)
        # Drain the asynchronous load ONCE, here. Its completion callback
        # replaces spectrogram.annotations wholesale, so it has to finish before
        # any test seeds them — and it must not be pumped later: a review
        # decision reselects a row, which queues a spectrogram seek that
        # segfaults the offscreen platform the moment the loop runs again.
        settle()
        _GUI = (w, wavs, app)
    return _GUI[0], _GUI[1]


def seed(w, n_pred=5):
    """Put pending predictions on the loaded file, as a run would."""
    sg = w.spectrogram
    sg.annotations = []
    for i in range(n_pred):
        # Grid indices, not seconds: annotations live on the spectrogram grid
        # (see _annotation_from_example).
        sg.annotations.append({
            'id': f'p{i}', 'status': 'prediction', 'category': 'USV',
            'score': 0.9, 'blob_id': i,
            't0': 200 + 100 * i, 't1': 260 + 100 * i,
            'f0': 120, 'f1': 200,
        })
    w._refresh_annotation_list()
    return sg


def badge(w, idx=None):
    """The (accepted, pending, rejected) tuple the row is displaying."""
    idx = w.current_file_idx if idx is None else idx
    base = os.path.basename(w.audio_files[idx])
    return w._file_count_cache.get(base)


# ----------------------------------------------------------------------
def test_the_badge_starts_from_what_inference_found():
    w, _ = gui()
    seed(w, 5)
    assert badge(w) == (0, 5, 0), badge(w)


def test_accepting_moves_a_call_out_of_pending_on_the_fast_path():
    """The keystroke case: _touch_annotation_rows succeeds, so the full list
    rebuild -- and with it the old badge update -- never runs."""
    w, _ = gui()
    seed(w, 5)
    assert w._touch_annotation_rows(['p0']), \
        "setup: this test is about the fast path"
    w._after_review_decision('p0', was_pending=True)
    # No event pumping here: auto-advance can queue a file switch, and running
    # it mid-test loads the next recording out from under the ones that follow.
    # _touch_current_file_badge is synchronous, so there is nothing to wait for.
    assert badge(w) == w._mem_status_counts()


def test_a_real_accept_shows_up_in_the_badge():
    w, _ = gui()
    sg = seed(w, 5)
    sg.annotations[0]['status'] = 'accepted'
    w._after_review_decision('p0', was_pending=True)
    assert badge(w) == (1, 4, 0), badge(w)


def test_a_real_reject_shows_up_in_the_badge():
    w, _ = gui()
    sg = seed(w, 5)
    sg.annotations[1]['status'] = 'rejected'
    w._after_review_decision('p1', was_pending=True)
    assert badge(w) == (0, 4, 1), badge(w)


def test_the_badge_follows_a_run_of_decisions():
    w, _ = gui()
    sg = seed(w, 6)
    for i, st in enumerate(['accepted', 'accepted', 'rejected', 'accepted']):
        sg.annotations[i]['status'] = st
        w._after_review_decision(f'p{i}', was_pending=True)
    assert badge(w) == (3, 2, 1), badge(w)


def test_bulk_paths_were_already_covered():
    """_refresh_annotation_list ends in _update_file_list_counts(), so
    box-select and Accept All never had the problem. Pinned so a future
    refactor cannot quietly remove it."""
    import inspect
    src = inspect.getsource(M.MADMainWindow._refresh_annotation_list)
    assert "self._update_file_list_counts()" in src


def test_only_the_current_row_is_touched():
    """The other recording's badge must not move -- and on a six-thousand-file
    project, walking every row per keystroke is the thing being avoided."""
    w, wavs = gui()
    seed(w, 4)
    other = os.path.basename(wavs[1])
    w._file_count_cache[other] = (7, 8, 9)
    w.spectrogram.annotations[0]['status'] = 'accepted'
    w._after_review_decision('p0', was_pending=True)
    assert w._file_count_cache[other] == (7, 8, 9)


def test_nothing_is_badged_while_a_file_is_still_loading():
    """current_file_idx points at the new recording before its annotations
    arrive; badging then writes the old file's counts onto the new row."""
    w, wavs = gui()
    seed(w, 4)
    before = dict(w._file_count_cache)
    keep = w._loading_path
    try:
        w._loading_path = wavs[1]
        w._touch_current_file_badge()
    finally:
        w._loading_path = keep
    assert w._file_count_cache == before


def test_a_mismatched_loaded_path_is_ignored():
    w, wavs = gui()
    seed(w, 4)
    before = dict(w._file_count_cache)
    keep = w._loaded_wav_path
    try:
        w._loaded_wav_path = wavs[1]      # annotations belong to the other file
        w._touch_current_file_badge()
    finally:
        w._loaded_wav_path = keep
    assert w._file_count_cache == before


def test_an_emptied_file_that_was_analyzed_keeps_a_zero_badge():
    """All-zero means 'analyzed, found nothing'. Dropping the entry would
    relabel the recording as never analyzed."""
    w, _ = gui()
    seed(w, 2)
    base = os.path.basename(w.audio_files[w.current_file_idx])
    w._file_run_info = {base: {'last_infer_model': 'm'}}
    w.spectrogram.annotations = []
    w._touch_current_file_badge()
    assert badge(w) == (0, 0, 0)


def test_an_emptied_file_that_was_never_analyzed_loses_its_badge():
    w, _ = gui()
    seed(w, 2)
    base = os.path.basename(w.audio_files[w.current_file_idx])
    w._file_run_info = {}
    w.spectrogram.annotations = []
    w._touch_current_file_badge()
    assert badge(w) is None


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
