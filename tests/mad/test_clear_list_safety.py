"""Pruning the Audio list never drops a recording it hasn't looked at.

What happened, on a real 443-recording project: files were added, which
restarts the background sidecar scan from nothing; seconds later "Clear All" ->
"Clear N with no detections" was pressed. The dialog honestly reported what the
count cache held, and the count cache held almost nothing, because the scan
reads sidecars two per event-loop turn over a network share. 437 recordings
went, 45 of them carrying finished QC -- 273 accepted and 135 rejected calls.

The defect is that a missing cache entry was read as "never analyzed". It
actually covers three states:

  * no sidecar at all      -- decided, and genuinely empty
  * the scan hasn't got to it yet
  * the read threw

Only the first is a statement about the recording. ``_counts_scanned`` records
what the scan actually decided, and emptiness is established from that.

Runs under pytest, or directly.
"""
import json
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
    """One window, six recordings, no project scan run."""
    global _GUI
    if _GUI is None:
        root = tempfile.mkdtemp(prefix="mad_clear_")
        wavs = []
        for i in range(6):
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


def scan_state(w, scanned_paths, counts=None):
    """Put the window in the middle of a scan: only ``scanned_paths`` decided."""
    w._counts_scanned = {os.path.basename(p) for p in scanned_paths}
    w._file_count_cache = dict(counts or {})
    w._counts_complete = False


def names(paths):
    return sorted(os.path.basename(p) for p in paths)


# --------------------------------------------------------- the reported loss
def test_a_recording_the_scan_has_not_reached_is_never_clearable():
    """The bug, in one line: absence from the cache used to mean 'empty'."""
    w, wavs = gui()
    scan_state(w, [])                       # scan just restarted, knows nothing
    assert w._files_without_detections() == [], \
        "offered to clear recordings nothing had looked at yet"


def test_only_the_part_the_scan_has_decided_is_offered():
    w, wavs = gui()
    scan_state(w, wavs[:2])                 # two probed, no sidecar on either
    assert names(w._files_without_detections()) == names(wavs[:2])


def test_a_labelled_recording_is_kept_once_it_is_read():
    w, wavs = gui()
    scan_state(w, wavs, {os.path.basename(wavs[3]): (12, 0, 4)})
    out = names(w._files_without_detections())
    assert os.path.basename(wavs[3]) not in out
    assert len(out) == 5


def test_rejections_alone_keep_a_recording():
    """Rejections are hard negatives -- the most valuable supervision here."""
    w, wavs = gui()
    scan_state(w, wavs, {os.path.basename(wavs[1]): (0, 0, 9)})
    assert os.path.basename(wavs[1]) not in names(w._files_without_detections())


def test_an_analyzed_empty_recording_is_still_prunable():
    """The feature has to keep working: that is the whole point of pruning a
    sampled 24/7 set down to the few files with anything in them."""
    w, wavs = gui()
    scan_state(w, wavs, {os.path.basename(wavs[0]): (0, 0, 0)})
    assert os.path.basename(wavs[0]) in names(w._files_without_detections())


def test_a_failed_sidecar_read_leaves_the_recording_unknown():
    """One dropped read on a network share must not make a labelled file look
    empty. _read_sidecar_chunk adds to _counts_scanned only after the read
    returns, so a throw leaves the basename out."""
    w, wavs = gui()
    scan_state(w, [p for p in wavs if p is not wavs[2]])
    assert os.path.basename(wavs[2]) not in names(w._files_without_detections())


# ------------------------------------------------------------- scan lifecycle
def test_restarting_the_scan_forgets_what_it_knew():
    """Adding files rebuilds the list and restarts the scan. Carrying the old
    decisions forward would leave the new files looking decided-and-empty."""
    w, wavs = gui()
    scan_state(w, wavs, {os.path.basename(wavs[0]): (3, 0, 0)})
    w._counts_complete = True
    w._scan_all_file_counts()
    assert w._counts_scanned == set()
    assert w._counts_complete is False
    assert w._files_without_detections() == []


def test_an_empty_list_is_immediately_complete():
    w, _ = gui()
    saved = list(w.audio_files)
    try:
        w.audio_files = []
        w._scan_all_file_counts()
        assert w._counts_complete is True
    finally:
        w.audio_files = saved


# ------------------------------------------------------------------ the undo
def test_removing_recordings_parks_a_copy_of_the_list_first():
    w, wavs = gui()
    bak = w._audio_list_backup_path()
    assert bak, "no project, no backup path"
    if os.path.isfile(bak):
        os.remove(bak)
    w._snapshot_audio_list(3)
    assert os.path.isfile(bak)
    with open(bak) as f:
        data = json.load(f)
    assert data['n_before'] == len(wavs)
    assert data['n_removed'] == 3
    assert len(data['entries']) == len(wavs)
    assert all(e.get('path') for e in data['entries'])


def test_the_backup_records_paths_not_copies():
    """It has to be cheap enough to write on every removal without thinking."""
    w, wavs = gui()
    w._snapshot_audio_list(1)
    size = os.path.getsize(w._audio_list_backup_path())
    assert size < 200_000, size


def test_a_failed_backup_never_blocks_the_removal():
    """Insurance that throws is worse than no insurance: it would make Remove
    File(s) fail on a read-only or full project directory."""
    w, wavs = gui()
    real = w._project.project_dir
    try:
        w._project.project_dir = os.path.join(real, "does", "not", "exist")
        w._snapshot_audio_list(1)           # must not raise
    finally:
        w._project.project_dir = real


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
