"""A focused session: MAD opened on a handful of a project's recordings.

For targeted QC ("label these 11 files"): ``python -m fnt.usv.mad_pyqt
--project P --focus list.txt`` shows only those recordings, so nobody scrolls
through hundreds to find them.

The hard requirement is that a focus is a VIEW. MAD has already lost a
project's audio list once (437 recordings cleared to 6), and the Audio list and
the project registry look like the same thing in a lot of code. So:

* the registry never shrinks, whatever is done in a focus;
* training still reads every registered recording;
* actions that act on "the list" either act on the visible subset by design or
  are refused (Empty the Audio List);
* File > Show All Recordings brings everything back.

Runs under pytest, or directly.
"""
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PyQt5.QtWidgets import QApplication, QMessageBox  # noqa: E402
from scipy.io import wavfile  # noqa: E402

import fnt.usv.mad_pyqt as M  # noqa: E402
from fnt.usv.usv_detector.mad_project import (  # noqa: E402
    MADProjectConfig, create_mad_project)

SR = 250_000
_KEEP = []


def make(n=8):
    root = tempfile.mkdtemp(prefix="mad_focus_")
    wavs = []
    for i in range(n):
        p = os.path.join(root, f"ch{i % 4 + 1}2026-04-30_0{i}-00-00_{i:05d}.wav")
        wavfile.write(p, SR, (np.random.default_rng(i).normal(0, .02, SR)
                              * 32767).astype(np.int16))
        wavs.append(p)
    proj = os.path.join(root, "p")
    create_mad_project(proj)
    app = QApplication.instance() or QApplication([])
    M.MADMainWindow._apply_dark_theme()
    w = M.MADMainWindow()
    _KEEP.append((app, w))
    w._activate_project(MADProjectConfig.load(proj))
    w._register_audio_files(wavs)
    w._rescan_project_wavs()
    w._wait_for_audio_load()
    return w, wavs, root, proj


def registry(w):
    return [e.path for e in w._project.audio_entries()]


def names(paths):
    return [os.path.basename(p) for p in paths]


# ----------------------------------------------------------------- showing
def test_bare_filenames_focus_the_list():
    w, wavs, _, _ = make()
    shown, missing = w.focus_recordings([os.path.basename(wavs[2]),
                                         os.path.basename(wavs[5])])
    assert shown == 2 and missing == []
    assert names(w.audio_files) == names([wavs[2], wavs[5]])


def test_a_list_file_with_paths_and_comments():
    w, wavs, root, _ = make()
    lst = os.path.join(root, "qc.txt")
    with open(lst, "w") as f:
        f.write("# tone files for QC\n\n" + wavs[1] + "\n"
                + os.path.basename(wavs[6]) + "   # by name\n")
    shown, _ = w.focus_recordings([lst])
    assert names(w.audio_files) == names([wavs[1], wavs[6]])


def test_names_not_in_the_project_are_reported_not_invented():
    w, wavs, _, _ = make()
    shown, missing = w.focus_recordings([os.path.basename(wavs[0]),
                                         "nope_00001.wav"])
    assert shown == 1 and missing == ["nope_00001.wav"]


def test_a_wav_the_project_lacks_is_registered_so_its_labels_count():
    w, wavs, root, _ = make()
    extra = os.path.join(root, "ch12026-05-01_00-00-00_09999.wav")
    wavfile.write(extra, SR, np.zeros(SR, np.int16))
    shown, missing = w.focus_recordings([extra])
    assert shown == 1 and missing == []
    assert extra in registry(w)


def test_the_window_says_it_is_a_focus():
    w, wavs, _, _ = make()
    w.focus_recordings([os.path.basename(wavs[3])])
    assert "focus: 1 of 8" in w.windowTitle()
    assert "focus" in w.lbl_project_status.text()
    assert w.act_show_all.isEnabled()


# ------------------------------------------------- the registry is untouched
def test_a_focus_never_shrinks_the_registry():
    w, wavs, _, proj = make()
    w.focus_recordings([os.path.basename(wavs[0])])
    assert len(registry(w)) == 8
    assert len(MADProjectConfig.load(proj).audio_entries()) == 8


def test_training_still_reads_every_recording():
    w, wavs, _, _ = make()
    w.focus_recordings([os.path.basename(wavs[0])])
    assert sorted(w._training_source_paths()) == sorted(wavs)


def test_emptying_the_list_is_refused_in_a_focus(monkeypatch):
    w, wavs, _, _ = make()
    w.focus_recordings([os.path.basename(wavs[0]), os.path.basename(wavs[1])])
    told = []
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: told.append(a[1])))
    w._empty_audio_list()
    assert told and len(registry(w)) == 8
    assert not w.act_empty_list.isEnabled()


def test_restoring_cleared_recordings_does_not_duplicate_hidden_ones(
        monkeypatch):
    """Restore compared against the visible list, so in a focus it saw the
    hidden recordings as missing and registered them all a second time."""
    w, wavs, _, _ = make()
    w._snapshot_audio_list(0)
    w.focus_recordings([os.path.basename(wavs[0])])
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.Yes))
    w._restore_audio_list()
    assert len(registry(w)) == 8


# ------------------------------------------------------------- leaving it
def test_show_all_brings_everything_back_and_keeps_the_open_file():
    w, wavs, _, _ = make()
    w.focus_recordings([os.path.basename(wavs[4]), os.path.basename(wavs[6])])
    w._on_file_selected(1)                  # wavs[6]
    w._wait_for_audio_load()
    w.clear_focus()
    assert names(w.audio_files) == names(wavs)
    assert w.audio_files[w.current_file_idx] == wavs[6]
    assert "focus" not in w.windowTitle()
    assert not w.act_show_all.isEnabled()


def test_a_rescan_keeps_the_focus():
    w, wavs, _, _ = make()
    w.focus_recordings([os.path.basename(wavs[2])])
    w._rescan_project_wavs()
    assert names(w.audio_files) == [os.path.basename(wavs[2])]


def test_closing_the_project_ends_the_focus():
    w, wavs, _, proj = make()
    w.focus_recordings([os.path.basename(wavs[2])])
    w._close_project(silent=True)
    w._activate_project(MADProjectConfig.load(proj))
    assert len(w.audio_files) == 8


def test_focus_needs_a_project_on_launch():
    with pytest.raises(SystemExit):
        M.main(["--focus", "x.wav"])


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
