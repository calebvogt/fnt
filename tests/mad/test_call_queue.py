"""A call queue: a focus list that names detections, reviewed call by call.

``recording<TAB>detection<TAB>note`` lines in a ``--focus`` file make Next /
Back walk the listed detections across recordings, in the file's order:
each opens its recording, centred and selected. Deciding the queue's call
moves on through the queue (Auto-advance), not through the rest of that
recording, and the position is remembered for the next session.

Runs under pytest, or directly.
"""
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402
from scipy.io import wavfile  # noqa: E402

import fnt.usv.mad_pyqt as M  # noqa: E402
from fnt.usv.usv_detector import fnt_mask_store as MS  # noqa: E402
from fnt.usv.usv_detector.mad_project import (  # noqa: E402
    MADProjectConfig, create_mad_project)

SR, HOP = 250_000, 128
_KEEP = []


def make():
    """Three 2-s recordings, each with three detections (blob_ids 1-3)."""
    root = tempfile.mkdtemp(prefix="mad_queue_")
    wavs = []
    for i in range(3):
        p = os.path.join(root, f"rec{i}.wav")
        wavfile.write(p, SR, (np.random.default_rng(i).normal(0, .02, 2 * SR)
                              * 32767).astype(np.int16))
        h5 = MS.masks_sibling_path(p)
        MS.set_grid_attrs(h5, sample_rate=SR, nperseg=512, noverlap=384, nfft=1024,
                          n_freq_bins=513, n_time_frames=2 * SR // HOP)
        MS.write_pred_masks(h5, [{"blob_id": b, "mask": np.ones((20, 40), np.uint8),
                                  "f_off": 250, "t_off": 600 * b, "score": 0.9}
                                 for b in (1, 2, 3)])
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
    return w, wavs, root


def settle(w):
    for _ in range(5):
        w._wait_for_audio_load()
        QApplication.instance().processEvents()


def selected(w):
    """(recording, blob_id) of the selected annotation."""
    sg = w.spectrogram
    i = sg._selected_ann_idx
    assert i is not None
    return (os.path.basename(w.audio_files[w.current_file_idx]),
            str(sg.annotations[i].get("blob_id")))


def queue_file(root, wavs, rows):
    q = os.path.join(root, "queue.txt")
    with open(q, "w") as f:
        f.write("# call queue\n")
        for k, b, note in rows:
            f.write(f"{wavs[k]}\t{b}\t{note}\n")
    return q


def test_lines_without_a_detection_stay_a_plain_focus():
    w, wavs, root = make()
    q = os.path.join(root, "plain.txt")
    with open(q, "w") as f:
        f.write(wavs[0] + "\n" + wavs[2] + "\n")
    w.focus_recordings([q])
    settle(w)
    assert not w._queue_active()
    assert len(w.audio_files) == 2


def test_next_and_back_walk_the_queue_across_recordings():
    w, wavs, root = make()
    q = queue_file(root, wavs, [(2, 3, "wheel?"), (0, 1, ""), (0, 3, ""), (1, 2, "")])
    w._settings.remove(f"mad/call_queue_pos/{os.path.normcase(os.path.abspath(q))}")
    w.focus_recordings([q])
    settle(w)
    assert w._queue_active() and len(w._call_queue) == 4
    assert selected(w) == ("rec2.wav", "3")
    w._pred_next()
    settle(w)
    assert selected(w) == ("rec0.wav", "1")
    w._pred_next()
    settle(w)
    assert selected(w) == ("rec0.wav", "3")            # same recording: no reload
    w._pred_prev()
    settle(w)
    assert selected(w) == ("rec0.wav", "1")
    w._update_pred_review_widgets()
    assert w._review_sets[0]["nav_lbl"].text() == "Call 2/4"


def test_deciding_the_queued_call_moves_on_through_the_queue():
    w, wavs, root = make()
    q = queue_file(root, wavs, [(1, 2, ""), (2, 1, "")])
    w._settings.remove(f"mad/call_queue_pos/{os.path.normcase(os.path.abspath(q))}")
    w.focus_recordings([q])
    settle(w)
    assert selected(w) == ("rec1.wav", "2")
    w._auto_advance = True
    w._reject_current_pred()
    settle(w)
    assert selected(w) == ("rec2.wav", "1")            # not rec1's next detection


def test_accepting_the_queued_call_moves_on_through_the_queue():
    """Accept saves the call under a new id; the queue must still see it as
    its call (2026-11-05: Accept stayed inside the recording)."""
    w, wavs, root = make()
    q = queue_file(root, wavs, [(0, 2, ""), (1, 1, ""), (2, 3, "")])
    w._settings.remove(f"mad/call_queue_pos/{os.path.normcase(os.path.abspath(q))}")
    w.focus_recordings([q])
    settle(w)
    w._auto_advance = True
    w._pred_next()                                     # reach call 2 the usual way
    settle(w)
    assert selected(w) == ("rec1.wav", "1")
    w._accept_current_pred()
    settle(w)
    assert selected(w) == ("rec2.wav", "3") and w._call_queue_pos == 2


def test_the_position_is_remembered():
    w, wavs, root = make()
    q = queue_file(root, wavs, [(0, 1, ""), (1, 1, ""), (2, 1, "")])
    w._settings.remove(f"mad/call_queue_pos/{os.path.normcase(os.path.abspath(q))}")
    w.focus_recordings([q])
    settle(w)
    w._pred_next()
    w._pred_next()
    settle(w)
    w.clear_focus()
    settle(w)
    assert not w._queue_active()
    w.focus_recordings([q])
    settle(w)
    assert w._call_queue_pos == 2 and selected(w) == ("rec2.wav", "1")


def test_the_queued_call_is_marked_everywhere():
    """Spectrogram frame, Detections row and header all point at the queued
    call -- a call queue must not look like a per-recording focus."""
    w, wavs, root = make()
    q = queue_file(root, wavs, [(1, 2, "MMZ ch1")])
    w._settings.remove(f"mad/call_queue_pos/{os.path.normcase(os.path.abspath(q))}")
    w.focus_recordings([q])
    settle(w)
    qid = w._call_queue_ann_id
    assert qid is not None and w.spectrogram.queue_ann_id == qid
    tree = w.annotation_list
    marked = [tree.topLevelItem(r) for r in range(tree.topLevelItemCount())
              if tree.topLevelItem(r).text(0).startswith("▶")]
    assert len(marked) == 1 and marked[0].data(0, M.Qt.UserRole)[2] == qid
    w._update_view_header()
    head = w.lbl_view_header.text()
    assert "CALL QUEUE 1/1" in head and "MMZ ch1" in head
    assert "queued call" in w._grp_detections.title()
    assert "call queue" in w.windowTitle()
    w.clear_focus()
    settle(w)
    assert w.spectrogram.queue_ann_id is None
    assert w._grp_detections.title() == "Detections"


def test_skip_and_file_navigation_follow_the_queue():
    w, wavs, root = make()
    q = queue_file(root, wavs, [(2, 3, ""), (0, 1, ""), (1, 2, "")])
    w._settings.remove(f"mad/call_queue_pos/{os.path.normcase(os.path.abspath(q))}")
    w.focus_recordings([q])
    settle(w)
    assert selected(w) == ("rec2.wav", "3")
    w._skip_current_pred()                              # Skip = next queued call
    settle(w)
    assert selected(w) == ("rec0.wav", "1") and w._call_queue_pos == 1
    w._next_file()                                      # file Next = next queued call
    settle(w)
    assert selected(w) == ("rec1.wav", "2") and w._call_queue_pos == 2
    w._prev_file()
    settle(w)
    assert selected(w) == ("rec0.wav", "1") and w._call_queue_pos == 1
    # Clicking a recording in the Audio list opens its queued call
    row = next(i for i, p in enumerate(w.audio_files) if p.endswith("rec2.wav"))
    w.file_list.setCurrentRow(row)
    settle(w)
    assert selected(w) == ("rec2.wav", "3") and w._call_queue_pos == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
