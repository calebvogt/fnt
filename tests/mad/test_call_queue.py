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


def make(t_offs=None):
    """Three 2-s recordings, each with three detections (blob_ids 1-3, 40
    frames wide at frames 600, 1200, 1800) -- or, with ``t_offs``, one
    detection per offset (blob_ids 1..n) in every recording."""
    t_offs = t_offs or [600, 1200, 1800]
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
                                  "f_off": 250, "t_off": t, "score": 0.9}
                                 for b, t in enumerate(t_offs, start=1)])
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


def test_deciding_the_queued_call_first_clears_its_training_window():
    """Deciding a queued call inside a bout puts its neighbours into training
    as background unless they are decided too (2026-11-08). So Auto-advance
    first walks every pending detection in the queued call's training window,
    which widens with each decision (a chain of calls), and only then moves on
    through the queue. A detection beyond the chain is left alone."""
    # 40-frame calls 100 frames apart chain through the 128-frame context;
    # blob 4 (950) is reached only once blob 3 (800-840, window to 968) is
    # decided; blob 5 at 1600 is beyond the last window (990 + 128 = 1118).
    w, wavs, root = make([600, 700, 800, 950, 1600])
    q = queue_file(root, wavs, [(0, 2, ""), (1, 1, "")])
    w._settings.remove(f"mad/call_queue_pos/{os.path.normcase(os.path.abspath(q))}")
    w.focus_recordings([q])
    settle(w)
    assert selected(w) == ("rec0.wav", "2")
    assert w._queue_window_pending() == []           # queued call still pending
    w._auto_advance = True
    w._reject_current_pred()
    settle(w)
    seen = []
    while selected(w)[0] == "rec0.wav":
        seen.append(selected(w)[1])
        assert "in its training window" in w._queue_header_html()
        w._reject_current_pred()
        settle(w)
        assert len(seen) < 6
    assert sorted(seen) == ["1", "3", "4"]            # the chain, not blob 5
    assert seen[-1] == "4"                           # reached only via blob 3's window
    assert selected(w) == ("rec1.wav", "1") and w._call_queue_pos == 1
    pending = [a.get("blob_id") for a in w.spectrogram.annotations
               if a.get("status") == "prediction"]
    assert pending == [1, 2, 3, 4, 5]                # rec1 untouched


def test_skip_leaves_the_training_window_and_moves_on():
    w, wavs, root = make([600, 700, 1600])
    q = queue_file(root, wavs, [(0, 1, ""), (2, 3, "")])
    w._settings.remove(f"mad/call_queue_pos/{os.path.normcase(os.path.abspath(q))}")
    w.focus_recordings([q])
    settle(w)
    w._auto_advance = True
    w._reject_current_pred()
    settle(w)
    assert selected(w) == ("rec0.wav", "2")           # its window holds blob 2
    w._skip_current_pred()
    settle(w)
    assert selected(w) == ("rec2.wav", "3") and not w._queue_window_mode


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


def _edit_shape(w, idx, paint=None):
    """Right-click -> edit shape on call ``idx``, paint, Enter."""
    sg = w.spectrogram
    sg.start_edit(idx)
    assert sg.is_editing()
    if paint is not None:
        sg.paint_mode = 'brush'
        sg.brush_radius_px = 2
        sg._stamp(*paint)
    w._confirm_pending()
    settle(w)


def test_editing_the_queued_call_keeps_the_queue_and_walks_its_window():
    """An edit saves the call under a new id; the queue must follow it, and
    since the edit accepts a pending call it moves on like Accept: through the
    call's training window, then the queue (2026-11-08: after an edit the
    header lost the queued call and the window walk became the whole file)."""
    w, wavs, root = make([600, 700, 800, 950, 1600])
    q = queue_file(root, wavs, [(0, 2, ""), (1, 1, "")])
    w._settings.remove(f"mad/call_queue_pos/{os.path.normcase(os.path.abspath(q))}")
    w.focus_recordings([q])
    settle(w)
    assert selected(w) == ("rec0.wav", "2")
    w._auto_advance = True
    sg = w.spectrogram
    _edit_shape(w, sg._selected_ann_idx, paint=(720, 271))
    qa = w._queue_ann()
    assert qa is not None and qa['status'] == 'accepted'
    assert sg.queue_ann_id == w._call_queue_ann_id == qa['id']
    assert str(qa['blob_id']) == "2"                # still the model's detection
    assert w._queue_window_mode
    assert "in its training window" in w._queue_header_html()
    seen = []
    while selected(w)[0] == "rec0.wav":
        seen.append(selected(w)[1])
        w._reject_current_pred()
        settle(w)
        assert len(seen) < 6
    assert sorted(seen) == ["1", "3", "4"]
    assert selected(w) == ("rec1.wav", "1") and w._call_queue_pos == 1
    w._pred_prev()                                   # Back finds the edited call
    settle(w)
    assert selected(w) == ("rec0.wav", "2")
    assert w._queue_ann().get('status') in (None, 'accepted')   # a label now


def test_editing_a_neighbour_during_the_window_walk_continues_the_walk():
    w, wavs, root = make([600, 700, 800, 950, 1600])
    q = queue_file(root, wavs, [(0, 2, ""), (1, 1, "")])
    w._settings.remove(f"mad/call_queue_pos/{os.path.normcase(os.path.abspath(q))}")
    w.focus_recordings([q])
    settle(w)
    w._auto_advance = True
    w._reject_current_pred()                         # decide the queued call
    settle(w)
    first = selected(w)
    assert first[0] == "rec0.wav" and first[1] in ("1", "3")
    _edit_shape(w, w.spectrogram._selected_ann_idx)  # edit = accept the neighbour
    nxt = selected(w)
    assert nxt[0] == "rec0.wav" and nxt[1] != first[1]   # the walk went on
    assert w._queue_window_mode
    seen = [first[1]]
    while selected(w)[0] == "rec0.wav":
        seen.append(selected(w)[1])
        w._reject_current_pred()
        settle(w)
        assert len(seen) < 6
    assert sorted(seen) == ["1", "3", "4"]
    assert selected(w) == ("rec1.wav", "1") and w._call_queue_pos == 1


def _fresh_load(path):
    """What a normal open decodes for ``path`` (the same worker, run inline)."""
    wk = M.MADAudioLoadWorker(path, 0)
    got = {}
    wk.loaded.connect(lambda t, a, sr, e: got.update(a=a, sr=sr),
                      M.Qt.DirectConnection)
    wk.run()
    return got['a'], got['sr']


def _until(cond, timeout=20.0):
    import time
    end = time.time() + timeout
    while time.time() < end:
        QApplication.instance().processEvents()
        if cond():
            return True
        time.sleep(0.01)
    return False


def _count_decodes(monkeypatch, slow=None):
    """Recordings decoded, in order (``slow`` takes half a second)."""
    import time
    decoded = []
    orig = M.load_audio

    def load(p, *a, **k):
        name = os.path.basename(p)
        decoded.append(name)
        if name == slow:
            time.sleep(0.5)
        return orig(p, *a, **k)
    monkeypatch.setattr(M, "load_audio", load)
    return decoded


def _prefetched(w, name):
    pf = w._prefetch
    return pf is not None and pf['done'] and pf['path'].endswith(name)


def test_the_next_queued_recording_is_decoded_ahead(monkeypatch):
    """Stepping to a call in another recording decoded the whole file on the
    spot (~3-5 s for 600 s of ADPCM) -- twice, side by side. The queue now
    reads its next recording while you work and the open takes that read
    over: one decode, the same samples, and the .mad's crops read once."""
    w, wavs, root = make()
    for p in wavs:                      # stamped: opening writes nothing
        MS.mark_ridges_complete(MS.masks_sibling_path(p))
    decoded = _count_decodes(monkeypatch)
    reads = []
    orig_read = MS.read_all_pred_masks

    def read(h5, *a, **k):
        reads.append(os.path.basename(h5))
        return orig_read(h5, *a, **k)
    monkeypatch.setattr(MS, "read_all_pred_masks", read)
    mad2 = os.path.basename(MS.masks_sibling_path(wavs[2]))
    q = queue_file(root, wavs, [(0, 1, ""), (0, 3, ""), (2, 2, ""), (1, 1, "")])
    w._settings.remove(f"mad/call_queue_pos/{os.path.normcase(os.path.abspath(q))}")
    w.focus_recordings([q])
    settle(w)
    assert selected(w) == ("rec0.wav", "1")
    assert decoded.count("rec0.wav") == 1           # one decode per open
    # The next call in ANOTHER recording is rec2's: rec0's second is skipped.
    assert _until(lambda: _prefetched(w, "rec2.wav"))
    assert decoded.count("rec2.wav") == 1 and reads.count(mad2) == 1
    w._pred_next()
    settle(w)
    assert selected(w) == ("rec0.wav", "3")         # same recording, no load
    hits = w._prefetch_hits
    w._pred_next()
    settle(w)
    assert selected(w) == ("rec2.wav", "2")
    assert w._prefetch_hits == hits + 1
    assert decoded.count("rec2.wav") == 1           # served from the prefetch
    assert reads.count(mad2) == 1                   # and so were its crops
    audio, sr = _fresh_load(wavs[2])
    assert w.audio_data.dtype == audio.dtype
    assert np.array_equal(w.audio_data, audio) and w.sample_rate == sr
    pend = sorted(a.get("blob_id") for a in w.spectrogram.annotations
                  if a.get("status") == "prediction")
    assert pend == [1, 2, 3]
    # rec1 is read ahead now; Back to rec0, which is not, decodes it once.
    assert _until(lambda: _prefetched(w, "rec1.wav"))
    before = decoded.count("rec0.wav")
    w._pred_prev()
    settle(w)
    assert selected(w) == ("rec0.wav", "3")
    assert decoded.count("rec0.wav") == before + 1
    w.clear_focus()
    settle(w)
    assert w._prefetch is None


def test_an_open_takes_over_a_prefetch_still_running(monkeypatch):
    w, wavs, root = make()
    decoded = _count_decodes(monkeypatch, slow="rec1.wav")
    q = queue_file(root, wavs, [(0, 1, ""), (1, 2, ""), (2, 3, "")])
    w._settings.remove(f"mad/call_queue_pos/{os.path.normcase(os.path.abspath(q))}")
    w.focus_recordings([q])
    settle(w)
    assert selected(w) == ("rec0.wav", "1")
    assert _until(lambda: "rec1.wav" in decoded)
    assert w._prefetch is not None and not w._prefetch['done']
    w._pred_next()                                   # while rec1 still decodes
    settle(w)
    assert selected(w) == ("rec1.wav", "2")
    assert decoded.count("rec1.wav") == 1
    audio, _sr = _fresh_load(wavs[1])
    assert np.array_equal(w.audio_data, audio)


def test_a_mad_changed_since_the_prefetch_is_read_again():
    w, wavs, root = make()
    for p in wavs:
        MS.mark_ridges_complete(MS.masks_sibling_path(p))
    q = queue_file(root, wavs, [(0, 1, ""), (1, 2, "")])
    w._settings.remove(f"mad/call_queue_pos/{os.path.normcase(os.path.abspath(q))}")
    w.focus_recordings([q])
    settle(w)
    assert _until(lambda: _prefetched(w, "rec1.wav"))
    MS.write_pred_masks(MS.masks_sibling_path(wavs[1]),
                        [{"blob_id": 2, "mask": np.ones((20, 40), np.uint8),
                          "f_off": 250, "t_off": 1200, "score": 0.9}])
    w._pred_next()
    settle(w)
    assert selected(w) == ("rec1.wav", "2")
    pend = [a.get("blob_id") for a in w.spectrogram.annotations
            if a.get("status") == "prediction"]
    assert pend == [2]                               # the .mad as it is now


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)


def _decide_on_disk(w, wav, blob_ids, kind="negative"):
    """Record a decision for ``blob_ids`` the way an earlier session left it:
    an example carrying the blob_id in the recording's .mad."""
    from fnt.usv.usv_detector.fnt_mask_store import td_save_example
    h5 = MS.masks_sibling_path(wav)
    for b in blob_ids:
        t = 600 * b                                   # make()'s default layout
        meta = {"id": f"prev_{b}", "blob_id": b, "source_wav": os.path.basename(wav),
                "t_start_s": t * HOP / SR, "t_stop_s": (t + 40) * HOP / SR,
                "f_low_hz": 250 * SR / 1024, "f_high_hz": 270 * SR / 1024,
                "patch_t_off": t, "patch_f_off": 250}
        if kind != "label":
            meta["kind"] = kind
        td_save_example(h5, np.zeros((20, 40), np.float32),
                        np.ones((20, 40), np.uint8), meta, example_id=f"prev_{b}")


def test_skip_passes_over_decided_calls_and_opening_resumes_on_an_open_one():
    """2026-11-09: Skip and auto-advance walked onto calls already decided
    (in the window walk or an earlier session), and reopening started at the
    saved position. Now both go to the next call still to decide, and the end
    of the queue wraps to the calls skipped earlier."""
    w, wavs, root = make()
    rows = [(0, 1, ""), (1, 1, ""), (2, 1, ""), (0, 2, ""), (1, 2, ""), (2, 2, "")]
    q = queue_file(root, wavs, rows)
    _decide_on_disk(w, wavs[0], [1])
    _decide_on_disk(w, wavs[1], [1, 2])
    w._settings.remove(f"mad/call_queue_pos/{os.path.normcase(os.path.abspath(q))}")
    w.focus_recordings([q])
    settle(w)
    assert w._call_queue_pos == 2 and selected(w) == ("rec2.wav", "1")
    assert "4 to decide" not in w._queue_header_html()     # 3 open
    assert "3 to decide" in w._queue_header_html()
    w._skip_current_pred()                                  # rec2/1 stays open
    settle(w)
    assert selected(w) == ("rec0.wav", "2") and w._call_queue_pos == 3
    w._auto_advance = True
    w._reject_current_pred()                                # passes rec1/2 (decided)
    settle(w)
    assert w._call_queue_pos == 5 and selected(w) == ("rec2.wav", "2")
    w._skip_current_pred()                                  # end: wrap to the skipped one
    settle(w)
    assert w._call_queue_pos == 2 and selected(w) == ("rec2.wav", "1")


def test_a_stale_detection_row_does_not_move_the_queue():
    """A Detections row naming the recording being replaced must not reopen
    that recording's first queued call (the queue jumped back to its start)."""
    w, wavs, root = make()
    rows = [(0, 1, ""), (1, 1, ""), (0, 2, ""), (1, 2, ""), (0, 3, "")]
    q = queue_file(root, wavs, rows)
    w._settings.remove(f"mad/call_queue_pos/{os.path.normcase(os.path.abspath(q))}")
    w.focus_recordings([q])
    settle(w)
    for _ in range(3):
        w._skip_current_pred()
        settle(w)
    assert w._call_queue_pos == 3 and selected(w) == ("rec1.wav", "2")
    tree = w.annotation_list
    item = tree.topLevelItem(0)
    wav, t0, eid, st = item.data(0, M.Qt.UserRole)
    item.setData(0, M.Qt.UserRole, ("rec0.wav", t0, eid, st))   # a row left from rec0
    w._on_annotation_list_selected(item)
    settle(w)
    assert w._call_queue_pos == 3 and selected(w) == ("rec1.wav", "2")
    # Picking rec0 in the Audio list opens its next open call, not its first
    row = next(i for i, p in enumerate(w.audio_files) if p.endswith("rec0.wav"))
    w.file_list.setCurrentRow(row)
    settle(w)
    assert w._call_queue_pos == 4 and selected(w) == ("rec0.wav", "3")
