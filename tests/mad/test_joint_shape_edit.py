"""Editing several calls' shapes together: select the pieces, right-click one,
Edit N shapes together, paint the joins, Enter.

Field request (2026-11-09): faint, oscillating calls come out of the model as
several broken pieces. Joining them took three steps -- paint a bridge and
Enter (its own call), select every piece and the bridge, Merge. A joint edit
puts every selected call into one shape edit; on Enter, calls that touch
become one merged call and everything else stays as it was.

What has to hold:

* the edited pixels are split into 8-connected components (the "touch" Merge
  accepts); calls sharing one become ONE accepted call, stored as one label
  example and still tied to one model detection (the right-clicked one's, or
  in a call queue the queued call's);
* a call left untouched and unconnected keeps its status -- a pending
  detection stays pending; a lone call whose shape changed is saved as an
  edited shape (accepted), as Edit shape + Enter does;
* a stroke touching none of the calls keeps the edit open (a bridge that fell
  short is not a call);
* Esc changes nothing; one Ctrl+Z undoes the whole edit, store included;
* in a call queue, a joint edit holding the queued call moves the queue on.

Runs under pytest, or directly.
"""
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from PyQt5.QtCore import QPoint  # noqa: E402
from PyQt5.QtWidgets import QApplication, QMenu  # noqa: E402
from scipy.io import wavfile  # noqa: E402

import fnt.usv.mad_pyqt as M  # noqa: E402
import fnt.usv.usv_detector.fnt_mask_store as MS  # noqa: E402
from fnt.usv.usv_detector.mad_project import (  # noqa: E402
    MADProjectConfig, create_mad_project)

SR = 250_000
_APP = None
_WINDOWS = []

# Three pieces: A and B side by side with a gap the bridge fills, C well away.
A = dict(blob_id=901, f_off=300, t_off=1000)
B = dict(blob_id=902, f_off=300, t_off=1040)
C = dict(blob_id=903, f_off=300, t_off=1600)
#: The painted stretch joining A (t 1001-1018) and B (t 1041-1058).
BRIDGE = (303, 307, 1015, 1045)          # f0, f1, t0, t1


def _piece():
    m = np.zeros((10, 20), np.uint8)
    m[2:8, 1:19] = 1
    return m


def _session(accept=()):
    """A project with pending detections A, B, C on one recording; the blob
    ids in ``accept`` are accepted first."""
    global _APP
    root = tempfile.mkdtemp(prefix="mad_joint_")
    wav = os.path.join(root, "rec.wav")
    wavfile.write(wav, SR, (np.random.default_rng(0).normal(0, .05, SR * 2)
                            * 32767).astype(np.int16))
    proj = os.path.join(root, "p")
    create_mad_project(proj)
    _APP = QApplication.instance() or QApplication([])
    M.MADMainWindow._apply_dark_theme()
    w = M.MADMainWindow()
    w._activate_project(MADProjectConfig.load(proj))
    w._register_audio_files([wav])
    w._append_audio_paths([wav])
    w._wait_for_audio_load()
    w._ask_class_for_confirm = lambda k, c, l: 'USV'
    w._show_first_label_info = lambda: None
    w._show_labeling_instructions = lambda: None    # a modal tip, once
    MS.write_pred_masks(MS.masks_sibling_path(wav), [
        {**p, 'mask': _piece(), 'score': 0.8, 'class': 'USV'}
        for p in (A, B, C)])
    _reopen(w)
    for b in accept:
        w._accept_prediction(_idx_of_blob(w.spectrogram, b))
    w._undo_stack = []
    _WINDOWS.append(w)
    return w, wav


def _reopen(w):
    w._loaded_wav_path = None
    w.audio_data = None
    w.current_file_idx = 0
    w._load_current_file()
    w._wait_for_audio_load()
    out = {}
    for a in w.spectrogram.annotations:
        k = a.get('status') or 'confirmed'
        out[k] = out.get(k, 0) + 1
    return out


def _idx_of_blob(sg, b):
    return next(i for i, a in enumerate(sg.annotations)
                if str(a.get('blob_id')) == str(b))


def _pixels(a):
    fs, ts = np.nonzero(np.asarray(a['mask']))
    return set(zip((fs + a['f0']).tolist(), (ts + a['t0']).tolist()))


def _box_pixels(f0, f1, t0, t1):
    return {(f, t) for f in range(f0, f1) for t in range(t0, t1)}


def _labels(wav):
    """This recording's label examples: id -> (meta, pixels)."""
    out = {}
    for ex in MS.td_iter_file_examples(MS.masks_sibling_path(wav),
                                       os.path.basename(wav),
                                       with_spec=False):
        meta = ex['meta']
        if MS.example_kind(meta) != 'label':
            continue
        fs, ts = np.nonzero(ex['mask'])
        out[meta['id']] = (meta, set(zip(
            (fs + int(meta.get('patch_f_off') or 0)).tolist(),
            (ts + int(meta.get('patch_t_off') or 0)).tolist())))
    return out


def _crops(wav):
    return set(MS.read_all_pred_masks(MS.masks_sibling_path(wav)))


def _state(sg):
    """Everything about the calls a joint edit could change, in list order."""
    return [(a.get('id'), str(a.get('blob_id')), a.get('status'),
             a['f0'], a['f1'], a['t0'], a['t1'],
             np.asarray(a['mask']).tobytes()) for a in sg.annotations]


def _menu(w, ann_idx, pick=None):
    """Open the call menu without blocking: list its entries, choose one."""
    seen = {}
    orig = QMenu.exec_

    def fake(menu, *a, **k):
        acts = [x for x in menu.actions() if not x.isSeparator()]
        seen['items'] = [(x.text(), x.isEnabled()) for x in acts]
        for x in acts:
            if pick and x.text().startswith(pick):
                return x
        return None
    QMenu.exec_ = fake
    try:
        w._on_annotation_context_menu(ann_idx, QPoint(0, 0))
    finally:
        QMenu.exec_ = orig
    return seen.get('items', [])


def _joint_edit(w, blobs, anchor):
    """Shift-select the calls with ``blobs``, right-click ``anchor`` and pick
    Edit N shapes together."""
    sg = w.spectrogram
    idx = [_idx_of_blob(sg, b) for b in blobs]
    w._on_box_selection(idx)
    items = _menu(w, _idx_of_blob(sg, anchor),
                  pick=f"Edit {len(blobs)} shapes together")
    assert (f"Edit {len(blobs)} shapes together", True) in items, items
    assert sg.is_joint_editing() and sg.paint_mode == 'brush'
    assert not any(str(a.get('blob_id')) in {str(b) for b in blobs}
                   for a in sg.annotations), "the edited calls leave the list"


def _paint(sg, box):
    """A brush stroke filling ``box`` (as the brush records it)."""
    f0, f1, t0, t1 = box
    sg._pending[f0:f1, t0:t1] = 1
    sg._note_pending_region(f0, f1, t0, t1)


# ----------------------------------------------------------------------
# Sorting the edited pixels into calls
# ----------------------------------------------------------------------
def _ann(f0, t0, mask):
    mask = np.asarray(mask, bool)
    return {'f0': f0, 'f1': f0 + mask.shape[0], 't0': t0,
            't1': t0 + mask.shape[1], 'mask': mask}


def _union(anns, extra=()):
    f0 = min(a['f0'] for a in anns)
    t0 = min(a['t0'] for a in anns)
    f1 = max(a['f1'] for a in anns)
    t1 = max(a['t1'] for a in anns)
    m = np.zeros((f1 - f0, t1 - t0), bool)
    for a in anns:
        m[a['f0'] - f0:a['f1'] - f0, a['t0'] - t0:a['t1'] - t0] |= a['mask']
    for f, t in extra:
        m[f - f0, t - t0] = True
    return (f0, f1, t0, t1), m


def test_groups_follow_touch_and_chains():
    a, b, c, d = (_ann(0, 0, np.ones((2, 3))), _ann(2, 3, np.ones((2, 3))),
                  _ann(0, 6, np.ones((2, 3))), _ann(0, 20, np.ones((2, 3))))
    # a-b touch diagonally, b-c through b's corner; d is apart.
    box, m = _union([a, b, c, d])
    groups, strays, gone = M.joint_edit_groups([a, b, c, d], box, m)
    assert [g for g, _s in groups] == [(0, 1, 2), (3,)]
    assert strays == [] and gone == []
    assert M.same_call_shape(d, groups[1][1])


def test_stray_strokes_and_erased_calls_are_reported():
    a, b = _ann(0, 0, np.ones((2, 3))), _ann(0, 10, np.ones((2, 3)))
    box, m = _union([a, b])
    m[:, 10 - box[2]:] = False                 # b erased away
    m[0, 6] = True                             # a dot touching nothing
    groups, strays, gone = M.joint_edit_groups([a, b], box, m)
    assert [g for g, _s in groups] == [(0,)] and gone == [1]
    assert len(strays) == 1 and strays[0][:4] == (0, 1, 6, 7)


# ----------------------------------------------------------------------
# The edit in the window
# ----------------------------------------------------------------------
def test_a_bridge_joins_two_pieces_and_the_third_stays_pending():
    w, wav = _session()
    sg = w.spectrogram
    c_before = [s for s in _state(sg) if s[1] == '903']
    a_px = _pixels(sg.annotations[_idx_of_blob(sg, 901)])
    b_px = _pixels(sg.annotations[_idx_of_blob(sg, 902)])
    _joint_edit(w, (901, 902, 903), anchor=902)
    _paint(sg, BRIDGE)
    w._confirm_pending()

    assert not sg.is_editing()
    union = a_px | b_px | _box_pixels(*BRIDGE)
    joined = [a for a in sg.annotations if _pixels(a) & union]
    assert len(joined) == 1, [a.get('id') for a in joined]
    m = joined[0]
    assert _pixels(m) == union
    assert m['status'] == 'accepted' and m['blob_id'] == 902, \
        "the right-clicked piece's detection is kept"
    assert [s for s in _state(sg) if s[1] == '903'] == c_before, \
        "the untouched call is exactly as it was, still pending"
    labels = _labels(wav)
    assert len(labels) == 1, sorted(labels)
    (meta, px), = labels.values()
    assert px == union and meta.get('blob_id') == 902
    assert _crops(wav) == {'902', '903'}, "A's crop goes, B's and C's stay"

    assert _reopen(w) == {'confirmed': 1, 'prediction': 1}
    shown = [a for a in sg.annotations if _pixels(a) & union]
    assert len(shown) == 1 and _pixels(shown[0]) == union


def test_escape_changes_nothing():
    w, wav = _session(accept=(903,))
    sg = w.spectrogram
    before = _state(sg)
    labels = _labels(wav)
    _joint_edit(w, (901, 902, 903), anchor=901)
    _paint(sg, BRIDGE)
    _paint(sg, (300, 304, 1590, 1603))              # and grow C
    w._clear_pending()
    assert not sg.is_editing() and sg.paint_mode is None
    assert _state(sg) == before
    assert _labels(wav).keys() == labels.keys()
    assert _crops(wav) == {'901', '902', '903'}
    assert w._undo_stack == []
    assert sg.pending_bbox() is None


def test_one_undo_puts_all_three_back():
    w, wav = _session()
    sg = w.spectrogram
    before = sorted((a.get('status'), str(a.get('blob_id')))
                    for a in sg.annotations)
    _joint_edit(w, (901, 902, 903), anchor=901)
    _paint(sg, BRIDGE)
    w._confirm_pending()
    assert len(_labels(wav)) == 1

    w._undo_review_action()

    assert sorted((a.get('status'), str(a.get('blob_id')))
                  for a in sg.annotations) == before
    assert _labels(wav) == {}
    assert _crops(wav) == {'901', '902', '903'}
    assert _reopen(w) == {'prediction': 3}


def test_undo_restores_accepted_pieces_in_the_store():
    """Accepted detections re-save under their deterministic ids, so the
    order of destroying and saving decides whether undo gets them back."""
    w, wav = _session(accept=(901, 902, 903))
    sg = w.spectrogram
    before = _labels(wav)
    assert len(before) == 3
    _joint_edit(w, (901, 902, 903), anchor=901)
    _paint(sg, BRIDGE)                              # join A and B
    _paint(sg, (300, 304, 1590, 1603))              # and grow C on its own
    w._confirm_pending()
    after = _labels(wav)
    assert len(after) == 2
    c = sg.annotations[_idx_of_blob(sg, 903)]
    assert _box_pixels(300, 304, 1590, 1600) <= _pixels(c)

    w._undo_review_action()

    got = _labels(wav)
    assert set(got) == set(before)
    for k, (_meta, px) in before.items():
        assert got[k][1] == px, f"{k} came back with another shape"
    assert _crops(wav) == {'901', '902', '903'}
    assert _reopen(w) == {'confirmed': 3}


def test_a_lone_call_whose_shape_changed_is_accepted_the_others_stay():
    w, wav = _session()
    sg = w.spectrogram
    a_before = [s for s in _state(sg) if s[1] in ('901', '902')]
    _joint_edit(w, (901, 902, 903), anchor=903)
    _paint(sg, (300, 304, 1590, 1603))              # touches C only
    w._confirm_pending()
    c = sg.annotations[_idx_of_blob(sg, 903)]
    assert c['status'] == 'accepted' and c['blob_id'] == 903
    assert _box_pixels(300, 304, 1590, 1600) <= _pixels(c)
    assert [s for s in _state(sg) if s[1] in ('901', '902')] == a_before
    assert len(_labels(wav)) == 1
    assert _reopen(w) == {'confirmed': 1, 'prediction': 2}


def test_unchanged_shapes_save_nothing():
    w, wav = _session()
    sg = w.spectrogram
    before = _state(sg)
    _joint_edit(w, (901, 902), anchor=901)
    w._confirm_pending()
    assert not sg.is_editing()
    assert _state(sg) == before and _labels(wav) == {}
    assert w._undo_stack == []


def test_a_stroke_that_touches_nothing_keeps_the_edit_open():
    w, wav = _session()
    sg = w.spectrogram
    before = _state(sg)
    _joint_edit(w, (901, 902), anchor=901)
    _paint(sg, (303, 307, 1025, 1030))              # short of both pieces
    w._confirm_pending()
    assert sg.is_joint_editing(), "Enter refused: the stroke is a stray"
    assert "touch none of the calls" in w.status_bar.currentMessage()
    assert _labels(wav) == {}
    _paint(sg, BRIDGE)                              # now it joins them
    w._confirm_pending()
    assert not sg.is_editing()
    assert len(_labels(wav)) == 1
    assert len(sg.annotations) == len(before) - 1


def test_ctrl_z_during_the_edit_does_not_restore_underneath_it():
    w, wav = _session(accept=(903,))
    sg = w.spectrogram
    w._accept_prediction(_idx_of_blob(sg, 901))     # something to undo
    before = _state(sg)
    _joint_edit(w, (901, 902), anchor=901)
    w._undo_review_action()
    assert sg.is_joint_editing()
    w._undo_last()                                  # U: no stroke to take back
    w._clear_pending()
    assert _state(sg) == before


# ----------------------------------------------------------------------
# In a call queue
# ----------------------------------------------------------------------
def test_a_joint_edit_holding_the_queued_call_moves_the_queue_on():
    from test_call_queue import make, queue_file, selected, settle
    w, wavs, root = make()
    _WINDOWS.append(w)
    w._show_labeling_instructions = lambda: None
    q = queue_file(root, wavs, [(0, 2, ""), (1, 1, "")])
    w._settings.remove(
        f"mad/call_queue_pos/{os.path.normcase(os.path.abspath(q))}")
    w.focus_recordings([q])
    settle(w)
    assert selected(w) == ("rec0.wav", "2")
    w._auto_advance = True
    sg = w.spectrogram
    # Queued call 2 (frames 1200-1240) and call 3 (1800-1840), joined by a
    # stroke; right-clicked on 3, so only the queue decides whose detection
    # the merged call keeps.
    _joint_edit(w, (2, 3), anchor=3)
    _paint(sg, (255, 260, 1235, 1805))
    w._confirm_pending()
    settle(w)
    assert selected(w) == ("rec1.wav", "1") and w._call_queue_pos == 1
    w._pred_prev()
    settle(w)
    assert selected(w) == ("rec0.wav", "2")
    qa = w._queue_ann()
    assert qa is not None and qa.get('status') in (None, 'accepted')
    assert str(qa['blob_id']) == "2", "the queued call's detection is kept"
    assert qa['t0'] == 1200 and qa['t1'] == 1840, "one call: 2 + stroke + 3"
    assert not any(str(a.get('blob_id')) == "3"
                   for a in sg.annotations)


if __name__ == "__main__":
    import sys
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:      # noqa: BLE001
                fails += 1
                print("FAIL", name, "-", type(e).__name__, e)
    sys.exit(1 if fails else 0)
