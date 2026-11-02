"""Merging calls: shift-click several, right-click one of them, Merge.

Field request (2026-10-13): a faint inverted-U sweep came out of the model as
two detections. The reviewer accepted both and painted the faint stretch
between them, which left three calls where there is one vocalisation. Merge
turns calls that overlap or touch into one accepted call.

What has to hold:

* shift-click adds a call to the selection (or takes it out), and a
  right-click on a selected call keeps the selection for the menu;
* the merged call is the union of the members' pixels, stored as ONE label
  example, and still tied to one model detection (QC counts hits by blob_id);
* reopening the file shows the merged call once, with no member coming back
  as a pending ghost;
* Ctrl+Z puts every member back, in the store as well as on screen;
* calls that do not touch are never merged.

Runs under pytest, or directly.
"""
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QApplication, QMenu
from scipy.io import wavfile

import fnt.usv.mad_pyqt as M
import fnt.usv.usv_detector.fnt_mask_store as MS
from fnt.usv.usv_detector.mad_project import MADProjectConfig, create_mad_project

SR = 250_000
_APP = None
#: Windows kept alive for the run: a collected MADMainWindow takes the
#: process down with it (see test_csv_independence).
_WINDOWS = []

# Two pieces of one call, side by side in time, and a third call well away.
A = dict(blob_id=901, f_off=300, t_off=1000)
B = dict(blob_id=902, f_off=300, t_off=1040)
C = dict(blob_id=903, f_off=300, t_off=1600)
#: B moved left until it overlaps A: the model's own pieces of one call.
B_ON_A = dict(blob_id=902, f_off=300, t_off=1015)
#: The painted stretch joining A and B: overlaps both.
BRIDGE = (302, 308, 1015, 1045)          # f0, f1, t0, t1


def _piece():
    m = np.zeros((10, 20), np.uint8)
    m[2:8, 1:19] = 1
    return m


def _session(accept=(901, 902), bridge=True, preds=(A, B, C)):
    """A project with predictions A, B, C on one recording; ``accept`` are
    accepted, and ``bridge`` paints the stretch joining A and B."""
    global _APP
    root = tempfile.mkdtemp(prefix="mad_merge_")
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
    MS.write_pred_masks(MS.masks_sibling_path(wav), [
        {**p, 'mask': _piece(), 'score': 0.8, 'class': 'USV'}
        for p in preds])
    _reopen(w)
    sg = w.spectrogram
    for b in accept:
        w._accept_prediction(_idx_of_blob(sg, b))
    if bridge:
        f0, f1, t0, t1 = BRIDGE
        sg._pending = np.zeros((sg.n_freq_bins, sg.n_time_frames), np.uint8)
        sg._pending[f0:f1, t0:t1] = 1
        sg._note_pending_region(f0, f1, t0, t1)
        w._confirm_pending()
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


def _idx_painted(sg):
    return next(i for i, a in enumerate(sg.annotations)
                if a.get('source') == 'label' and a.get('status') is None)


def _pixels(a):
    """A call's pixels as a set of (f, t), wherever its box sits."""
    fs, ts = np.nonzero(np.asarray(a['mask']))
    return set(zip((fs + a['f0']).tolist(), (ts + a['t0']).tolist()))


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


# ----------------------------------------------------------------------
# The union rule
# ----------------------------------------------------------------------
def _ann(f0, t0, mask):
    mask = np.asarray(mask, bool)
    return {'f0': f0, 'f1': f0 + mask.shape[0], 't0': t0,
            't1': t0 + mask.shape[1], 'mask': mask}


def test_overlapping_calls_merge_into_their_union():
    a, b = _ann(0, 0, np.ones((3, 4))), _ann(1, 2, np.ones((3, 4)))
    f0, f1, t0, t1, m = M.merge_call_masks([a, b])
    assert (f0, f1, t0, t1) == (0, 4, 0, 6)
    assert set(zip(*np.nonzero(m))) == _pixels(a) | _pixels(b)


def test_touching_calls_merge_but_separate_ones_do_not():
    a = _ann(0, 0, np.ones((3, 4)))
    touching = _ann(3, 4, np.ones((3, 4)))       # diagonal neighbour
    apart = _ann(0, 6, np.ones((3, 4)))          # one empty column between
    assert M.merge_call_masks([a, touching]) is not None
    assert M.merge_call_masks([a, apart]) is None


def test_a_chain_merges_even_where_its_ends_do_not_meet():
    a, b, c = (_ann(0, 0, np.ones((2, 4))), _ann(0, 3, np.ones((2, 4))),
               _ann(0, 6, np.ones((2, 4))))
    assert M.merge_call_masks([a, c]) is None
    assert M.merge_call_masks([a, b, c]) is not None


def test_boxes_that_overlap_are_not_enough():
    """Two L-shapes whose boxes overlap but whose pixels never meet."""
    m1 = np.zeros((6, 6), bool)
    m1[0, :] = True
    m1[:, 0] = True
    m2 = np.zeros((6, 6), bool)
    m2[5, :] = True
    m2[:, 5] = True
    assert M.merge_call_masks([_ann(0, 0, m1), _ann(2, 2, m2)]) is None


# ----------------------------------------------------------------------
# Selecting on the canvas
# ----------------------------------------------------------------------
def _canvas():
    """A bare canvas showing three calls, the third well apart."""
    global _APP
    _APP = QApplication.instance() or QApplication([])
    sg = M.MADSpectrogramWidget()
    sg.resize(1100, 700)
    audio = np.random.default_rng(0).normal(0, .05, SR).astype(np.float32)
    sg.set_audio_data(audio, SR, preserve_view=False)
    sg.init_mask(audio_len=len(audio), sample_rate=SR, nperseg=512,
                 noverlap=384, nfft=512)
    sg.view_start, sg.view_end = 0.25, 0.45
    sg.min_freq, sg.max_freq = 0.0, SR / 2.0
    sg.annotations = [
        {'id': f'c{i}', 'status': 'accepted', 'category': 'USV',
         **_ann(100, t0, np.ones((20, 30)))}
        for i, t0 in enumerate((600, 620, 760))]
    _WINDOWS.append(sg)
    return sg


def _point(sg, t, f):
    rect = sg._get_spec_rect()
    x, y = sg._grid_to_screen(t, f, rect, sg._visible_spec_bounds())
    return QPointF(x, y)


def _press(sg, pos, button=Qt.LeftButton, mods=Qt.NoModifier):
    sg.mousePressEvent(QMouseEvent(QEvent.MouseButtonPress, pos, button,
                                   button, mods))


def _release(sg, pos, button=Qt.LeftButton, mods=Qt.NoModifier):
    sg.mouseReleaseEvent(QMouseEvent(QEvent.MouseButtonRelease, pos, button,
                                     Qt.NoButton, mods))


def test_shift_click_adds_and_removes_calls():
    sg = _canvas()
    emitted = []
    sg.annotations_selected.connect(lambda ix: emitted.append(list(ix)))
    _press(sg, _point(sg, 610, 110))                         # plain click
    assert sg.selected_indices() == {0}
    _press(sg, _point(sg, 645, 110), mods=Qt.ShiftModifier)
    assert sg.selected_indices() == {0, 1}
    assert emitted[-1] == [0, 1]
    _press(sg, _point(sg, 610, 110), mods=Qt.ShiftModifier)  # take it out
    assert sg.selected_indices() == {1}
    _press(sg, _point(sg, 700, 200), mods=Qt.ShiftModifier)  # a miss
    assert sg.selected_indices() == {1}, "a shift-click on nothing keeps it"


def test_shift_drag_adds_a_box_to_the_selection():
    sg = _canvas()
    _press(sg, _point(sg, 610, 110))
    lo, hi = _point(sg, 750, 90), _point(sg, 800, 130)
    _press(sg, lo, mods=Qt.ShiftModifier)
    _release(sg, hi, mods=Qt.ShiftModifier)
    assert sg.selected_indices() == {0, 2}


def test_right_click_on_a_selected_call_keeps_the_selection():
    sg = _canvas()
    menus = []
    sg.request_context_menu.connect(lambda ai, _p: menus.append(ai))
    _press(sg, _point(sg, 610, 110))
    _press(sg, _point(sg, 645, 110), mods=Qt.ShiftModifier)
    _press(sg, _point(sg, 645, 110), button=Qt.RightButton)
    assert menus == [1]
    assert sg.selected_indices() == {0, 1}
    # On a call outside the selection the menu is about that call alone.
    _press(sg, _point(sg, 770, 110), button=Qt.RightButton)
    assert menus == [1, 2]
    assert sg.selected_indices() == set()


# ----------------------------------------------------------------------
# The merge itself
# ----------------------------------------------------------------------
def test_the_field_case_two_accepted_pieces_and_a_painted_bridge():
    w, wav = _session()
    sg = w.spectrogram
    idx = [_idx_of_blob(sg, 901), _idx_of_blob(sg, 902), _idx_painted(sg)]
    union = set().union(*(_pixels(sg.annotations[i]) for i in idx))
    w._on_box_selection(idx)
    items = _menu(w, idx[0], pick="Merge")
    assert items[0] == ("Merge 3 selected calls", True), items

    merged = [a for a in sg.annotations if _pixels(a) & union]
    assert len(merged) == 1, [a.get('id') for a in merged]
    m = merged[0]
    assert _pixels(m) == union
    assert m['status'] == 'accepted' and m['blob_id'] == 901
    labels = _labels(wav)
    assert len(labels) == 1, sorted(labels)
    (meta, px), = labels.values()
    assert px == union, "the stored mask is the union"
    assert meta.get('blob_id') == 901, "still tied to the model's detection"
    assert _crops(wav) == {'901', '903'}, "B's crop goes, A's stays"
    assert w._box_sel_ids == []

    got = _reopen(w)
    assert got == {'confirmed': 1, 'prediction': 1}, got
    shown = [a for a in sg.annotations if _pixels(a) & union]
    assert len(shown) == 1 and _pixels(shown[0]) == union


def test_undo_puts_every_piece_back():
    w, wav = _session()
    sg = w.spectrogram
    before_labels = _labels(wav)
    before = sorted((str(a.get('status')), str(a.get('blob_id')))
                    for a in sg.annotations)
    idx = [_idx_of_blob(sg, 901), _idx_of_blob(sg, 902), _idx_painted(sg)]
    assert w._merge_calls(idx, idx[0])

    w._undo_review_action()

    assert sorted((str(a.get('status')), str(a.get('blob_id')))
                  for a in sg.annotations) == before
    after_labels = _labels(wav)
    assert set(after_labels) == set(before_labels)
    for k, (_meta, px) in before_labels.items():
        assert after_labels[k][1] == px, f"{k} came back with another shape"
    assert _crops(wav) == {'901', '902', '903'}
    assert _reopen(w) == {'confirmed': 3, 'prediction': 1}


def test_pending_pieces_merge_into_an_accepted_call():
    w, wav = _session(accept=(), bridge=False, preds=(A, B_ON_A, C))
    sg = w.spectrogram
    a = sg.annotations[_idx_of_blob(sg, 901)]
    b = sg.annotations[_idx_of_blob(sg, 902)]
    assert M.merge_call_masks([a, b]) is not None
    assert w._merge_calls([_idx_of_blob(sg, 901), _idx_of_blob(sg, 902)],
                          _idx_of_blob(sg, 902))
    labels = _labels(wav)
    assert len(labels) == 1
    (meta, _px), = labels.values()
    assert meta.get('blob_id') == 902, "the right-clicked piece is kept"
    assert _crops(wav) == {'902', '903'}
    assert _reopen(w) == {'confirmed': 1, 'prediction': 1}


def test_calls_that_do_not_touch_are_left_alone():
    w, wav = _session(accept=(901, 902, 903), bridge=False)
    sg = w.spectrogram
    idx = [_idx_of_blob(sg, 901), _idx_of_blob(sg, 903)]
    labels = _labels(wav)
    w._on_box_selection(idx)
    items = _menu(w, idx[0], pick="Merge")
    assert items[0] == ("Merge 2 selected calls (they must overlap or touch)",
                        False), items
    assert not w._merge_calls(idx, idx[0])
    assert _labels(wav).keys() == labels.keys()
    assert _crops(wav) == {'901', '902', '903'}


def test_no_merge_entry_outside_a_multi_selection():
    w, _wav = _session(accept=(901, 902), bridge=False)
    sg = w.spectrogram
    w._on_box_selection([_idx_of_blob(sg, 901), _idx_of_blob(sg, 902)])
    items = _menu(w, _idx_of_blob(sg, 903))
    assert not any(t.startswith("Merge") for t, _e in items), items


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
