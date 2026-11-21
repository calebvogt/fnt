"""Edit Shape (right-click -> edit, paint, Enter / Esc) touches only the edited
call's region of the full-file buffers.

On a 10-minute recording the spectrogram grid is 513 x 1.17M (601 MB). Edit
Shape zeroed the pending buffer, copied all of it as the Esc backup and rebuilt
the confirmed mask from scratch when it started; Enter scanned the whole buffer
twice (``any`` + ``nonzero``, ~3.5 s) and rebuilt the mask twice more; Esc
zeroed and rebuilt again (2026-11-08: the window froze while editing). Strokes
from an earlier edit also stayed on the undo stack, so every later stroke's
component count labelled a crop spanning both edits.

The bounded versions must leave exactly what the full-grid ones did: Esc puts
the call back untouched, Enter commits the painted pixels, erasing the call to
nothing cancels, and the confirmed mask always equals a full rebuild.

"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

pytest.importorskip("PyQt5")
from PyQt5.QtWidgets import QApplication  # noqa: E402

import fnt.usv.mad_pyqt as M  # noqa: E402

NF, NT = 513, 20000


class _Tripwire(np.ndarray):
    """A buffer that counts whole-grid scans, fills and copies."""

    def __array_finalize__(self, obj):
        self.full_ops = getattr(obj, 'full_ops', None)

    def _note(self):
        if self.shape == (NF, NT) and self.full_ops is not None:
            self.full_ops[0] += 1

    def any(self, *a, **k):
        self._note()
        return super().any(*a, **k)

    def nonzero(self):
        self._note()
        return super().nonzero()

    def copy(self, *a, **k):
        self._note()
        return super().copy(*a, **k)

    def __setitem__(self, k, v):
        if isinstance(k, slice) and k == slice(None):
            self._note()
        super().__setitem__(k, v)


def _tripwire(a):
    t = a.view(_Tripwire)
    t.full_ops = [0]
    return t


def _ann(i, f0, t0, h=20, w=40, status='prediction', hole=False):
    m = np.ones((h, w), bool)
    if hole:
        m[5:8, 10:14] = False             # a hole the edit must not fill
    return {'id': f'a{i}', 'blob_id': i, 'f0': f0, 'f1': f0 + h,
            't0': t0, 't1': t0 + w, 'mask': m, 'status': status,
            'category': 'USV'}


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def sg(qapp):
    w = M.MADSpectrogramWidget()
    w.n_freq_bins, w.n_time_frames = NF, NT
    w.mask = np.zeros((NF, NT), np.uint8)
    w._pending = np.zeros((NF, NT), np.uint8)
    w._pending_bbox = None
    w._pending_stack = []
    # Two overlapping calls and a far one: a restamp that clears the edited
    # call's box must put the neighbour's shared pixels back.
    w.set_annotations([_ann(1, 200, 1000, hole=True), _ann(2, 210, 1030),
                       _ann(3, 300, 15000)])
    return w


def _full_rebuild(sg):
    ref = np.zeros_like(np.asarray(sg.mask))
    for a in sg.annotations:
        ref[a['f0']:a['f1'], a['t0']:a['t1']][a['mask']] = M.MASK_POSITIVE
    return ref


def _arm(sg):
    sg.mask = _tripwire(np.asarray(sg.mask))
    sg._pending = _tripwire(np.asarray(sg._pending))
    return sg.mask.full_ops, sg._pending.full_ops


def _paint(sg, t, f, mode='brush', r=2):
    sg.paint_mode = mode
    sg.brush_radius_px = r
    sg._stroke_fset = set() if mode == 'brush' else None
    sg._stamp(t, f)
    if sg._stroke_fset:
        fs = np.array([p[0] for p in sg._stroke_fset], int)
        ts = np.array([p[1] for p in sg._stroke_fset], int)
        sg._pending_stack.append((fs, ts))
    sg._stroke_fset = None


def test_escape_restores_the_call_exactly_without_touching_the_grid(sg):
    before = {k: (v.copy() if isinstance(v, np.ndarray) else v)
              for k, v in sg.annotations[0].items()}
    mops, pops = _arm(sg)
    sg.start_edit(0)
    assert sg.is_editing() and len(sg.annotations) == 2
    assert np.array_equal(np.asarray(sg.mask), _full_rebuild(sg))
    _paint(sg, 1020, 230)                       # a stroke outside the call
    _paint(sg, 1010, 205, 'eraser')             # and an erase inside it
    sg.cancel_edit()
    assert not sg.is_editing()
    back = next(a for a in sg.annotations if a['id'] == 'a1')
    for k, v in before.items():
        if isinstance(v, np.ndarray):
            assert np.array_equal(back[k], v)
        else:
            assert back[k] == v
    assert not np.asarray(sg._pending).any()    # nothing left half-drawn
    assert sg._pending_bbox is None and sg._pending_stack == []
    assert np.array_equal(np.asarray(sg.mask), _full_rebuild(sg))
    assert mops[0] == 0 and pops[0] == 0


def test_enter_commits_the_painted_pixels_exactly(sg):
    mops, pops = _arm(sg)
    sg.start_edit(0)
    _paint(sg, 1045, 215, r=3)                  # grow the call to the right
    _paint(sg, 1002, 202, 'eraser', r=1)        # nibble its corner
    want = np.asarray(sg._pending).copy()       # plain ndarray: not counted
    fs, ts = np.nonzero(want)
    got = []
    sg.annotation_edit_finished.connect(got.append)
    sg.finish_edit()
    assert got and not sg.is_editing()
    ann = sg.annotations[got[0]]
    assert (ann['f0'], ann['f1'], ann['t0'], ann['t1']) == (
        fs.min(), fs.max() + 1, ts.min(), ts.max() + 1)
    assert np.array_equal(ann['mask'],
                          want[ann['f0']:ann['f1'], ann['t0']:ann['t1']] > 0)
    assert not ann['mask'][5, 10]               # the hole survived the edit
    assert not np.asarray(sg._pending).any()
    assert np.array_equal(np.asarray(sg.mask), _full_rebuild(sg))
    assert mops[0] == 0 and pops[0] == 0


def test_erasing_the_call_to_nothing_cancels_and_keeps_the_call(sg):
    sg.start_edit(2)
    sg._pending[300:320, 15000:15040] = 0       # all of it erased
    sg.finish_edit()
    assert not sg.is_editing()
    ids = sorted(a['id'] for a in sg.annotations)
    assert ids == ['a1', 'a2', 'a3']            # it used to vanish here
    a3 = next(a for a in sg.annotations if a['id'] == 'a3')
    assert a3['mask'].all() and (a3['t0'], a3['t1']) == (15000, 15040)
    assert np.array_equal(np.asarray(sg.mask), _full_rebuild(sg))


def test_strokes_from_an_earlier_edit_do_not_follow_into_the_next(sg):
    sg.start_edit(2)
    _paint(sg, 15020, 325)
    sg.finish_edit()
    assert sg._pending_stack == []
    i = next(k for k, a in enumerate(sg.annotations) if a['id'] == 'a1')
    sg.start_edit(i)
    _paint(sg, 1020, 219)                       # touching the call
    comps = sg.pending_components()
    # One call, its crop near the edited call -- not a box reaching 14,000
    # frames back to the first edit's stroke.
    assert len(comps) == 1
    assert comps[0][3] - comps[0][2] < 100
    sg.cancel_edit()


def test_the_component_count_includes_the_loaded_call(sg):
    """With a stroke on the stack the crop came from the strokes alone, which
    cut the loaded call out of the count."""
    sg.start_edit(2)
    _paint(sg, 15100, 310)                      # a separate blob, 60 frames on
    comps = sg.pending_components()
    assert len(comps) == 2
    sg.cancel_edit()


def test_pending_bbox_scans_only_the_dirty_box(sg):
    _arm(sg)
    assert sg.pending_bbox() is None
    sg._pending[100:110, 500:520] = 1
    sg._note_pending_region(100, 110, 500, 520)
    assert sg.pending_bbox() == (100, 110, 500, 520)
    assert sg._pending.full_ops[0] == 0
    sg.clear_pending()
    assert sg._pending.full_ops[0] == 0
    assert sg.pending_bbox() is None and not np.asarray(sg._pending).any()

