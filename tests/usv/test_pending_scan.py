"""Counting in-progress drawings must not scan the whole spectrogram grid.

``pending_components`` fell back to ``p.any(axis=1)`` whenever there was no
stroke history — which is the normal state during review, since nothing is
being drawn. So every accept/reject scanned the full-file pending buffer
(513 x 1.17M booleans on a 10-minute recording) to count drawings that were not
there. Measured at a flat ~322 ms per keystroke, constant with file duration
and independent of detection count, and the largest cost left in a decision
after the label-count cache landed.

The early return cannot simply be "no stroke history means nothing pending":
Edit Shape loads a call straight into the buffer without pushing a stroke. A
tracked dirty box covers both.
"""
import numpy as np
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def sg(qapp):
    from fnt.usv.mad_pyqt import MADSpectrogramWidget
    w = MADSpectrogramWidget()
    w.n_freq_bins, w.n_time_frames = 513, 20000
    w.mask = np.zeros((513, 20000), dtype=np.uint8)
    w._pending = np.zeros_like(w.mask)
    w._pending_bbox = None
    w._pending_stack = []
    return w


class _Tripwire:
    """A pending buffer that screams if anything scans it whole."""

    def __init__(self, real):
        self._real = real
        self.full_scans = 0

    def any(self, axis=None):
        self.full_scans += 1
        return self._real.any(axis=axis)

    def __getitem__(self, k):
        return self._real[k]

    def __setitem__(self, k, v):
        self._real[k] = v

    @property
    def shape(self):
        return self._real.shape


# --------------------------------------------------------- the fix
def test_an_untouched_buffer_is_not_scanned(sg):
    """The review case: nothing drawn, so nothing to look at."""
    sg._pending = _Tripwire(sg._pending)
    assert sg.pending_components() == []
    assert sg._pending.full_scans == 0


def test_has_pending_does_not_scan_either(sg):
    sg._pending = _Tripwire(sg._pending)
    assert sg.has_pending() is False
    assert sg._pending.full_scans == 0


# ------------------------------------------------- still correct
def test_a_painted_stroke_is_found(sg):
    fg, tg = np.mgrid[100:110, 500:520]
    fg, tg = fg.ravel(), tg.ravel()
    sg._pending[fg, tg] = 1
    sg._pending_stack = [(fg, tg)]          # the coords actually painted
    comps = sg.pending_components()
    assert len(comps) == 1
    f0, f1, t0, t1, local = comps[0]
    assert f0 == 100 and f1 == 110
    assert local.sum() == 10 * 20


def test_edit_shape_is_found_without_any_stroke_history(sg):
    """The case that makes a bare 'no strokes means nothing' wrong."""
    sg._pending[200:220, 900:940] = 1
    sg._pending_stack = []                      # Edit Shape pushes none
    sg._note_pending_region(200, 220, 900, 940)
    comps = sg.pending_components()
    assert len(comps) == 1
    assert comps[0][0] == 200 and comps[0][2] == 900


def test_two_separate_drawings_stay_separate(sg):
    sg._pending[100:110, 500:520] = 1
    sg._pending[300:310, 5000:5020] = 1
    sg._note_pending_region(100, 110, 500, 520)
    sg._note_pending_region(300, 310, 5000, 5020)
    assert len(sg.pending_components()) == 2


def test_the_box_grows_to_cover_everything_written(sg):
    sg._note_pending_region(100, 110, 500, 520)
    sg._note_pending_region(300, 310, 5000, 5020)
    assert sg._pending_bbox == (100, 310, 500, 5020)


def test_a_superset_box_is_harmless(sg):
    """Undo shrinks the content but not the box; only the crop widens."""
    sg._pending[100:110, 500:520] = 1
    sg._note_pending_region(0, 513, 0, 20000)   # deliberately the whole grid
    comps = sg.pending_components()
    assert len(comps) == 1
    assert comps[0][0] == 100                   # still the true extent


# ------------------------------------------------- box lifecycle
def test_clearing_resets_the_box(sg):
    sg._pending[100:110, 500:520] = 1
    sg._note_pending_region(100, 110, 500, 520)
    sg.clear_pending()
    assert sg._pending_bbox is None
    assert sg.pending_components() == []


def test_painting_records_the_box(sg):
    """Whatever route adds pixels must leave the box able to find them.

    The three coords are paired by fancy indexing, so they are a diagonal of
    isolated pixels — three components, not one. Worth stating: it is the
    box being right that matters here, not the component count.
    """
    fg = np.array([10, 11, 12])
    tg = np.array([70, 71, 72])
    sg._pending[fg, tg] = 1
    sg._note_pending_region(fg.min(), fg.max() + 1, tg.min(), tg.max() + 1)
    assert sg._pending_bbox == (10, 13, 70, 73)
    assert len(sg.pending_components()) == 3


def test_a_cleared_then_redrawn_buffer_finds_only_the_new_work(sg):
    sg._pending[100:110, 500:520] = 1
    sg._note_pending_region(100, 110, 500, 520)
    sg.clear_pending()
    sg._pending[400:410, 8000:8020] = 1
    sg._note_pending_region(400, 410, 8000, 8020)
    comps = sg.pending_components()
    assert len(comps) == 1
    assert comps[0][0] == 400


def test_no_pending_buffer_at_all_is_survivable(sg):
    sg._pending = None
    assert sg.pending_components() == []
    assert sg.has_pending() is False


# --------------------------------------- the brush must record its box
"""The dirty box is not an optimisation detail — Escape depends on it.

``_stamp`` writes brush pixels straight into ``_pending``. When it did not also
record the region, the box stayed empty, ``has_pending()`` reported nothing
painted, and Escape silently refused to clear a stroke plainly visible on
screen. Enter still worked, because ``pending_components`` falls back to the
stroke history — so the two disagreed about whether anything was drawn.
"""


def _stamp_at(sg, t_idx, f_idx, mode='brush'):
    sg.paint_mode = mode
    sg._stamp(t_idx, f_idx)


def test_a_brush_stroke_records_its_box(sg):
    sg.brush_radius_px = 3
    _stamp_at(sg, 500, 100)
    assert sg._pending_bbox is not None
    assert sg.has_pending() is True


def test_escape_can_clear_a_brush_stroke(sg):
    """The reported bug, end to end at the widget level."""
    sg.brush_radius_px = 3
    _stamp_at(sg, 500, 100)
    assert sg.has_pending() is True
    sg.clear_pending()
    assert sg.has_pending() is False
    assert sg._pending_bbox is None


def test_a_brush_stroke_is_found_as_a_component(sg):
    sg.brush_radius_px = 2
    _stamp_at(sg, 400, 60)
    assert len(sg.pending_components()) == 1


def test_the_eraser_does_not_need_to_grow_the_box(sg):
    """It only removes pixels; a box that is a superset stays correct."""
    sg.brush_radius_px = 3
    _stamp_at(sg, 500, 100)
    box = sg._pending_bbox
    _stamp_at(sg, 500, 100, mode='erase')
    assert sg._pending_bbox == box
    assert sg.has_pending() is False        # everything painted was erased


def test_two_strokes_grow_the_box_to_cover_both(sg):
    sg.brush_radius_px = 2
    _stamp_at(sg, 300, 50)
    _stamp_at(sg, 900, 200)
    f0, f1, t0, t1 = sg._pending_bbox
    assert f0 <= 48 and f1 >= 202
    assert t0 <= 298 and t1 >= 902
