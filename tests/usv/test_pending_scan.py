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
