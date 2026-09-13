"""Skip wraps to the first pending detection at the end of the list.

Skipping defers a decision, so the calls you skipped past are exactly the ones
you still owe. Stopping dead at the last row left the key doing nothing while
work was outstanding, with no way back to it but the mouse.

Accept/Reject deliberately does NOT wrap: that advance has to be able to
finish, or review would never end and "all reviewed" would never fire.
"""
import inspect

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from fnt.usv.mad_pyqt import MADMainWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(qapp):
    class Spec:
        def __init__(self):
            self.annotations = []
            self._selected_ann_idx = None

    class W:
        _select_first_pending = MADMainWindow._select_first_pending
        _select_next_pending_after_id = (
            MADMainWindow._select_next_pending_after_id)

        def __init__(self):
            self.spectrogram = Spec()
            self.display = []

        def _review_order(self):
            return list(self.display)

        def _select_review_pos(self, pos):
            self.spectrogram._selected_ann_idx = self.display[pos]

        def add(self, aid, status='prediction'):
            self.spectrogram.annotations.append({'id': aid, 'status': status})
            self.display.append(len(self.spectrogram.annotations) - 1)

        def selected_id(self):
            i = self.spectrogram._selected_ann_idx
            return None if i is None else self.spectrogram.annotations[i]['id']

    return W()


# ------------------------------------------------------------ wrapping
def test_skipping_the_last_pending_goes_back_to_the_first(win):
    """The reported behaviour: the key used to do nothing here."""
    for i in range(3):
        win.add(f"p{i}")
    assert win._select_next_pending_after_id("p2") is False   # nothing after
    assert win._select_first_pending() is True
    assert win.selected_id() == "p0"


def test_the_wrap_skips_over_decided_rows(win):
    win.add("done", status='accepted')
    win.add("gone", status='rejected')
    win.add("p1")
    assert win._select_first_pending() is True
    assert win.selected_id() == "p1"


def test_nothing_pending_does_not_wrap(win):
    """With no work left there is nowhere to wrap to, and the caller says so."""
    win.add("a", status='accepted')
    win.add("b", status='rejected')
    assert win._select_first_pending() is False
    assert win.selected_id() is None


def test_an_empty_file_is_survivable(win):
    assert win._select_first_pending() is False


def test_a_single_pending_call_wraps_to_itself(win):
    """One left: skipping should land back on it, not go silent."""
    win.add("only")
    assert win._select_next_pending_after_id("only") is False
    assert win._select_first_pending() is True
    assert win.selected_id() == "only"


def test_the_wrap_follows_the_display_order_not_the_annotation_order(win):
    """Sorting the list by Score must change where 'first' is."""
    win.add("a")
    win.add("b")
    win.add("c")
    win.display = [2, 0, 1]          # user sorted the table
    assert win._select_first_pending() is True
    assert win.selected_id() == "c"


# --------------------------------------------------- not the other advance
def test_accept_and_reject_still_stop_at_the_end():
    """Wrapping that advance would make review unfinishable."""
    src = inspect.getsource(MADMainWindow._after_review_decision)
    assert "_select_first_pending" not in src


def test_only_skip_wraps():
    src = inspect.getsource(MADMainWindow._skip_current_pred)
    assert "_select_first_pending" in src
    assert "Back to the first pending detection" in src


def test_the_dead_end_message_is_gone():
    """It reported a state the user can now always get out of."""
    src = inspect.getsource(MADMainWindow._skip_current_pred)
    assert "No more pending predictions after this" not in src
