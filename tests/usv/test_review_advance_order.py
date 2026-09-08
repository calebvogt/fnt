"""Rejecting a call must not drag the cursor to wherever the row re-sorts to.

With the Detections list sorted by Status, deciding a call moves its row into
the accepted or rejected block. The advance used to be expressed as "the next
pending after *this id*", resolved against the list **after** the move — so
rejecting sent the cursor chasing the dismissed call down into the rejected
group, instead of carrying on through the pending ones.

The fix is to remember the neighbour before the decision is applied. "The row
after the one I was on" means the same thing under every sort order, and does
not depend on where the decided row ends up.
"""
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(qapp):
    from fnt.usv.mad_pyqt import MADMainWindow

    class Spec:
        def __init__(self):
            self.annotations = []
            self._selected_ann_idx = None

        def _score_hidden(self, ann):
            return False

    class W:
        """Annotations plus a display order we can reorder at will."""
        _successor_id = MADMainWindow._successor_id
        _select_next_pending_from_id = MADMainWindow._select_next_pending_from_id
        _select_next_pending_after_id = MADMainWindow._select_next_pending_after_id

        def __init__(self):
            self.spectrogram = Spec()
            self.display = []          # annotation indices, in display order

        # Stand in for the tree: whatever we say the display order is.
        def _review_order(self):
            return list(self.display)

        def _select_review_pos(self, pos):
            self.spectrogram._selected_ann_idx = self.display[pos]

        def add(self, aid, status='prediction'):
            self.spectrogram.annotations.append({'id': aid, 'status': status})
            self.display.append(len(self.spectrogram.annotations) - 1)

        def sort_by_status(self):
            rank = {'prediction': 0, 'accepted': 1, 'rejected': 2}
            anns = self.spectrogram.annotations
            self.display.sort(key=lambda i: rank.get(anns[i]['status'], 1))

        def selected_id(self):
            i = self.spectrogram._selected_ann_idx
            return None if i is None else self.spectrogram.annotations[i]['id']

    return W()


def test_the_neighbour_is_captured_before_the_row_moves(win):
    for i in range(4):
        win.add(f"p{i}")
    assert win._successor_id("p1") == "p2"


def test_the_last_row_has_no_successor(win):
    win.add("a")
    win.add("b")
    assert win._successor_id("b") is None


def test_rejecting_under_a_status_sort_advances_to_the_next_pending(win):
    """The reported bug, end to end."""
    for i in range(5):
        win.add(f"p{i}")
    successor = win._successor_id("p1")          # captured BEFORE the change
    win.spectrogram.annotations[1]['status'] = 'rejected'
    win.sort_by_status()                          # p1 drops to the bottom
    assert win._select_next_pending_from_id(successor) is True
    assert win.selected_id() == "p2"              # not p1, not the rejected block


def test_the_old_behaviour_would_have_jumped(win):
    """Documents what went wrong, so the regression is unmistakable."""
    for i in range(5):
        win.add(f"p{i}")
    win.spectrogram.annotations[1]['status'] = 'rejected'
    win.sort_by_status()
    # Resolving "next pending AFTER p1" against the re-sorted list finds
    # nothing after it, because p1 is now last.
    assert win._select_next_pending_after_id("p1") is False


def test_time_order_is_unaffected(win):
    """Under the default Time sort nothing moves, and the advance is the same."""
    for i in range(4):
        win.add(f"p{i}")
    successor = win._successor_id("p0")
    win.spectrogram.annotations[0]['status'] = 'rejected'
    assert win._select_next_pending_from_id(successor) is True
    assert win.selected_id() == "p1"


def test_it_skips_over_already_decided_neighbours(win):
    """The successor may itself be accepted; the cursor wants the next pending."""
    win.add("p0")
    win.add("done", status='accepted')
    win.add("p2")
    successor = win._successor_id("p0")
    assert successor == "done"
    assert win._select_next_pending_from_id(successor) is True
    assert win.selected_id() == "p2"


def test_nothing_pending_after_it_reports_failure(win):
    """So the caller can leave the selection where it is."""
    win.add("p0")
    win.add("done", status='accepted')
    successor = win._successor_id("p0")
    assert win._select_next_pending_from_id(successor) is False


def test_a_missing_successor_reports_failure(win):
    win.add("p0")
    assert win._select_next_pending_from_id(None) is False
    assert win._select_next_pending_from_id("gone") is False


def test_the_successor_is_itself_the_first_candidate(win):
    """'At or after', not 'after' — the neighbour is where review resumes."""
    win.add("p0")
    win.add("p1")
    win.add("p2")
    successor = win._successor_id("p0")
    assert win._select_next_pending_from_id(successor) is True
    assert win.selected_id() == "p1"
