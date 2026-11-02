"""Accept/Reject cycles back to pending calls left earlier in the list.

Reviewing a file is a sweep, and Skip is how you defer a call you are not sure
about. Those deferred calls sit *behind* the cursor, so an advance that only
ever moves forward runs off the end of the list and parks on a decided row with
work still outstanding -- the only way back being a manual click, on a file
whose badge still says pending. Auto-advance now wraps to the first call still
pending, and keeps cycling until the file is genuinely clear.

Wrapping is safe here in a way it would not be for Skip's cursor alone: every
Accept/Reject/Delete takes one call *out* of pending, so the cycle strictly
shrinks and "all reviewed" still fires. test_the_cycle_always_terminates is the
test that holds that guarantee.
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
    """A window stripped to what the advance actually touches."""

    class Spec:
        def __init__(self):
            self.annotations = []
            self._selected_ann_idx = None

    class Bar:
        def __init__(self):
            self.messages = []

        def showMessage(self, msg, *a):
            self.messages.append(msg)

        def last(self):
            return self.messages[-1] if self.messages else ""

    class W:
        # The code under test, verbatim.
        _after_review_decision = MADMainWindow._after_review_decision
        _successor_id = MADMainWindow._successor_id
        _select_first_pending = MADMainWindow._select_first_pending
        _select_next_pending_from_id = (
            MADMainWindow._select_next_pending_from_id)
        _select_next_pending_after_id = (
            MADMainWindow._select_next_pending_after_id)
        _reselect_by_id = MADMainWindow._reselect_by_id

        def __init__(self):
            self.spectrogram = Spec()
            self.status_bar = Bar()
            self.display = []
            self._auto_advance = True
            self._reviewed_count = 0
            self.prompted = 0
            self.rebuilds = 0

        # -- stubs for everything the tail calls on its way past -----------
        def _mark(self, *a):
            pass

        def _invalidate_label_count(self, *a):
            pass

        def _active_review_wav_path(self):
            return "r.wav"

        def _touch_annotation_rows(self, ids):
            return True                     # fast path; no rebuild needed

        def _refresh_annotation_list(self):
            self.rebuilds += 1

        def _touch_current_file_badge(self):
            pass

        def _update_view_header(self):
            pass

        def _maybe_prompt_next_file(self):
            self.prompted += 1

        # -- the bits the real window computes from its widgets ------------
        def _review_order(self):
            return list(self.display)

        def _select_review_pos(self, pos):
            self.spectrogram._selected_ann_idx = self.display[pos]

        def _center_and_select_ann(self, ann_idx):
            self.spectrogram._selected_ann_idx = ann_idx

        def _pred_indices(self):
            return [i for i in self.display
                    if self.spectrogram.annotations[i]['status'] == 'prediction']

        # -- test helpers ---------------------------------------------------
        def add(self, aid, status='prediction'):
            self.spectrogram.annotations.append({'id': aid, 'status': status})
            self.display.append(len(self.spectrogram.annotations) - 1)

        def selected_id(self):
            i = self.spectrogram._selected_ann_idx
            return None if i is None else self.spectrogram.annotations[i]['id']

        def select(self, aid):
            for i, a in enumerate(self.spectrogram.annotations):
                if a['id'] == aid:
                    self.spectrogram._selected_ann_idx = i
                    return
            raise AssertionError(aid + " is not in the list")

        def decide(self, aid, status='accepted'):
            """Accept/Reject `aid` the way the real handlers do: write the new
            status, then run the shared tail."""
            for a in self.spectrogram.annotations:
                if a['id'] == aid:
                    was_pending = a['status'] == 'prediction'
                    a['status'] = status
                    self._after_review_decision(aid, was_pending)
                    return
            raise AssertionError(aid + " is not in the list")

    return W()


# --------------------------------------------------------------- the ask
def test_deciding_the_last_pending_goes_back_to_one_skipped_earlier(win):
    """The reported behaviour: p0 was skipped, p2 is the last row, and the
    sweep used to stop there with p0 still owed."""
    for i in range(3):
        win.add(f"p{i}")
    win.decide("p1")                      # p0 skipped, cursor moved to p2
    assert win.selected_id() == "p2"
    win.decide("p2")                      # nothing after it
    assert win.selected_id() == "p0", "the skipped call was abandoned"


def test_the_wrap_says_so_and_counts_what_is_left(win):
    for i in range(3):
        win.add(f"p{i}")
    win.decide("p1")
    win.decide("p2")
    msg = win.status_bar.last()
    assert "Back to the first" in msg and "1 pending" in msg, msg


def test_it_keeps_cycling_until_the_file_is_clear(win):
    """Two skipped, two passes: the second wrap has to happen as well."""
    for i in range(4):
        win.add(f"p{i}")
    win.decide("p2")
    win.decide("p3")
    assert win.selected_id() == "p0"      # first wrap
    win.decide("p0")
    assert win.selected_id() == "p1"      # forward again
    win.decide("p1")
    assert not win._pred_indices()


def test_a_pending_call_still_ahead_is_not_skipped_over(win):
    """The wrap must be a last resort, not a reset: forward first."""
    for i in range(3):
        win.add(f"p{i}")
    win.select("p0")
    win.decide("p0")
    assert win.selected_id() == "p1"
    assert "Back to the first" not in win.status_bar.last()


def test_the_wrap_steps_over_decided_rows(win):
    win.add("old", status='rejected')
    win.add("p0")
    win.add("p1")
    win.decide("p1")
    assert win.selected_id() == "p0"


def test_deciding_the_very_last_pending_call_stays_put_and_prompts(win):
    """Nothing to wrap to: the file is done, and that is when the
    next-file prompt is allowed to fire."""
    win.add("p0")
    win.decide("p0")
    assert win.selected_id() == "p0", "moved somewhere with nothing pending"
    assert win.prompted == 1
    assert "All predictions reviewed" in win.status_bar.messages


def test_the_cycle_always_terminates(win):
    """The safety property the wrap rests on. Every decision removes a call
    from pending, so cycling can never spin: 12 calls, 12 decisions, done.

    Guards the difference from Skip, which changes no status and so could
    cycle forever if it were ever driven automatically.
    """
    for i in range(12):
        win.add(f"p{i}")
    win.select("p7")                      # start mid-list, as a user would
    seen = []
    for _ in range(12):
        pending = win._pred_indices()
        assert pending, "ran out of work before the loop was done"
        aid = win.selected_id()
        if win.spectrogram.annotations[
                win.spectrogram._selected_ann_idx]['status'] != 'prediction':
            aid = win.spectrogram.annotations[pending[0]]['id']
        seen.append(aid)
        win.decide(aid, 'accepted' if len(seen) % 2 else 'rejected')
    assert sorted(seen) == sorted(f"p{i}" for i in range(12))
    assert not win._pred_indices()
    assert win.prompted == 1


# ------------------------------------------------------- what must not move
def test_auto_advance_off_still_stays_on_the_decided_call(win):
    """Off means 'don't move me'. Wrapping would be a surprise jump."""
    for i in range(3):
        win.add(f"p{i}")
    win._auto_advance = False
    win.select("p2")
    win.decide("p2")
    assert win.selected_id() == "p2"
    assert "Back to the first" not in win.status_bar.last()


def test_re_deciding_a_settled_call_does_not_wrap(win):
    """Changing your mind about an accepted call is a deliberate visit -- the
    selection stays on it so the change is visible."""
    win.add("p0")
    win.add("done", status='accepted')
    win.select("done")
    win.decide("done", 'rejected')        # was_pending is False
    assert win.selected_id() == "done"
    assert win._reviewed_count == 0


def test_the_wrap_follows_the_display_order(win):
    """Sorted by Score, 'first pending' is the first pending *on screen*."""
    win.add("a")
    win.add("b")
    win.add("c")
    win.display = [2, 0, 1]               # user sorted the table
    win.decide("b")                       # last row in this order
    assert win.selected_id() == "c"


def test_reviewed_count_only_grows_on_real_decisions(win):
    """It drives the 'all reviewed' prompt; cycling must not inflate it."""
    for i in range(3):
        win.add(f"p{i}")
    win.decide("p1")
    win.decide("p2")
    win.decide("p0")
    assert win._reviewed_count == 3


# ------------------------------------------------------------ delete too
def test_delete_cycles_back_as_well():
    """D is the third key of the same sweep. Source-level only: the real
    handler writes CSVs, crops and undo snapshots, but a silent removal of the
    wrap would leave D inconsistent with A and R."""
    src = inspect.getsource(MADMainWindow._delete_selected_annotation)
    assert "_select_first_pending" in src
