"""Deleting a mask in the gallery must correct the Detections list at once.

Reported from real use: ten calls in the gallery, ten in the Detections list,
delete one in the gallery — still ten in the list, still marked confirmed. The
store was right (the example really was gone); the on-screen list was not.

The old refresh called ``_load_current_file()``, an asynchronous multi-second
re-read of the audio and its spectrogram. Until that landed the list still
showed the deleted call, which is indistinguishable from the delete having done
nothing — and reloading a ten-minute recording to remove one row is the wrong
mechanism regardless. A confirmed call carries its example's id, so the row can
be removed directly.
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
            self.rebuilt = 0

        def remove_annotation(self, idx):
            return self.annotations.pop(idx)

        def _rebuild_confirmed_mask(self):
            self.rebuilt += 1

        def update(self):
            pass

    class W:
        drop_annotations_by_example_id = (
            MADMainWindow.drop_annotations_by_example_id)

        def __init__(self):
            self.spectrogram = Spec()
            self.refreshed = 0
            self.bumped = 0

        def _refresh_annotation_list(self):
            self.refreshed += 1

        def _bump_review_token(self):
            self.bumped += 1

        def add(self, *ids):
            for i in ids:
                self.spectrogram.annotations.append(
                    {'id': i, 'status': 'accepted'})

    return W()


def test_deleting_one_removes_exactly_one_row(win):
    """The reported case: ten in, delete one, nine left."""
    win.add(*[f"ex{i}" for i in range(10)])
    assert win.drop_annotations_by_example_id(["ex4"]) == 1
    ids = [a['id'] for a in win.spectrogram.annotations]
    assert len(ids) == 9
    assert "ex4" not in ids


def test_the_list_is_refreshed_so_the_change_is_visible(win):
    win.add("a", "b")
    win.drop_annotations_by_example_id(["a"])
    assert win.refreshed == 1
    assert win.spectrogram.rebuilt == 1


def test_several_ids_at_once(win):
    win.add(*[f"ex{i}" for i in range(6)])
    assert win.drop_annotations_by_example_id(["ex1", "ex3", "ex5"]) == 3
    assert [a['id'] for a in win.spectrogram.annotations] == [
        "ex0", "ex2", "ex4"]


def test_removal_is_back_to_front_so_indices_stay_valid(win):
    """Popping front-first would shift the later indices and delete wrong rows."""
    win.add(*[f"ex{i}" for i in range(5)])
    win.drop_annotations_by_example_id(["ex0", "ex1", "ex2", "ex3", "ex4"])
    assert win.spectrogram.annotations == []


def test_an_unknown_id_removes_nothing(win):
    win.add("a", "b")
    assert win.drop_annotations_by_example_id(["nope"]) == 0
    assert len(win.spectrogram.annotations) == 2
    assert win.refreshed == 0          # nothing changed, nothing rebuilt


def test_no_ids_is_a_no_op(win):
    win.add("a")
    assert win.drop_annotations_by_example_id([]) == 0
    assert win.drop_annotations_by_example_id([None]) == 0
    assert len(win.spectrogram.annotations) == 1


def test_the_selection_is_cleared_so_it_cannot_dangle(win):
    """A stale index would point at a different call after the removal."""
    win.add("a", "b", "c")
    win.spectrogram._selected_ann_idx = 2
    win.drop_annotations_by_example_id(["a"])
    assert win.spectrogram._selected_ann_idx is None


def test_ids_are_compared_as_strings(win):
    """Example ids are strings on disk but may arrive as other types."""
    win.spectrogram.annotations.append({'id': 42, 'status': 'accepted'})
    assert win.drop_annotations_by_example_id(["42"]) == 1
