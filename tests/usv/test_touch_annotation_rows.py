"""Restyling one detection row instead of rebuilding the whole list.

Accepting or rejecting changes one row, but the list was rebuilt whole each
time — 85 ms on a 2,508-detection file, per keystroke, on exactly the files
where a reviewer holds the key down.

The danger in a fast path like this is not that it is wrong on the case it was
written for; it is that it silently applies to a case it does not handle, and
leaves a row saying something untrue. So most of what follows is about when it
must REFUSE and let the full rebuild happen.
"""
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import Qt  # noqa: E402
from PyQt5.QtWidgets import QApplication, QTreeWidget, QTreeWidgetItem  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(qapp):
    """A stand-in exposing only what _touch_annotation_rows touches."""
    from fnt.usv.mad_pyqt import MADMainWindow

    class Spec:
        def __init__(self):
            self.annotations = []

        def pending_components(self):
            return []

    class Combo:
        def __init__(self, text="All"):
            self._t = text

        def currentText(self):
            return self._t

    class W:
        _touch_annotation_rows = MADMainWindow._touch_annotation_rows
        _refresh_annotation_counts = MADMainWindow._refresh_annotation_counts

        def __init__(self):
            self.spectrogram = Spec()
            self.annotation_list = QTreeWidget()
            self.annotation_list.setColumnCount(7)
            self.combo_det_filter = Combo()
            self._ann_row_index = {}
            self._min_score_value = 0.0
            self._n_score_hidden = 0
            self.lbl_annotation_count = None
            self.tail = 0

        def _min_score(self):
            return self._min_score_value

        def _overlay_palette(self):
            return {'pending': (255, 210, 60), 'rejected': (214, 69, 69),
                    'confirmed': (80, 220, 120), 'drawing': (255, 0, 255)}

        # The tail a real refresh runs; counted, not exercised.
        def _update_pred_review_widgets(self): self.tail += 1
        def _update_train_button_count(self): self.tail += 1
        def _update_file_list_counts(self): self.tail += 1
        def _update_overview_marks(self): self.tail += 1
        def _refresh_open_gallery(self): self.tail += 1

        def add(self, aid, status='prediction', score=0.9):
            self.spectrogram.annotations.append(
                {'id': aid, 'status': status, 'score': score,
                 'category': 'USV'})
            it = QTreeWidgetItem(["○", "1.00s", "USV", "12ms", "40-60",
                                  "180", f"{score:.2f}"])
            it.setData(0, Qt.UserRole, ("rec.wav", 1.0, aid, 'prediction'))
            self.annotation_list.addTopLevelItem(it)
            self._ann_row_index[aid] = it
            return it

    return W()


def test_rejecting_restyles_the_row_in_place(win):
    it = win.add("a")
    win.spectrogram.annotations[0]['status'] = 'rejected'
    assert win._touch_annotation_rows(["a"]) is True
    assert it.text(0) == "✕"
    assert it.text(2) == "Reject"
    assert it.text(6) == ""                      # score is the model's, not yours
    assert it.data(0, Qt.UserRole)[3] == 'rejected'


def test_accepting_restyles_the_row_in_place(win):
    it = win.add("a")
    win.spectrogram.annotations[0]['status'] = 'accepted'
    assert win._touch_annotation_rows(["a"]) is True
    assert it.text(0) == "●"
    assert it.text(2) == "USV"
    assert it.data(0, Qt.UserRole)[3] == 'confirmed'


def test_only_the_named_row_changes(win):
    a, b = win.add("a"), win.add("b")
    win.spectrogram.annotations[0]['status'] = 'rejected'
    assert win._touch_annotation_rows(["a"]) is True
    assert a.text(0) == "✕" and b.text(0) == "○"


def test_the_time_column_is_left_alone(win):
    """Accepting never moves a call, so its sort key must not shift."""
    it = win.add("a")
    win.spectrogram.annotations[0]['status'] = 'accepted'
    win._touch_annotation_rows(["a"])
    assert it.text(1) == "1.00s"
    assert it.data(0, Qt.UserRole)[1] == 1.0


def test_it_refuses_under_a_status_filter(win):
    """A decided row has to LEAVE a Pending list — that needs a rebuild."""
    win.add("a")
    win.combo_det_filter._t = "Pending"
    win.spectrogram.annotations[0]['status'] = 'rejected'
    assert win._touch_annotation_rows(["a"]) is False


@pytest.mark.parametrize("flt", ["Pending", "Confirmed", "Rejected"])
def test_it_refuses_under_every_status_filter(win, flt):
    win.add("a")
    win.combo_det_filter._t = flt
    assert win._touch_annotation_rows(["a"]) is False


def test_it_refuses_when_the_score_slider_is_engaged(win):
    """Deciding a call changes what the hidden tally means."""
    win.add("a")
    win._min_score_value = 0.5
    assert win._touch_annotation_rows(["a"]) is False


def test_it_refuses_for_an_unknown_id(win):
    win.add("a")
    assert win._touch_annotation_rows(["nope"]) is False


def test_it_refuses_when_the_annotation_is_gone(win):
    """Delete removes the call; the row must go too, so rebuild."""
    win.add("a")
    win.spectrogram.annotations.clear()
    assert win._touch_annotation_rows(["a"]) is False


def test_it_refuses_with_no_index_yet(win):
    win._ann_row_index = {}
    assert win._touch_annotation_rows(["a"]) is False


def test_a_none_id_is_skipped_not_fatal(win):
    win.add("a")
    win.spectrogram.annotations[0]['status'] = 'rejected'
    assert win._touch_annotation_rows([None, "a"]) is True


def test_several_rows_can_be_touched_at_once(win):
    a, b = win.add("a"), win.add("b")
    for ann in win.spectrogram.annotations:
        ann['status'] = 'rejected'
    assert win._touch_annotation_rows(["a", "b"]) is True
    assert a.text(0) == "✕" and b.text(0) == "✕"


def test_the_tail_still_runs_on_the_fast_path(win):
    """Skipping it would leave the Audio list badges and ticks stale."""
    win.add("a")
    win.spectrogram.annotations[0]['status'] = 'rejected'
    win._touch_annotation_rows(["a"])
    assert win.tail == 5


def test_a_stale_index_falls_back_instead_of_crashing(win):
    """Items deleted by a rebuild leave dangling C++ pointers."""
    win.add("a")
    win.annotation_list.clear()          # destroys the item behind the index
    win.spectrogram.annotations[0]['status'] = 'rejected'
    assert win._touch_annotation_rows(["a"]) is False
    assert win._ann_row_index == {}
