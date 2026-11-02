"""Clicking a call's class label selects the call — and its harmonics.

A faint USV is a few pixels tall and hard to hit; its "USV" label sits right
above it in clear space and was inert. Making the label part of the call's
click target costs nothing and is usually the easier aim.

Clicking it selects the whole harmonic stack, because a fundamental and its
harmonics are one vocalisation — selecting the fundamental alone leaves the
rest of the same call unhighlighted.
"""
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import QPointF, QRectF  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def sg(qapp):
    from fnt.usv.mad_pyqt import MADSpectrogramWidget

    class W:
        label_at = MADSpectrogramWidget.label_at
        harmonic_group_of = MADSpectrogramWidget.harmonic_group_of

        def __init__(self):
            self.annotations = []
            self.harmonic_links = []
            self._label_hits = []

        def add(self, aid, rect=None):
            self.annotations.append({'id': aid, 'category': 'USV'})
            i = len(self.annotations) - 1
            if rect is not None:
                self._label_hits.append((rect, i))
            return i

    return W()


# ---------------------------------------------------------------- hit test
def test_a_click_inside_a_label_finds_its_call(sg):
    sg.add("a", QRectF(10, 10, 40, 14))
    assert sg.label_at(QPointF(20, 15)) == 0


def test_a_click_outside_every_label_finds_nothing(sg):
    sg.add("a", QRectF(10, 10, 40, 14))
    assert sg.label_at(QPointF(200, 200)) is None


def test_no_labels_drawn_yet_is_not_an_error(sg):
    assert sg.label_at(QPointF(5, 5)) is None


def test_the_topmost_label_wins_where_two_overlap(sg):
    """Later in the list is drawn later, so it is the one visible there."""
    sg.add("under", QRectF(10, 10, 40, 14))
    sg.add("over", QRectF(20, 10, 40, 14))
    assert sg.label_at(QPointF(30, 15)) == 1


def test_a_stale_rect_for_a_removed_call_is_ignored(sg):
    """Hits are rebuilt every paint, but never trust the index blindly."""
    sg.add("a", QRectF(10, 10, 40, 14))
    sg.annotations.clear()
    assert sg.label_at(QPointF(20, 15)) is None


# ------------------------------------------------------------ harmonics
def test_a_call_with_no_harmonics_selects_only_itself(sg):
    sg.add("a")
    assert sg.harmonic_group_of(0) == {0}


def test_clicking_a_fundamental_selects_its_harmonics(sg):
    sg.add("f")
    sg.add("h2")
    sg.harmonic_links = [("f", "h2", 2)]
    assert sg.harmonic_group_of(0) == {0, 1}


def test_clicking_a_harmonic_selects_the_fundamental_too(sg):
    """Whichever member is clicked, the stack comes back whole."""
    sg.add("f")
    sg.add("h2")
    sg.harmonic_links = [("f", "h2", 2)]
    assert sg.harmonic_group_of(1) == {0, 1}


def test_a_three_high_stack_is_walked_transitively(sg):
    sg.add("f")
    sg.add("h2")
    sg.add("h3")
    sg.harmonic_links = [("f", "h2", 2), ("h2", "h3", 3)]
    for member in (0, 1, 2):
        assert sg.harmonic_group_of(member) == {0, 1, 2}


def test_a_separate_stack_is_not_pulled_in(sg):
    sg.add("f1")
    sg.add("h1")
    sg.add("f2")
    sg.add("h2")
    sg.harmonic_links = [("f1", "h1", 2), ("f2", "h2", 2)]
    assert sg.harmonic_group_of(0) == {0, 1}
    assert sg.harmonic_group_of(2) == {2, 3}


def test_a_link_naming_a_call_that_is_gone_is_survivable(sg):
    """Links persist across reloads; the annotation may not."""
    sg.add("f")
    sg.harmonic_links = [("f", "vanished", 2)]
    assert sg.harmonic_group_of(0) == {0}


def test_an_out_of_range_index_returns_nothing(sg):
    sg.add("a")
    assert sg.harmonic_group_of(99) == set()


def test_ids_are_matched_as_strings(sg):
    """Link ids and annotation ids can differ in type across the store."""
    sg.annotations.append({'id': 7, 'category': 'USV'})
    sg.annotations.append({'id': 8, 'category': 'USV'})
    sg.harmonic_links = [("7", "8", 2)]
    assert sg.harmonic_group_of(0) == {0, 1}
