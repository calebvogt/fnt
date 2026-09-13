"""Notes and tags on a call.

Motivating case: a USV appearing right after a high-amplitude noise event,
noticed twice. The observation is worth keeping, and worth finding again — a
note you cannot retrieve is a note that was never taken, which is why tags and
a "Has note" filter are part of this and not a later idea.

Restricted to ACCEPTED and REJECTED calls on purpose. A pending detection is
replaced wholesale by the next inference run, so a note on one would either
vanish or would have to pin the detection down and stop it being overwritten —
and pending detections are meant to be disposable. Accepted and rejected calls
are stored as examples, which already survive a re-run, so nothing special is
needed to keep a note attached.
"""
import numpy as np
import pytest

pytest.importorskip("PyQt5")
h5py = pytest.importorskip("h5py")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from fnt.usv.mad_pyqt import MADMainWindow, _status_icon  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


# --------------------------------------------------- who may be noted
@pytest.mark.parametrize("status,ok", [
    ('accepted', True), ('rejected', True), (None, True),
    ('prediction', False),
])
def test_only_judged_calls_can_carry_a_note(status, ok):
    assert MADMainWindow._is_notable({'status': status}) is ok


def test_a_pending_call_is_refused_even_with_an_id(qapp, tmp_path):
    """The whole reason for the restriction: the next run overwrites it."""
    win = _win(tmp_path)
    ann = {'id': 'ex1', 'status': 'prediction'}
    assert win._write_call_annotation(ann, note="hello") is False
    assert 'note' not in ann


# ------------------------------------------------------ tag spelling
@pytest.mark.parametrize("raw,want", [
    ("post-noise", "#post-noise"),
    ("#post-noise", "#post-noise"),
    ("post noise response", "#post-noise-response"),
    ("  review  ", "#review"),
    ("###weird", "#weird"),
    ("has spaces and!punct", "#has-spaces-andpunct"),
    ("", ""),
    ("#", ""),
    ("   ", ""),
])
def test_tags_get_one_canonical_spelling(raw, want):
    """Otherwise '#post-noise' and '#postnoise' quietly become two things."""
    assert MADMainWindow._normalize_tag(raw) == want


# ----------------------------------------------------- store round trip
def _win(tmp_path, project=None):
    store = str(tmp_path / "training_data.h5")

    class W:
        # staticmethod() matters: a plain assignment would rebind these as
        # instance methods and silently pass `self` as the first argument.
        _is_notable = staticmethod(MADMainWindow._is_notable)
        _write_call_annotation = MADMainWindow._write_call_annotation
        _normalize_tag = staticmethod(MADMainWindow._normalize_tag)
        _known_tags = MADMainWindow._known_tags
        _remember_tag = MADMainWindow._remember_tag

        def __init__(self):
            self._project = project
            self.store = store

        def _active_review_wav_path(self):
            return "rec.wav"

        def _example_store_paths(self, wav=None):
            return [self.store]

    return W()


def _seed_example(store, eid="ex1"):
    from fnt.usv.usv_detector.fnt_mask_store import td_save_example
    spec = np.zeros((8, 8), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    td_save_example(store, spec, mask,
                    {'id': eid, 'class': 'USV', 'source_wav': 'rec.wav',
                     'kind': 'label'}, example_id=eid)


def test_a_note_round_trips_through_the_mad(qapp, tmp_path):
    from fnt.usv.usv_detector.fnt_mask_store import td_iter_meta
    win = _win(tmp_path)
    _seed_example(win.store)
    ann = {'id': 'ex1', 'status': 'accepted'}
    assert win._write_call_annotation(
        ann, note="follows a loud broadband event") is True
    metas = {m['id']: m for m in td_iter_meta(win.store)}
    assert metas['ex1']['note'] == "follows a loud broadband event"
    assert ann['note'] == "follows a loud broadband event"


def test_tags_round_trip_as_a_list(qapp, tmp_path):
    from fnt.usv.usv_detector.fnt_mask_store import td_iter_meta
    win = _win(tmp_path)
    _seed_example(win.store)
    ann = {'id': 'ex1', 'status': 'accepted'}
    win._write_call_annotation(ann, tags=["#post-noise", "#review"])
    metas = {m['id']: m for m in td_iter_meta(win.store)}
    assert metas['ex1']['tags'] == ["#post-noise", "#review"]


def test_a_note_does_not_disturb_the_mask(qapp, tmp_path):
    """td_update_meta must leave the pixels alone — the call is still a label."""
    from fnt.usv.usv_detector.fnt_mask_store import td_read_example
    win = _win(tmp_path)
    _seed_example(win.store)
    before = td_read_example(win.store, "ex1")['mask'].sum()
    win._write_call_annotation({'id': 'ex1', 'status': 'accepted'}, note="x")
    assert td_read_example(win.store, "ex1")['mask'].sum() == before


def test_an_empty_note_clears_rather_than_being_refused(qapp, tmp_path):
    from fnt.usv.usv_detector.fnt_mask_store import td_iter_meta
    win = _win(tmp_path)
    _seed_example(win.store)
    ann = {'id': 'ex1', 'status': 'accepted'}
    win._write_call_annotation(ann, note="something")
    win._write_call_annotation(ann, note="")
    metas = {m['id']: m for m in td_iter_meta(win.store)}
    assert metas['ex1']['note'] == ""


def test_a_missing_example_is_reported_not_crashed(qapp, tmp_path):
    win = _win(tmp_path)
    _seed_example(win.store)
    assert win._write_call_annotation(
        {'id': 'nope', 'status': 'accepted'}, note="x") is False


# ------------------------------------------------- the tag vocabulary
class _Proj:
    def __init__(self):
        self.tags = []
        self.saved = 0

    def save(self):
        self.saved += 1


def test_a_new_tag_is_remembered_in_the_project(qapp, tmp_path):
    """So it can be offered next time instead of retyped from memory."""
    proj = _Proj()
    win = _win(tmp_path, project=proj)
    win._remember_tag("post noise")
    assert proj.tags == ["#post-noise"]
    assert proj.saved == 1
    assert win._known_tags() == ["#post-noise"]


def test_remembering_the_same_tag_twice_does_not_duplicate(qapp, tmp_path):
    proj = _Proj()
    win = _win(tmp_path, project=proj)
    win._remember_tag("#review")
    win._remember_tag("review")
    assert proj.tags == ["#review"]


def test_an_empty_tag_is_not_remembered(qapp, tmp_path):
    proj = _Proj()
    win = _win(tmp_path, project=proj)
    win._remember_tag("   ")
    assert proj.tags == []


def test_without_a_project_tags_live_for_the_session(qapp, tmp_path):
    win = _win(tmp_path, project=None)
    win._remember_tag("scratch")
    assert win._known_tags() == ["#scratch"]


def test_the_project_carries_a_tags_field():
    from fnt.usv.usv_detector.mad_project import MADProjectConfig
    import dataclasses
    names = {f.name for f in dataclasses.fields(MADProjectConfig)}
    assert 'tags' in names


# ------------------------------------------------------- list marker
def test_a_noted_call_is_marked_in_the_detections_list():
    assert _status_icon({'note': 'x'}, False, False) == "● ✎"
    assert _status_icon({'tags': ['#a']}, False, False) == "● ✎"
    assert _status_icon({'note': '   '}, False, False) == "●"
    assert _status_icon({}, False, False) == "●"
    assert _status_icon({'note': 'x'}, False, True) == "✕ ✎"


def test_the_show_filter_offers_has_note():
    import inspect
    src = inspect.getsource(MADMainWindow)
    assert '"Has note"' in src
    # and the terminology fix came with it
    assert '"Accepted", "Rejected", "Has note"' in src


# --------------------------------------------------- canvas plumbing
def test_note_markers_hit_test_apart_from_labels(qapp):
    """Clicking the label selects the call and its harmonic stack; a note
    must not ride on that."""
    from PyQt5.QtCore import QPointF, QRectF
    from fnt.usv.mad_pyqt import MADSpectrogramWidget

    class W:
        note_marker_at = MADSpectrogramWidget.note_marker_at
        label_at = MADSpectrogramWidget.label_at

        def __init__(self):
            self.annotations = [{'id': 'a'}]
            self._note_hits = [(QRectF(50, 10, 12, 12), 0)]
            self._label_hits = [(QRectF(10, 10, 40, 12), 0)]

    w = W()
    assert w.note_marker_at(QPointF(55, 15)) == 0
    assert w.note_marker_at(QPointF(20, 15)) is None   # that is the label
    assert w.label_at(QPointF(20, 15)) == 0


def test_a_stale_note_hit_is_ignored(qapp):
    from PyQt5.QtCore import QPointF, QRectF
    from fnt.usv.mad_pyqt import MADSpectrogramWidget

    class W:
        note_marker_at = MADSpectrogramWidget.note_marker_at

        def __init__(self):
            self.annotations = []
            self._note_hits = [(QRectF(50, 10, 12, 12), 0)]

    assert W().note_marker_at(QPointF(55, 15)) is None


def test_notes_and_tags_reach_the_exported_rows():
    """A note nobody can get at from the data was never taken."""
    from fnt.usv.usv_detector.mad_csv_rebuild import _AUTHORITATIVE
    assert 'note' in _AUTHORITATIVE and 'tags' in _AUTHORITATIVE
