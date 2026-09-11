"""Scoring only where the user has actually judged the audio.

Whole-file scoring counts every detection in an unreviewed region as a false
positive. Those are unjudged, not wrong, and on a partially reviewed recording
they are most of what the model finds — so precision reads far below what the
model deserves, and reads worse the more unlabelled audio there is.

Restricting to judged spans also makes the eval cheap enough to run after every
training run, because the tiled forward pass only has to cover a few percent of
each recording.

Recall is the asymmetric case and gets its own treatment: truth is hand labels
plus accepted predictions, so a call the model never proposed and nobody drew
is invisible and missing it is free. Only a recording the user has declared
exhaustively reviewed can count a miss.
"""
import pytest

from fnt.usv.usv_detector.mad_eval import (
    Box, _merge_spans, expand_windows, overlaps_any, reviewed_windows,
    score_in_windows, windows_duration,
)


def _row(t0, t1, blob_id=1, status='pending', f0=40000.0, f1=60000.0,
         score=0.9):
    return {'blob_id': blob_id, 'status': status, 'start_s': t0, 'stop_s': t1,
            'min_freq_hz': f0, 'max_freq_hz': f1, 'score': score}


def _label_row(t0, t1, **kw):
    """A hand-drawn label: string blob_id."""
    return _row(t0, t1, blob_id='lbl_1', status='accepted', **kw)


# ------------------------------------------------------- judged spans
def test_an_accepted_prediction_makes_its_span_judged():
    assert reviewed_windows([_row(1.0, 1.2, status='accepted')]) == [(1.0, 1.2)]


def test_a_hand_label_makes_its_span_judged():
    assert reviewed_windows([_label_row(2.0, 2.3)]) == [(2.0, 2.3)]


def test_a_rejection_also_makes_its_span_judged():
    """A rejection is a recorded 'no call here' — the evidence a false
    positive has to be measured against."""
    assert reviewed_windows([_row(5.0, 5.4, status='rejected')]) == [(5.0, 5.4)]


def test_a_pending_prediction_judges_nothing():
    """Nobody has said what it is."""
    assert reviewed_windows([_row(3.0, 3.5, status='pending')]) == []


def test_adjacent_spans_merge():
    got = reviewed_windows([_row(1.0, 1.2, status='accepted'),
                            _row(1.3, 1.5, status='accepted')],
                           merge_gap_s=0.5)
    assert got == [(1.0, 1.5)]


def test_distant_spans_stay_separate():
    got = reviewed_windows([_row(1.0, 1.2, status='accepted'),
                            _row(9.0, 9.2, status='accepted')],
                           merge_gap_s=0.5)
    assert got == [(1.0, 1.2), (9.0, 9.2)]


def test_overlapping_spans_become_one():
    assert _merge_spans([(1.0, 3.0), (2.0, 4.0)], 0.0) == [(1.0, 4.0)]


def test_a_contained_span_does_not_shrink_its_container():
    assert _merge_spans([(1.0, 9.0), (2.0, 3.0)], 0.0) == [(1.0, 9.0)]


def test_a_degenerate_span_is_dropped():
    assert reviewed_windows([_row(4.0, 4.0, status='accepted')]) == []


def test_no_rows_is_no_windows():
    assert reviewed_windows([]) == []


# ------------------------------------------------ inference windows
def test_context_is_added_for_inference_only():
    assert expand_windows([(10.0, 10.2)], context_s=1.0) == [(9.0, 11.2)]


def test_context_is_clamped_to_the_recording():
    lo = expand_windows([(0.1, 0.2)], context_s=1.0, duration_s=5.0)
    assert lo[0][0] == 0.0 and lo[0][1] == pytest.approx(1.2)
    hi = expand_windows([(4.9, 5.0)], context_s=1.0, duration_s=5.0)
    assert hi[0][0] == pytest.approx(3.9) and hi[0][1] == 5.0


def test_context_can_merge_windows_that_scoring_keeps_apart():
    """The model sees one stretch; scoring still uses the two exact spans."""
    judged = [(10.0, 10.2), (11.0, 11.2)]
    assert len(expand_windows(judged, context_s=1.0)) == 1
    assert len(judged) == 2


def test_scored_duration_is_the_unpadded_total():
    assert windows_duration([(1.0, 1.5), (3.0, 3.25)]) == pytest.approx(0.75)


# --------------------------------------------------------- scoping
def test_a_detection_outside_every_judged_span_is_not_counted():
    """The reported problem: unreviewed regions dominating precision."""
    labels = [Box(1.0, 1.2, 40000, 60000)]
    preds = [Box(1.0, 1.2, 40000, 60000),      # matches the label
             Box(50.0, 50.2, 40000, 60000)]    # in unreviewed audio
    c, _ = score_in_windows(preds, labels, [(1.0, 1.2)])
    assert (c.tp, c.fp, c.fn) == (1, 0, 0)
    assert c.precision == 1.0


def test_whole_file_scoring_would_have_called_it_a_false_positive():
    """States the old behaviour, so the difference is unmistakable."""
    labels = [Box(1.0, 1.2, 40000, 60000)]
    preds = [Box(1.0, 1.2, 40000, 60000), Box(50.0, 50.2, 40000, 60000)]
    c, _ = score_in_windows(preds, labels, None)
    assert (c.tp, c.fp) == (1, 1)
    assert c.precision == 0.5


def test_a_detection_inside_a_judged_span_still_counts_against_you():
    """Scoping must not launder real errors away."""
    labels = [Box(1.0, 1.2, 40000, 60000)]
    preds = [Box(1.0, 1.2, 40000, 60000),
             Box(1.05, 1.15, 90000, 110000)]   # same time, wrong band
    c, _ = score_in_windows(preds, labels, [(1.0, 1.2)])
    assert (c.tp, c.fp) == (1, 1)


def test_a_detection_straddling_the_edge_counts():
    """The user would have been shown it."""
    assert overlaps_any(Box(0.9, 1.05, 0, 1), [(1.0, 1.2)]) is True


def test_a_detection_touching_only_the_boundary_does_not():
    assert overlaps_any(Box(0.5, 1.0, 0, 1), [(1.0, 1.2)]) is False


def test_a_missed_label_inside_a_judged_span_is_a_false_negative():
    labels = [Box(1.0, 1.2, 40000, 60000)]
    c, _ = score_in_windows([], labels, [(1.0, 1.2)])
    assert (c.tp, c.fp, c.fn) == (0, 0, 1)


def test_windows_none_scores_everything():
    """What an exhaustively reviewed recording wants."""
    labels = [Box(1.0, 1.2, 0, 1), Box(50.0, 50.2, 0, 1)]
    c, _ = score_in_windows([], labels, None)
    assert c.fn == 2


# ------------------------------------------------- exhaustive flag
def test_the_flag_round_trips(tmp_path):
    h5py = pytest.importorskip("h5py")
    from fnt.usv.usv_detector.fnt_mask_store import (
        is_review_complete, review_complete_at, set_review_complete)
    p = str(tmp_path / "rec_FNT.mad")
    assert is_review_complete(p) is False        # absent file, no crash
    set_review_complete(p, True)
    assert is_review_complete(p) is True
    assert review_complete_at(p)


def test_the_flag_can_be_withdrawn(tmp_path):
    pytest.importorskip("h5py")
    from fnt.usv.usv_detector.fnt_mask_store import (
        is_review_complete, review_complete_at, set_review_complete)
    p = str(tmp_path / "rec_FNT.mad")
    set_review_complete(p, True)
    set_review_complete(p, False)
    assert is_review_complete(p) is False
    # No date left claiming a completeness that was just withdrawn.
    assert review_complete_at(p) == ""


def test_an_unflagged_recording_reads_false(tmp_path):
    pytest.importorskip("h5py")
    import h5py
    from fnt.usv.usv_detector.fnt_mask_store import is_review_complete
    p = str(tmp_path / "rec_FNT.mad")
    with h5py.File(p, "w"):
        pass
    assert is_review_complete(p) is False
