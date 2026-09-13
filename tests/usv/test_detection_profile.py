"""The run summary describes its OWN detections, and borrows nothing.

The rejected alternative was projecting a reject rate measured on PREVIOUS
models onto the freshly trained one. That assumes the two behave alike, which
is the one thing retraining is meant to change, and it errs in the flattering
direction exactly when training worked: a genuinely better model would be
credited with the old model's junk.

So every figure here is computed from the run's own output. Nothing in it has
been judged by a human, and nothing claims to know which detections are wrong.
"""
import pytest

from fnt.usv.usv_detector.mad_inference import (
    DUR_EDGES_MS, N_SCORE_BINS, SCORE_BIN, detection_profile,
)


def _row(score=0.9, dur_ms=30.0, t0=1.0):
    return {'score': score, 'start_s': t0, 'stop_s': t0 + dur_ms / 1000.0}


# ------------------------------------------------------- the counts
def test_a_detection_lands_in_its_score_bin():
    p = detection_profile([_row(score=0.93)])
    assert p['score_hist'][int(0.93 / SCORE_BIN)] == 1
    assert sum(p['score_hist']) == 1


def test_a_detection_lands_in_its_duration_bin():
    p = detection_profile([_row(dur_ms=30.0)])          # 20 <= 30 < 40
    assert p['dur_hist'][DUR_EDGES_MS.index(40.0)] == 1


def test_a_perfect_score_does_not_fall_off_the_end():
    p = detection_profile([_row(score=1.0)])
    assert p['score_hist'][N_SCORE_BINS - 1] == 1


def test_a_very_long_detection_lands_in_the_last_bin():
    p = detection_profile([_row(dur_ms=5000.0)])
    assert p['dur_hist'][-1] == 1


def test_counts_are_poolable_across_files():
    """The whole point of histograms over quantiles: a median of per-file
    medians is not a median, but these simply add."""
    a = detection_profile([_row(score=0.3), _row(score=0.9)])
    b = detection_profile([_row(score=0.9)])
    pooled = [x + y for x, y in zip(a['score_hist'], b['score_hist'])]
    direct = detection_profile([_row(score=0.3), _row(score=0.9),
                                _row(score=0.9)])['score_hist']
    assert pooled == direct


def test_no_detections_is_all_zeros():
    p = detection_profile([])
    assert p['n'] == 0 and sum(p['score_hist']) == 0


def test_a_malformed_row_does_not_crash_the_run():
    p = detection_profile([{'score': None, 'start_s': None, 'stop_s': 'x'}])
    assert p['n'] == 1


# ------------------------------------------------------ the reporting
@pytest.fixture
def win():
    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from fnt.usv.mad_pyqt import MADMainWindow

    class W:
        _detection_profile_lines = MADMainWindow._detection_profile_lines

    return W()


def _results(rows_per_file):
    return [{'det_stats': detection_profile(rows)} for rows in rows_per_file]


def test_nothing_is_reported_without_detections(win):
    assert win._detection_profile_lines(_results([[]]), {}) == []


def test_the_threshold_trade_is_reported(win):
    """The only lever left after a run: scores are stored, so the cutoff can be
    raised on existing detections but not lowered without re-running."""
    rows = [_row(score=0.55)] * 40 + [_row(score=0.95)] * 60
    out = " ".join(win._detection_profile_lines(_results([rows]),
                                               {'best_threshold': 0.5}))
    assert "Raising the threshold to 0.70" in out
    assert "60" in out and "60%" in out


def test_cutoffs_at_or_below_the_run_threshold_are_not_offered(win):
    """Lowering needs a re-run, so offering it would be a lie."""
    rows = [_row(score=0.95)] * 10
    out = " ".join(win._detection_profile_lines(_results([rows]),
                                               {'best_threshold': 0.8}))
    assert "to 0.70" not in out and "to 0.80" not in out
    assert "to 0.90" in out


def test_duration_outliers_are_surfaced(win):
    """A mass of 2 ms or 1 s detections is the shape of noise, and needs no
    judgement to see."""
    rows = ([_row(dur_ms=2.0)] * 30 + [_row(dur_ms=30.0)] * 60
            + [_row(dur_ms=400.0)] * 10)
    out = " ".join(win._detection_profile_lines(_results([rows]), {}))
    assert "under 5 ms" in out and "over 160 ms" in out


def test_no_outlier_line_when_durations_are_sane(win):
    rows = [_row(dur_ms=30.0)] * 50
    out = " ".join(win._detection_profile_lines(_results([rows]), {}))
    assert "Duration outliers" not in out


def test_concentration_is_flagged(win):
    """Output dominated by a few recordings is usually a few noisy
    recordings, not a productive model."""
    files = [[_row()] * 500] + [[_row()] * 2 for _ in range(19)]
    out = " ".join(win._detection_profile_lines(_results(files), {}))
    assert "Concentrated" in out


def test_an_even_spread_is_not_flagged(win):
    files = [[_row()] * 20 for _ in range(20)]
    out = " ".join(win._detection_profile_lines(_results(files), {}))
    assert "Concentrated" not in out


def test_empty_recordings_are_counted(win):
    files = [[_row()] * 5, [], []]
    out = " ".join(win._detection_profile_lines(_results(files), {}))
    assert "2 recording(s) produced no detections" in out


def test_nothing_claims_to_know_what_is_junk(win):
    """The line that was removed. Nothing here predicts the user's judgement."""
    rows = [_row(score=0.55)] * 40 + [_row(dur_ms=2.0)] * 10
    out = " ".join(win._detection_profile_lines(_results([rows]),
                                               {'best_threshold': 0.5})).lower()
    for word in ("junk", "expected to be", "reject rate", "false positive"):
        assert word not in out


def test_the_summary_no_longer_projects_a_past_reject_rate():
    import inspect
    from fnt.usv.mad_pyqt import MADMainWindow
    src = inspect.getsource(MADMainWindow._show_run_summary_dialog)
    code = [ln for ln in src.splitlines() if not ln.strip().startswith("#")]
    assert not any("_recent_reject_rate" in ln for ln in code)
