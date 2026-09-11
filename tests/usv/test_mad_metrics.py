"""Storing evaluations and reading them back as a round-over-round trend.

The decision this supports is "is another round of labelling worth it?", which
is a question about the difference between rounds. A single F1 cannot answer
it; a flat F1 across a round that added 60 labels can.
"""
import json
import os

import pytest

from fnt.usv.usv_detector import mad_metrics as MM
from fnt.usv.usv_detector.mad_eval import Counts, EvalResult


def _result(f1_by_thr=((0.5, 0.8),), n_labels=100, exhaustive=None, **kw):
    per = [{'threshold': t, 'precision': f1, 'recall': f1, 'f1': f1,
            'tp': 10, 'fp': 1, 'fn': 1, 'n_pred': 11}
           for t, f1 in f1_by_thr]
    r = EvalResult(thresholds=[t for t, _ in f1_by_thr], per_threshold=per,
                   n_labels=n_labels, n_files=3, scope='reviewed', **kw)
    if exhaustive is not None:
        r.per_threshold_exhaustive = [
            {'threshold': t, 'precision': v, 'recall': v, 'f1': v,
             'tp': 5, 'fp': 0, 'fn': 1, 'n_pred': 5}
            for t, v in exhaustive]
        r.n_exhaustive_files = 1
    return r


def _run(tmp_path, name, res=None):
    d = tmp_path / name
    d.mkdir()
    if res is not None:
        MM.save_eval(str(d), res)
    return d


# ------------------------------------------------------ persistence
def test_an_eval_round_trips(tmp_path):
    d = _run(tmp_path, "20260101_010101_unet_n=10", _result())
    back = MM.load_eval(str(d))
    assert back is not None
    assert back.n_labels == 100
    assert back.scope == 'reviewed'
    assert back.per_threshold[0]['f1'] == 0.8


def test_it_is_its_own_file_not_the_training_record(tmp_path):
    """Re-running an eval must not rewrite the account of what was trained."""
    d = _run(tmp_path, "20260101_010101_unet_n=10", _result())
    assert os.path.isfile(d / "eval.json")
    assert not os.path.exists(d / "training_summary.json")


def test_a_missing_eval_is_none_not_an_error(tmp_path):
    d = _run(tmp_path, "20260101_010101_unet_n=10")
    assert MM.load_eval(str(d)) is None


def test_a_corrupt_eval_is_none_not_an_error(tmp_path):
    d = _run(tmp_path, "20260101_010101_unet_n=10")
    (d / "eval.json").write_text("{not json", encoding="utf-8")
    assert MM.load_eval(str(d)) is None


def test_an_older_eval_json_still_loads(tmp_path):
    """The trend has to survive fields added after some runs were written."""
    d = _run(tmp_path, "20260101_010101_unet_n=10")
    (d / "eval.json").write_text(json.dumps({
        'model_name': 'weights', 'n_labels': 5, 'n_files': 1,
        'per_threshold': [{'threshold': 0.5, 'f1': 0.4, 'precision': 0.4,
                           'recall': 0.4}]}), encoding="utf-8")
    r = MM.load_eval(str(d))
    assert r is not None and r.n_labels == 5
    assert r.per_threshold_exhaustive == []       # defaulted, not crashed
    assert r.thresholds == [0.5]


# ---------------------------------------------------------- run names
def test_a_run_name_yields_the_label_count(tmp_path):
    assert MM.parse_run_name("20260908_203506_unet_n=161") == {
        'timestamp': '20260908_203506', 'arch': 'unet', 'n_labels': 161}


def test_a_hand_renamed_directory_is_skipped_not_fatal(tmp_path):
    assert MM.parse_run_name("my_best_model") == {}


def test_runs_are_ordered_by_name_not_mtime(tmp_path):
    """Re-running an eval touches the directory; the history must not reorder."""
    a = _run(tmp_path, "20260101_010101_unet_n=10", _result())
    b = _run(tmp_path, "20260202_020202_unet_n=20", _result())
    os.utime(a, None)                         # a is now the newest on disk
    assert [os.path.basename(p) for p in MM.model_run_dirs(str(tmp_path))] == \
        [a.name, b.name]


def test_non_run_directories_are_ignored(tmp_path):
    _run(tmp_path, "20260101_010101_unet_n=10", _result())
    (tmp_path / "training_data").mkdir()
    assert len(MM.model_run_dirs(str(tmp_path))) == 1


def test_a_missing_models_dir_is_empty(tmp_path):
    assert MM.model_run_dirs(str(tmp_path / "nope")) == []


# -------------------------------------------------------------- trend
def test_the_trend_is_one_row_per_evaluated_run(tmp_path):
    _run(tmp_path, "20260101_010101_unet_n=10", _result(n_labels=10))
    _run(tmp_path, "20260202_020202_unet_n=60", _result(n_labels=60))
    t = MM.eval_trend(str(tmp_path))
    assert [r['n_labels'] for r in t] == [10, 60]


def test_an_unevaluated_run_is_omitted_not_zeroed(tmp_path):
    """Never evaluated and scored zero are different claims."""
    _run(tmp_path, "20260101_010101_unet_n=10", _result())
    _run(tmp_path, "20260202_020202_unet_n=60")       # no eval.json
    assert len(MM.eval_trend(str(tmp_path))) == 1


def test_each_run_is_reported_at_its_own_best_threshold(tmp_path):
    """A fixed cutoff penalises whichever model is calibrated differently."""
    _run(tmp_path, "20260101_010101_unet_n=10",
         _result(f1_by_thr=((0.3, 0.5), (0.8, 0.9))))
    row = MM.eval_trend(str(tmp_path))[0]
    assert row['threshold'] == 0.8 and row['f1'] == 0.9


def test_the_delta_is_the_stop_signal(tmp_path):
    _run(tmp_path, "20260101_010101_unet_n=10", _result(f1_by_thr=((0.5, 0.60),)))
    _run(tmp_path, "20260202_020202_unet_n=60", _result(f1_by_thr=((0.5, 0.82),)))
    t = MM.eval_trend(str(tmp_path))
    assert MM.trend_delta(t) == pytest.approx(0.22)


def test_a_flat_round_shows_as_no_gain(tmp_path):
    """What 'stop labelling' actually looks like."""
    _run(tmp_path, "20260101_010101_unet_n=10", _result(f1_by_thr=((0.5, 0.81),)))
    _run(tmp_path, "20260202_020202_unet_n=90", _result(f1_by_thr=((0.5, 0.81),)))
    assert MM.trend_delta(MM.eval_trend(str(tmp_path))) == pytest.approx(0.0)


def test_one_run_has_no_delta(tmp_path):
    _run(tmp_path, "20260101_010101_unet_n=10", _result())
    assert MM.trend_delta(MM.eval_trend(str(tmp_path))) is None


def test_exhaustive_recall_is_carried_separately(tmp_path):
    _run(tmp_path, "20260101_010101_unet_n=10",
         _result(f1_by_thr=((0.5, 0.9),), exhaustive=((0.5, 0.6),)))
    row = MM.eval_trend(str(tmp_path))[0]
    # The optimistic number and the believable one are not the same field.
    assert row['recall'] == 0.9
    assert row['exhaustive_recall'] == 0.6
    assert row['n_exhaustive_files'] == 1


def test_no_exhaustive_files_leaves_the_field_absent(tmp_path):
    """Absent, not zero — nobody has declared a recording complete."""
    _run(tmp_path, "20260101_010101_unet_n=10", _result())
    assert 'exhaustive_recall' not in MM.eval_trend(str(tmp_path))[0]


# ----------------------------------------------- review outcome
def test_accept_and_reject_tally_per_model(monkeypatch, tmp_path):
    rows = [
        {'blob_id': 1, 'model_name': 'm1', 'status': 'accepted'},
        {'blob_id': 2, 'model_name': 'm1', 'status': 'rejected'},
        {'blob_id': 3, 'model_name': 'm1', 'status': 'rejected'},
        {'blob_id': 4, 'model_name': 'm1', 'status': 'pending'},
        {'blob_id': 'lbl_a', 'status': 'accepted'},      # hand-drawn
    ]
    import fnt.usv.usv_detector.mad_csv_rebuild as CR
    import fnt.usv.usv_detector.fnt_mask_store as MS
    monkeypatch.setattr(CR, 'rows_for_wav', lambda w: rows)
    monkeypatch.setattr(MS, 'masks_sibling_path', lambda w: str(tmp_path / "x"))
    (tmp_path / "x").write_text("", encoding="utf-8")

    out = MM.review_outcome_by_model(["a.wav"])
    assert out['m1']['accepted'] == 1
    assert out['m1']['rejected'] == 2
    assert out['m1']['pending'] == 1
    assert out['m1']['n_judged'] == 3
    assert out['m1']['reject_rate'] == pytest.approx(2 / 3)


def test_hand_labels_are_not_counted_as_proposals(monkeypatch, tmp_path):
    """The metric is 'of what the model proposed, how much was real'."""
    import fnt.usv.usv_detector.mad_csv_rebuild as CR
    import fnt.usv.usv_detector.fnt_mask_store as MS
    monkeypatch.setattr(CR, 'rows_for_wav',
                        lambda w: [{'blob_id': 'lbl_a', 'status': 'accepted'}])
    monkeypatch.setattr(MS, 'masks_sibling_path', lambda w: str(tmp_path / "x"))
    (tmp_path / "x").write_text("", encoding="utf-8")
    assert MM.review_outcome_by_model(["a.wav"]) == {}


def test_an_all_pending_model_has_no_rate(monkeypatch, tmp_path):
    """Unjudged is not the same as wrong."""
    import fnt.usv.usv_detector.mad_csv_rebuild as CR
    import fnt.usv.usv_detector.fnt_mask_store as MS
    monkeypatch.setattr(CR, 'rows_for_wav', lambda w: [
        {'blob_id': 1, 'model_name': 'm1', 'status': 'pending'}])
    monkeypatch.setattr(MS, 'masks_sibling_path', lambda w: str(tmp_path / "x"))
    (tmp_path / "x").write_text("", encoding="utf-8")
    out = MM.review_outcome_by_model(["a.wav"])
    assert out['m1']['reject_rate'] is None


def test_a_recording_with_no_store_is_skipped(tmp_path):
    assert MM.review_outcome_by_model([str(tmp_path / "missing.wav")]) == {}
