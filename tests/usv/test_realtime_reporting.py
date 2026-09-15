"""One name, one meaning: ``realtime_factor`` is audio over *total* time.

It used to mean two things at once. ``run_inference_on_file`` computed
``audio_dur / t_infer`` — the tile scan alone — while ``mad_batch.summarize``
computed ``audio / wall`` over every stage. Both were called
``realtime_factor``, and the Batch Run Summary showed them in the same window:
rows at one definition, the header at the other, ~35% apart.

The per-file log line was the damaging one. It read

    600.0s audio in 84.47s [spec 20.46s · scan 62.84s · blobs 1.17s]
      → 9.55× realtime

where 600/84.47 is 7.1, not 9.55. Estimating an 11,402-file batch from that
number gave 8.3 days against a real 12.5.

Both rates are worth having, so both are reported — under names that say which
is which. And the per-file line now carries the detection count, which the
manifest had all along: on a multi-day run it is what shows a bad threshold at
hour one, while there is still a run to stop.
"""
import pytest

from fnt.usv.usv_detector.mad_batch import RunManifest, summarize


# ------------------------------------------------- the two rates
def _timing(audio=600.0, spec=20.0, scan=60.0, blobs=4.0):
    """What run_inference_on_file builds, by the same arithmetic."""
    total = spec + scan + blobs
    return {
        'audio_dur_s': audio,
        't_spec': spec, 't_infer': scan, 't_blobs': blobs,
        't_total': round(total, 2),
        'realtime_factor': round(audio / total, 2) if total > 0 else 0.0,
        'scan_realtime_factor': round(audio / scan, 2) if scan > 0 else 0.0,
        'device': 'cuda',
    }


def test_the_headline_rate_covers_every_stage():
    t = _timing(audio=600.0, spec=20.0, scan=60.0, blobs=4.0)
    assert t['realtime_factor'] == pytest.approx(600 / 84.0, rel=1e-3)


def test_the_scan_rate_is_kept_under_its_own_name():
    t = _timing(scan=60.0)
    assert t['scan_realtime_factor'] == pytest.approx(10.0, rel=1e-3)


def test_the_rate_divides_into_the_numbers_printed_beside_it():
    """The defect in one line: 'audio in t_total → X× realtime' has to be a
    division a reader can check."""
    t = _timing()
    assert t['realtime_factor'] == pytest.approx(
        t['audio_dur_s'] / t['t_total'], rel=1e-2)


def test_the_two_rates_differ_enough_to_matter():
    t = _timing(audio=600.0, spec=20.46, scan=62.84, blobs=1.17)
    assert t['scan_realtime_factor'] / t['realtime_factor'] > 1.3


def test_a_file_that_produced_no_timing_does_not_divide_by_zero():
    t = _timing(audio=0.0, spec=0.0, scan=0.0, blobs=0.0)
    assert t['realtime_factor'] == 0.0 and t['scan_realtime_factor'] == 0.0


def test_inference_reports_both():
    import inspect
    from fnt.usv.usv_detector.mad_inference import run_inference_on_file
    src = inspect.getsource(run_inference_on_file)
    assert "'realtime_factor': round(rt_factor, 2)" in src
    assert "'scan_realtime_factor': round(scan_rt_factor, 2)" in src
    assert "rt_factor = (audio_dur / total)" in src
    assert "scan_rt_factor = (audio_dur / t_infer)" in src


def test_the_per_file_rate_now_agrees_with_the_run_total(tmp_path):
    """The whole point: a row and the header are the same measurement."""
    m = RunManifest(str(tmp_path)).open()
    for _ in range(3):
        m.record({'wav_path': str(tmp_path / 'a.wav'), 'n_blobs': 5,
                  'timing': _timing()})
    m.close()
    recs = [r for r in _read(tmp_path)]
    s = summarize(recs)
    assert s['realtime_factor'] == pytest.approx(
        recs[0]['realtime_factor'], rel=1e-2)


def _read(d):
    import json
    with open(d / 'manifest.jsonl') as f:
        return [json.loads(ln) for ln in f if ln.strip()]


# ------------------------------------------------- the manifest
def test_the_manifest_carries_the_scan_rate_too(tmp_path):
    """So comparing devices or batch sizes across runs doesn't mean
    re-deriving it from stage times the manifest never stored."""
    m = RunManifest(str(tmp_path)).open()
    m.record({'wav_path': str(tmp_path / 'a.wav'), 'n_blobs': 5,
              'timing': _timing()})
    m.close()
    r = _read(tmp_path)[0]
    assert r['scan_realtime_factor'] == pytest.approx(10.0, rel=1e-3)
    assert r['realtime_factor'] != r['scan_realtime_factor']


def test_the_manifest_still_records_the_detection_count(tmp_path):
    m = RunManifest(str(tmp_path)).open()
    m.record({'wav_path': str(tmp_path / 'a.wav'), 'n_blobs': 29,
              'timing': _timing()})
    m.close()
    assert _read(tmp_path)[0]['n_detections'] == 29


# ------------------------------------------------- the log line
def _on_file_done_source():
    import inspect
    from fnt.usv.mad_pyqt import MADMainWindow
    src = inspect.getsource(MADMainWindow._start_inference)
    body = src[src.index("def on_file_done"):]
    return "\n".join(ln for ln in body.splitlines()
                     if not ln.strip().startswith("#"))


def test_the_log_line_states_how_many_detections(_=None):
    code = _on_file_done_source()
    assert "summary.get('n_blobs')" in code
    assert "det · " in code


def test_the_log_line_reports_the_end_to_end_rate():
    code = _on_file_done_source()
    assert "t.get('realtime_factor')" in code
    assert "× realtime on " in code


def test_the_scan_rate_is_labelled_where_it_appears():
    """It is still useful — just not printable as bare '× realtime'."""
    code = _on_file_done_source()
    assert "scan_realtime_factor" in code
    assert "× scan)" in code


def test_a_file_with_no_detections_still_says_so():
    """0 is a result, not a missing value — a run of zeros is the signal that
    the threshold is wrong."""
    code = _on_file_done_source()
    assert "if n_det is None else" in code


# ------------------------------------------------- the summary dialog
def test_the_wall_time_column_is_named_for_what_it_holds():
    """It shows t_total — spectrogram, scan and blobs — and was called
    'Scan time' next to a scan-only rate that didn't divide into it."""
    from fnt.usv.mad_pyqt import MADRunSummaryDialog
    assert "Wall time" in MADRunSummaryDialog.COLS
    assert "Scan time" not in MADRunSummaryDialog.COLS


def test_rows_recompute_the_rate_so_old_manifests_stay_consistent():
    import inspect
    from fnt.usv.mad_pyqt import MADRunSummaryDialog
    src = inspect.getsource(MADRunSummaryDialog._render)
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.strip().startswith("#"))
    assert "rt = (dur / wall) if wall > 0 else 0.0" in code
    assert "r.get('realtime_factor')" not in code


def test_the_finished_line_leads_with_the_rate_that_predicts_a_corpus():
    import inspect
    from fnt.usv.mad_pyqt import MADMainWindow
    src = inspect.getsource(MADMainWindow._start_inference)
    assert src.index("{rt:.1f}× realtime ") < src.index("tile scan alone")
