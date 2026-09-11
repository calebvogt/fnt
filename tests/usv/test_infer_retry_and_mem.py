"""A batch must measure its memory and re-attempt what it dropped.

An overnight run lost 72 of 262 recordings to "Unable to allocate 572 MiB".
The files were fine — same format, sample rate and duration as 190 that had
just succeeded, each dying on its first allocation before a sample was read.
The process had committed 78.6 GiB and the system commit charge was at its
limit. Two things were missing, and both are cheap:

* a memory reading per file, so the growth is visible while it happens rather
  than inferred from an event-log entry afterwards, and
* one retry pass at the end, when every per-file grid has been released.
"""
import numpy as np
import pytest

from fnt.usv.usv_detector import mad_inference as MI


# --------------------------------------------------------------- memory
def test_a_snapshot_reports_commit_not_just_rss():
    """RSS held steady across the leaking run; commit is what grew."""
    snap = MI._mem_snapshot()
    if not snap:
        pytest.skip("psutil not installed")
    assert snap['commit_mb'] > 0
    assert snap['rss_mb'] > 0


def test_the_system_headroom_is_reported_where_available():
    sysm = MI._system_commit()
    if not sysm:
        pytest.skip("not Windows")
    assert sysm['sys_commit_limit_mb'] >= sysm['sys_commit_avail_mb'] > 0


def test_a_record_carries_the_delta_for_this_file():
    """The absolute number hides a slow leak; the per-file delta does not."""
    rec = MI._mem_record({'commit_mb': 1000.0})
    if not rec:
        pytest.skip("psutil not installed")
    assert 'delta_commit_mb' in rec
    assert rec['delta_commit_mb'] == pytest.approx(
        rec['commit_mb'] - 1000.0, abs=0.11)


def test_a_record_survives_a_missing_baseline():
    rec = MI._mem_record({})
    assert 'delta_commit_mb' not in rec


# ---------------------------------------------------------------- retry
class _Runner:
    """Stands in for run_inference_on_file: fails listed names once."""

    def __init__(self, fail_once=(), fail_always=()):
        self.fail_once = set(fail_once)
        self.fail_always = set(fail_always)
        self.calls = []

    def __call__(self, wav, cfg, model=None, ckpt=None, device=None,
                 progress=None, wait_if_paused=None):
        self.calls.append(wav)
        if wav in self.fail_always:
            raise MemoryError(f"Unable to allocate 572. MiB for {wav}")
        if wav in self.fail_once:
            self.fail_once.discard(wav)
            raise MemoryError(f"Unable to allocate 572. MiB for {wav}")
        return {'wav_path': wav, 'n_blobs': 3, 'timing': {'t_total': 1.0}}


@pytest.fixture
def batch(monkeypatch):
    monkeypatch.setattr(MI, 'load_model', lambda p, d: (object(), {}, 'cpu'))
    monkeypatch.setattr(MI, '_release_between_files', lambda: None)

    def run(wavs, runner, **kw):
        monkeypatch.setattr(MI, 'run_inference_on_file', runner)
        cfg = MI.MADInferenceConfig(model_path='m.pt', **kw)
        seen = []
        return MI.run_inference_on_files(
            wavs, cfg, on_file_done=seen.append), seen

    return run


def test_a_transient_failure_is_recovered(batch):
    """The reported bug: a good file lost because of when it ran."""
    r = _Runner(fail_once=['b.wav'])
    results, _ = batch(['a.wav', 'b.wav', 'c.wav'], r)
    assert [x['wav_path'] for x in results] == ['a.wav', 'b.wav', 'c.wav']
    assert not any('error' in x for x in results)
    assert results[1]['retry'] is True
    assert r.calls == ['a.wav', 'b.wav', 'c.wav', 'b.wav']


def test_the_retry_runs_after_the_whole_pass_not_immediately(batch):
    """Retrying in place would hit the same wall; the point is the headroom."""
    r = _Runner(fail_once=['a.wav'])
    batch(['a.wav', 'b.wav', 'c.wav'], r)
    assert r.calls == ['a.wav', 'b.wav', 'c.wav', 'a.wav']


def test_order_is_preserved_when_a_file_is_recovered(batch):
    r = _Runner(fail_once=['a.wav', 'c.wav'])
    results, _ = batch(['a.wav', 'b.wav', 'c.wav'], r)
    assert [x['wav_path'] for x in results] == ['a.wav', 'b.wav', 'c.wav']


def test_a_file_that_fails_twice_keeps_its_first_error(batch):
    """One pass only — a second failure is not about memory."""
    r = _Runner(fail_always=['b.wav'])
    results, _ = batch(['a.wav', 'b.wav'], r)
    assert 'error' in results[1]
    assert results[1]['wav_path'] == 'b.wav'
    assert r.calls.count('b.wav') == 2       # tried exactly twice


def test_retry_can_be_switched_off(batch):
    r = _Runner(fail_once=['b.wav'])
    results, _ = batch(['a.wav', 'b.wav'], r, retry_failed=False)
    assert 'error' in results[1]
    assert r.calls == ['a.wav', 'b.wav']


def test_a_clean_run_starts_no_retry_pass(batch, monkeypatch):
    fired = []
    monkeypatch.setattr(MI, 'load_model', lambda p, d: (object(), {}, 'cpu'))
    monkeypatch.setattr(MI, 'run_inference_on_file', _Runner())
    cfg = MI.MADInferenceConfig(model_path='m.pt')
    MI.run_inference_on_files(['a.wav'], cfg,
                              on_retry_start=lambda n: fired.append(n))
    assert fired == []


def test_the_retry_pass_announces_its_size(monkeypatch):
    monkeypatch.setattr(MI, 'load_model', lambda p, d: (object(), {}, 'cpu'))
    monkeypatch.setattr(MI, '_release_between_files', lambda: None)
    monkeypatch.setattr(MI, 'run_inference_on_file',
                        _Runner(fail_once=['a.wav', 'b.wav']))
    cfg = MI.MADInferenceConfig(model_path='m.pt')
    fired = []
    MI.run_inference_on_files(['a.wav', 'b.wav', 'c.wav'], cfg,
                              on_retry_start=lambda n: fired.append(n))
    assert fired == [2]


def test_every_summary_carries_a_memory_reading(batch):
    r = _Runner(fail_always=['b.wav'])
    results, _ = batch(['a.wav', 'b.wav'], r)
    for x in results:
        assert 'mem' in x, x
    # The failure's reading is the one that says WHY it failed.
    if MI._mem_snapshot():
        assert results[1]['mem'].get('commit_mb', 0) > 0


def test_the_caller_is_told_about_the_retry_attempt(batch):
    """on_file_done fires for retries too, so the log shows the recovery."""
    r = _Runner(fail_once=['b.wav'])
    _, seen = batch(['a.wav', 'b.wav'], r)
    assert [s['wav_path'] for s in seen] == ['a.wav', 'b.wav', 'b.wav']
    assert seen[1].get('error') and not seen[2].get('error')


def test_a_stop_request_skips_the_retry_pass(monkeypatch):
    """Cancelling means stop, not 'stop after re-doing the failures'."""
    monkeypatch.setattr(MI, 'load_model', lambda p, d: (object(), {}, 'cpu'))
    monkeypatch.setattr(MI, 'run_inference_on_file', _Runner(fail_always=['a.wav']))
    cfg = MI.MADInferenceConfig(model_path='m.pt')
    fired = []
    MI.run_inference_on_files(['a.wav'], cfg, should_stop=lambda: True,
                              on_retry_start=lambda n: fired.append(n))
    assert fired == []


# ------------------------------------------------------------- manifest
def test_the_manifest_records_memory_and_the_retry_flag(tmp_path):
    from fnt.usv.usv_detector.mad_batch import RunManifest
    m = RunManifest(str(tmp_path)).open()
    m.record({'wav_path': str(tmp_path / 'a.wav'), 'n_blobs': 2,
              'mem': {'commit_mb': 3300.0, 'delta_commit_mb': 12.0},
              'retry': True, 'timing': {'audio_dur_s': 600.0}})
    m.record({'wav_path': str(tmp_path / 'b.wav'), 'error': 'Unable to allocate',
              'mem': {'commit_mb': 78000.0, 'sys_commit_avail_mb': 60.0}})
    m.close()
    recs = RunManifest(str(tmp_path)).records()
    assert recs[0]['mem']['delta_commit_mb'] == 12.0
    assert recs[0]['retry'] is True
    # An errored file still carries its reading — that is the diagnostic one.
    assert recs[1]['status'] == 'error'
    assert recs[1]['mem']['sys_commit_avail_mb'] == 60.0
    assert recs[1]['retry'] is False


def test_a_summary_without_memory_still_records(tmp_path):
    """psutil is optional; the manifest must not depend on it."""
    from fnt.usv.usv_detector.mad_batch import RunManifest
    m = RunManifest(str(tmp_path)).open()
    m.record({'wav_path': str(tmp_path / 'a.wav'), 'n_blobs': 1})
    m.close()
    assert RunManifest(str(tmp_path)).records()[0]['mem'] is None


# ------------------------------------------------------------ log line
def test_the_log_line_shows_growth_and_headroom():
    pytest.importorskip("PyQt5")
    from fnt.usv.mad_pyqt import _mem_suffix
    s = _mem_suffix({'mem': {'commit_mb': 3276.8, 'delta_commit_mb': 410.0,
                             'sys_commit_avail_mb': 48230.4}})
    assert "mem 3.2G" in s and "+410M" in s and "free 47.1G" in s


def test_the_log_line_is_empty_without_a_reading():
    pytest.importorskip("PyQt5")
    from fnt.usv.mad_pyqt import _mem_suffix
    assert _mem_suffix({}) == ""
    assert _mem_suffix({'mem': {}}) == ""
