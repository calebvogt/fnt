"""A batch builds the next file's spectrogram while this one scans.

Per 10-minute recording, inference was ~16 s of CPU (1.2 s decode + 14.6 s
spectrogram) then ~12 s of GPU, one after the other. The batch now prepares
file i+1 on a background thread during file i's scan. Output is unchanged:
the same prepare_file runs, just earlier.

Runs under pytest, or directly.
"""
import threading
import time

import numpy as np
import pytest

from fnt.usv.usv_detector import mad_inference as MI

CPU_S = 0.3        # stand-in for the spectrogram build
GPU_S = 0.3        # stand-in for the scan


class _Fake:
    """prepare_file and run_inference_on_file stand-ins that log timings."""

    def __init__(self, fail_prepare=()):
        self.events = []
        self.got = {}
        self.fail_prepare = set(fail_prepare)
        self.lock = threading.Lock()

    def prepare(self, wav, cfg, ckpt):
        t0 = time.perf_counter()
        time.sleep(CPU_S)
        if wav in self.fail_prepare:
            raise OSError(f"cannot read {wav}")
        with self.lock:
            self.events.append(("prep", wav, t0, time.perf_counter()))
        return {'wav_path': wav, 'spec': f"spec:{wav}", 'sr': 250_000,
                'audio_dur': 600.0, 'db_min': -100.0, 'db_max': -20.0,
                't_spec': CPU_S}

    def run(self, wav, cfg, model=None, ckpt=None, device=None,
            progress=None, wait_if_paused=None, prepared=None):
        t0 = time.perf_counter()
        if prepared is None:            # the foreground path builds it
            prepared = self.prepare(wav, cfg, ckpt)
        time.sleep(GPU_S)
        with self.lock:
            self.got[wav] = prepared
            self.events.append(("run", wav, t0, time.perf_counter()))
        return {'wav_path': wav, 'timing': {'t_total': 1.0}}


@pytest.fixture
def batch(monkeypatch):
    monkeypatch.setattr(MI, 'load_model', lambda p, d: (object(), {}, 'cpu'))
    monkeypatch.setattr(MI, '_release_between_files', lambda: None)

    def go(wavs, fake, **kw):
        monkeypatch.setattr(MI, 'prepare_file', fake.prepare)
        monkeypatch.setattr(MI, 'run_inference_on_file', fake.run)
        cfg = MI.MADInferenceConfig(model_path='m.pt',
                                    **{k: v for k, v in kw.items()
                                       if k != 'should_stop'})
        t0 = time.perf_counter()
        out = MI.run_inference_on_files(wavs, cfg,
                                        should_stop=kw.get('should_stop'))
        return out, time.perf_counter() - t0

    return go


WAVS = ["a.wav", "b.wav", "c.wav", "d.wav"]


def test_each_file_scans_its_own_spectrogram(batch):
    f = _Fake()
    out, _ = batch(WAVS, f)
    assert [r['wav_path'] for r in out] == WAVS
    assert all(f.got[w]['spec'] == f"spec:{w}" for w in WAVS)


def test_the_next_build_runs_during_this_scan(batch):
    f = _Fake()
    batch(WAVS, f)
    ev = {(k, w): (a, b) for k, w, a, b in f.events}
    for this, nxt in zip(WAVS, WAVS[1:]):
        p0, p1 = ev[("prep", nxt)]
        r0, r1 = ev[("run", this)]
        assert p0 < r1, f"{nxt} was not built while {this} scanned"
        # Overlap, not total wall time: a loaded machine stretches both, but
        # the two still run side by side for most of the scan.
        assert min(p1, r1) - max(p0, r0) > 0.5 * GPU_S, \
            f"{nxt}'s build barely overlapped {this}'s scan"


def test_prefetch_can_be_turned_off(batch):
    f = _Fake()
    _, wall = batch(WAVS, f, prefetch_next=False)
    assert wall >= len(WAVS) * (CPU_S + GPU_S) * 0.95


def test_a_failed_background_build_is_redone_in_the_foreground(batch):
    f = _Fake(fail_prepare={"b.wav"})
    out, _ = batch(WAVS, f)
    bad = next(r for r in out if r['wav_path'] == "b.wav")
    assert "cannot read b.wav" in bad.get('error', ""), \
        "the error must be the file's own, reported as before"
    assert all(f.got[w]['spec'] == f"spec:{w}" for w in ("a.wav", "c.wav",
                                                          "d.wav"))


def test_a_stop_leaves_no_prefetch_thread_behind(batch):
    f = _Fake()
    calls = []

    def stop():
        calls.append(1)
        return len(calls) > 2           # stop before the third file

    out, _ = batch(WAVS, f, should_stop=stop)
    assert len(out) == 2
    assert not [t for t in threading.enumerate()
                if t.name.startswith("mad-prefetch")]


def test_a_supplied_spectrogram_is_used_not_rebuilt(monkeypatch):
    """The real run_inference_on_file: given a bundle, it must scan that
    spectrogram and not decode the recording again."""
    spec = np.zeros((513, 4000), np.uint8)
    seen = {}

    def no_decode(*a, **k):
        raise AssertionError("decoded again despite a prepared spectrogram")

    class Stop(Exception):
        pass

    def infer(model, s, **k):
        seen['spec'] = s
        raise Stop()                    # the scan is all this test needs

    monkeypatch.setattr(MI, 'load_audio', no_decode)
    monkeypatch.setattr(MI, 'infer_probability_mask', infer)
    cfg = MI.MADInferenceConfig(model_path='m.pt')
    bundle = {'wav_path': 'x.wav', 'spec': spec, 'sr': 250_000,
              'audio_dur': 2.0, 'db_min': -100.0, 'db_max': -20.0,
              't_spec': 1.0, 't_wait': 0.0}
    with pytest.raises(Stop):
        MI.run_inference_on_file('x.wav', cfg, model=object(), ckpt={},
                                 device='cpu', prepared=bundle)
    assert seen['spec'] is spec


def test_a_bundle_for_another_file_is_ignored(monkeypatch):
    built = []

    class Stop(Exception):
        pass

    def prep(wav, cfg, ckpt):
        built.append(wav)
        return {'wav_path': wav, 'spec': np.zeros((4, 4), np.uint8),
                'sr': 250_000, 'audio_dur': 1.0, 'db_min': -100.0,
                'db_max': -20.0, 't_spec': 0.0}

    monkeypatch.setattr(MI, 'prepare_file', prep)
    monkeypatch.setattr(MI, 'infer_probability_mask',
                        lambda *a, **k: (_ for _ in ()).throw(Stop()))
    cfg = MI.MADInferenceConfig(model_path='m.pt')
    with pytest.raises(Stop):
        MI.run_inference_on_file('x.wav', cfg, model=object(), ckpt={},
                                 device='cpu',
                                 prepared={'wav_path': 'other.wav',
                                           'spec': None})
    assert built == ['x.wav']


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
