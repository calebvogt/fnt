"""CAD harmonic links and the Freq Samples setting.

* Automatic harmonics used to carry ``harmonic_of = id(fund_call)``, a Python
  object address, which CAD's backfill never corrected: the CSV pointed at no
  call. The detector now leaves the link to call_group_id and CAD resolves
  harmonic_of to the fundamental's call_id (and repairs old CSVs on load).
* The first save after a run lacked harmonic_of_call_number.
* The GUI's Freq Samples setting never reached the detector: with it off the
  run still wrote peak_freq_1..5. Off now means: contour measured, its columns
  left out (as for a hand-measured call).

Runs under pytest, or directly.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from fnt.usv.usv_detector.config import USVDetectorConfig  # noqa: E402
from fnt.usv.usv_detector.dsp_detector import DSPDetector  # noqa: E402
from fnt.usv.classic_audio_detector import ClassicAudioDetectorWindow as CAD  # noqa: E402

SR = 250_000


def _call_with_harmonic(seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(0.5 * SR)) / SR
    a = rng.normal(0, 0.001, len(t))
    seg = (t > 0.2) & (t < 0.26)
    a[seg] += 0.5 * np.sin(2 * np.pi * 25_000 * t[seg]) + 0.3 * np.sin(2 * np.pi * 50_000 * t[seg])
    return a


def _detect(**kw):
    cfg = USVDetectorConfig(min_freq_hz=15_000, max_freq_hz=80_000,
                            max_bandwidth_hz=60_000, min_tonality=0, **kw)
    return pd.DataFrame(DSPDetector(cfg).detect(_call_with_harmonic(), SR))


def test_detector_links_by_group_not_by_address():
    df = _detect()
    assert df['is_harmonic'].any()
    assert (df['harmonic_of'] == -1).all()          # no object address
    # every harmonic shares its group with a fundamental
    fund_groups = set(df.loc[~df['is_harmonic'], 'call_group_id'])
    assert set(df.loc[df['is_harmonic'], 'call_group_id']) <= fund_groups
    assert not any(c.startswith('_') for c in df.columns)


def test_cad_resolves_the_fundamentals_call_id_and_number():
    df = _detect()
    CAD._ensure_call_ids(df)
    df.insert(0, 'call_number', range(1, len(df) + 1))
    CAD._add_harmonic_call_numbers(df)
    fund = df[~df['is_harmonic']].iloc[0]
    harm = df[df['is_harmonic']].iloc[0]
    assert harm['harmonic_of'] == fund['call_id']
    assert harm['harmonic_rank'] == 1
    assert harm['harmonic_of_call_number'] == fund['call_number']
    assert fund['harmonic_of'] == -1 and fund['harmonic_of_call_number'] == ''


def test_old_csv_with_an_address_is_repaired_on_load():
    df = _detect()
    df.loc[df['is_harmonic'], 'harmonic_of'] = 1702443950144   # what old runs wrote
    CAD._ensure_call_ids(df)
    fund_id = df.loc[~df['is_harmonic'], 'call_id'].iloc[0]
    assert (df.loc[df['is_harmonic'], 'harmonic_of'] == fund_id).all()


def test_hand_link_is_kept():
    df = _detect()
    CAD._ensure_call_ids(df)
    harm_id = df.loc[df['is_harmonic'], 'call_id'].iloc[0]
    fund_id = df.loc[~df['is_harmonic'], 'call_id'].iloc[0]
    df.loc[df['call_id'] == harm_id, ['harmonic_of', 'harmonic_rank']] = [fund_id, 2]
    CAD._ensure_call_ids(df)
    row = df[df['call_id'] == harm_id].iloc[0]
    assert (row['harmonic_of'], row['harmonic_rank']) == (fund_id, 2)


@pytest.mark.parametrize("n", [0, 5, 8])
def test_freq_samples_setting(n):
    df = _detect(freq_samples=n)
    cols = [c for c in df.columns if c.startswith('peak_freq_') and c[10:].isdigit()]
    assert len(cols) == n
    # The contour is measured either way, so its features and filters work.
    assert {'mean_sweep_rate_khz_ms', 'contour_smoothness'} <= set(df.columns)
    assert df['mean_sweep_rate_khz_ms'].notna().all()


def test_worker_passes_freq_samples_through(monkeypatch):
    import fnt.usv.classic_audio_detector as cad
    seen = {}

    class Stop(Exception):
        pass

    def fake_detector(config):
        seen['n'] = config.freq_samples
        raise Stop

    import fnt.usv.usv_detector.dsp_detector as dsp
    monkeypatch.setattr(dsp, 'DSPDetector', fake_detector)
    w = cad.DSPDetectionWorker([], {'freq_samples': 0})
    with pytest.raises(Stop):
        w.run()
    assert seen['n'] == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
