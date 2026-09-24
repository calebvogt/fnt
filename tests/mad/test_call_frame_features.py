"""Per-frame call features must agree with the per-call metrics exactly.

``call_frame_features`` exists so a per-frame view of a call (the 3D trajectory
window) and the metric CSV come from one computation. Two guarantees follow:

* refactoring ``compute_call_metrics`` onto the shared core must not move a
  single CSV number — checked against a frozen copy of the old function;
* the frame arrays must reduce to those same numbers, so the trajectory a user
  looks at is the call the CSV describes.
"""
import numpy as np
import pytest

from _call_metrics_oracle import legacy_compute_call_metrics
from fnt.usv.usv_detector.mad_inference import (
    call_frame_features, compute_call_metrics,
)

SR = 250_000
NFFT = 1024
DF = (SR / 2.0) / (NFFT // 2)
DT = 128 / SR
F_FULL = NFFT // 2 + 1
DB_MIN, DB_MAX = -100.0, -20.0


def _call(seed, f_low=80, H=40, W=60, sweep=True, gap=False, noise_db=-95.0):
    """Synthetic full-frequency dB columns with an FM call and its mask."""
    rng = np.random.default_rng(seed)
    cols = noise_db + 4.0 * rng.standard_normal((F_FULL, W))
    mask = np.zeros((H, W), dtype=bool)
    for t in range(W):
        row = int(5 + (H - 10) * t / max(1, W - 1)) if sweep else H // 2
        row = min(H - 3, max(2, row + int(rng.integers(-1, 2))))
        cols[f_low + row - 1:f_low + row + 2, t] = -30.0 + rng.normal(0, 2, 3)
        mask[row - 2:row + 3, t] = True
    if gap:
        mask[:, W // 3:W // 3 + 4] = False
    return cols, mask, f_low


CASES = {
    'upsweep': dict(seed=1),
    'flat': dict(seed=2, sweep=False),
    'gapped': dict(seed=3, gap=True),
    'short': dict(seed=4, W=3),
    'single_column': dict(seed=5, W=1),
    'clipped_loud': dict(seed=6, noise_db=-10.0),   # everything above db_max
    'band_top': dict(seed=7, f_low=F_FULL - 40),    # mask touches the top
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_metrics_unchanged_by_the_refactor(name):
    cols, mask, f_low = _call(**CASES[name])
    old = legacy_compute_call_metrics(cols, mask, f_low, DF, DT, DB_MIN, DB_MAX)
    new = compute_call_metrics(cols, mask, f_low, DF, DT, DB_MIN, DB_MAX)
    assert new == old


def test_empty_and_misfit_masks_still_return_nothing():
    cols, mask, f_low = _call(seed=8)
    assert compute_call_metrics(cols, np.zeros_like(mask), f_low,
                                DF, DT, DB_MIN, DB_MAX) == {}
    assert call_frame_features(cols, np.zeros_like(mask), f_low,
                               DF, DB_MIN, DB_MAX) is None
    # Mask runs off the top of the spectrum.
    assert call_frame_features(cols, mask, F_FULL - 5,
                               DF, DB_MIN, DB_MAX) is None


def test_frames_reduce_to_the_csv_metrics():
    cols, mask, f_low = _call(seed=9, gap=True)
    fr = call_frame_features(cols, mask, f_low, DF, DB_MIN, DB_MAX)
    m = compute_call_metrics(cols, mask, f_low, DF, DT, DB_MIN, DB_MAX)

    contour = fr['peak_freq_hz'][fr['has_mask']]
    assert round(float(contour[0]), 2) == m['start_freq_hz']
    assert round(float(contour[-1]), 2) == m['end_freq_hz']
    assert round(float(contour.mean()), 2) == m['mean_freq_hz']
    assert round(float(fr['entropy'].mean()), 4) == m['spectral_entropy']
    assert round(float(fr['tonality'].mean()), 4) == m['tonality']


def test_frame_arrays_are_aligned_and_nan_off_mask():
    cols, mask, f_low = _call(seed=10, gap=True)
    fr = call_frame_features(cols, mask, f_low, DF, DB_MIN, DB_MAX)
    W = mask.shape[1]

    for key in ('has_mask', 'peak_freq_hz', 'centroid_hz', 'bandwidth_hz',
                'energy', 'power_db', 'entropy', 'tonality'):
        assert fr[key].shape == (W,), key

    off = ~fr['has_mask']
    assert off.any(), "fixture should have a gap"
    for key in ('peak_freq_hz', 'centroid_hz', 'bandwidth_hz', 'power_db'):
        assert np.isnan(fr[key][off]).all(), key
        assert np.isfinite(fr[key][~off]).all(), key
    # Full-column features are defined everywhere.
    assert np.isfinite(fr['entropy']).all()
    assert np.isfinite(fr['tonality']).all()


def test_the_contour_follows_the_call():
    """An upsweep's per-frame pitch should rise; a flat call's should not."""
    cols, mask, f_low = _call(seed=11)
    up = call_frame_features(cols, mask, f_low, DF, DB_MIN, DB_MAX)
    pitch = up['peak_freq_hz'][up['has_mask']]
    assert pitch[-5:].mean() > pitch[:5].mean() + 10 * DF

    cols, mask, f_low = _call(seed=12, sweep=False)
    flat = call_frame_features(cols, mask, f_low, DF, DB_MIN, DB_MAX)
    pitch = flat['peak_freq_hz'][flat['has_mask']]
    # ±1 row of jitter on a 3-row loud band: the peak may sit 2 bins either side.
    assert np.ptp(pitch) <= 4 * DF
