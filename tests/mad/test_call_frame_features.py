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


# ----------------------------------------------------------------------
# Sub-bin peak interpolation (the 3D view's pitch; never the CSV's)
# ----------------------------------------------------------------------
def _tone_columns(freq_hz, amp=0.05, seed=0, dur=0.02):
    """Real STFT dB columns of a steady tone, MAD's grid, plus light noise."""
    from scipy import signal
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * dur)) / SR
    x = amp * np.sin(2 * np.pi * freq_hz * t) + 1e-4 * rng.standard_normal(t.size)
    _f, _t, S = signal.spectrogram(x, fs=SR, nperseg=512, noverlap=384,
                                   nfft=NFFT, window='hann')
    return 10 * np.log10(S + 1e-10)


def _band_mask(cols, centre_bin, half=5):
    f_low = centre_bin - half
    return np.ones((2 * half + 1, cols.shape[1]), dtype=bool), f_low


@pytest.mark.parametrize("offset_bins", [0.0, 0.2, 0.37, 0.5, -0.3])
def test_interpolated_peak_lands_between_bins(offset_bins):
    """A tone off the bin grid: the nearest bin can be up to half a bin
    (122 Hz) out; the parabola must do much better."""
    k = 180
    f = (k + offset_bins) * DF
    cols = _tone_columns(f)
    mask, f_low = _band_mask(cols, k)
    fr = call_frame_features(cols, mask, f_low, DF, DB_MIN, DB_MAX)
    err_interp = np.abs(fr['peak_freq_interp_hz'] - f).max()
    assert err_interp < 0.1 * DF, err_interp          # < ~24 Hz
    # And never meaningfully worse than the nearest bin it refines. (A tone
    # exactly on a bin interpolates to within a fraction of a hertz — the
    # parabola is a model of the Hann peak, not the peak itself.)
    err_near = np.abs(fr['peak_freq_hz'] - f).max()
    assert err_interp <= err_near + 0.01 * DF


def test_interpolation_survives_a_call_louder_than_db_max():
    """Clipped at db_max the peak is a flat plateau — no curvature, tied
    bins. The interpolation reads unclipped dB, so it is unaffected."""
    k = 150
    f = (k + 0.3) * DF
    loud = _tone_columns(f, amp=20.0)                # far above -20 dB
    assert loud.max() > DB_MAX
    mask, f_low = _band_mask(loud, k)
    fr = call_frame_features(loud, mask, f_low, DF, DB_MIN, DB_MAX)
    assert np.abs(fr['peak_freq_interp_hz'] - f).max() < 0.1 * DF


def test_interpolated_contour_is_nan_off_mask_like_the_rest():
    cols, mask, f_low = _call(seed=13, gap=True)
    fr = call_frame_features(cols, mask, f_low, DF, DB_MIN, DB_MAX)
    off = ~fr['has_mask']
    assert np.isnan(fr['peak_freq_interp_hz'][off]).all()
    assert np.isfinite(fr['peak_freq_interp_hz'][~off]).all()
    # Within half a bin of the nearest-bin contour it refines.
    d = np.abs(fr['peak_freq_interp_hz'] - fr['peak_freq_hz'])[~off]
    assert d.max() <= 0.5 * DF + 1e-6


def test_the_csv_contour_does_not_use_the_interpolated_pitch():
    """Switching the CSV to sub-bin pitch would move every existing
    start/end/mean frequency; the metrics must keep the nearest-bin contour."""
    k = 170
    f = (k + 0.4) * DF
    cols = _tone_columns(f)
    mask, f_low = _band_mask(cols, k)
    m = compute_call_metrics(cols, mask, f_low, DF, DT, DB_MIN, DB_MAX)
    # The nearest bin, not the tone's true 0.4-bin-off frequency.
    assert m['start_freq_hz'] == round(k * DF, 2)
