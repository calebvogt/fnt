"""The 3D view's numbers: calls become paths, and paths fit the box.

The window only draws what ``mad_trajectory`` returns, so the properties that
decide whether the picture is honest are tested here, without a display:
a sweep's path must actually move the way the sweep does, gaps in a mask must
not invent motion, and every point must land inside the axes it is drawn in.
"""
import numpy as np
import pytest

from fnt.usv.usv_detector.mad_trajectory import (
    DEFAULT_AXES, FEATURES, axis_ranges, call_trajectory, direction_alpha,
    finite_rows, label_for, playback_progress, to_unit_cube,
)

DT = 128 / 250_000          # 0.512 ms per frame at MAD's defaults


def _frames(pitch_hz, has=None, entropy=0.3):
    """A call_frame_features-shaped dict from a pitch contour."""
    pitch_hz = np.asarray(pitch_hz, dtype=np.float64)
    W = pitch_hz.size
    has = np.ones(W, dtype=bool) if has is None else np.asarray(has, bool)
    nan_off = lambda a: np.where(has, a, np.nan)   # noqa: E731
    return {
        'has_mask': has,
        'peak_freq_hz': nan_off(pitch_hz),
        'centroid_hz': nan_off(pitch_hz + 500.0),
        'bandwidth_hz': nan_off(np.full(W, 2000.0)),
        'energy': np.where(has, 1e-4, 0.0),
        'power_db': nan_off(np.full(W, -40.0)),
        'entropy': np.full(W, entropy),
        'tonality': np.full(W, 0.8),
    }


def test_every_feature_key_is_returned_aligned():
    tr = call_trajectory(_frames(np.linspace(40e3, 60e3, 50)), DT,
                         frame_offset=1000)
    for key in FEATURES:
        assert tr[key].shape == (50,), key
    assert tr['frame'][0] == 1000 and tr['frame'][-1] == 1049


def test_an_upsweep_has_positive_pitch_rate_in_kHz_per_ms():
    # 20 kHz over 49 frames * 0.512 ms ≈ 25 ms → ~0.8 kHz/ms.
    tr = call_trajectory(_frames(np.linspace(40e3, 60e3, 50)), DT)
    expected = 20.0 / (49 * DT * 1000.0)
    assert np.median(tr['pitch_rate']) == pytest.approx(expected, rel=0.02)
    assert tr['pitch'][0] == pytest.approx(40.0, abs=0.5)


def test_a_downsweep_is_the_mirror_image():
    tr = call_trajectory(_frames(np.linspace(60e3, 40e3, 50)), DT)
    assert np.median(tr['pitch_rate']) < 0


def test_smoothing_tames_bin_quantisation():
    """A slow sweep quantised to 244 Hz bins staircases; unsmoothed, its rate
    is mostly spikes and zeros. Smoothed, it should sit near the true slope."""
    df = 244.140625
    true = np.linspace(40e3, 42e3, 80)
    stair = np.round(true / df) * df
    raw = call_trajectory(_frames(stair), DT, smooth_frames=1)
    smooth = call_trajectory(_frames(stair), DT)
    assert np.std(smooth['pitch_rate']) < 0.5 * np.std(raw['pitch_rate'])


def test_a_mask_gap_is_skipped_not_interpolated():
    has = np.ones(40, dtype=bool)
    has[10:15] = False
    tr = call_trajectory(_frames(np.linspace(40e3, 50e3, 40), has), DT)
    assert tr['pitch'].size == 35
    assert 12 not in set(tr['frame'].tolist())
    # Time keeps its real spacing across the gap.
    assert np.diff(tr['time']).max() == pytest.approx(6 * DT * 1000.0)


@pytest.mark.parametrize("n", [0, 1])
def test_too_few_frames_is_no_path(n):
    assert call_trajectory(_frames(np.full(max(n, 1), 50e3),
                                   has=np.ones(max(n, 1), bool) if n else
                                   np.zeros(1, bool)), DT) is None
    assert call_trajectory({}, DT) is None


def test_points_fill_but_never_leave_the_unit_cube():
    trs = [call_trajectory(_frames(np.linspace(a, b, 40), entropy=e), DT)
           for a, b, e in [(30e3, 60e3, 0.2), (70e3, 45e3, 0.5),
                           (50e3, 50e3, 0.35)]]
    ranges = axis_ranges(trs, DEFAULT_AXES)
    allp = np.vstack([to_unit_cube(t, DEFAULT_AXES, ranges) for t in trs])
    assert np.all(allp >= -1.0) and np.all(allp <= 1.0)
    # Padded min/max: the extremes sit just inside the faces, not at the centre.
    assert allp[:, 0].min() < -0.8 and allp[:, 0].max() > 0.8


def test_a_flat_axis_is_widened_not_divided_by_zero():
    tr = call_trajectory(_frames(np.linspace(40e3, 60e3, 30)), DT)
    ranges = axis_ranges([tr], ['bandwidth'])     # constant 2 kHz
    lo, hi = ranges['bandwidth']
    assert lo < 2.0 < hi
    pts = to_unit_cube(tr, ['bandwidth', 'pitch', 'time'],
                       {**ranges, **axis_ranges([tr], ['pitch', 'time'])})
    assert np.isfinite(pts).all()
    assert np.allclose(pts[:, 0], 0.0)


def test_direction_alpha_rises_onset_to_offset():
    a = direction_alpha(10)
    assert a[0] < a[-1] and np.all(np.diff(a) > 0)
    assert direction_alpha(1).tolist() == [1.0]
    assert direction_alpha(0).size == 0


def test_finite_rows_is_a_mask_the_caller_can_reuse():
    pts = np.array([[0, 0, 0], [np.nan, 0, 0], [1, 1, 1]], dtype=float)
    assert finite_rows(pts).tolist() == [True, False, True]


def test_labels_carry_units():
    assert label_for('pitch') == 'Pitch (kHz)'
    assert label_for('entropy') == 'Spectral entropy'


# ----------------------------------------------------------------------
# Playback: how much of a call the playhead has drawn
# ----------------------------------------------------------------------
@pytest.mark.parametrize("pos, expected", [
    (0.05, (0, False)),      # before the call: nothing drawn, not live
    (0.10, (1, True)),       # exactly at onset: first point, live
    (0.1225, (5, True)),     # mid-call: five of nine points, live
    (0.14, (9, True)),       # exactly at offset: all of it, still live
    (0.20, (9, False)),      # after: complete, finished (drawn dim)
])
def test_playback_progress_follows_the_playhead(pos, expected):
    # A 40 ms call, one point every 5 ms. Probes sit on the endpoints and
    # between points — not ON an interior point, where linspace's rounding
    # (0.12000000000000001) would make the test about floating point.
    t = np.linspace(0.10, 0.14, 9)
    assert playback_progress(t, pos) == expected


def test_playback_progress_of_nothing():
    assert playback_progress(np.array([]), 1.0) == (0, False)


def test_progress_only_grows_as_the_playhead_advances():
    t = np.sort(np.random.default_rng(3).uniform(0.2, 0.3, 50))
    ns = [playback_progress(t, p)[0] for p in np.linspace(0.15, 0.35, 200)]
    assert ns == sorted(ns) and ns[0] == 0 and ns[-1] == 50
