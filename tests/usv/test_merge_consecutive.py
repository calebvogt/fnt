"""When two detections are fragments of one call, and when they are not.

The case that drove this: a real 81.5 -> 63.5 kHz downsweep in the 2021_8x8 set
came back as two detections, 73.2-81.5 kHz followed by 63.5-72.3 kHz. They
overlap in time by 2.6 ms and miss in frequency by 0.98 kHz — four bins — so
the old "bands must overlap" rule refused to join them, and one 71 ms call was
reported as two calls of 32 and 41 ms with half the bandwidth each.

The gating still has to keep a harmonic separate from its fundamental. The
property that separates the two cases is time, not frequency: a harmonic sounds
*with* its fundamental, sweep fragments sound *after* one another.
"""
import numpy as np
import pytest

from fnt.usv.usv_detector.mad_inference import (
    merge_consecutive_blobs, _bands_joinable, _freq_gap_bins,
    _time_overlap_fraction,
)


def blob(t0, t1, f0, f1, score=0.9):
    return {
        't_start': t0, 't_end_exclusive': t1,
        'f_low': f0, 'f_high_exclusive': f1,
        'area_pixels': (t1 - t0) * (f1 - f0), 'score': score,
        'mask': np.ones((f1 - f0, t1 - t0), dtype=bool),
    }


#: The real pair, in frames/bins (dt = 0.512 ms, df = 244.14 Hz).
SWEEP_A = blob(825286, 825348, 300, 334)     # 422.5469-422.5787 s, 73.2-81.5 kHz
SWEEP_B = blob(825343, 825423, 260, 296)     # 422.5761-422.6171 s, 63.5-72.3 kHz


def test_the_real_downsweep_now_merges():
    out = merge_consecutive_blobs([SWEEP_A, SWEEP_B], max_gap_frames=20)
    assert len(out) == 1
    m = out[0]
    assert m['f_low'] == 260 and m['f_high_exclusive'] == 334
    assert m['t_start'] == 825286 and m['t_end_exclusive'] == 825423


def test_that_pair_really_does_not_overlap_in_frequency():
    """Guards the premise: it is 4 bins apart, not overlapping."""
    assert _freq_gap_bins(SWEEP_A, SWEEP_B) == 4
    assert _time_overlap_fraction(SWEEP_A, SWEEP_B) < 0.1


def test_a_harmonic_stays_separate():
    """Concurrent and frequency-separated: a fundamental and its harmonic."""
    fund = blob(1000, 1100, 200, 240)
    harm = blob(1000, 1100, 400, 440)          # same time, double frequency
    out = merge_consecutive_blobs([fund, harm], max_gap_frames=20)
    assert len(out) == 2


def test_a_near_harmonic_stays_separate_even_when_bands_are_close():
    """Concurrency wins over adjacency — the whole point of the rule."""
    a = blob(1000, 1100, 200, 240)
    b = blob(1000, 1100, 242, 280)             # only 2 bins apart, but SIMULTANEOUS
    assert _freq_gap_bins(a, b) == 2
    out = merge_consecutive_blobs([a, b], max_gap_frames=20)
    assert len(out) == 2


def test_sequential_fragments_too_far_apart_stay_separate():
    a = blob(1000, 1050, 200, 240)
    b = blob(1055, 1100, 300, 340)             # 60 bins away
    out = merge_consecutive_blobs([a, b], max_gap_frames=20)
    assert len(out) == 2


def test_overlapping_bands_still_merge_as_before():
    a = blob(1000, 1050, 200, 250)
    b = blob(1055, 1100, 240, 290)
    out = merge_consecutive_blobs([a, b], max_gap_frames=20)
    assert len(out) == 1


def test_a_time_gap_beyond_the_limit_still_blocks_the_merge():
    """The frequency rule loosened; the time rule did not."""
    out = merge_consecutive_blobs([SWEEP_A, blob(825600, 825680, 260, 296)],
                                  max_gap_frames=20)
    assert len(out) == 2


def test_disabling_the_frequency_gate_merges_everything_adjacent():
    fund = blob(1000, 1100, 200, 240)
    harm = blob(1000, 1100, 400, 440)
    out = merge_consecutive_blobs([fund, harm], max_gap_frames=20,
                                  require_freq_overlap=False)
    assert len(out) == 1


def test_the_gap_tolerance_is_honoured():
    a = blob(1000, 1050, 200, 240)
    b = blob(1055, 1100, 246, 286)             # 6 bins apart, sequential
    assert len(merge_consecutive_blobs([a, b], 20, max_freq_gap_bins=8)) == 1
    assert len(merge_consecutive_blobs([a, b], 20, max_freq_gap_bins=4)) == 2


@pytest.mark.parametrize("frac, expect", [(0.9, 1), (0.05, 2)])
def test_the_concurrency_threshold_decides_the_marginal_case(frac, expect):
    """SWEEP_A/B overlap ~8% in time: a strict threshold calls them sequential
    (merge), a very loose one calls them concurrent (keep apart)."""
    out = merge_consecutive_blobs([SWEEP_A, SWEEP_B], 20,
                                  concurrent_fraction=frac)
    assert len(out) == expect


def test_merged_mask_is_the_union_in_the_union_box():
    out = merge_consecutive_blobs([SWEEP_A, SWEEP_B], max_gap_frames=20)
    m = out[0]
    assert m['mask'].shape == (334 - 260, 825423 - 825286)
    assert m['area_pixels'] == int(m['mask'].sum())
    assert m['area_pixels'] >= max(SWEEP_A['area_pixels'], SWEEP_B['area_pixels'])


def test_a_single_blob_is_returned_untouched():
    out = merge_consecutive_blobs([SWEEP_A], max_gap_frames=20)
    assert len(out) == 1 and out[0]['t_start'] == SWEEP_A['t_start']


def test_input_blobs_are_not_mutated():
    before = dict(SWEEP_A)
    merge_consecutive_blobs([SWEEP_A, SWEEP_B], max_gap_frames=20)
    assert SWEEP_A['t_end_exclusive'] == before['t_end_exclusive']
    assert SWEEP_A['f_high_exclusive'] == before['f_high_exclusive']


def test_bands_joinable_is_symmetric():
    for a, b in ((SWEEP_A, SWEEP_B), (SWEEP_B, SWEEP_A)):
        assert _bands_joinable(a, b, 8, 0.5) is True
