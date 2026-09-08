"""Chunked connected-component labelling must equal the whole-grid version.

``ndi.label`` returns int32, so labelling a recording in one go needs a
contiguous block four times the size of the probability grid: 2.4 GB for ten
minutes, 14.4 GB for an hour, 29 GB for two. That is a ceiling, not a cost — a
217-file batch lost 29 files to "Unable to allocate 572 MiB" with 25 GB free,
and an hour-long recording would not run at all on a 16 GB machine.

Chunking the time axis fixes the memory. The risk it introduces is the seams: a
call crossing a chunk boundary is labelled separately on each side, and if the
halves are not rejoined the detector silently reports two calls where there was
one — on long files only, and never on the short files anyone tests with.

So the bar here is equality, not plausibility: same blobs, same boxes, same
areas, same scores, same masks, at every chunk size.
"""
import numpy as np
import pytest

from fnt.usv.usv_detector.mad_inference import (
    extract_blobs, extract_blobs_chunked,
)

KEYS = ('t_start', 't_end_exclusive', 'f_low', 'f_high_exclusive',
        'area_pixels')


def assert_same(a, b, ctx=""):
    assert len(a) == len(b), f"{ctx}: {len(a)} blobs vs {len(b)}"
    for x, y in zip(a, b):
        for k in KEYS:
            assert x[k] == y[k], f"{ctx}: {k} {x[k]} vs {y[k]}"
        assert abs(x['score'] - y['score']) < 1e-6, f"{ctx}: score"
        assert ('mask' in x) == ('mask' in y), f"{ctx}: mask presence"
        if 'mask' in x:
            assert np.array_equal(x['mask'], y['mask']), f"{ctx}: mask pixels"


def test_a_call_spanning_many_chunks_stays_one_blob():
    """The whole point. One streak, 490 columns wide, chunked into 49 pieces."""
    g = np.zeros((20, 500), dtype=np.uint8)
    g[9:12, 5:495] = 255
    ref = extract_blobs(g, 0.5, 1, include_mask=True)
    assert len(ref) == 1
    for chunk in (3, 10, 37, 100, 499, 500, 100000):
        got = extract_blobs_chunked(g, 0.5, 1, include_mask=True,
                                    chunk_frames=chunk)
        assert len(got) == 1, f"chunk={chunk} split the call into {len(got)}"
        assert_same(ref, got, f"chunk={chunk}")


def test_eight_connectivity_holds_across_a_seam():
    """A diagonal touches only corner-to-corner at the boundary."""
    g = np.zeros((60, 300), dtype=np.uint8)
    for i in range(250):
        g[5 + i // 5, 10 + i] = 255
    ref = extract_blobs(g, 0.5, 1, include_mask=True)
    for chunk in (2, 5, 11, 64, 100000):
        assert_same(ref, extract_blobs_chunked(g, 0.5, 1, include_mask=True,
                                               chunk_frames=chunk),
                    f"chunk={chunk}")


def test_two_calls_separated_by_one_column_stay_separate():
    """The opposite error: merging things that only look adjacent."""
    g = np.zeros((20, 60), dtype=np.uint8)
    g[9:12, 5:25] = 255
    g[9:12, 27:50] = 255          # a clear two-column gap
    ref = extract_blobs(g, 0.5, 1, include_mask=True)
    assert len(ref) == 2
    for chunk in (4, 26, 27, 100000):   # seams landing inside the gap too
        got = extract_blobs_chunked(g, 0.5, 1, include_mask=True,
                                    chunk_frames=chunk)
        assert len(got) == 2, f"chunk={chunk} merged two calls"
        assert_same(ref, got, f"chunk={chunk}")


def test_a_seam_landing_exactly_on_a_boundary_column():
    """Off-by-one territory: the chunk edge at the blob's first/last column."""
    g = np.zeros((10, 40), dtype=np.uint8)
    g[4:6, 10:20] = 255
    ref = extract_blobs(g, 0.5, 1, include_mask=True)
    for chunk in (9, 10, 11, 19, 20, 21):
        assert_same(ref, extract_blobs_chunked(g, 0.5, 1, include_mask=True,
                                               chunk_frames=chunk),
                    f"chunk={chunk}")


@pytest.mark.parametrize("seed", range(6))
def test_random_grids_match_at_every_chunk_size(seed):
    rng = np.random.default_rng(seed)
    H, W = int(rng.integers(8, 30)), int(rng.integers(60, 200))
    g = (rng.random((H, W)) * 255).astype(np.uint8)
    for _ in range(int(rng.integers(1, 6))):
        r = int(rng.integers(0, H))
        c = int(rng.integers(0, max(1, W - 50)))
        g[max(0, r - 1):r + 2, c:c + int(rng.integers(10, 50))] = 255
    for chunk in (7, 31, 100000):
        for mn in (1, 20):
            assert_same(extract_blobs(g, 0.6, mn, include_mask=True),
                        extract_blobs_chunked(g, 0.6, mn, include_mask=True,
                                              chunk_frames=chunk),
                        f"seed={seed} chunk={chunk} min={mn}")


def test_min_blob_pixels_applies_to_the_merged_area():
    """A call split across a seam must be measured whole before filtering —
    otherwise each half is judged alone and both may be discarded."""
    g = np.zeros((10, 40), dtype=np.uint8)
    g[4:6, 8:28] = 255                     # 40 px total, 20 either side of 18
    assert extract_blobs_chunked(g, 0.5, 30, chunk_frames=18) != []
    assert extract_blobs_chunked(g, 0.5, 50, chunk_frames=18) == []


def test_an_empty_grid_returns_nothing():
    g = np.zeros((10, 50), dtype=np.uint8)
    for chunk in (4, 50, 100000):
        assert extract_blobs_chunked(g, 0.5, 1, chunk_frames=chunk) == []


def test_a_fully_saturated_grid_is_one_blob():
    g = np.full((10, 50), 255, dtype=np.uint8)
    ref = extract_blobs(g, 0.5, 1, include_mask=True)
    for chunk in (4, 50, 100000):
        assert_same(ref, extract_blobs_chunked(g, 0.5, 1, include_mask=True,
                                               chunk_frames=chunk),
                    f"chunk={chunk}")


def test_an_empty_chunk_between_two_calls_breaks_the_run():
    """A blank chunk must not chain the call before it to the one after."""
    g = np.zeros((10, 90), dtype=np.uint8)
    g[4:6, 2:10] = 255
    g[4:6, 80:88] = 255
    got = extract_blobs_chunked(g, 0.5, 1, include_mask=True, chunk_frames=30)
    assert len(got) == 2
    assert_same(extract_blobs(g, 0.5, 1, include_mask=True), got)


def test_float_grids_work_too():
    """The interactive view path passes float probabilities, not uint8."""
    g = np.zeros((12, 80), dtype=np.float32)
    g[5:8, 10:70] = 0.9
    ref = extract_blobs(g, 0.5, 1, include_mask=True)
    for chunk in (7, 100000):
        assert_same(ref, extract_blobs_chunked(g, 0.5, 1, include_mask=True,
                                               chunk_frames=chunk),
                    f"chunk={chunk}")


def test_masks_are_only_built_when_asked_for():
    g = np.zeros((10, 40), dtype=np.uint8)
    g[4:6, 10:20] = 255
    assert 'mask' not in extract_blobs_chunked(g, 0.5, 1, chunk_frames=7)[0]
    assert 'mask' in extract_blobs_chunked(g, 0.5, 1, include_mask=True,
                                           chunk_frames=7)[0]
