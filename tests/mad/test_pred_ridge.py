"""The brightness ridge of each detection: its frequency contour from the pixels under its mask.

A mask drawn as a blob over a fast trill has a flat centroid; the brightest continuous path
through the image inside it still oscillates. Covers the algorithm on a synthetic trill (turns,
rate, depth, trill call; the ridge follows the bright line, not the blob), the zigzag pivots,
and the store: computed whenever a crop is written with its spectrogram, read back with the
crop, recomputed on an incremental write, deleted with its crop, backfilled for files that have
spectrograms but no ridge, absent (without error) where the file has no grid, and carried into
the CSV export.

The algorithm and constants match ridge_of / zigzag in the 2021_8x8_Mus analysis (R/09_usv.R);
the two agree exactly on all 1,300 verified calls there (checked when this was written).

Runs under pytest, or directly.
"""
import os
import tempfile

import numpy as np
import pytest

pytest.importorskip("h5py")

from fnt.usv.usv_detector import fnt_mask_store as ms  # noqa: E402
from fnt.usv.usv_detector import mad_ridge as mr  # noqa: E402

SR, NPERSEG, NOVERLAP, NFFT = 250_000, 512, 384, 1024
DF, DT = mr.grid_units(SR, NPERSEG, NOVERLAP, NFFT)     # 0.244 kHz/bin, 0.512 ms/frame


def trill(n_frames=120, f_mid=40, amp_bins=24, period=24):
    """(level, mask), frequency x time: a bright sinusoidal line (amp_bins up and down,
    `period` frames per cycle) inside a blob mask covering the whole excursion."""
    h = f_mid + amp_bins + 12
    level = np.full((h, n_frames), 40, np.float64)
    rows = np.round(f_mid + amp_bins * np.sin(2 * np.pi * np.arange(n_frames) / period)).astype(int)
    for t, r in enumerate(rows):
        level[r, t] = 250
        level[r - 1, t] = level[r + 1, t] = 180
    mask = np.zeros((h, n_frames), bool)
    mask[f_mid - amp_bins - 3:f_mid + amp_bins + 4, :] = True
    return level.astype(np.uint8), mask, rows


def test_the_ridge_follows_the_bright_line_not_the_blob():
    level, mask, rows = trill()
    frames, bins = mr.ridge_path(level, mask)
    assert len(frames) == mask.shape[1]
    assert np.max(np.abs(bins - rows)) <= 1
    # The blob's mean-frequency contour is flat: no turns at a 3 kHz excursion.
    centroid_khz = (np.arange(mask.shape[0])[:, None] * mask).sum(0) / mask.sum(0) * DF
    assert len(mr.zigzag(centroid_khz, mr.RIDGE_H_KHZ)) == 0


def test_turns_rate_depth_and_trill():
    level, mask, _ = trill()                       # 5 cycles in 61 ms: ~10 turns, ~80 Hz
    res = mr.ridge_measures(level, mask, DF, DT, f_off=100)
    assert 8 <= res["ridge_turns"] <= 10
    dur_s = mask.shape[1] * DT / 1000
    assert res["ridge_rate_hz"] == pytest.approx(res["ridge_turns"] / 2 / dur_s)
    assert res["ridge_depth_khz"] == pytest.approx(48 * DF, abs=2 * DF)   # peak to trough
    assert res["ridge_trill"] == 1
    assert np.nanmin(res["khz"]) == pytest.approx((100 + 40 - 24) * DF, abs=DF)   # f_off counted


def test_a_slow_sweep_is_not_a_trill():
    level, mask, _ = trill(n_frames=120, period=240, amp_bins=10)   # half a cycle
    res = mr.ridge_measures(level, mask, DF, DT)
    assert res["ridge_turns"] <= 1 and res["ridge_trill"] == 0


def test_zigzag_needs_the_excursion():
    x = [0, 2, 0, 2, 0]                            # 2 kHz wiggles
    assert len(mr.zigzag(x, 3)) == 0
    assert list(mr.zigzag([0, 5, 0, 5, 0], 3)) == [5, 0, 5]


def test_an_empty_mask_has_no_ridge():
    assert mr.ridge_measures(np.zeros((5, 5)), np.zeros((5, 5), bool), DF, DT) is None


def _file(d, with_grid=True):
    h5 = os.path.join(d, "r_FNT.mad")
    if with_grid:
        ms.set_grid_attrs(h5, sample_rate=SR, nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT,
                          n_freq_bins=513, n_time_frames=5000)
    return h5


def _crop(key, level, mask, spec=True):
    c = {"blob_id": key, "mask": mask.astype(np.uint8), "f_off": 100, "t_off": 7}
    if spec:
        c["spec"] = level
    return c


def test_the_store_computes_reads_and_deletes_ridges():
    level, mask, _ = trill()
    with tempfile.TemporaryDirectory() as d:
        h5 = _file(d)
        ms.write_pred_masks(h5, [_crop(1, level, mask), _crop(2, level, mask, spec=False)])
        got = ms.read_all_pred_masks(h5, include_ridge=True)
        assert got["1"]["ridge_trill"] == 1 and "ridge_khz" in got["1"]
        assert "ridge_turns" not in got["2"]               # no spectrogram, no ridge
        assert len(got["1"]["ridge_khz"]) == mask.shape[1]
        # an incremental write that reshapes the crop drops the stale ridge
        small = mask.copy(); small[:, 60:] = False
        ms.write_pred_mask_subset(h5, [{"blob_id": 1, "mask": small[:, :60].astype(np.uint8),
                                        "f_off": 100, "t_off": 7}])
        assert "ridge_turns" not in ms.read_all_pred_masks(h5)["1"]
        ms.write_pred_mask_subset(h5, [_crop(1, level, mask)])          # back with its image
        assert ms.read_all_pred_masks(h5)["1"]["ridge_trill"] == 1
        ms.delete_pred_mask(h5, 1)
        import h5py
        with h5py.File(h5, "r") as f:
            assert "1" not in f.get(ms.PRED_RIDGE_GROUP, {})


def test_ridges_are_backfilled_for_files_with_spectrograms():
    level, mask, _ = trill()
    with tempfile.TemporaryDirectory() as d:
        h5 = _file(d)
        ms.write_pred_masks(h5, [_crop(1, level, mask)])
        import h5py
        with h5py.File(h5, "a") as f:                      # as written before ridges existed
            del f[ms.PRED_RIDGE_GROUP]
            for k in ("ridge_turns", "ridge_rate_hz", "ridge_depth_khz", "ridge_trill"):
                del f[ms.PRED_GROUP]["1"].attrs[k]
        assert "ridge_turns" not in ms.read_all_pred_masks(h5)["1"]
        assert ms.add_pred_ridges(h5) == 1
        assert ms.read_all_pred_masks(h5)["1"]["ridge_trill"] == 1
        assert ms.add_pred_ridges(h5) == 0                 # already there


def test_no_grid_no_ridge_no_error():
    level, mask, _ = trill()
    with tempfile.TemporaryDirectory() as d:
        h5 = _file(d, with_grid=False)
        ms.write_pred_masks(h5, [_crop(1, level, mask)])
        assert "ridge_turns" not in ms.read_all_pred_masks(h5)["1"]
        assert ms.add_pred_ridges(h5) == 0


def test_the_csv_export_carries_the_ridge():
    from fnt.usv.usv_detector import mad_csv_rebuild as cr
    from fnt.usv.usv_detector.mad_inference import CSV_FIELDNAMES
    assert {"ridge_turns", "ridge_rate_hz", "ridge_depth_khz", "ridge_trill"} <= set(CSV_FIELDNAMES)
    level, mask, _ = trill()
    with tempfile.TemporaryDirectory() as d:
        h5 = _file(d)
        ms.write_pred_masks(h5, [_crop(1, level, mask)])
        rows = {"1": {"blob_id": 1, "status": "pending"}, "9": {"blob_id": 9, "status": "accepted"}}
        cr._add_ridge_fields(h5, rows)
        assert rows["1"]["ridge_trill"] == 1 and rows["1"]["ridge_turns"] >= 8
        assert "ridge_turns" not in rows["9"]              # a painted call: no crop


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
