"""One brightness ridge per piece of a mask, and MAD filling in the ridges older .mad files lack.

A mask in several pieces (two detections merged into one call, or a call painted in two strokes)
used to get one path through all of them, so the gap between pieces counted as a frequency step.
Each piece now has its own ridge: turns add up, the gap is not a turn and not part of the
duration, and pieces sounding together (a merged harmonic) keep a row each. A one-piece mask gives
exactly what it did before.

The fill: ``ridge_status`` says what a file lacks, ``fill_ridges`` completes it (from the stored
spectrograms, or from the audio first) and stamps it with the ridge version, so a stamped file is
never opened twice; an unstamped file also has its multi-piece masks recomputed. Inference stamps
what it writes, a resumed run fills the recordings it skips, and the project's ``ridge`` setting
(on by default) turns all of it off.

Runs under pytest, or directly.
"""
import os
import tempfile

import numpy as np
import pytest

pytest.importorskip("h5py")

from fnt.usv.usv_detector import fnt_mask_store as ms  # noqa: E402
from fnt.usv.usv_detector import mad_inference as mi  # noqa: E402
from fnt.usv.usv_detector import mad_ridge as mr  # noqa: E402

SR, NPERSEG, NOVERLAP, NFFT = 250_000, 512, 384, 1024
DF, DT = mr.grid_units(SR, NPERSEG, NOVERLAP, NFFT)


@pytest.fixture(autouse=True)
def _ridges_on():
    ms.set_ridge_enabled(True)
    yield
    ms.set_ridge_enabled(True)


def old_measures(level, mask, f_off=0):
    """The ridge measures as computed before pieces were handled (one path through the mask)."""
    frames, bins = mr.ridge_path(level, mask)
    khz_on = (bins + f_off) * DF
    piv = mr.zigzag(khz_on, mr.RIDGE_H_KHZ)
    dur_s = mask.shape[1] * DT / 1000
    khz = np.full(mask.shape[1], np.nan, np.float32)
    khz[frames] = khz_on
    return {"khz": khz, "ridge_turns": len(piv), "ridge_rate_hz": len(piv) / 2 / dur_s,
            "ridge_depth_khz": float(np.median(np.abs(np.diff(piv)))) if len(piv) >= 2 else np.nan}


def lines(width, segments, h=60):
    """(level, mask): bright flat lines, each ``(row, c0, c1)``, masked 2 bins either side."""
    level = np.full((h, width), 30, np.uint8)
    mask = np.zeros((h, width), bool)
    for row, c0, c1 in segments:
        level[row, c0:c1] = 250
        mask[row - 2:row + 3, c0:c1] = True
    return level, mask


def test_a_one_piece_mask_measures_as_before():
    rng = np.random.default_rng(1)
    n = 0
    for _ in range(40):
        h, w = 50, int(rng.integers(20, 120))
        level = rng.integers(0, 255, (h, w)).astype(np.uint8)
        centre = np.clip(25 + np.cumsum(rng.integers(-2, 3, w)), 8, 41)
        mask = np.zeros((h, w), bool)
        for t, c in enumerate(centre):
            mask[c - int(rng.integers(2, 7)):c + int(rng.integers(2, 7)), t] = True
        assert len(mr.mask_pieces(mask)) == 1
        new, old = mr.ridge_measures(level, mask, DF, DT, f_off=7), old_measures(level, mask, 7)
        assert new["ridge_turns"] == old["ridge_turns"]
        assert new["ridge_rate_hz"] == pytest.approx(old["ridge_rate_hz"])
        assert np.array_equal(np.isnan(new["khz"]), np.isnan(old["khz"]))
        assert np.allclose(new["khz"], old["khz"], equal_nan=True)
        assert (np.isnan(new["ridge_depth_khz"]) and np.isnan(old["ridge_depth_khz"])
                or new["ridge_depth_khz"] == pytest.approx(old["ridge_depth_khz"]))
        assert new["pieces"] == 1
        n += old["ridge_turns"] > 0
    assert n > 5                       # the comparison covered calls with turns


def test_the_ridge_never_jumps_between_pieces():
    # three flat pieces, low - high - low, with gaps: one path turns at the high one
    level, mask = lines(70, [(10, 0, 20), (45, 25, 45), (10, 50, 70)])
    old = old_measures(level, mask)
    assert old["ridge_turns"] == 1                     # the jump read as a turn
    res = mr.ridge_measures(level, mask, DF, DT)
    assert res["pieces"] == 3 and res["ridge_turns"] == 0 and res["ridge_trill"] == 0
    assert res["khz"].ndim == 1                        # pieces don't share a column
    assert np.isnan(res["khz"][20:25]).all() and np.isnan(res["khz"][45:50]).all()
    assert np.allclose(res["khz"][25:45], 45 * DF)


def test_turns_add_up_over_pieces_and_the_gap_is_not_duration():
    def zig(c0, w, period=24, amp=14, base=25):
        return [(int(round(base + amp * np.sin(2 * np.pi * t / period))), c0 + t, c0 + t + 1)
                for t in range(w)]
    level, mask = lines(100, zig(0, 40) + zig(60, 40))
    one = mr.ridge_measures(*lines(40, zig(0, 40)), DF, DT)
    res = mr.ridge_measures(level, mask, DF, DT)
    assert res["pieces"] == 2
    assert res["ridge_turns"] == 2 * one["ridge_turns"] > 0
    assert res["ridge_rate_hz"] == pytest.approx(res["ridge_turns"] / 2 / (80 * DT / 1000))
    assert res["ridge_depth_khz"] == pytest.approx(one["ridge_depth_khz"])


def test_pieces_sounding_together_keep_a_row_each():
    level, mask = lines(40, [(10, 0, 40), (45, 5, 35)])        # a harmonic over its fundamental
    res = mr.ridge_measures(level, mask, DF, DT)
    assert res["khz"].shape == (2, 40)
    assert np.allclose(res["khz"][0], 10 * DF)
    assert np.allclose(res["khz"][1][5:35], 45 * DF) and np.isnan(res["khz"][1][:5]).all()


def _file(d):
    wav = os.path.join(d, "r.wav")
    open(wav, "wb").close()
    h5 = ms.masks_sibling_path(wav)
    ms.set_grid_attrs(h5, sample_rate=SR, nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT,
                      n_freq_bins=513, n_time_frames=5000)
    return wav, h5


def _crop(key, level, mask, spec=True):
    c = {"blob_id": key, "mask": mask.astype(np.uint8), "f_off": 100, "t_off": 7}
    if spec:
        c["spec"] = level
    return c


def _strip_ridges(h5):
    """Make the file look written before ridges (and before the stamp) existed."""
    import h5py
    with h5py.File(h5, "a") as f:
        if ms.PRED_RIDGE_GROUP in f:
            del f[ms.PRED_RIDGE_GROUP]
        for ds in f[ms.PRED_GROUP].values():
            for k in mr.RIDGE_ATTRS:
                if k in ds.attrs:
                    del ds.attrs[k]
        if ms.RIDGE_VERSION_ATTR in f.attrs:
            del f.attrs[ms.RIDGE_VERSION_ATTR]


def test_a_two_piece_ridge_is_stored_and_read_back():
    level, mask = lines(40, [(10, 0, 40), (45, 5, 35)])
    with tempfile.TemporaryDirectory() as d:
        _, h5 = _file(d)
        ms.write_pred_masks(h5, [_crop(1, level, mask)])
        got = ms.read_all_pred_masks(h5, include_ridge=True)["1"]
    assert got["ridge_khz"].shape == (2, 40)


def test_fill_completes_a_file_from_its_spectrograms_and_stamps_it():
    level, mask = lines(70, [(10, 0, 20), (45, 25, 45), (10, 50, 70)])
    with tempfile.TemporaryDirectory() as d:
        wav, h5 = _file(d)
        ms.write_pred_masks(h5, [_crop(1, level, mask), _crop(2, level, mask)])
        _strip_ridges(h5)
        st = ms.ridge_status(h5)
        assert st["needed"] and st["pred"] == 2 and st["pred_no_spec"] == 0
        r = mi.fill_ridges(wav, use_audio=False)
        assert r["pred"] == 2 and r["stamped"]
        assert ms.read_all_pred_masks(h5)["1"]["ridge_turns"] == 0
        assert not ms.ridge_status(h5)["needed"]
        assert mi.fill_ridges(wav)["pred"] == 0               # stamped: not reopened


def test_fill_recomputes_multi_piece_ridges_of_an_unstamped_file_only():
    import h5py
    level3, mask3 = lines(70, [(10, 0, 20), (45, 25, 45), (10, 50, 70)])
    level1, mask1 = lines(70, [(10, 0, 70)])
    with tempfile.TemporaryDirectory() as d:
        wav, h5 = _file(d)
        ms.write_pred_masks(h5, [_crop(1, level3, mask3), _crop(2, level1, mask1)])
        with h5py.File(h5, "a") as f:                         # ridges as the old rule left them
            f[ms.PRED_GROUP]["1"].attrs["ridge_turns"] = 1
            f[ms.PRED_GROUP]["2"].attrs["ridge_turns"] = 99   # a sentinel: must stay
        assert ms.ridge_status(h5)["needed"]
        mi.fill_ridges(wav, use_audio=False)
        got = ms.read_all_pred_masks(h5)
    assert got["1"]["ridge_turns"] == 0
    assert got["2"]["ridge_turns"] == 99


def test_without_the_audio_spectrogram_less_detections_wait_and_the_file_stays_unstamped():
    level, mask = lines(40, [(10, 0, 40)])
    with tempfile.TemporaryDirectory() as d:
        wav, h5 = _file(d)
        ms.write_pred_masks(h5, [_crop(1, level, mask), _crop(2, level, mask, spec=False)])
        _strip_ridges(h5)
        st = ms.ridge_status(h5)
        assert st["pred"] == 2 and st["pred_no_spec"] == 1
        r = mi.fill_ridges(wav, use_audio=False)
        assert r["pred"] == 1 and not r["stamped"]
        assert ms.ridge_status(h5)["needed"]


def test_with_the_audio_the_spectrograms_come_first_with_the_models_settings(monkeypatch):
    level, mask = lines(40, [(10, 0, 40)])
    seen = {}

    def fake_backfill(wav, db_min=None, db_max=None, db_norm=None, overwrite=False, ridges=True):
        seen.update(db_min=db_min, db_max=db_max, db_norm=db_norm)
        import h5py
        with h5py.File(ms.masks_sibling_path(wav), "a") as f:
            f.require_group(ms.PRED_SPEC_GROUP).create_dataset("2", data=level)
        return 1
    monkeypatch.setattr(mi, "backfill_pred_specs", fake_backfill)
    with tempfile.TemporaryDirectory() as d:
        wav, h5 = _file(d)
        ms.write_pred_masks(h5, [_crop(1, level, mask), _crop(2, level, mask, spec=False)])
        _strip_ridges(h5)
        r = mi.fill_ridges(wav, db_min=-90.0, db_max=-30.0, db_norm="per_file")
        assert r == {"needed": True, "pred": 2, "examples": 0, "specs": 1, "stamped": True}
    assert seen == {"db_min": -90.0, "db_max": -30.0, "db_norm": "per_file"}


def test_ridges_off_stores_none_and_fills_nothing():
    level, mask = lines(40, [(10, 0, 40)])
    with tempfile.TemporaryDirectory() as d:
        wav, h5 = _file(d)
        ms.set_ridge_enabled(False)
        ms.write_pred_masks(h5, [_crop(1, level, mask)])
        assert "ridge_turns" not in ms.read_all_pred_masks(h5)["1"]
        assert ms.add_pred_ridges(h5) == 0
        assert mi.fill_ridges(wav)["pred"] == 0
        ms.set_ridge_enabled(True)
        assert mi.fill_ridges(wav, use_audio=False)["pred"] == 1


def test_a_write_with_ridges_off_unstamps_the_file():
    level, mask = lines(40, [(10, 0, 40)])
    with tempfile.TemporaryDirectory() as d:
        wav, h5 = _file(d)
        ms.write_pred_masks(h5, [_crop(1, level, mask)])
        mi.fill_ridges(wav, use_audio=False)
        assert not ms.ridge_status(h5)["needed"]
        ms.set_ridge_enabled(False)
        ms.write_pred_mask_subset(h5, [_crop(2, level, mask)])
        ms.set_ridge_enabled(True)
        assert ms.ridge_status(h5)["needed"]                  # crop 2 has no ridge yet
        assert mi.fill_ridges(wav, use_audio=False)["pred"] == 1


def test_a_run_fills_the_recordings_it_skipped(monkeypatch):
    calls, stages = [], []
    monkeypatch.setattr(mi, "load_model", lambda *a, **k: (None, {"db_min": -95.0}, "cpu"))

    def fake_fill(wav, db_min=None, db_max=None, db_norm=None, use_audio=True):
        calls.append((os.path.basename(wav), db_min, ms.ridge_enabled()))
        return {"pred": 3, "examples": 0, "specs": 0, "stamped": True}
    monkeypatch.setattr(mi, "fill_ridges", fake_fill)
    done = []
    cfg = mi.MADInferenceConfig(model_path="m.pt")
    out = mi.run_inference_on_files([], cfg, ridge_fill=["a.wav", "b.wav"],
                                    progress=lambda i, n, name, st, *_: stages.append((i, n, st)),
                                    on_ridge_done=done.append)
    assert out == []
    assert calls == [("a.wav", -95.0, True), ("b.wav", -95.0, True)]
    assert stages == [(0, 2, "ridge"), (1, 2, "ridge")]
    assert [r["wav_path"] for r in done] == ["a.wav", "b.wav"]
    # off: no fill, and the switch is back as it was afterwards
    calls.clear()
    mi.run_inference_on_files([], mi.MADInferenceConfig(model_path="m.pt", ridge=False),
                              ridge_fill=["a.wav"])
    assert calls == [] and ms.ridge_enabled()


def test_the_project_carries_the_ridge_setting():
    from fnt.usv.usv_detector.mad_project import MADProjectConfig, create_mad_project
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "p")
        create_mad_project(p)
        cfg = MADProjectConfig.load(p)
        assert cfg.ridge is True
        cfg.ridge = False
        cfg.save()
        assert MADProjectConfig.load(p).ridge is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
