"""The brightness ridge of every labelled call (training example), not only of detections.

A verified call's ridge is what the 2021_8x8_Mus analysis measures trills on, and the labels
live in the training examples, not in /pred_calls. Covers: the ridge is computed from the
call's own pixels -- the mask inside its time and frequency bounds, the cut R/01_build_usv.R
makes, so a composited (older) mask's neighbours never enter it -- whenever an example is
saved or committed, its mask is edited, or its bounds move (and not for other metadata);
backfilled by add_example_ridges and by `add-ridge` for a recording's .mad or a training store
given directly; absent without error when the metadata carries no grid; and carried into the
CSV export, where a label's own ridge outranks its detection crop's.

Runs under pytest, or directly.
"""
import json
import os
import tempfile

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from fnt.usv.usv_detector import fnt_mask_store as ms  # noqa: E402
from fnt.usv.usv_detector import mad_ridge as mr  # noqa: E402

SR, NPERSEG, NOVERLAP, NFFT = 250_000, 512, 384, 1024
DF_HZ = SR / NFFT                      # 244.14 Hz per bin
DT_S = (NPERSEG - NOVERLAP) / SR       # 0.512 ms per frame
DF_KHZ, DT_MS = mr.grid_units(SR, NPERSEG, NOVERLAP, NFFT)
N_FREQ, W = NFFT // 2 + 1, 200
FB, T0, T1 = 150, 40, 160              # the trill's rows start at FB; the call spans T0..T1
PATCH_T0 = 10.0


def _patch(neighbour=False):
    """(spec float [0,1], mask, meta): a trill in a full-height patch. ``neighbour`` adds a
    brighter straight call in the mask outside the call's bounds (an older composited mask)."""
    level = np.full((N_FREQ, W), 40, np.uint8)
    mask = np.zeros((N_FREQ, W), np.uint8)
    rows = np.round(FB + 40 + 24 * np.sin(2 * np.pi * np.arange(T1 - T0) / 24)).astype(int)
    for j, r in enumerate(rows):
        level[r - 1:r + 2, T0 + j] = (180, 250, 180)
    mask[FB + 13:FB + 68, T0:T1] = 1
    if neighbour:
        level[300:303, T0:T1] = 255
        mask[298:305, T0:T1] = 1
    meta = {"source_wav": "rec.wav", "class": "USV",
            "sample_rate": SR, "nperseg": NPERSEG, "noverlap": NOVERLAP, "nfft": NFFT,
            "patch_t0_s": PATCH_T0,
            "t_start_s": round(PATCH_T0 + T0 * DT_S, 6), "t_stop_s": round(PATCH_T0 + T1 * DT_S, 6),
            "f_low_hz": round((FB + 13) * DF_HZ, 2), "f_high_hz": round((FB + 68) * DF_HZ, 2)}
    return level / 255.0, mask, meta


def _expected(spec, mask):
    u8 = (np.clip(spec, 0, 1) * 255).round().astype(np.uint8)
    f0, f1 = FB + 13, FB + 68
    return mr.ridge_measures(u8[f0:f1, T0:T1], mask[f0:f1, T0:T1], DF_KHZ, DT_MS, f_off=f0)


def _attrs(h5, eid):
    with h5py.File(h5, "r") as f:
        g = f["examples"][eid]
        out = {k: g.attrs[k] for k in mr.RIDGE_ATTRS + ("ridge_t_off",) if k in g.attrs}
        if ms.EXAMPLE_RIDGE_DS in g:
            out["khz"] = g[ms.EXAMPLE_RIDGE_DS][()]
        return out


def test_a_saved_label_carries_its_ridge():
    spec, mask, meta = _patch()
    exp = _expected(spec, mask)
    with tempfile.TemporaryDirectory() as d:
        h5 = os.path.join(d, "rec_FNT.mad")
        eid = ms.td_save_example(h5, spec, mask, meta)
        got = _attrs(h5, eid)
        assert got["ridge_trill"] == 1 and exp["ridge_trill"] == 1
        for k in mr.RIDGE_ATTRS:
            assert got[k] == pytest.approx(exp[k], nan_ok=True), k
        assert got["ridge_t_off"] == T0 and len(got["khz"]) == T1 - T0
        np.testing.assert_allclose(got["khz"], exp["khz"])


def test_neighbours_in_an_older_composited_mask_stay_out():
    spec, mask, meta = _patch(neighbour=True)
    clean = _expected(*_patch()[:2])
    with tempfile.TemporaryDirectory() as d:
        h5 = os.path.join(d, "rec_FNT.mad")
        eid = ms.td_save_example(h5, spec, mask, meta)
        np.testing.assert_allclose(_attrs(h5, eid)["khz"], clean["khz"])


def test_commit_returns_the_ridge_and_edits_recompute_it():
    spec, mask, meta = _patch()
    with tempfile.TemporaryDirectory() as d:
        h5 = os.path.join(d, "rec_FNT.mad")
        res = ms.td_commit_example(h5, spec, mask, meta, "e1")
        assert res["ridge"]["ridge_trill"] == 1
        assert _attrs(h5, "e1")["ridge_trill"] == 1
        # a mask edit to the first 30 frames: about one cycle, no trill
        short = mask.copy()
        short[:, T0 + 30:] = 0
        ms.td_update_mask(h5, "e1", short)
        got = _attrs(h5, "e1")
        assert got["ridge_trill"] == 0 and np.isnan(got["khz"][30:]).all()
        # an erased mask: no ridge left behind
        ms.td_update_mask(h5, "e1", np.zeros_like(mask))
        assert _attrs(h5, "e1") == {}


def test_moved_bounds_recompute_and_other_metadata_does_not():
    spec, mask, meta = _patch()
    with tempfile.TemporaryDirectory() as d:
        h5 = os.path.join(d, "rec_FNT.mad")
        ms.td_save_example(h5, spec, mask, meta, example_id="e1")
        with h5py.File(h5, "a") as f:                 # mark it, to see whether it is redone
            f["examples"]["e1"].attrs["ridge_turns"] = -1
        ms.td_update_meta(h5, "e1", {"note": "nice trill", "harmonic_n": 1})
        assert _attrs(h5, "e1")["ridge_turns"] == -1
        ms.td_update_meta(h5, "e1", {"t_stop_s": round(PATCH_T0 + (T0 + 30) * DT_S, 6)})
        got = _attrs(h5, "e1")
        assert got["ridge_turns"] >= 0 and len(got["khz"]) == 30


def test_backfill_and_no_grid_no_error():
    spec, mask, meta = _patch()
    with tempfile.TemporaryDirectory() as d:
        h5 = os.path.join(d, "rec_FNT.mad")
        ms.td_save_example(h5, spec, mask, meta, example_id="e1")
        bare = {k: v for k, v in meta.items() if k not in ("sample_rate", "nfft")}
        ms.td_save_example(h5, spec, mask, bare, example_id="e2")
        assert _attrs(h5, "e2") == {}
        with h5py.File(h5, "a") as f:                 # as written before ridges existed
            g = f["examples"]["e1"]
            del g[ms.EXAMPLE_RIDGE_DS]
            for k in mr.RIDGE_ATTRS + ("ridge_t_off",):
                del g.attrs[k]
        assert ms.add_example_ridges(h5) == 1
        assert _attrs(h5, "e1")["ridge_trill"] == 1
        assert ms.add_example_ridges(h5) == 0         # already there; e2 still has no grid
        assert set(ms.read_example_ridges(h5)) == {"e1"}


def test_add_ridge_cli_fills_recordings_and_stores():
    from fnt.usv.usv_detector import cli
    spec, mask, meta = _patch()
    with tempfile.TemporaryDirectory() as d:
        wav = os.path.join(d, "rec.wav")
        open(wav, "wb").close()
        h5 = ms.masks_sibling_path(wav)
        store = os.path.join(d, "training_data.h5")
        for path in (h5, store):
            ms.td_save_example(path, spec, mask, meta, example_id="e1")
            with h5py.File(path, "a") as f:
                g = f["examples"]["e1"]
                del g[ms.EXAMPLE_RIDGE_DS]
                for k in mr.RIDGE_ATTRS + ("ridge_t_off",):
                    del g.attrs[k]
        assert cli.main(["add-ridge", "--input", d, store]) == 0
        assert _attrs(h5, "e1")["ridge_trill"] == 1
        assert _attrs(store, "e1")["ridge_trill"] == 1


def test_the_csv_export_carries_a_labels_ridge():
    from fnt.usv.usv_detector import mad_csv_rebuild as cr
    spec, mask, meta = _patch()
    with tempfile.TemporaryDirectory() as d:
        h5 = os.path.join(d, "rec_FNT.mad")
        ms.set_grid_attrs(h5, sample_rate=SR, nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT,
                          n_freq_bins=N_FREQ, n_time_frames=5000)
        # detection 7: its crop holds a flat line; the accepted label (the edited mask) the trill
        flat = np.full((20, 120), 40, np.uint8)
        flat[10, :] = 250
        ms.write_pred_masks(h5, [{"blob_id": 7, "mask": np.ones((20, 120), np.uint8),
                                  "spec": flat, "f_off": 100, "t_off": 7}])
        ms.td_save_example(h5, spec, mask, dict(meta, blob_id=7), example_id="acc7")
        ms.td_save_example(h5, spec, mask, meta, example_id="painted")
        ms.td_save_example(h5, spec, mask, dict(meta, blob_id=7, kind="negative"),
                           example_id="neg7")
        rows = {"7": {"blob_id": 7, "status": "accepted"},
                "painted": {"blob_id": "painted", "status": "accepted"}}
        cr._add_ridge_fields(h5, rows)
        assert rows["7"]["ridge_trill"] == 1          # the label's, not the crop's (0)
        assert rows["painted"]["ridge_trill"] == 1    # painted by hand: no crop at all


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
