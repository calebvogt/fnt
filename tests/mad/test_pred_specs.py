"""Each stored detection keeps the spectrogram under its mask (/pred_specs).

A mask says where a call is, not what it looks like inside: a mask drawn as a
blob over a fast trill has a flat centroid, while the image inside it carries
the oscillation. Saving the image under every detection lets contour measures
run from the .mad file without the audio. Covers the store (round trip; kept
through rewrites that do not reshape a crop; dropped or replaced when they
do; deleted with its crop), the inference write path (spectrogram and model
stubbed), and the backfill for files analysed before.

Runs under pytest, or directly.
"""
import os
import tempfile

import numpy as np
import pytest

pytest.importorskip("h5py")

from fnt.usv.usv_detector import fnt_mask_store as ms  # noqa: E402
from fnt.usv.usv_detector import mad_inference as mi  # noqa: E402

SR = 250_000
H, T = 513, 3000


def grid():
    """A full-file image whose every pixel is distinguishable (uint8)."""
    f, t = np.mgrid[0:H, 0:T]
    return ((f * 7 + t * 3) % 251).astype(np.uint8)


def crop(fo, to, h=10, w=40, spec=None):
    c = {"blob_id": f"{fo}_{to}", "mask": np.ones((h, w), np.uint8), "f_off": fo, "t_off": to}
    if spec is not None:
        c["spec"] = ms.spec_crop(spec, fo, to, (h, w))
    return c


def test_spec_crop_takes_the_window_and_scales_floats():
    g = grid()
    assert np.array_equal(ms.spec_crop(g, 5, 9, (3, 4)), g[5:8, 9:13])
    f = np.full((4, 4), 0.5, np.float32)
    assert ms.spec_crop(f, 0, 0, (2, 2)).tolist() == [[128, 128], [128, 128]]


def test_a_crop_s_spec_round_trips_and_is_only_read_on_request():
    g = grid()
    with tempfile.TemporaryDirectory() as d:
        h5 = os.path.join(d, "r_FNT.mad")
        ms.write_pred_masks(h5, [crop(100, 200, spec=g)])
        assert "spec" not in ms.read_all_pred_masks(h5)["100_200"]
        got = ms.read_all_pred_masks(h5, include_spec=True)["100_200"]["spec"]
    assert np.array_equal(got, g[100:110, 200:240])


def test_a_rewrite_without_specs_keeps_them_for_unchanged_crops_only():
    """The GUI rewrites crops without ever holding the image."""
    g = grid()
    with tempfile.TemporaryDirectory() as d:
        h5 = os.path.join(d, "r_FNT.mad")
        ms.write_pred_masks(h5, [crop(100, 200, spec=g), crop(300, 900, spec=g)])
        moved = crop(300, 900)
        moved["t_off"] = 950                          # reshaped / moved by an edit
        ms.write_pred_masks(h5, [crop(100, 200), moved])
        got = ms.read_all_pred_masks(h5, include_spec=True)
    assert np.array_equal(got["100_200"]["spec"], g[100:110, 200:240])
    assert "spec" not in got["300_900"]               # the old window would be wrong


def test_subset_writes_and_deletes_keep_specs_in_step():
    g = grid()
    with tempfile.TemporaryDirectory() as d:
        h5 = os.path.join(d, "r_FNT.mad")
        ms.write_pred_masks(h5, [crop(100, 200, spec=g), crop(300, 900, spec=g),
                                 crop(50, 1500, spec=g)])
        ms.write_pred_mask_subset(h5, [crop(100, 200)])               # unchanged
        wider = crop(300, 900, w=60)                                  # reshaped
        ms.write_pred_mask_subset(h5, [wider])
        ms.delete_pred_mask(h5, "50_1500")
        got = ms.read_all_pred_masks(h5, include_spec=True)
        import h5py
        with h5py.File(h5, "r") as f:
            spec_keys = sorted(f[ms.PRED_SPEC_GROUP].keys())
    assert np.array_equal(got["100_200"]["spec"], g[100:110, 200:240])
    assert "spec" not in got["300_900"]
    assert spec_keys == ["100_200"]                   # none left behind


def test_inference_saves_the_image_under_every_new_detection(monkeypatch):
    g = grid()
    prob = np.zeros((H, T), np.float32)
    boxes = [(170, 700), (180, 2200)]
    for fo, to in boxes:
        prob[fo:fo + 10, to:to + 40] = 0.95
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "r.wav")
        open(wav, "wb").close()
        mad = ms.masks_sibling_path(wav)
        ms.set_grid_attrs(mad, sample_rate=SR, nperseg=512, noverlap=384,
                          nfft=1024, n_freq_bins=H, n_time_frames=T)
        monkeypatch.setattr(mi, "load_audio", lambda p, *a, **k: (np.zeros(T * 128, np.float32), SR))
        monkeypatch.setattr(mi, "compute_full_spec_image", lambda *a, **k: g.copy())
        monkeypatch.setattr(mi, "infer_probability_mask", lambda *a, **k: prob.copy())
        td = os.path.join(tmp, "training_data")
        os.makedirs(td)
        cfg = mi.MADInferenceConfig(model_path=os.path.join(tmp, "run", "weights.pt"),
                                    threshold=0.5, min_blob_pixels=8, merge_consecutive=False,
                                    save_blob_csv=False, training_data_dir=td)
        mi.run_inference_on_file(wav, cfg, model=object(), ckpt={}, device="cpu")
        got = ms.read_all_pred_masks(mad, include_spec=True)
    assert len(got) == 2
    for c in got.values():
        h, w = c["mask"].shape
        assert np.array_equal(c["spec"], g[c["f_off"]:c["f_off"] + h, c["t_off"]:c["t_off"] + w])


def test_backfill_adds_the_image_to_files_analysed_before(monkeypatch):
    g = grid()
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "r.wav")
        open(wav, "wb").close()
        mad = ms.masks_sibling_path(wav)
        ms.set_grid_attrs(mad, sample_rate=SR, nperseg=512, noverlap=384,
                          nfft=1024, n_freq_bins=H, n_time_frames=T)
        ms.write_pred_masks(mad, [crop(100, 200), crop(300, 900)])   # no specs: an old file
        monkeypatch.setattr(mi, "load_audio", lambda p, *a, **k: (np.zeros(T * 128, np.float32), SR))
        monkeypatch.setattr(mi, "compute_full_spec_image", lambda *a, **k: g.copy())
        assert mi.backfill_pred_specs(wav) == 2
        assert mi.backfill_pred_specs(wav) == 0                        # already filled
        got = ms.read_all_pred_masks(mad, include_spec=True)
    assert np.array_equal(got["300_900"]["spec"], g[300:310, 900:940])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
