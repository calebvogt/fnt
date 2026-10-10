"""Re-running inference must respect the decisions already stored.

Two things a re-run used to learn from the CSV, and silently stopped knowing
when the CSV became an export:

* **Which ids are taken.** New detections were numbered from 0 every run, so
  a fresh detection inherited the id of an old, unrelated accept or reject.
  Review hid it as "already loaded" and the store counted it as decided --
  456 detections across 52 anosmia recordings, 404 of them behind an accepted
  label, invisible to QC.
* **Which calls were rejected.** Every run re-proposed calls the reviewer had
  turned down; 19 rejections in the anosmia sidecars duplicate an earlier
  rejection of the same call.

The store's examples are the record of every decision, so both now come from
there. Runs the real ``run_inference_on_file`` write path with the model and
spectrogram stubbed.

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
H, T = 513, 3000          # full-file grid: freq bins x frames
WAV = "r.wav"


def _meta(t_off, f_off, h, w, kind=None, blob_id=None, ex_id=None):
    dt = 128 / SR
    m = {"source_wav": WAV, "patch_t_off": t_off, "patch_f_off": f_off,
         "t_start_s": t_off * dt, "t_stop_s": (t_off + w) * dt,
         "f_low_hz": f_off * SR / 1024, "f_high_hz": (f_off + h) * SR / 1024,
         "sample_rate": SR, "nperseg": 512, "noverlap": 384, "nfft": 1024,
         "class": "USV"}
    if kind:
        m["kind"] = kind
    if blob_id is not None:
        m["blob_id"] = blob_id
    if ex_id:
        m["id"] = ex_id
    return m


# Regions on the grid (f0, t0) of 10 x 40 boxes.
LABELLED = (150, 200)       # accepted earlier, blob_id 0
REJECTED = (160, 1200)      # rejected earlier, blob_id 1
NEW_A = (170, 700)          # genuinely new
NEW_B = (180, 2200)         # genuinely new


def _box(prob, where, v=0.95):
    f, t = where
    prob[f:f + 10, t:t + 40] = v


def run(tmp, monkeypatch, *, preserve=True):
    wav = os.path.join(tmp, WAV)
    open(wav, "wb").close()                     # load_audio is stubbed
    mad = ms.masks_sibling_path(wav)
    ms.set_grid_attrs(mad, sample_rate=SR, nperseg=512, noverlap=384,
                      nfft=1024, n_freq_bins=H, n_time_frames=T)
    mask = np.ones((10, 40), np.uint8)
    ms.td_save_example(mad, np.zeros((10, 40), np.float32), mask,
                       _meta(LABELLED[1], LABELLED[0], 10, 40, blob_id=0,
                             ex_id="acc0"), "acc0")
    ms.td_save_example(mad, np.zeros((10, 40), np.float32), mask,
                       _meta(REJECTED[1], REJECTED[0], 10, 40,
                             kind="negative", blob_id=1, ex_id="rej1"),
                       "rej1")
    td = os.path.join(tmp, "training_data")
    os.makedirs(td, exist_ok=True)

    prob = np.zeros((H, T), np.float32)
    for where in (LABELLED, REJECTED, NEW_A, NEW_B):
        _box(prob, where)
    monkeypatch.setattr(mi, "load_audio",
                        lambda p, *a, **k: (np.zeros(T * 128, np.float32), SR))
    monkeypatch.setattr(mi, "compute_full_spec_image",
                        lambda *a, **k: np.zeros((H, T), np.float32))
    monkeypatch.setattr(mi, "infer_probability_mask",
                        lambda *a, **k: prob.copy())
    cfg = mi.MADInferenceConfig(model_path=os.path.join(tmp, "run",
                                                        "weights.pt"),
                                threshold=0.5, min_blob_pixels=8,
                                merge_consecutive=False, save_blob_csv=False,
                                preserve_labels=preserve,
                                training_data_dir=td)
    mi.run_inference_on_file(wav, cfg, model=object(), ckpt={},
                             device="cpu")
    return mad


def crops_at(mad):
    return {k: (c["f_off"], c["t_off"])
            for k, c in ms.read_all_pred_masks(mad).items()}


def near(a, b, tol=6):
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


# ----------------------------------------------------------------------
def test_new_detections_never_take_a_stored_decisions_id(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        mad = run(tmp, monkeypatch)
        ids = {int(k) for k in crops_at(mad)}
    assert ids, "nothing was written"
    assert not ids & {0, 1}, f"a new detection reused a decided id: {ids}"


def test_a_rejected_call_is_not_proposed_again(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        got = crops_at(run(tmp, monkeypatch))
    assert not any(near(v, REJECTED) for v in got.values()), got


def test_genuinely_new_calls_still_come_through(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        got = list(crops_at(run(tmp, monkeypatch)).values())
    assert any(near(v, NEW_A) for v in got) and \
        any(near(v, NEW_B) for v in got), got
    assert len(got) == 2, "the labelled and the rejected call must not return"


def test_every_new_detection_counts_as_pending(monkeypatch):
    """The badge and 'next file with pending' read this: a new detection that
    inherited a decision's id read as already decided."""
    with tempfile.TemporaryDirectory() as tmp:
        mad = run(tmp, monkeypatch)
        claimed = {str(m.get("blob_id")) for m in ms.td_iter_meta(mad)
                   if m.get("blob_id") is not None}
        pending = [k for k in ms.list_pred_ids(mad) if k not in claimed]
    assert len(pending) == 2


def test_re_detect_from_scratch_ignores_rejections(monkeypatch):
    """preserve_labels off means look everywhere again, rejections included."""
    with tempfile.TemporaryDirectory() as tmp:
        got = crops_at(run(tmp, monkeypatch, preserve=False))
    assert any(near(v, REJECTED) for v in got.values()), got


def test_decision_ids_are_read_from_the_store():
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, WAV)
        mad = ms.masks_sibling_path(wav)
        z = np.zeros((4, 4), np.float32)
        o = np.ones((4, 4), np.uint8)
        ms.td_save_example(mad, z, o, _meta(0, 0, 4, 4, blob_id=7,
                                            ex_id="a"), "a")
        ms.td_save_example(mad, z, o, _meta(9, 0, 4, 4, blob_id="12",
                                            ex_id="b"), "b")
        # A hand label's prefixed id can never collide with a run's integers.
        ms.td_save_example(mad, z, o, _meta(20, 0, 4, 4,
                                            blob_id="r_5", ex_id="c"), "c")
        ms.td_save_example(mad, z, o, dict(_meta(30, 0, 4, 4, blob_id=99,
                                                 ex_id="d"),
                                           source_wav="other.wav"), "d")
        assert sorted(mi.stored_decision_ids(wav)) == [7, 12]


def test_rejections_can_be_painted_but_are_not_labels():
    """reconstruct_file_mask paints labels by default (the label shield) and
    rejections only when asked (the rejection suppressor)."""
    from fnt.usv.usv_detector.mad_examples import reconstruct_file_mask
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, WAV)
        mad = ms.masks_sibling_path(wav)
        o = np.ones((10, 40), np.uint8)
        z = np.zeros((10, 40), np.float32)
        ms.td_save_example(mad, z, o, _meta(100, 50, 10, 40, ex_id="L"), "L")
        ms.td_save_example(mad, z, o, _meta(500, 50, 10, 40, kind="negative",
                                            ex_id="N"), "N")
        labels = reconstruct_file_mask("", WAV, (H, T), extra_stores=[mad])
        rej = reconstruct_file_mask("", WAV, (H, T), extra_stores=[mad],
                                    kinds=ms.REJECTED_KINDS)
    assert labels[:, 100:140].any() and not labels[:, 500:540].any()
    assert rej[:, 500:540].any() and not rej[:, 100:140].any()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
