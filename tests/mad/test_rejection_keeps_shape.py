"""A rejection is stored with its own shape, like every other example.

Reported from the field (2026-10-11): rejected detections reopened as filled
bounding boxes. A rejected prediction used to be saved with the OTHER confirmed
calls in its window as ``mask`` and its own shape nowhere, which broke three
things at once:

* reopening a file drew the rejection as a box (nothing else to draw), or as
  the outline of a confirmed neighbour;
* re-runs suppress detections on stored rejections by painting each one's
  ``mask``, so they painted nothing -- or the neighbour;
* ``neighbors`` is written exclusive of ``mask``, so it came out empty and the
  confirmed neighbour trained as background.

Now ``mask`` is the rejected call, ``neighbors`` the confirmed calls, and
:mod:`mad_fix_rejection_shapes` repairs the ones saved the old way.

Runs under pytest, or directly.
"""
import json
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np

from fnt.usv.usv_detector import fnt_mask_store as ms
from fnt.usv.usv_detector import mad_examples as mx
from fnt.usv.usv_detector import mad_fix_rejection_shapes as fix
from fnt.usv.usv_detector.mad_migrate_composites import split_example

H, W = 64, 80
SR, NPERSEG, NOVERLAP, NFFT = 250_000, 512, 384, 1024
DT = (NPERSEG - NOVERLAP) / SR
DF = (SR / 2.0) / (NFFT // 2)


def meta_for(t0f, t1f, f0, f1, kind="negative", patch_t_off=0, **extra):
    m = dict(
        **{"class": "USV"}, source_wav="r.wav",
        patch_t_off=patch_t_off, patch_f_off=0,
        t_start_s=round((patch_t_off + t0f) * DT, 9),
        t_stop_s=round((patch_t_off + t1f) * DT, 9),
        patch_t0_s=round(patch_t_off * DT, 9),
        patch_t1_s=round((patch_t_off + W) * DT, 9),
        f_low_hz=round(f0 * DF, 4), f_high_hz=round(f1 * DF, 4),
        patch_t_frames=W, f_bins=H,
        nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT, sample_rate=SR,
        db_min=-100.0, db_max=-20.0, db_norm="fixed", blob_id=7,
    )
    if kind != "label":
        m["kind"] = kind
    m.update(extra)
    return m


def tone_patch(rng=None):
    """Noise with a stationary tone across the whole patch at rows 30-32 --
    the ~21 kHz electrical tone that most rejections are."""
    rng = rng or np.random.default_rng(0)
    spec = rng.uniform(0.20, 0.35, size=(H, W)).astype(np.float32)
    spec[30:33, :] = 0.8
    return (spec * 255).astype(np.uint8)


def neighbour():
    nb = np.zeros((H, W), bool)
    nb[45:55, 60:70] = True
    return nb


def crop_shape():
    m = np.zeros((H, W), bool)
    m[29:34, 20:40] = True
    m[29, 20:25] = False          # a non-rectangular outline
    return m


# ----------------------------------------------------------------------
# Saving
# ----------------------------------------------------------------------
_GUI = None


def gui():
    global _GUI
    if _GUI is None:
        from PyQt5.QtWidgets import QApplication
        from scipy.io import wavfile
        import fnt.usv.mad_pyqt as M
        from fnt.usv.usv_detector.mad_project import (
            MADProjectConfig, create_mad_project)
        root = tempfile.mkdtemp(prefix="mad_rejshape_")
        wav = os.path.join(root, "rec", "a.wav")
        os.makedirs(os.path.dirname(wav), exist_ok=True)
        wavfile.write(wav, SR, (np.random.default_rng(0).normal(0, .05, SR * 2)
                                * 32767).astype(np.int16))
        proj = os.path.join(root, "proj")
        create_mad_project(proj)
        app = QApplication.instance() or QApplication([])
        w = M.MADMainWindow()
        w._activate_project(MADProjectConfig.load(proj))
        w._register_audio_files([wav])
        w._append_audio_paths([wav])
        w._wait_for_audio_load()
        _GUI = (w, wav, app, M)
    return _GUI


def _reject_with_a_confirmed_neighbour():
    w, wav, _app, M = gui()
    h5 = ms.masks_sibling_path(wav)
    for i in ms.td_list_ids(h5):
        ms.td_delete(h5, i)
    own = np.zeros((12, 20), bool)
    own[3:9, 4:16] = True
    own[3, 4:8] = False
    conf = np.ones((10, 15), bool)
    sg = w.spectrogram
    pred = {'id': M._new_ann_id('pred'), 'blob_id': 7, 'category': 'USV',
            'status': 'prediction', 'score': 0.9,
            'f0': 100, 'f1': 112, 't0': 300, 't1': 320, 'mask': own}
    confirmed = {'id': 'lbl1', 'category': 'USV', 'status': None,
                 'f0': 150, 'f1': 160, 't0': 360, 't1': 375, 'mask': conf}
    sg.annotations = [pred, confirmed]
    sg._rebuild_confirmed_mask()
    ex_id = w._save_rejection_negative(pred)
    assert ex_id
    ex = ms.td_read_example(h5, ex_id)
    return ex, own, conf


def test_a_new_rejection_stores_its_own_shape():
    ex, own, _conf = _reject_with_a_confirmed_neighbour()
    m = ex["mask"] > 0
    t_off = ex["meta"]["patch_t_off"]
    got = m[100:112, 300 - t_off:320 - t_off]
    assert np.array_equal(got, own), "mask must be the rejected call's shape"
    assert m.sum() == own.sum(), "and nothing else"
    assert ex["meta"].get("mask_is_own") is True


def test_a_new_rejection_keeps_its_confirmed_neighbour_apart():
    ex, _own, conf = _reject_with_a_confirmed_neighbour()
    t_off = ex["meta"]["patch_t_off"]
    assert ex["neighbors"] is not None, "the neighbour was dropped"
    nb = ex["neighbors"] > 0
    assert nb[150:160, 360 - t_off:375 - t_off].all()
    assert not (nb & (ex["mask"] > 0)).any()


def test_a_new_rejection_reopens_as_its_shape_not_a_box():
    """What the user saw: rejects drawn as boxes after reopening a file."""
    w, wav, _app, _M = gui()
    ex, own, _ = _reject_with_a_confirmed_neighbour()
    anns = list(mx._examples_to_annotations(
        ms.td_iter_file_examples(ms.masks_sibling_path(wav),
                                 os.path.basename(wav), with_spec=False,
                                 kinds=ms.REJECTED_KINDS),
        os.path.basename(wav),
        (w.spectrogram.n_freq_bins, w.spectrogram.n_time_frames),
        kinds=ms.REJECTED_KINDS))
    assert len(anns) == 1
    a = anns[0]
    assert (a["f0"], a["t0"]) == (103, 304)
    assert not a["mask"].all(), "drawn as a filled box"
    assert a["mask"].sum() == own.sum()


# ----------------------------------------------------------------------
# What reads a rejection
# ----------------------------------------------------------------------
def _store_one(d, mask, meta, neighbors=None):
    p = os.path.join(d, "r_FNT.mad")
    ms.td_save_example(p, tone_patch(), mask, meta, "r_neg_7",
                       neighbors_patch=neighbors)
    return p


def test_training_still_sees_a_rejection_as_all_background():
    own = crop_shape()
    with tempfile.TemporaryDirectory() as d:
        os.rename(_store_one(d, own, meta_for(20, 40, 29, 34), neighbour()),
                  os.path.join(d, "training_data.h5"))
        _s, t, w = mx.collect_training_examples(
            d, tile_time_frames=W, tile_freq_bins=H, placements=1, seed=0)
    assert t[0].max() < 0.5, "a rejection's own shape must not become a target"
    assert w[0][45:55, 60:70].max() < 0.5, "confirmed neighbour must be ignored"
    assert w[0][29:34, 20:40].min() > 0.5, "the rejected call stays supervised"


def test_rerun_suppression_paints_the_rejected_call():
    own = crop_shape()
    with tempfile.TemporaryDirectory() as d:
        p = _store_one(d, own, meta_for(20, 40, 29, 34, patch_t_off=500),
                       neighbour())
        rej = mx.reconstruct_file_mask("", "r.wav", (H, 1000),
                                       extra_stores=[p],
                                       kinds=ms.REJECTED_KINDS)
    assert np.array_equal(rej[:, 500:500 + W] > 0, own)
    assert not rej[45:55, 560:570].any(), "the neighbour is not a rejection"


# ----------------------------------------------------------------------
# Re-tracing
# ----------------------------------------------------------------------
def test_trace_follows_a_stationary_tone_not_the_box():
    """A per-row floor would subtract a tone that spans the patch; the trace
    has to find it anyway, and stay a shape rather than the rectangle."""
    box = (25, 38, 20, 40)
    shape = fix.trace_shape(tone_patch(), box)
    assert shape[30:33, 20:40].mean() > 0.9, "the tone itself was missed"
    assert shape.sum() < 0.6 * (13 * 20), "traced the box, not the tone"
    assert not shape[:, :20].any() and not shape[:, 40:].any()


def test_trace_never_claims_a_confirmed_neighbour():
    box = (25, 38, 20, 40)
    avoid = np.zeros((H, W), bool)
    avoid[30:33, 30:40] = True
    shape = fix.trace_shape(tone_patch(), box, avoid=avoid)
    assert not (shape & avoid).any()
    assert shape[30:33, 20:30].any()


def test_trace_of_pure_noise_is_still_a_shape():
    spec = (np.random.default_rng(3).uniform(0.2, 0.4, (H, W)) * 255
            ).astype(np.uint8)
    box = (10, 30, 10, 50)
    shape = fix.trace_shape(spec, box)
    assert shape.any(), "a rejection must never be left with nothing to draw"
    assert shape.sum() < (20 * 40), "never the filled box"


# ----------------------------------------------------------------------
# Repairing rejections saved the old way
# ----------------------------------------------------------------------
def _legacy_store(d, mask, *, crop=None, meta=None):
    """A rejection as the old code wrote it: no shape of its own."""
    p = os.path.join(d, "r_FNT.mad")
    meta = meta or meta_for(20, 40, 29, 34)
    ms.td_save_example(p, tone_patch(), mask, meta, "r_neg_7")
    if crop is not None:
        ms.write_pred_masks(p, [crop])
    return p


def test_repair_takes_the_models_crop_when_it_is_still_there():
    own = crop_shape()
    crop = {"blob_id": 7, "mask": own[29:34, 20:40], "f_off": 29,
            "t_off": 20}
    with tempfile.TemporaryDirectory() as d:
        p = _legacy_store(d, np.zeros((H, W), bool), crop=crop)
        rep = fix.repair_store(p, dry_run=False, backup=False)
        ex = ms.td_read_example(p, "r_neg_7")
    assert rep.repaired == rep.from_crop == 1
    assert np.array_equal(ex["mask"] > 0, own)
    assert ex["meta"]["shape_source"] == "pred_crop"
    assert ex["meta"]["mask_is_own"] is True


def test_repair_ignores_a_crop_that_is_a_different_detection():
    """Ids are reused across runs; a crop elsewhere must not lend its shape."""
    elsewhere = {"blob_id": 7, "mask": np.ones((5, 10), bool), "f_off": 5,
                 "t_off": 60}
    with tempfile.TemporaryDirectory() as d:
        p = _legacy_store(d, np.zeros((H, W), bool), crop=elsewhere)
        rep = fix.repair_store(p, dry_run=False, backup=False)
        ex = ms.td_read_example(p, "r_neg_7")
    assert rep.traced == 1
    assert not (ex["mask"] > 0)[5:10, 60:70].any()
    assert (ex["mask"] > 0)[30:33, 20:40].mean() > 0.9


def test_repair_moves_the_old_neighbours_out_of_mask():
    """The 592 rejections whose `mask` held a confirmed call: that call goes
    to `neighbors`, where training ignores it instead of teaching it as noise."""
    nb = neighbour()
    with tempfile.TemporaryDirectory() as d:
        p = _legacy_store(d, nb)
        rep = fix.repair_store(p, dry_run=False, backup=False)
        ex = ms.td_read_example(p, "r_neg_7")
    assert rep.neighbors_moved == 1
    assert np.array_equal(ex["neighbors"] > 0, nb)
    m = ex["mask"] > 0
    assert not (m & nb).any()
    assert m[30:33, 20:40].mean() > 0.9


def test_repair_dry_run_writes_nothing_and_apply_is_idempotent():
    with tempfile.TemporaryDirectory() as d:
        p = _legacy_store(d, np.zeros((H, W), bool))
        dry = fix.repair_store(p, dry_run=True, backup=False)
        assert dry.repaired == 1
        assert not (ms.td_read_example(p, "r_neg_7")["mask"] > 0).any()
        fix.repair_store(p, dry_run=False, backup=False)
        again = fix.repair_store(p, dry_run=False, backup=False)
    assert again.repaired == 0 and again.negatives == 1


def test_repair_backs_up_before_writing():
    with tempfile.TemporaryDirectory() as d:
        p = _legacy_store(d, np.zeros((H, W), bool))
        rep = fix.repair_store(p, dry_run=False, backup=True)
        assert rep.backup == p + fix.BACKUP_SUFFIX
        old = ms.td_read_example(rep.backup, "r_neg_7")
    assert not (old["mask"] > 0).any(), "the backup must be the original"


def test_repair_leaves_labels_and_demoted_calls_alone():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "r_FNT.mad")
        own = crop_shape()
        ms.td_save_example(p, tone_patch(), own,
                           meta_for(20, 40, 29, 34, kind="label"), "lbl")
        ms.td_save_example(p, tone_patch(), own,
                           meta_for(20, 40, 29, 34, kind="rejected"), "dem")
        rep = fix.repair_store(p, dry_run=False, backup=False)
        assert rep.repaired == 0
        with h5py.File(p, "r") as f:
            for k in ("lbl", "dem"):
                meta = json.loads(f["examples"][k].attrs["meta_json"])
                assert "shape_source" not in meta


def test_unplaceable_metadata_is_reported_not_guessed():
    bad = meta_for(20, 40, 29, 34)
    del bad["nfft"]
    with tempfile.TemporaryDirectory() as d:
        p = _legacy_store(d, np.zeros((H, W), bool), meta=bad)
        rep = fix.repair_store(p, dry_run=False, backup=False)
        assert rep.unplaceable == ["r_neg_7"] and rep.repaired == 0
        assert not (ms.td_read_example(p, "r_neg_7")["mask"] > 0).any()


def test_the_composite_migration_leaves_a_repaired_rejection_alone():
    """Re-running the older migration must not move the rejected shape into
    `neighbors`, which would protect it like a confirmed call."""
    own = crop_shape()
    res = split_example(own, meta_for(20, 40, 29, 34, mask_is_own=True))
    assert np.array_equal(res["own"], own)
    assert not res["neighbors"].any()


def test_a_registry_project_finds_its_stores():
    """``audio_files`` holds registry entries now, not path strings; reading
    the raw JSON found zero stores on the anosmia project."""
    from scipy.io import wavfile
    from fnt.usv.usv_detector.mad_migrate_composites import project_stores
    from fnt.usv.usv_detector.mad_project import (
        MADProjectConfig, create_mad_project)
    from fnt.usv.usv_detector.mad_registry import RegisteredFile
    with tempfile.TemporaryDirectory() as d:
        wav = os.path.join(d, "rec", "a.wav")
        os.makedirs(os.path.dirname(wav))
        wavfile.write(wav, SR, np.zeros(SR // 10, np.int16))
        _legacy_store(os.path.dirname(wav), np.zeros((H, W), bool))
        os.replace(os.path.join(os.path.dirname(wav), "r_FNT.mad"),
                   ms.masks_sibling_path(wav))
        proj = os.path.join(d, "proj")
        create_mad_project(proj)
        cfg = MADProjectConfig.load(proj)
        cfg.set_audio_entries([RegisteredFile(path=wav,
                                              basename=os.path.basename(wav))])
        cfg.save()
        assert project_stores(proj) == [ms.masks_sibling_path(wav)]
        reps = fix.repair_project(proj, dry_run=True)
    assert sum(r.repaired for r in reps) == 1


if __name__ == "__main__":
    import sys
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print("  OK   " + name, flush=True)
        except Exception:
            fails += 1
            print("  FAIL " + name, flush=True)
            traceback.print_exc()
    print("")
    print("ALL OK" if not fails else str(fails) + " FAILURE(S)", flush=True)
    sys.stdout.flush()
    os._exit(1 if fails else 0)
