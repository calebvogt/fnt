"""Join fragments: ``mad join-fragments`` makes pending detections that are
pieces of one call into one detection, after inference (mad_join, 2026-11-09).

The 2021_8x8 production run stored a faint call broken by the threshold as
several detections. The join gates on the pitch carrying on across the gap
(the earlier piece's end frequency vs the later one's start, from the
brightness ridge), and must:

* join a sweep split in two, and a call split in three (a chain);
* not join pieces further apart than the gap, or whose pitch jumps;
* not join a harmonic stacked above its fundamental, nor two concurrent
  calls in separate bands;
* never touch a reviewed call;
* undo exactly (datasets, attrs, filters and root attrs as they were), and on
  a rerun start from the originals;
* leave a file the GUI opens with the joined call as an ordinary pending
  detection.

Runs under pytest, or directly.
"""
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from fnt.usv.usv_detector import fnt_mask_store as MS  # noqa: E402
from fnt.usv.usv_detector import mad_join as J  # noqa: E402
from fnt.usv.usv_detector.cli import main as mad_main  # noqa: E402

SR, HOP, NFFT = 250_000, 128, 1024
DF_KHZ = SR / NFFT / 1000.0              # 0.244 kHz per bin


def sweep(t0, t1, f_start, slope, thick=3):
    """A crop of a straight FM line from frame t0 to t1 (exclusive) starting at
    bin ``f_start`` and moving ``slope`` bins per frame: (mask, spec, f_off,
    t_off), the spectrogram bright on the line and dim around it."""
    n = t1 - t0
    rows = np.round(f_start + slope * np.arange(n)).astype(int)
    f_lo, f_hi = rows.min() - 2, rows.max() + thick + 2
    mask = np.zeros((f_hi - f_lo, n), np.uint8)
    spec = np.full(mask.shape, 40, np.uint8)
    for j, r in enumerate(rows):
        mask[r - f_lo:r - f_lo + thick, j] = 1
        spec[r - f_lo:r - f_lo + thick, j] = (150, 220, 150)[:thick]
    return mask, spec, int(f_lo), int(t0)


def crop(blob_id, t0, t1, f_start, slope=0.0, score=0.9, harmonic_n=1, call=None):
    mask, spec, f_off, t_off = sweep(t0, t1, f_start, slope)
    return {"blob_id": blob_id, "mask": mask, "spec": spec, "f_off": f_off,
            "t_off": t_off, "score": score, "class": "USV",
            "model_name": "unit", "threshold": 0.5, "min_blob_pixels": 50,
            "harmonic_n": harmonic_n,
            "harmonic_call_id": str(call if call is not None else blob_id)}


def make(crops, name="rec"):
    root = tempfile.mkdtemp(prefix="mad_join_")
    h5 = os.path.join(root, name + MS.MAD_SUFFIX)
    MS.set_grid_attrs(h5, sample_rate=SR, nperseg=512, noverlap=384, nfft=NFFT,
                      n_freq_bins=513, n_time_frames=20000)
    MS.set_infer_run_attrs(h5, last_infer_model="unit", last_infer_at="2026-11-09T10:00:00")
    with h5py.File(h5, "a") as f:
        f.attrs["source_wav"] = name + ".wav"
    MS.write_pred_masks(h5, crops)
    return h5


def ids(h5):
    return set(MS.list_pred_ids(h5))


def attrs_of(h5, blob_id):
    with h5py.File(h5, "r") as f:
        return {k: J._attr(v) for k, v in f[MS.PRED_GROUP][str(blob_id)].attrs.items()}


def snapshot(h5):
    """Every dataset under the prediction groups and examples, with its
    dtype, filters, bytes and attrs, plus the root attrs."""
    out = {}
    with h5py.File(h5, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                out[name] = (str(obj.dtype), obj.shape, obj.compression,
                             obj.compression_opts, obj[()].tobytes(),
                             sorted((k, repr(J._attr(v))) for k, v in obj.attrs.items()))
            else:
                out[name + "/"] = sorted((k, repr(J._attr(v))) for k, v in obj.attrs.items())
        f.visititems(visit)
        out["/"] = sorted((k, repr(J._attr(v))) for k, v in f.attrs.items())
    return out


def label(h5, blob_id, t0, t1, f0, f1, kind=None):
    """An example in the .mad, as review leaves one."""
    t_org = 512 / (2.0 * SR)
    meta = {"id": f"ex_{blob_id}", "blob_id": blob_id, "source_wav": "rec.wav",
            "t_start_s": t_org + t0 * HOP / SR, "t_stop_s": t_org + t1 * HOP / SR,
            "f_low_hz": f0 * DF_KHZ * 1000, "f_high_hz": f1 * DF_KHZ * 1000,
            "patch_t_off": t0, "patch_f_off": f0}
    if kind:
        meta["kind"] = kind
    MS.td_save_example(h5, np.zeros((f1 - f0, t1 - t0), np.float32),
                       np.ones((f1 - f0, t1 - t0), np.uint8), meta, example_id=f"ex_{blob_id}")


# A downsweep from bin 300 at 0.5 bins/frame, broken at frames 1040-1044
# (a 2 ms gap): piece 2 starts where piece 1 left off.
SPLIT = [crop(1, 1000, 1040, 300, -0.5), crop(2, 1044, 1080, 300 - 0.5 * 44, -0.5)]


def test_a_sweep_split_in_two_is_joined():
    h5 = make(SPLIT + [crop(3, 5000, 5040, 250)])
    r = J.join_file(h5)
    assert r["before"] == 3 and r["after"] == 2 and r["pieces"] == [2]
    assert ids(h5) == {"1", "3"}
    a = attrs_of(h5, 1)
    assert a["joined_from"] == "1,2" and a["n_pieces"] == 2
    assert a["t_off"] == 1000 and a["model_name"] == "unit"
    assert abs(a["score"] - 0.9) < 1e-6
    rec = MS.read_all_pred_masks(h5, include_spec=True, include_ridge=True)["1"]
    assert rec["mask"].shape[1] == 80
    assert not rec["mask"][:, 40:44].any(), "the gap stays empty"
    assert rec["mask"].sum() == SPLIT[0]["mask"].sum() + SPLIT[1]["mask"].sum()
    assert rec["spec"].shape == rec["mask"].shape and not rec["spec"][:, 40:44].any()
    assert "ridge_turns" in rec and np.isfinite(rec["ridge_khz"][:40]).all()
    with h5py.File(h5, "r") as f:
        assert f.attrs["join_gap_s"] == 0.01 and f.attrs["join_max_df_khz"] == 5.0
        assert f.attrs["n_pred_blobs"] == 2 and "joined_at" in f.attrs
        assert "join_fnt_version" in f.attrs
        assert set(f[J.UNJOINED_GROUP][MS.PRED_GROUP]) == {"1", "2"}


def test_pieces_further_apart_than_the_gap_stay_apart():
    # Same pitch, 30 frames (15 ms) apart: two calls at 10 ms, one at 20.
    far = [crop(1, 1000, 1040, 300), crop(2, 1070, 1110, 300)]
    h5 = make(far)
    assert J.join_file(h5)["n_joined"] == 0 and ids(h5) == {"1", "2"}
    with h5py.File(h5, "r") as f:
        assert J.UNJOINED_GROUP not in f
    assert J.join_file(h5, J.JoinConfig(max_gap_s=0.020))["n_joined"] == 1


def test_a_pitch_jump_larger_than_the_limit_is_not_joined():
    # 2 frames apart, but piece 2 starts 30 bins (7.3 kHz) below piece 1's end.
    jump = [crop(1, 1000, 1040, 300), crop(2, 1042, 1080, 270)]
    h5 = make(jump)
    assert J.join_file(h5)["n_joined"] == 0 and ids(h5) == {"1", "2"}
    assert J.join_file(h5, J.JoinConfig(max_df_khz=8.0))["n_joined"] == 1


def test_a_harmonic_above_its_fundamental_is_not_joined():
    stack = [crop(1, 1000, 1080, 200), crop(2, 1000, 1080, 400, harmonic_n=2, call=1)]
    h5 = make(stack)
    assert J.join_file(h5, J.JoinConfig(max_df_khz=100))["n_joined"] == 0
    # Not yet grouped (both fundamentals) and close enough in pitch, but
    # sounding together in separate bands: two calls.
    together = [crop(1, 1000, 1080, 200), crop(2, 1002, 1080, 208)]
    h5 = make(together)
    assert J.join_file(h5)["n_joined"] == 0 and ids(h5) == {"1", "2"}


def test_a_reviewed_call_is_never_touched():
    for kind in (None, "negative"):
        h5 = make(SPLIT)
        label(h5, 2, 1044, 1080, 270, 285, kind=kind)
        before = snapshot(h5)
        r = J.join_file(h5)
        assert r["n_joined"] == 0 and snapshot(h5) == before
    # A hand-drawn label (no blob_id) over a piece claims it too.
    h5 = make(SPLIT)
    label(h5, None, 1000, 1040, 270, 305)
    assert J.join_file(h5)["n_joined"] == 0


def test_a_call_in_three_pieces_is_joined_as_a_chain():
    three = SPLIT + [crop(4, 1086, 1120, 300 - 0.5 * 86, -0.5), crop(9, 1300, 1340, 100)]
    h5 = make(three)
    r = J.join_file(h5)
    assert r["pieces"] == [3] and ids(h5) == {"1", "9"}
    assert attrs_of(h5, 1)["joined_from"] == "1,2,4"


def test_undo_puts_the_originals_back_exactly():
    three = SPLIT + [crop(4, 1086, 1120, 300 - 0.5 * 86, -0.5), crop(9, 1300, 1340, 100)]
    h5 = make(three)
    before = snapshot(h5)
    J.join_file(h5)
    assert snapshot(h5) != before
    r = J.undo_file(h5)
    assert r["joins"] == 1 and r["restored"] == 3
    assert snapshot(h5) == before
    assert J.undo_file(h5)["status"] == "not_joined"


def test_undo_keeps_a_stale_detection_count_as_it_was():
    # The GUI does not always keep n_pred_blobs in step; undo restores the
    # value the file had, not a recount (found on the 2021_8x8 copies).
    h5 = make(SPLIT)
    with h5py.File(h5, "a") as f:
        f.attrs["n_pred_blobs"] = 7
    before = snapshot(h5)
    J.join_file(h5)
    with h5py.File(h5, "r") as f:
        assert f.attrs["n_pred_blobs"] == 1
    J.undo_file(h5)
    assert snapshot(h5) == before
    J.join_file(h5, J.JoinConfig(max_gap_s=0.001))      # joins nothing
    assert snapshot(h5) == before


def test_a_rerun_starts_from_the_originals():
    pieces = SPLIT + [crop(5, 3000, 3040, 300), crop(6, 3070, 3110, 300)]
    h5 = make(pieces)
    before = snapshot(h5)
    assert J.join_file(h5)["pieces"] == [2]                     # 10 ms: 1+2 only
    r = J.join_file(h5, J.JoinConfig(max_gap_s=0.020))          # 20 ms: 5+6 too
    assert sorted(r["pieces"]) == [2, 2] and r["before"] == 4
    assert attrs_of(h5, 5)["joined_from"] == "5,6"
    assert attrs_of(h5, 1)["joined_from"] == "1,2", "not a join of a joined call"
    with h5py.File(h5, "r") as f:
        assert abs(f.attrs["join_gap_s"] - 0.02) < 1e-12
    assert J.join_file(h5, J.JoinConfig(max_gap_s=0.001))["n_joined"] == 0
    assert snapshot(h5) == before, "a rerun that joins nothing leaves the original"


def test_dry_run_writes_nothing_and_reports_from_the_originals():
    h5 = make(SPLIT)
    J.join_file(h5)
    joined = snapshot(h5)
    r = J.join_file(h5, J.JoinConfig(max_gap_s=0.001), dry_run=True)
    assert r["before"] == 2 and r["after"] == 2 and r["n_joined"] == 0
    assert snapshot(h5) == joined


def test_a_call_reviewed_after_the_join_stays_joined_on_undo():
    h5 = make(SPLIT + [crop(5, 3000, 3040, 300), crop(6, 3044, 3080, 300)])
    J.join_file(h5)
    label(h5, 1, 1000, 1080, 270, 305)                 # the joined call accepted
    r = J.undo_file(h5)
    assert r["joins"] == 1 and r["kept"] == 1
    assert ids(h5) == {"1", "5", "6"}
    assert attrs_of(h5, 1)["n_pieces"] == 2
    with h5py.File(h5, "r") as f:
        assert set(f[J.UNJOINED_GROUP][MS.PRED_GROUP]) == {"1", "2"}


def test_a_reinference_makes_the_copies_stale():
    h5 = make(SPLIT)
    J.join_file(h5)
    MS.write_pred_masks(h5, SPLIT)                     # inference writes anew
    MS.set_infer_run_attrs(h5, last_infer_at="2026-11-10T10:00:00")
    fresh = snapshot(h5)
    assert J.undo_file(h5)["status"] == "stale"
    assert ids(h5) == {"1", "2"}
    with h5py.File(h5, "r") as f:
        assert J.UNJOINED_GROUP not in f and "join_gap_s" not in f.attrs
    assert {k: v for k, v in snapshot(h5).items() if not k.startswith(J.UNJOINED_GROUP)
            and k != "/"} == {k: v for k, v in fresh.items()
                              if not k.startswith(J.UNJOINED_GROUP) and k != "/"}


def test_harmonics_of_a_joined_call_follow_it():
    # 1+2 is a sweep; 7 is the second harmonic of its second half, grouped
    # with piece 2 at inference. Once 2 is part of 1, 7 must not name it.
    fund = [crop(1, 1000, 1040, 150, -0.25), crop(2, 1044, 1080, 150 - 0.25 * 44, -0.25)]
    h = crop(7, 1044, 1080, 2 * (150 - 0.25 * 44), -0.5, harmonic_n=2, call=2)
    h5 = make(fund + [h])
    r = J.join_file(h5)
    assert r["n_joined"] == 1 and r["regrouped"] >= 1
    a7 = attrs_of(h5, 7)
    assert a7["harmonic_call_id"] == "1" and a7["harmonic_n"] == 2
    assert attrs_of(h5, 1)["harmonic_n"] == 1
    J.undo_file(h5)
    assert attrs_of(h5, 7)["harmonic_call_id"] == "2"


def test_cli_joins_and_undoes_a_folder():
    h5 = make(SPLIT)
    before = snapshot(h5)
    folder = os.path.dirname(h5)
    assert mad_main(["join-fragments", folder, "--settle-minutes", "0", "--dry-run"]) == 0
    assert snapshot(h5) == before
    assert mad_main(["join-fragments", folder, "--settle-minutes", "0",
                     "--gap-ms", "10", "--max-df-khz", "5"]) == 0
    assert ids(h5) == {"1"}
    assert mad_main(["join-fragments", h5, "--settle-minutes", "0", "--undo"]) == 0
    assert snapshot(h5) == before


def test_the_gui_shows_a_joined_call_as_a_pending_detection():
    from PyQt5.QtWidgets import QApplication
    from scipy.io import wavfile
    import fnt.usv.mad_pyqt as M
    from fnt.usv.usv_detector.mad_project import MADProjectConfig, create_mad_project
    root = tempfile.mkdtemp(prefix="mad_join_gui_")
    wav = os.path.join(root, "rec.wav")
    wavfile.write(wav, SR, (np.random.default_rng(0).normal(0, .02, 2 * SR)
                            * 32767).astype(np.int16))
    h5 = MS.masks_sibling_path(wav)
    MS.set_grid_attrs(h5, sample_rate=SR, nperseg=512, noverlap=384, nfft=NFFT,
                      n_freq_bins=513, n_time_frames=2 * SR // HOP)
    MS.set_infer_run_attrs(h5, last_infer_model="unit", last_infer_at="2026-11-09T10:00:00")
    MS.write_pred_masks(h5, SPLIT + [crop(3, 2000, 2040, 250)])
    assert J.join_file(h5)["n_joined"] == 1
    proj = os.path.join(root, "p")
    create_mad_project(proj)
    app = QApplication.instance() or QApplication([])
    M.MADMainWindow._apply_dark_theme()
    w = M.MADMainWindow()
    _KEEP.append((app, w))
    w._activate_project(MADProjectConfig.load(proj))
    w._register_audio_files([wav])
    w._rescan_project_wavs()
    w._wait_for_audio_load()
    anns = w.spectrogram.annotations
    by_blob = {str(a.get("blob_id")): a for a in anns}
    assert set(by_blob) == {"1", "3"}, by_blob.keys()
    assert all(a.get("status") == "prediction" for a in anns)
    a = by_blob["1"]
    assert int(np.asarray(a["mask"]).sum()) == int(SPLIT[0]["mask"].sum()
                                                   + SPLIT[1]["mask"].sum())


_KEEP = []


if __name__ == "__main__":
    import sys
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok", name)
            except Exception as e:                       # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
