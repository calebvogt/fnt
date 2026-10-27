"""Folder-wide tables from the .mad files (mad_export, ``mad export``, File > Export...).

Replaces three scripts the 2025_VoleCosm analysis carried (usv_detections.py,
usv_labeled_calls.py, usv_call_images.py) with generic tables FNT writes itself.
Covers: the columns of each table; the model filter (the most common by default,
a name or the unique start of one); fundamentals only unless harmonics are asked
for; review status from the stored examples; the file-name fields (Avisoft start
and channel); the contour measures, against the scripts' own formula; the
incremental export -- unchanged recordings copied from the previous tables and
never opened, a changed one read, a table cut short read again, a change of
settings reading everything -- and recent or unreadable files left out; Stop
leaving the previous tables whole; labelled calls measured with MAD's call
metrics from the audio, with their images; and the CLI.

Runs under pytest, or directly.
"""
import csv
import json
import os
import tempfile
import time

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from fnt.usv.usv_detector import cli  # noqa: E402
from fnt.usv.usv_detector import fnt_mask_store as ms  # noqa: E402
from fnt.usv.usv_detector import mad_export as mx  # noqa: E402
from fnt.usv.usv_detector.mad_inference import CALL_METRIC_KEYS, frames_to_seconds  # noqa: E402
from fnt.usv.usv_detector.mad_ridge import grid_units  # noqa: E402

SR, NPERSEG, NOVERLAP, NFFT = 250_000, 512, 384, 1024
DF_KHZ, DT_MS = grid_units(SR, NPERSEG, NOVERLAP, NFFT)
OLD = time.time() - 3600                     # well past the 10-minute settle
TRIAL = "VC-T001_prairie_anosmic-saline"
A1, A2, B1 = ("ch12025-10-14_12-00-00_0001.wav", "ch22025-10-14_12-00-00_0001.wav",
              "ch32025-10-14_12-00-00_0001.wav")


def _crop(bid, t_off, harmonic_n=1, score=0.9):
    m = np.zeros((10, 30), np.uint8)
    m[3:6, :] = 1
    m[6, 10:20] = 1                          # a little shape
    spec = np.full((10, 30), 40, np.uint8)
    spec[4, :] = 250
    return {"blob_id": bid, "mask": m, "spec": spec, "f_off": 150, "t_off": t_off,
            "score": score, "harmonic_n": harmonic_n, "harmonic_call_id": str(bid)}


def _age(path, t=OLD):
    os.utime(path, (t, t))


def _recording(folder, name, model, crops):
    os.makedirs(folder, exist_ok=True)
    wav = os.path.join(folder, name)
    open(wav, "wb").close()
    h5 = ms.masks_sibling_path(wav)
    ms.set_grid_attrs(h5, sample_rate=SR, nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT,
                      n_freq_bins=513, n_time_frames=1_171_875, source_wav=name)
    ms.write_pred_masks(h5, crops)
    ms.set_infer_run_attrs(h5, last_infer_at="2026-10-20T10:00:00", last_infer_model=model,
                           last_infer_threshold=0.5, last_infer_min_blob=8,
                           last_infer_merge=1, last_infer_n=len(crops))
    _age(h5)
    return wav, h5


def _run_folder():
    """Two recordings of model m_a (one with a harmonic), one of m_b."""
    root = tempfile.mkdtemp(prefix="mad_export_")
    d = os.path.join(root, TRIAL)
    _recording(d, A1, "m_a", [_crop(0, 1000), _crop(1, 2000), _crop(2, 2000, harmonic_n=2)])
    _recording(d, A2, "m_a", [_crop(5, 500)])
    _recording(d, B1, "m_b", [_crop(0, 100)])
    return root


def _table(path):
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        return header, [dict(zip(header, row)) for row in r]


def _export(root, out, **kw):
    lines = []
    rep = mx.export(mx.ExportOptions(out_dir=out, folders=[root], **kw), log=lines.append)
    return rep, lines


# ----------------------------------------------------------------------
# Recordings and detections
# ----------------------------------------------------------------------
def test_the_tables_hold_the_most_common_models_recordings_and_their_fundamentals():
    root = _run_folder()
    out = os.path.join(root, "out")
    rep, _ = _export(root, out)
    assert rep["model"] == "m_a" and rep["models"] == {"m_a": 2, "m_b": 1}
    head, recs = _table(os.path.join(out, mx.RECORDINGS_CSV))
    assert head == mx.REC_COLS
    assert [r["recording"] for r in recs] == [A1, A2]
    r = recs[0]
    assert r["folder"] == TRIAL and r["path"] == os.path.join(root, TRIAL, A1)
    assert (r["channel"], r["start"]) == ("1", "2025-10-14 12:00:00")
    assert float(r["duration_s"]) == pytest.approx(1_171_875 * 128 / SR, abs=1e-3)
    assert (r["model"], r["inferred_at"], r["threshold"]) == ("m_a", "2026-10-20T10:00:00", "0.5")
    assert (r["detections"], r["calls"]) == ("3", "2")          # the harmonic is not a call
    head, dets = _table(os.path.join(out, mx.DETECTIONS_CSV))
    assert head == mx.DET_COLS
    assert [(d["recording"], d["blob_id"]) for d in dets] == [(A1, "0"), (A1, "1"), (A2, "5")]
    d = dets[0]
    assert float(d["start_s"]) == pytest.approx(frames_to_seconds(1000, NPERSEG, NOVERLAP, SR))
    assert (d["score"], d["status"], d["harmonic_n"]) == ("0.9", "pending", "1")
    assert float(d["f_min_khz"]) == pytest.approx(153 * DF_KHZ, abs=1e-4)
    assert float(d["f_max_khz"]) == pytest.approx(157 * DF_KHZ, abs=1e-4)
    assert float(d["duration_ms"]) == pytest.approx(30 * DT_MS, abs=1e-4)
    assert int(d["area_px"]) == 100
    assert d["ridge_turns"] == "0" and d["ridge_trill"] == "0"   # stored with the crop
    settings = json.load(open(os.path.join(out, mx.SETTINGS_JSON)))
    assert settings["detections"]["model"] == "m_a"
    assert settings["detections"]["harmonics"] is False


def test_harmonics_on_request():
    root = _run_folder()
    out = os.path.join(root, "out")
    _export(root, out, harmonics=True)
    _, dets = _table(os.path.join(out, mx.DETECTIONS_CSV))
    assert [(d["blob_id"], d["harmonic_n"]) for d in dets if d["recording"] == A1] == \
        [("0", "1"), ("1", "1"), ("2", "2")]


def test_a_model_by_name_or_the_start_of_one():
    root = _run_folder()
    out = os.path.join(root, "out")
    rep, _ = _export(root, out, model="m_b")
    assert rep["model"] == "m_b"
    _, recs = _table(os.path.join(out, mx.RECORDINGS_CSV))
    assert [r["recording"] for r in recs] == [B1]
    from collections import Counter
    counts = Counter({"20261014_181834_unet_n=3366": 5, "20261001_090000_unet_n=900": 2})
    assert mx.resolve_model("20261014_181834", counts) == "20261014_181834_unet_n=3366"
    assert mx.resolve_model(None, counts) == "20261014_181834_unet_n=3366"
    with pytest.raises(ValueError, match="Models found"):
        mx.resolve_model("2026", counts)                   # not unique
    with pytest.raises(ValueError):
        _export(root, os.path.join(root, "out2"), model="nope")


def test_review_status_comes_from_the_examples():
    root = _run_folder()
    h5 = ms.masks_sibling_path(os.path.join(root, TRIAL, A1))
    patch, mask = np.zeros((513, 40), np.uint8), np.zeros((513, 40), np.uint8)
    mask[150:155, 5:30] = 1
    meta = {"source_wav": A1, "sample_rate": SR, "nperseg": NPERSEG, "noverlap": NOVERLAP,
            "nfft": NFFT, "patch_t_off": 990, "patch_f_off": 0}
    ms.td_save_example(h5, patch, mask, dict(meta, blob_id=0, harmonic_n=1), "acc0")
    ms.td_save_example(h5, patch, mask, dict(meta, blob_id=1, kind="negative"), "neg1")
    _age(h5)
    out = os.path.join(root, "out")
    _export(root, out)
    _, dets = _table(os.path.join(out, mx.DETECTIONS_CSV))
    assert {d["blob_id"]: d["status"] for d in dets if d["recording"] == A1} == \
        {"0": "accepted", "1": "rejected"}


def test_name_fields():
    assert mx.name_fields(A2) == (2, "2025-10-14 12:00:00")
    assert mx.name_fields("T014_ch3_T0000042.wav") == (3, "")
    assert mx.name_fields("rec.wav") == (None, "")


def test_contour_measures_match_the_scripts_formula():
    """The measures of 2025_VoleCosm python/usv_call_images.py (and 2021_8x8_Mus
    R/01_build_usv.R), restated here as written there."""
    def oracle(m, f_off):                            # m: time x frequency
        m = m > 0
        px = m.sum(1)
        cont = ((m @ np.arange(1, m.shape[1] + 1))[px > 0] / px[px > 0] - 1 + f_off) * DF_KHZ
        d = np.diff(cont)
        steps = np.sign(d[np.abs(d) > DF_KHZ])
        fb = np.flatnonzero(m.sum(0) > 0) + 1
        return dict(duration_ms=m.shape[0] * DT_MS, f_mean_khz=cont.mean(),
                    f_min_khz=(fb.min() - 1 + f_off) * DF_KHZ, f_max_khz=(fb.max() + f_off) * DF_KHZ,
                    f_start_khz=cont[0], f_end_khz=cont[-1],
                    slope_khz_ms=(cont[-1] - cont[0]) / (m.shape[0] * DT_MS),
                    contour_sd_khz=cont.std(ddof=1) if len(cont) > 1 else 0.0,
                    turns=int((np.diff(steps) != 0).sum()) if len(steps) > 1 else 0,
                    max_jump_khz=np.abs(d).max() if len(d) else 0.0,
                    thickness_khz=px[px > 0].mean() * DF_KHZ, area_px=int(m.sum()))
    rng = np.random.default_rng(3)
    for _ in range(20):
        mask = rng.random((12, 40)) > 0.7                # frequency x time, as stored
        got = mx.contour_measures(mask, 140, DF_KHZ, DT_MS)
        want = oracle(mask.T, 140)
        assert set(got) == set(mx.CONTOUR_COLS)
        for k in want:
            assert got[k] == pytest.approx(want[k]), k
    assert mx.contour_measures(np.zeros((4, 4), bool), 0, DF_KHZ, DT_MS) is None


# ----------------------------------------------------------------------
# Incremental
# ----------------------------------------------------------------------
@pytest.fixture
def reads(monkeypatch):
    """The .mad files read in full (detections and examples), by name."""
    seen = []
    real = ms.read_store_summary

    def spy(path, preds=False, example_meta=False):
        if preds:
            seen.append(os.path.basename(path))
        return real(path, preds=preds, example_meta=example_meta)
    monkeypatch.setattr(ms, "read_store_summary", spy)
    return seen


def test_unchanged_recordings_are_copied_not_read(reads):
    root = _run_folder()
    out = os.path.join(root, "out")
    _export(root, out)
    first = [open(os.path.join(out, f), encoding="utf-8").read()
             for f in (mx.RECORDINGS_CSV, mx.DETECTIONS_CSV)]
    reads.clear()
    rep, _ = _export(root, out)
    assert reads == [] and rep["reused"] == 2
    assert [open(os.path.join(out, f), encoding="utf-8").read()
            for f in (mx.RECORDINGS_CSV, mx.DETECTIONS_CSV)] == first
    # a recording reviewed since: read again, the other copied
    h5 = ms.masks_sibling_path(os.path.join(root, TRIAL, A2))
    ms.update_pred_attrs(h5, {"5": {"score": 0.42}})
    _age(h5, OLD + 60)
    reads.clear()
    rep, _ = _export(root, out)
    assert reads == [os.path.basename(h5)] and rep["reused"] == 1
    _, dets = _table(os.path.join(out, mx.DETECTIONS_CSV))
    assert [d["score"] for d in dets] == ["0.9", "0.9", "0.42"]
    assert not [f for f in os.listdir(out) if f.endswith(".tmp")]


def test_a_table_cut_short_is_read_again(reads):
    """The recordings table says how many rows each recording has; rows lost
    (an export that died between replacing the two tables) are read back."""
    root = _run_folder()
    out = os.path.join(root, "out")
    _export(root, out)
    det = os.path.join(out, mx.DETECTIONS_CSV)
    lines = open(det, encoding="utf-8", newline="").readlines()
    with open(det, "w", encoding="utf-8", newline="") as f:
        f.writelines(lines[:-1])                    # A2's only row
    reads.clear()
    _export(root, out)
    assert reads == [os.path.basename(ms.masks_sibling_path(os.path.join(root, TRIAL, A2)))]
    _, dets = _table(det)
    assert len(dets) == 3


def test_new_settings_or_full_read_everything(reads):
    root = _run_folder()
    out = os.path.join(root, "out")
    _export(root, out)
    reads.clear()
    _export(root, out, harmonics=True)
    assert len(reads) == 2
    reads.clear()
    _export(root, out, harmonics=True, incremental=False)
    assert len(reads) == 2


def test_recent_and_unreadable_files_are_left_out():
    root = _run_folder()
    d = os.path.join(root, TRIAL)
    os.utime(ms.masks_sibling_path(os.path.join(d, A2)), None)     # just written
    bad = os.path.join(d, "ch42025-10-14_12-00-00_0001_FNT.mad")
    open(bad, "wb").write(b"not hdf5")
    _age(bad)
    out = os.path.join(root, "out")
    rep, lines = _export(root, out)
    assert rep["recent"] == 1 and rep["unreadable"] == 1
    assert any("left out" in m for m in lines)
    _, recs = _table(os.path.join(out, mx.RECORDINGS_CSV))
    assert [r["recording"] for r in recs] == [A1]
    rep, _ = _export(root, os.path.join(root, "out2"), settle_s=0)
    assert rep["recent"] == 0 and rep["recordings"] == 2


def test_stop_keeps_the_previous_tables():
    root = _run_folder()
    out = os.path.join(root, "out")
    _export(root, out)
    before = open(os.path.join(out, mx.DETECTIONS_CSV), encoding="utf-8").read()
    calls = []

    def progress(stage, i, n, name):
        calls.append(stage)
    rep = mx.export(mx.ExportOptions(out_dir=out, folders=[root], incremental=False),
                    progress=progress, should_stop=lambda: "detections" in calls)
    assert rep["stopped"]
    assert open(os.path.join(out, mx.DETECTIONS_CSV), encoding="utf-8").read() == before
    assert not [f for f in os.listdir(out) if f.endswith(".tmp")]


def test_a_projects_recordings_by_path():
    root = _run_folder()
    d = os.path.join(root, TRIAL)
    stores = mx.find_stores(wavs=[os.path.join(d, A1), os.path.join(d, B1),
                                  os.path.join(d, "never_analysed.wav")])
    assert [s.recording for s in stores] == [A1, B1]
    assert stores[0].mad == ms.masks_sibling_path(os.path.join(d, A1))


# ----------------------------------------------------------------------
# Labelled calls
# ----------------------------------------------------------------------
def _labelled_recording(root):
    """A real wav with one swept tone, labelled twice (accepted from detection
    7, and painted by hand) and rejected once, as the GUI saves them."""
    from scipy.io import wavfile
    from fnt.usv.usv_detector.mad_dataset import compute_full_spec_image
    hop, df = NPERSEG - NOVERLAP, SR / NFFT
    os.makedirs(root, exist_ok=True)
    wav = os.path.join(root, A1)
    t = np.arange(int(0.5 * SR)) / SR
    x = 0.001 * np.random.default_rng(0).standard_normal(len(t))
    on = (t > 0.1) & (t < 0.13)
    x[on] += 0.3 * np.sin(2 * np.pi * (40000 + 100000 * (t[on] - 0.1)) * t[on])
    wavfile.write(wav, SR, (x * 32767).astype(np.int16))
    spec = compute_full_spec_image(x.astype(np.float32), SR, NPERSEG, NOVERLAP, NFFT,
                                   -100, -20, as_uint8=True)
    n_freq, n_time = spec.shape
    t_org, dt = NPERSEG / (2 * SR), hop / SR
    f0, f1 = int(38000 / df), int(48000 / df)
    t0, t1 = int(round((0.1 - t_org) / dt)), int(round((0.13 - t_org) / dt))
    pt0 = max(0, t0 - 128)
    patch = spec[:, pt0:min(n_time, t1 + 128)]
    mask = np.zeros_like(patch)
    sub = patch[f0:f1, t0 - pt0:t1 - pt0]
    mask[f0:f1, t0 - pt0:t1 - pt0] = sub > np.percentile(sub, 80)
    fs, ts = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    meta = {"class": "USV", "source_wav": A1, "patch_t_off": pt0, "patch_f_off": 0,
            "t_start_s": round(t_org + (pt0 + ts[0]) * dt, 6),
            "t_stop_s": round(t_org + (pt0 + ts[-1] + 1) * dt, 6),
            "patch_t0_s": round(t_org + pt0 * dt, 6),
            "f_low_hz": round(fs[0] * df, 2), "f_high_hz": round((fs[-1] + 1) * df, 2),
            "nperseg": NPERSEG, "noverlap": NOVERLAP, "nfft": NFFT, "sample_rate": SR}
    h5 = ms.masks_sibling_path(wav)
    grid = dict(sample_rate=SR, nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT,
                n_freq_bins=n_freq, n_time_frames=n_time, source_wav=A1)
    ms.td_commit_example(h5, patch, mask, dict(meta, blob_id=7, harmonic_n=1), "acc7", grid=grid)
    ms.td_commit_example(h5, patch, mask, dict(meta, note="two\nlines"), "painted")
    ms.td_commit_example(h5, patch, mask, dict(meta, blob_id=9, harmonic_n=2), "harm9")
    ms.td_commit_example(h5, patch, mask, dict(meta, blob_id=8, kind="negative"), "neg8")
    _age(h5)
    return wav, h5, int(mask.sum())


def test_labelled_calls_are_measured_from_the_audio_with_their_images(monkeypatch):
    root = tempfile.mkdtemp(prefix="mad_export_lab_")
    wav, h5, n_px = _labelled_recording(os.path.join(root, TRIAL))
    out = os.path.join(root, "out")
    rep, _ = _export(root, out, detections=False, labels=True, images=True)
    assert rep["label_rows"] == 2 and not rep["error"]
    head, rows = _table(os.path.join(out, mx.LABELS_CSV))
    assert head == mx.LABEL_COLS
    by = {r["id"]: r for r in rows}
    assert set(by) == {"acc7", "painted"}           # no harmonic, no rejection
    a, p = by["acc7"], by["painted"]
    assert (a["blob_id"], a["painted"], p["blob_id"], p["painted"]) == ("7", "FALSE", "", "TRUE")
    assert (a["rec_start"], a["channel"], a["folder"]) == ("2025-10-14 12:00:00", "1", TRIAL)
    assert p["note"] == "two lines"                 # one line per row
    for k in ("peak_freq_hz", "snr_db", "tonality"):    # MAD's call metrics, from the audio
        assert a[k] != "", k
    assert 38 < float(a["f_mean_khz"]) < 48 and float(a["duration_ms"]) > 20
    from PIL import Image
    img = np.asarray(Image.open(os.path.join(out, mx.IMAGES_DIR, a["image"])))
    assert img.shape[0] == NFFT // 2 + 1 and img.shape[2] == 2
    assert int((img[..., 1] > 0).sum()) == n_px     # alpha: the call's own pixels
    margin = int(round(0.01 / (DT_MS / 1000)))
    assert int(a["img_col0"]) == margin + 1
    # unchanged: copied, the audio not read again
    import fnt.usv.usv_detector.mad_csv_rebuild as cr
    monkeypatch.setattr(cr, "_recompute_features",
                        lambda *a, **k: pytest.fail("read the audio again"))
    rep, _ = _export(root, out, detections=False, labels=True, images=True)
    assert rep["labels_reused"] == 1 and rep["label_rows"] == 2
    # a stale image goes
    open(os.path.join(out, mx.IMAGES_DIR, "gone.png"), "wb").write(b"x")
    _export(root, out, detections=False, labels=True, images=True)
    assert sorted(os.listdir(os.path.join(out, mx.IMAGES_DIR))) == ["acc7.png", "painted.png"]


def test_labelled_calls_without_the_audio_keep_their_rows():
    root = tempfile.mkdtemp(prefix="mad_export_lab_")
    wav, h5, _ = _labelled_recording(os.path.join(root, TRIAL))
    os.remove(wav)                                   # moved: the .mad still says enough
    out = os.path.join(root, "out")
    rep, _ = _export(root, out, detections=False, labels=True, harmonics=True)
    assert rep["labels_no_audio"] == 1 and rep["label_rows"] == 3
    _, rows = _table(os.path.join(out, mx.LABELS_CSV))
    assert all(r["peak_freq_hz"] == "" and r["f_mean_khz"] != "" for r in rows)


def test_labels_still_export_when_no_recording_was_analysed():
    root = tempfile.mkdtemp(prefix="mad_export_lab_")
    _labelled_recording(os.path.join(root, TRIAL))
    rep, _ = _export(root, os.path.join(root, "out"), detections=True, labels=True)
    assert "inference" in rep["error"] and rep["label_rows"] == 2
    with pytest.raises(ValueError):
        _export(root, os.path.join(root, "out2"), detections=True)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def test_cli_export_and_list_models(capsys):
    root = _run_folder()
    out = os.path.join(root, "out")
    assert cli.main(["export", "--input", root, "--list-models"]) == 0
    text = capsys.readouterr().out
    assert "m_a" in text and "m_b" in text
    assert cli.main(["export", "--input", root, "--out", out, "--model", "m_b"]) == 0
    _, recs = _table(os.path.join(out, mx.RECORDINGS_CSV))
    assert [r["recording"] for r in recs] == [B1]
    assert not os.path.exists(os.path.join(out, mx.LABELS_CSV))
    assert cli.main(["export", "--input", root, "--out", out, "--images"]) == 0
    assert os.path.isfile(os.path.join(out, mx.LABELS_CSV))   # images imply labels
    assert cli.main(["export", "--input", os.path.join(root, "nope"), "--out", out]) == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
