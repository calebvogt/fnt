"""The annotations CSV is a projection of the h5, not a second copy of the truth.

MAD wrote both files on every review action, and on a real project they drifted:
one recording's CSV listed 891 calls where its h5 held 906. Patching the CSV
incrementally cannot repair that, because each patch assumes the two were in
step. Regenerating it from the h5 makes drift impossible by construction.

The care is all in one place: a CSV row with no h5 record is NOT deleted.
Accepting from the gallery with no audio loaded deliberately records the
decision in the CSV and mints no example until the file is next opened, so
"absent from the h5" does not mean "deleted" — it can mean "not minted yet".

Runs under pytest, or directly.
"""
import os
import tempfile

import numpy as np

from fnt.usv.usv_detector import fnt_mask_store as MS
from fnt.usv.usv_detector.mad_csv_rebuild import (
    rebuild_annotations_csv, rebuild_folder)
from fnt.usv.usv_detector.mad_inference import read_blob_csv, write_blob_csv
from fnt.usv.usv_detector.mad_labels import annotations_csv_sibling_path

SR, NPERSEG, NOVERLAP, NFFT = 250_000, 512, 384, 1024


def _project():
    d = tempfile.mkdtemp(prefix="mad_rb_")
    wav = os.path.join(d, "rec.wav")
    open(wav, "wb").close()
    h5 = MS.masks_sibling_path(wav)
    MS.set_grid_attrs(h5, sample_rate=SR, nperseg=NPERSEG, noverlap=NOVERLAP,
                      nfft=NFFT, n_freq_bins=513, n_time_frames=4000)
    return wav, h5


def _add_example(h5, eid, *, kind="label", blob_id=None, t0=0.10, t1=0.13):
    spec = np.full((16, 20), 0.2, dtype=np.float32)
    mask = np.zeros((16, 20), dtype=np.uint8)
    mask[4:12, 5:15] = 1
    meta = {"class": "USV", "source_wav": "rec.wav", "sample_rate": SR,
            "nperseg": NPERSEG, "noverlap": NOVERLAP, "nfft": NFFT,
            "patch_f_off": 100, "patch_t_off": 200,
            "t_start_s": t0, "t_stop_s": t1,
            "f_low_hz": 25000.0, "f_high_hz": 45000.0}
    if kind != "label":
        meta["kind"] = kind
    if blob_id is not None:
        meta["blob_id"] = blob_id
    return MS.td_save_example(h5, spec, mask, meta, eid)


def _add_pred(h5, crops):
    MS.write_pred_masks(h5, crops)


def _crop(blob_id, **extra):
    m = np.zeros((10, 12), dtype=np.uint8)
    m[2:8, 3:9] = 1
    return {"blob_id": blob_id, "mask": m, "f_off": 120, "t_off": 400, **extra}


def _by_id(csv_path):
    return {str(r["blob_id"]): r for r in read_blob_csv(csv_path)}


# ----------------------------------------------------------------------
# Status comes out of the h5's structure
# ----------------------------------------------------------------------
def test_a_stored_label_becomes_an_accepted_row():
    wav, h5 = _project()
    _add_example(h5, "lab1")
    rep = rebuild_annotations_csv(wav)
    assert rep["accepted"] == 1 and rep["rejected"] == 0
    rows = _by_id(annotations_csv_sibling_path(wav))
    assert rows["lab1"]["status"] == "accepted"
    assert rows["lab1"]["class"] == "USV"


def test_a_demoted_example_becomes_a_rejected_row():
    wav, h5 = _project()
    _add_example(h5, "lab1")
    MS.td_set_kind(h5, "lab1", "rejected")
    rep = rebuild_annotations_csv(wav)
    assert rep["rejected"] == 1 and rep["accepted"] == 0
    assert _by_id(annotations_csv_sibling_path(wav))["lab1"]["status"] == "rejected"


def test_a_harvested_negative_becomes_a_rejected_row():
    """Rejecting a pending prediction stores a negative keyed to its blob_id."""
    wav, h5 = _project()
    _add_example(h5, "rec_neg_7", kind="negative", blob_id=7)
    rep = rebuild_annotations_csv(wav)
    assert rep["rejected"] == 1
    assert _by_id(annotations_csv_sibling_path(wav))["7"]["status"] == "rejected"


def test_an_unreviewed_crop_becomes_a_pending_row():
    wav, h5 = _project()
    _add_pred(h5, [_crop(1), _crop(2)])
    rep = rebuild_annotations_csv(wav)
    assert rep["pending"] == 2
    rows = _by_id(annotations_csv_sibling_path(wav))
    assert {r["status"] for r in rows.values()} == {"pending"}


def test_a_reviewed_crop_is_not_also_reported_pending():
    """The crop survives inference; the example is what says it was decided."""
    wav, h5 = _project()
    _add_pred(h5, [_crop(1), _crop(2)])
    _add_example(h5, "acc1", blob_id=1)
    rep = rebuild_annotations_csv(wav)
    assert (rep["accepted"], rep["pending"]) == (1, 1)
    rows = _by_id(annotations_csv_sibling_path(wav))
    assert rows["1"]["status"] == "accepted"
    assert rows["2"]["status"] == "pending"


def test_accepting_after_rejecting_lands_on_accepted():
    """Both records can exist; the label is the later decision."""
    wav, h5 = _project()
    _add_example(h5, "rec_neg_5", kind="negative", blob_id=5)
    _add_example(h5, "acc5", blob_id=5)
    rebuild_annotations_csv(wav)
    assert _by_id(annotations_csv_sibling_path(wav))["5"]["status"] == "accepted"


# ----------------------------------------------------------------------
# Score and provenance now survive in the h5
# ----------------------------------------------------------------------
def test_score_and_provenance_come_back_without_the_csv():
    """These lived only in the CSV, which is what made it a second truth."""
    wav, h5 = _project()
    _add_pred(h5, [_crop(3, score=0.87, **{"class": "USV"},
                         model_name="m1", threshold=0.5, min_blob_pixels=40)])
    rebuild_annotations_csv(wav)
    r = _by_id(annotations_csv_sibling_path(wav))["3"]
    assert abs(float(r["score"]) - 0.87) < 1e-6
    assert r["model_name"] == "m1"
    assert r["class"] == "USV"


def test_crop_attributes_round_trip_through_the_store():
    wav, h5 = _project()
    _add_pred(h5, [_crop(9, score=0.25, model_name="mdl", threshold=0.4,
                         min_blob_pixels=12)])
    got = MS.read_all_pred_masks(h5)["9"]
    assert abs(got["score"] - 0.25) < 1e-9
    assert got["model_name"] == "mdl"
    assert got["min_blob_pixels"] == 12


# ----------------------------------------------------------------------
# The thing that must not go wrong
# ----------------------------------------------------------------------
def test_a_row_with_no_h5_record_is_kept_not_deleted():
    """Gallery-accept with no audio loaded writes the CSV and mints no example
    on purpose. Treating that as "deleted" would discard a real decision."""
    wav, h5 = _project()
    _add_example(h5, "lab1")
    csv_path = annotations_csv_sibling_path(wav)
    write_blob_csv(csv_path, [
        {"blob_id": "lab1", "class": "USV", "start_s": 0.10, "stop_s": 0.13,
         "min_freq_hz": 25000.0, "max_freq_hz": 45000.0, "status": "accepted",
         "area_pixels": 80, "score": 1.0, "source": "label"},
        {"blob_id": "orphan", "class": "USV", "start_s": 0.5, "stop_s": 0.53,
         "min_freq_hz": 30000.0, "max_freq_hz": 40000.0, "status": "accepted",
         "area_pixels": 60, "score": 1.0, "source": "label"},
    ])
    rep = rebuild_annotations_csv(wav)
    assert rep["kept_without_h5"] == 1
    rows = _by_id(csv_path)
    assert "orphan" in rows, "a decision with no pixels yet was discarded"
    assert rows["orphan"]["status"] == "accepted"


def test_columns_nothing_reads_are_carried_through():
    """Derived features and harmonic grouping are export payload — expensive to
    recompute (they need the audio) and safe to preserve verbatim."""
    wav, h5 = _project()
    _add_example(h5, "lab1")
    csv_path = annotations_csv_sibling_path(wav)
    write_blob_csv(csv_path, [
        {"blob_id": "lab1", "class": "USV", "start_s": 0.10, "stop_s": 0.13,
         "min_freq_hz": 25000.0, "max_freq_hz": 45000.0, "status": "accepted",
         "area_pixels": 80, "score": 1.0, "source": "label",
         "sinuosity": 1.83, "harmonic_call_id": "lab1", "harmonic_n": 1},
    ])
    rebuild_annotations_csv(wav)
    r = _by_id(csv_path)["lab1"]
    assert abs(float(r["sinuosity"]) - 1.83) < 1e-6
    assert r["harmonic_call_id"] == "lab1"


def test_the_h5_wins_where_the_two_disagree():
    """The drift that started this: the CSV said one thing, the h5 another."""
    wav, h5 = _project()
    _add_example(h5, "lab1")
    MS.td_set_kind(h5, "lab1", "rejected")
    csv_path = annotations_csv_sibling_path(wav)
    write_blob_csv(csv_path, [
        {"blob_id": "lab1", "class": "USV", "start_s": 0.10, "stop_s": 0.13,
         "min_freq_hz": 25000.0, "max_freq_hz": 45000.0, "status": "accepted",
         "area_pixels": 80, "score": 1.0, "source": "label"}])
    rebuild_annotations_csv(wav)
    assert _by_id(csv_path)["lab1"]["status"] == "rejected"


def test_rebuilding_twice_changes_nothing():
    wav, h5 = _project()
    _add_example(h5, "lab1")
    _add_pred(h5, [_crop(2, score=0.5)])
    rebuild_annotations_csv(wav)
    first = read_blob_csv(annotations_csv_sibling_path(wav))
    rebuild_annotations_csv(wav)
    second = read_blob_csv(annotations_csv_sibling_path(wav))
    assert first == second


def test_dry_run_writes_nothing():
    wav, h5 = _project()
    _add_example(h5, "lab1")
    rep = rebuild_annotations_csv(wav, dry_run=True)
    assert rep["accepted"] == 1
    assert not os.path.isfile(annotations_csv_sibling_path(wav))


def test_a_recording_with_no_h5_is_reported_not_crashed():
    d = tempfile.mkdtemp()
    wav = os.path.join(d, "none.wav")
    open(wav, "wb").close()
    rep = rebuild_annotations_csv(wav)
    assert rep["h5_missing"] is True
    assert rep["written"] == 0


# ----------------------------------------------------------------------
# The bulk export
# ----------------------------------------------------------------------
def test_a_folder_can_be_rebuilt_in_one_pass():
    d = tempfile.mkdtemp(prefix="mad_folder_")
    for i in range(3):
        sub = os.path.join(d, f"day{i}")
        os.makedirs(sub, exist_ok=True)
        wav = os.path.join(sub, f"r{i}.wav")
        open(wav, "wb").close()
        h5 = MS.masks_sibling_path(wav)
        MS.set_grid_attrs(h5, sample_rate=SR, nperseg=NPERSEG,
                          noverlap=NOVERLAP, nfft=NFFT)
        _add_example(h5, f"lab{i}")
    seen = []
    reports = rebuild_folder(d, progress=lambda i, n, name: seen.append(name))
    assert len(reports) == 3
    assert all(r.get("accepted") == 1 for r in reports)
    assert len(seen) == 3
    for i in range(3):
        p = os.path.join(d, f"day{i}", f"r{i}_FNT_MAD_annotations.csv")
        assert os.path.isfile(p), p


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
