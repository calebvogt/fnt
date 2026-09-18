"""Harmonic grouping is a property of a detection run, not a per-file chore.

It used to be a "Detect Harmonics (H)" button in the Detections panel, which
meant a 263-recording batch would have needed 263 presses — so the recordings
that most needed grouping were the ones that never got it, and the
``harmonic_call_id``/``harmonic_n``/``f0_hz`` columns of the export stayed
blank. It is a toggle on both inference panels now, run per recording as the
run lands.

Making that work needed a store change. The GUI button wrote the assignment
into an *example's* metadata, and a freshly-inferred detection is not an
example — it is a prediction crop. So the grouping it computed for pending
detections lived in memory and was gone at the next file switch. Crops now
carry the harmonic fields, and ``update_pred_attrs`` writes them.

Runs under pytest, or directly.
"""
import os
import tempfile

import numpy as np
import pytest

pytest.importorskip("h5py")

from fnt.usv.usv_detector.fnt_mask_store import (
    PRED_ATTRS, read_pred_attrs, set_grid_attrs, update_pred_attrs,
    write_pred_masks)
from fnt.usv.usv_detector.mad_harmonics import group_recording

SR = 250_000
NFFT = 1024
NPERSEG = 512
NOVERLAP = 384
HZ_PER_BIN = SR / NFFT          # ~244 Hz


def store(tmp):
    """A .mad with MAD's real spectrogram grid and nothing else."""
    h5 = os.path.join(tmp, "r_FNT.mad")
    set_grid_attrs(h5, sample_rate=SR, nperseg=NPERSEG, noverlap=NOVERLAP,
                   nfft=NFFT, n_freq_bins=NFFT // 2 + 1, n_time_frames=4000)
    return h5


def tone(khz, t_frame, n_frames=40, thick=3):
    """A flat horizontal streak at ``khz`` — one element."""
    f_off = int(round(khz * 1000.0 / HZ_PER_BIN))
    return {"mask": np.ones((thick, n_frames), dtype=bool),
            "f_off": f_off, "t_off": t_frame}


def pred(h5, blobs):
    write_pred_masks(h5, [dict(b, blob_id=str(i)) for i, b in enumerate(blobs)])


def attrs_by_id(h5):
    return {r["blob_id"]: r for r in read_pred_attrs(h5)}


# ----------------------------------------------------------- the store change
def test_a_prediction_crop_can_carry_a_harmonic_assignment():
    """The gap that made batch grouping impossible: nowhere to put the answer
    on a detection nobody has reviewed yet."""
    for k in ("harmonic_call_id", "harmonic_n", "f0_hz"):
        assert k in PRED_ATTRS, k


def test_update_pred_attrs_writes_without_touching_masks():
    with tempfile.TemporaryDirectory() as tmp:
        h5 = store(tmp)
        pred(h5, [tone(40, 100), tone(80, 100)])
        assert update_pred_attrs(h5, {"0": {"harmonic_call_id": "0",
                                            "harmonic_n": 1}}) == 1
        got = attrs_by_id(h5)
        assert got["0"]["harmonic_call_id"] == "0"
        assert got["0"]["harmonic_n"] == 1
        assert "harmonic_call_id" not in got["1"]


def test_a_blank_value_clears_rather_than_storing_empty():
    """Regrouping has to be able to take an assignment away. Storing "" would
    read back as a real, blank call id."""
    with tempfile.TemporaryDirectory() as tmp:
        h5 = store(tmp)
        pred(h5, [tone(40, 100)])
        update_pred_attrs(h5, {"0": {"harmonic_call_id": "c1"}})
        update_pred_attrs(h5, {"0": {"harmonic_call_id": None}})
        assert "harmonic_call_id" not in attrs_by_id(h5)["0"]


def test_unknown_fields_are_refused():
    """PRED_ATTRS is what every reader iterates; a field outside it would be
    written and then never seen again."""
    with tempfile.TemporaryDirectory() as tmp:
        h5 = store(tmp)
        pred(h5, [tone(40, 100)])
        update_pred_attrs(h5, {"0": {"not_a_real_field": 7}})
        assert "not_a_real_field" not in attrs_by_id(h5)["0"]


def test_ids_that_are_gone_are_skipped_not_fatal():
    with tempfile.TemporaryDirectory() as tmp:
        h5 = store(tmp)
        pred(h5, [tone(40, 100)])
        assert update_pred_attrs(h5, {"nope": {"harmonic_n": 2}}) == 0


# -------------------------------------------------------- grouping a whole file
def test_a_stack_of_pending_detections_is_grouped_and_written():
    """40/80/120 kHz at the same time is a fundamental and two harmonics."""
    with tempfile.TemporaryDirectory() as tmp:
        h5 = store(tmp)
        pred(h5, [tone(40, 500), tone(80, 500), tone(120, 500)])
        res = group_recording(h5, "r.wav")
        assert res is not None
        assert res["n_elements"] == 3
        assert res["n_calls"] == 1, "the stack was not recognised as one call"
        assert res["n_harmonic"] == 2
        got = attrs_by_id(h5)
        assert len({g["harmonic_call_id"] for g in got.values()}) == 1
        assert sorted(g["harmonic_n"] for g in got.values()) == [1, 2, 3]


def test_calls_at_different_times_stay_separate():
    with tempfile.TemporaryDirectory() as tmp:
        h5 = store(tmp)
        pred(h5, [tone(40, 500), tone(40, 3000)])
        res = group_recording(h5, "r.wav")
        assert res["n_calls"] == 2
        got = attrs_by_id(h5)
        assert got["0"]["harmonic_call_id"] != got["1"]["harmonic_call_id"]


def test_f0_is_recorded_on_every_member_of_a_stack():
    """The export's f0_hz column. Per call, stamped on each element, so a row
    is self-describing without a join."""
    with tempfile.TemporaryDirectory() as tmp:
        h5 = store(tmp)
        pred(h5, [tone(40, 500), tone(80, 500)])
        group_recording(h5, "r.wav")
        f0s = {r["f0_hz"] for r in read_pred_attrs(h5)}
        assert len(f0s) == 1
        assert 35_000 < next(iter(f0s)) < 45_000, f0s


def test_an_empty_recording_is_a_no_op():
    with tempfile.TemporaryDirectory() as tmp:
        assert group_recording(store(tmp), "r.wav") is None


def test_a_store_with_no_grid_is_refused_not_guessed():
    """Without sample rate and nfft there is no frequency axis, and a grouping
    computed on bin indices would be confidently wrong."""
    with tempfile.TemporaryDirectory() as tmp:
        h5 = os.path.join(tmp, "r_FNT.mad")
        write_pred_masks(h5, [dict(tone(40, 500), blob_id="0")])
        assert group_recording(h5, "r.wav") is None


def test_regrouping_is_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        h5 = store(tmp)
        pred(h5, [tone(40, 500), tone(80, 500), tone(40, 3000)])
        first = group_recording(h5, "r.wav")
        before = attrs_by_id(h5)
        second = group_recording(h5, "r.wav")
        assert first == second
        assert attrs_by_id(h5) == before


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
    sys.exit(1 if fails else 0)
