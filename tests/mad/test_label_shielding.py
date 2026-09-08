"""Shielding a confirmed call must not shield the whole frequency axis.

``preserve_labels`` stops inference redrawing calls you already reviewed. It
did that by zeroing every time COLUMN containing a label::

    cols = (user_mask > 0).any(axis=0)
    prob[:, cols] = 0.0            # all 0-125 kHz

so a confirmed call at 25 kHz suppressed detection of everything stacked above
it for the call's whole duration. Rodent USVs overlap in time constantly — the
recording this was found on has calls at 25, 50 and 28-58 kHz inside one 120 ms
window — so it silently cost real detections. Measured on that 600 s file,
shielding 19,695 labelled pixels blanked 763,344 grid cells: 39x over-reach.

Now the label's own pixels are blanked, plus a small halo so a prediction
cannot hug the edge and return as a near-duplicate.

The second half: labels are read from the recording's own ``.mad``, which is
the MASTER set — the project's store is a cache rebuilt from the sidecars at
the start of each run. Shielding read the cache, which was stale: that file had
39 labels in its ``.mad`` and 28 in the cache, and the 11 missing ones were
exactly the labels predictions were drawn over. Unioning the two would be wrong
the other way, honouring a label already deleted from the sidecar.

Runs under pytest, or directly.
"""
import os
import tempfile

import numpy as np

from fnt.usv.usv_detector import fnt_mask_store as ms
from fnt.usv.usv_detector.mad_examples import reconstruct_file_mask
from fnt.usv.usv_detector.mad_inference import LABEL_SHIELD_PAD, _dilate_mask

H, W = 128, 200
WAV = "r.wav"


def meta(t_off, kind="label", ex_id=None, wav=WAV):
    m = {"class": "USV", "source_wav": wav, "patch_t_off": t_off,
         "patch_f_off": 0, "t_start_s": 0.0, "t_stop_s": 0.01,
         "f_low_hz": 0.0, "f_high_hz": 1000.0, "patch_t_frames": W,
         "f_bins": H, "nperseg": 512, "noverlap": 384, "nfft": 1024,
         "sample_rate": 250000}
    if kind != "label":
        m["kind"] = kind
    if ex_id:
        m["id"] = ex_id
    return m


def low_call():
    """A call low on the frequency axis, mid-patch."""
    m = np.zeros((H, W), bool)
    m[10:20, 80:120] = True
    return m


# ----------------------------------------------------------------------
# The shield covers the call, not the column
# ----------------------------------------------------------------------
def test_a_call_stacked_above_a_label_is_still_detectable():
    """THE bug. Same time span, different frequency — must not be blanked."""
    shield = _dilate_mask(low_call(), LABEL_SHIELD_PAD)
    assert shield[10:20, 80:120].all(), "the label itself must be shielded"
    assert not shield[90:110, 80:120].any(), \
        "a call stacked above the label was suppressed"


def test_the_shield_does_not_span_the_frequency_axis():
    shield = _dilate_mask(low_call(), LABEL_SHIELD_PAD)
    covered_rows = np.where(shield.any(axis=1))[0]
    assert covered_rows.min() == 10 - LABEL_SHIELD_PAD
    assert covered_rows.max() == 19 + LABEL_SHIELD_PAD
    assert shield.sum() < 0.05 * shield.size


def test_the_old_column_blanking_would_have_failed_these():
    """Pins what regressing to `prob[:, cols] = 0` would look like."""
    cols = low_call().any(axis=0)
    old = np.zeros((H, W), bool)
    old[:, cols] = True
    assert old[90:110, 80:120].all(), "column blanking hits the whole axis"
    # 7x on this 128-row toy grid; 13x on the real 513-row one (763,344 cells
    # blanked vs 57,672) because the ratio scales with the frequency axis.
    assert old.sum() > 5 * _dilate_mask(low_call(), LABEL_SHIELD_PAD).sum()


def test_the_halo_is_small_but_present():
    """Without one, a prediction hugs the label's edge and comes back as a
    near-duplicate of a call already reviewed."""
    m = low_call()
    shield = _dilate_mask(m, LABEL_SHIELD_PAD)
    assert shield[10 - LABEL_SHIELD_PAD, 100]
    assert not shield[10 - LABEL_SHIELD_PAD - 1, 100]
    assert 1 < shield.sum() / m.sum() < 3


def test_dilating_nothing_is_a_no_op():
    empty = np.zeros((H, W), bool)
    assert not _dilate_mask(empty, LABEL_SHIELD_PAD).any()
    m = low_call()
    assert np.array_equal(_dilate_mask(m, 0), m)


# ----------------------------------------------------------------------
# Labels come from both stores
# ----------------------------------------------------------------------
def _store(path, entries):
    for ex_id, mask, kind, t_off in entries:
        ms.td_save_example(path, np.zeros((H, W), np.uint8), mask,
                           meta(t_off, kind, ex_id), ex_id)


def test_a_label_only_in_the_mad_still_shields():
    """The case that let predictions be drawn over 11 confirmed calls."""
    with tempfile.TemporaryDirectory() as d:
        proj = os.path.join(d, "training_data")
        os.makedirs(proj)
        mad = os.path.join(d, "r_FNT.mad")
        _store(os.path.join(proj, "training_data.h5"),
               [("in_both", low_call(), "label", 0)])
        _store(mad, [("in_both", low_call(), "label", 0),
                     ("mad_only", low_call(), "label", 500)])

        only_proj = reconstruct_file_mask(proj, WAV, (H, 1000))
        both = reconstruct_file_mask(proj, WAV, (H, 1000), extra_stores=[mad])
    assert not only_proj[:, 500:1000].any(), "setup: mad-only label is elsewhere"
    assert both[:, 500:1000].any(), "the .mad-only label must shield too"
    assert both.sum() == 2 * only_proj.sum()


def test_an_example_in_both_stores_is_painted_once():
    with tempfile.TemporaryDirectory() as d:
        proj = os.path.join(d, "training_data")
        os.makedirs(proj)
        mad = os.path.join(d, "r_FNT.mad")
        _store(os.path.join(proj, "training_data.h5"),
               [("dup", low_call(), "label", 0)])
        _store(mad, [("dup", low_call(), "label", 0)])
        both = reconstruct_file_mask(proj, WAV, (H, 1000), extra_stores=[mad])
    assert both.sum() == low_call().sum()


def test_rejections_never_shield():
    """A rejected region is exactly where the model should look again."""
    with tempfile.TemporaryDirectory() as d:
        proj = os.path.join(d, "training_data")
        os.makedirs(proj)
        mad = os.path.join(d, "r_FNT.mad")
        _store(os.path.join(proj, "training_data.h5"), [])
        _store(mad, [("neg", low_call(), "negative", 0),
                     ("rej", low_call(), "rejected", 500)])
        got = reconstruct_file_mask(proj, WAV, (H, 1000), extra_stores=[mad])
    assert not got.any()


def test_another_recordings_labels_are_ignored():
    with tempfile.TemporaryDirectory() as d:
        proj = os.path.join(d, "training_data")
        os.makedirs(proj)
        mad = os.path.join(d, "other_FNT.mad")
        _store(os.path.join(proj, "training_data.h5"), [])
        ms.td_save_example(mad, np.zeros((H, W), np.uint8), low_call(),
                           meta(0, "label", "x", wav="other.wav"), "x")
        got = reconstruct_file_mask(proj, WAV, (H, 1000), extra_stores=[mad])
    assert not got.any()


def test_a_missing_extra_store_is_not_an_error():
    with tempfile.TemporaryDirectory() as d:
        proj = os.path.join(d, "training_data")
        os.makedirs(proj)
        _store(os.path.join(proj, "training_data.h5"),
               [("a", low_call(), "label", 0)])
        got = reconstruct_file_mask(proj, WAV, (H, 1000),
                                    extra_stores=[os.path.join(d, "nope.mad")])
    assert got.sum() == low_call().sum()


def test_the_sidecar_wins_over_a_stale_cache():
    """The .mad is the master; the project store is rebuilt from it each run.
    A label deleted from the sidecar lingers in the cache until then, and
    unioning the two would shield a call the user has already removed."""
    with tempfile.TemporaryDirectory() as d:
        proj = os.path.join(d, "training_data")
        os.makedirs(proj)
        mad = os.path.join(d, "r_FNT.mad")
        _store(os.path.join(proj, "training_data.h5"),
               [("deleted_since", low_call(), "label", 0)])
        _store(mad, [("still_here", low_call(), "label", 500)])
        sidecar_only = reconstruct_file_mask("", WAV, (H, 1000),
                                             extra_stores=[mad])
    assert not sidecar_only[:, 0:200].any(), "stale cached label still shields"
    assert sidecar_only[:, 500:1000].any(), "the live sidecar label must shield"


def test_the_cache_is_the_fallback_when_there_is_no_sidecar():
    with tempfile.TemporaryDirectory() as d:
        proj = os.path.join(d, "training_data")
        os.makedirs(proj)
        _store(os.path.join(proj, "training_data.h5"),
               [("a", low_call(), "label", 0)])
        got = reconstruct_file_mask(proj, WAV, (H, 1000), extra_stores=[])
    assert got.sum() == low_call().sum()


def test_inference_calls_the_shield_with_both_stores():
    """Guards the wiring, not just the helpers."""
    import inspect
    from fnt.usv.usv_detector import mad_inference as mi
    src = inspect.getsource(mi.analyze_wav) if hasattr(mi, "analyze_wav") \
        else inspect.getsource(mi)
    assert "extra_stores=sidecars" in src
    assert '"" if sidecars else cfg.training_data_dir' in src
    assert "prob[:, cols] = 0.0" not in src, "column blanking is back"
    assert "_dilate_mask(user_mask > 0, LABEL_SHIELD_PAD)" in src


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
