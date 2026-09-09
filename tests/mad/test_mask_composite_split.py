"""A stored mask is ONE call. Neighbours live beside it, not inside it.

Between 2026-09-05 and this suite, confirming a call ORed every confirmed pixel
in its patch window into the mask it saved. The intent was training-only — a
labelled neighbour must not be supervised as background — but the same array is
what the overlay, the confirmed-mask gallery, mask editing and the CSV geometry
read as *this call's shape*. Two symptoms, one cause:

* adjacent calls read as a single detection (one bounding box over both);
* re-confirming a call minted a bigger composite whose box painted over its
  neighbour.

And a third, worse one hiding underneath: a rejected prediction is stored with
an empty mask and supervised as all-background, but the composite gave it the
neighbouring confirmed call's pixels — so 15 of 35 rejections in the v5 project
were teaching the model that a call a human had just accepted was background.

The split: ``mask`` is this call, ``neighbors`` is everything else in the
window, and :func:`collect_training_examples` recombines them per kind.

Runs under pytest, or directly.
"""
import json
import os
import tempfile

import numpy as np

from fnt.usv.usv_detector import fnt_mask_store as ms
from fnt.usv.usv_detector import mad_examples as mx
from fnt.usv.usv_detector.mad_migrate_composites import (
    migrate_store, split_example)

H, W = 64, 40
SR, NPERSEG, NOVERLAP, NFFT = 250_000, 512, 384, 1024
HOP = NPERSEG - NOVERLAP
DT = HOP / SR
DF = (SR / 2.0) / (NFFT // 2)


def meta_for(t0f, t1f, f0, f1, kind="label", patch_t_off=0, **extra):
    """Metadata describing a call at patch-local frames/bins, as the GUI writes
    it — patch_t0_s and t_start_s share a frame-time origin that cancels."""
    m = dict(
        **{"class": "USV"}, source_wav="r.wav",
        patch_t_off=patch_t_off, patch_f_off=0,
        t_start_s=round(t0f * DT, 9), t_stop_s=round(t1f * DT, 9),
        patch_t0_s=0.0, patch_t1_s=round(W * DT, 9),
        f_low_hz=round(f0 * DF, 4), f_high_hz=round(f1 * DF, 4),
        patch_t_frames=W, f_bins=H,
        nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT, sample_rate=SR,
        db_min=-100.0, db_max=-20.0, db_norm="fixed",
    )
    if kind != "label":
        m["kind"] = kind
    m.update(extra)
    return m


def two_blobs():
    """A call at frames 5-12 and a neighbour at 25-33, disjoint."""
    own = np.zeros((H, W), bool)
    own[20:30, 5:12] = True
    nb = np.zeros((H, W), bool)
    nb[40:50, 25:33] = True
    return own, nb


# ----------------------------------------------------------------------
# Storage keeps them apart
# ----------------------------------------------------------------------
def test_the_mask_read_back_is_only_this_call():
    own, nb = two_blobs()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "training_data.h5")
        ms.td_save_example(p, np.zeros((H, W), np.uint8), own,
                           meta_for(5, 12, 20, 30), "ex1",
                           neighbors_patch=own | nb)
        got = list(ms.td_iter_examples(p))[0]
    assert np.array_equal(got["mask"] > 0, own)
    assert np.array_equal(got["neighbors"] > 0, nb)


def test_neighbors_are_stored_disjoint_from_the_mask():
    """Passing an overlapping composite must not double-count — the caller
    hands over `sg.mask`, which includes this call's own pixels."""
    own, nb = two_blobs()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "training_data.h5")
        ms.td_save_example(p, np.zeros((H, W), np.uint8), own,
                           meta_for(5, 12, 20, 30), "ex1",
                           neighbors_patch=own | nb)
        got = list(ms.td_iter_examples(p))[0]
    assert not ((got["mask"] > 0) & (got["neighbors"] > 0)).any()


def test_no_neighbours_writes_no_dataset():
    """The common case is a call alone in its window; an all-zero array per
    example is pure overhead."""
    import h5py
    own, _ = two_blobs()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "training_data.h5")
        ms.td_save_example(p, np.zeros((H, W), np.uint8), own,
                           meta_for(5, 12, 20, 30), "ex1")
        with h5py.File(p, "r") as f:
            assert "neighbors" not in f["examples"]["ex1"]
        assert list(ms.td_iter_examples(p))[0]["neighbors"] is None


def test_the_per_file_reader_never_returns_neighbours():
    """Every caller of td_iter_file_examples builds the overlay. Leaking a
    neighbour there is exactly the bug."""
    own, nb = two_blobs()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "training_data.h5")
        ms.td_save_example(p, np.zeros((H, W), np.uint8), own,
                           meta_for(5, 12, 20, 30), "ex1",
                           neighbors_patch=nb)
        got = list(ms.td_iter_file_examples(p, "r.wav"))[0]
    assert np.array_equal(np.asarray(got["mask"]) > 0, own)


def test_undo_restores_the_example_whole():
    own, nb = two_blobs()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "training_data.h5")
        ms.td_save_example(p, np.zeros((H, W), np.uint8), own,
                           meta_for(5, 12, 20, 30), "ex1",
                           neighbors_patch=nb)
        ex = ms.td_read_example(p, "ex1")
        assert ex["neighbors"] is not None
        ms.td_save_example(p, ex["spec"], ex["mask"], ex["meta"], "ex1",
                           neighbors_patch=ex["neighbors"])
        back = ms.td_read_example(p, "ex1")
    assert np.array_equal(back["neighbors"] > 0, nb)


# ----------------------------------------------------------------------
# Training recombines them, per kind
# ----------------------------------------------------------------------
def _collect(kind, own, nb):
    with tempfile.TemporaryDirectory() as d:
        ms.td_save_example(os.path.join(d, "training_data.h5"),
                           np.zeros((H, W), np.uint8),
                           np.zeros((H, W), bool) if kind == "negative" else own,
                           meta_for(5, 12, 20, 30, kind=kind), "ex1",
                           neighbors_patch=nb)
        # Tile at the patch's own size so output indices match patch indices;
        # the default 512-row tile pads from the bottom and shifts every index.
        s, t, w = mx.collect_training_examples(
            d, tile_time_frames=W, tile_freq_bins=H, placements=1, seed=0)
    return t[0], w[0]


def test_a_labels_neighbour_is_supervised_positive():
    """Restores the 2026-09-05 fix's intent: the weight mask is 1 across the
    patch, so a neighbour left at 0 would train as background."""
    own, nb = two_blobs()
    t, w = _collect("label", own, nb)
    assert t[40:50, 25:33].min() > 0.5, "neighbour lost from the target"
    assert t[20:30, 5:12].min() > 0.5, "own call lost from the target"
    assert w[40:50, 25:33].min() > 0.5, "neighbour should be weighted, not ignored"


def test_a_rejections_neighbour_is_excluded_from_the_loss():
    """The worse half of the bug: a negative's mask is zeroed wholesale, so a
    confirmed call caught in its window trained as background."""
    own, nb = two_blobs()
    t, w = _collect("negative", own, nb)
    assert t.max() < 0.5, "a negative must have an all-zero target"
    assert w[40:50, 25:33].max() < 0.5, "confirmed pixels must be ignored"
    assert w[0:10, 0:5].min() > 0.5, "the rest of the patch stays supervised"


def test_a_rejection_with_no_neighbour_is_still_fully_supervised():
    own, _ = two_blobs()
    t, w = _collect("negative", own, None)
    assert t.max() < 0.5
    assert w.min() > 0.5, "nothing to ignore, so nothing should be ignored"


# ----------------------------------------------------------------------
# The migration
# ----------------------------------------------------------------------
def test_split_keeps_the_component_over_the_calls_own_box():
    own, nb = two_blobs()
    res = split_example(own | nb, meta_for(5, 12, 20, 30))
    assert np.array_equal(res["own"], own)
    assert np.array_equal(res["neighbors"], nb)
    assert not res["ambiguous"]


def test_split_gives_a_negative_no_mask_of_its_own():
    """A rejected prediction is stored with an empty mask by design, so every
    pixel in its composite arrived from a neighbour."""
    own, nb = two_blobs()
    res = split_example(own | nb, meta_for(5, 12, 20, 30, kind="negative"))
    assert not res["own"].any()
    assert np.array_equal(res["neighbors"], own | nb)
    assert not res["ambiguous"]


def test_a_single_blob_is_left_alone():
    own, _ = two_blobs()
    res = split_example(own, meta_for(5, 12, 20, 30))
    assert np.array_equal(res["own"], own)
    assert not res["neighbors"].any()


def test_a_fused_blob_is_reported_not_guessed():
    """Two touching calls are one connected component; no metadata separates
    them, so the migration must say so rather than invent a boundary."""
    m = np.zeros((H, W), bool)
    m[20:30, 5:35] = True                    # 30 frames for a 7-frame call
    res = split_example(m, meta_for(5, 12, 20, 30))
    assert res["ambiguous"]
    assert "fused" in res["reason"]
    assert np.array_equal(res["own"], m), "an ambiguous case must not be altered"


def test_missing_geometry_changes_nothing():
    own, nb = two_blobs()
    bad = meta_for(5, 12, 20, 30)
    del bad["nfft"]
    res = split_example(own | nb, bad)
    assert res["ambiguous"]
    assert np.array_equal(res["own"], own | nb)


def test_migrate_store_rewrites_and_is_idempotent():
    own, nb = two_blobs()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "training_data.h5")
        ms.td_save_example(p, np.zeros((H, W), np.uint8), own | nb,
                           meta_for(5, 12, 20, 30), "ex1")   # composited
        dry = migrate_store(p, dry_run=True, backup=False)
        assert dry.changed == 1 and dry.px_moved == int(nb.sum())
        assert np.array_equal(list(ms.td_iter_examples(p))[0]["mask"] > 0,
                              own | nb), "a dry run must not write"

        rep = migrate_store(p, dry_run=False, backup=False)
        assert rep.changed == 1
        got = list(ms.td_iter_examples(p))[0]
        assert np.array_equal(got["mask"] > 0, own)
        assert np.array_equal(got["neighbors"] > 0, nb)

        again = migrate_store(p, dry_run=False, backup=False)
        assert again.changed == 0, "already-split examples must be skipped"


def test_migrate_never_empties_a_confirmed_call():
    """If the geometry cannot find this call, leaving the composite whole is
    strictly better than deleting a label."""
    own, nb = two_blobs()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "training_data.h5")
        # a box pointing at empty space
        ms.td_save_example(p, np.zeros((H, W), np.uint8), own | nb,
                           meta_for(35, 38, 55, 60), "ex1")
        rep = migrate_store(p, dry_run=False, backup=False)
        got = list(ms.td_iter_examples(p))[0]
    assert rep.changed == 0
    assert np.array_equal(got["mask"] > 0, own | nb)


def test_migrate_takes_a_backup_before_writing():
    own, nb = two_blobs()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "training_data.h5")
        ms.td_save_example(p, np.zeros((H, W), np.uint8), own | nb,
                           meta_for(5, 12, 20, 30), "ex1")
        rep = migrate_store(p, dry_run=False, backup=True)
        assert rep.backup and os.path.isfile(rep.backup)
        assert np.array_equal(
            list(ms.td_iter_examples(rep.backup))[0]["mask"] > 0, own | nb)


# ----------------------------------------------------------------------
# Pass 2: a component that IS another label
# ----------------------------------------------------------------------
def test_a_component_that_is_another_label_is_moved_out():
    """Geometry alone cannot fix an example whose own t_start/t_stop were
    themselves recomputed from a composite — its box already covers the
    neighbour. The neighbour's own label is the exact signal that does."""
    from fnt.usv.usv_detector.mad_migrate_composites import (
        subtract_sibling_labels)
    own, nb = two_blobs()
    entries = [
        # A: composite, and its metadata box spans the whole patch
        {"meta": meta_for(5, 33, 20, 50), "mask": own | nb},
        # B: the neighbour, labelled in its own right
        {"meta": meta_for(25, 33, 40, 50), "mask": nb},
    ]
    changes = subtract_sibling_labels(entries)
    assert len(changes) == 1 and changes[0]["index"] == 0
    assert np.array_equal(changes[0]["own"], own)
    assert np.array_equal(changes[0]["neighbors"], nb)


def test_patch_offsets_are_honoured_when_comparing_labels():
    """Two labels only share a coordinate system once their patch offsets are
    applied; comparing patch-local pixels would match unrelated calls."""
    from fnt.usv.usv_detector.mad_migrate_composites import (
        subtract_sibling_labels)
    own, nb = two_blobs()
    entries = [
        {"meta": meta_for(5, 33, 20, 50, patch_t_off=1000), "mask": own | nb},
        # same pixels, but 500 frames away on the file grid: NOT the same call
        {"meta": meta_for(25, 33, 40, 50, patch_t_off=1500), "mask": nb},
    ]
    assert subtract_sibling_labels(entries) == []


def test_a_partial_overlap_is_not_treated_as_the_same_object():
    """Stripping needs the component and the other label to be each other, in
    both directions — otherwise a big label eats a small one it merely touches."""
    from fnt.usv.usv_detector.mad_migrate_composites import (
        subtract_sibling_labels)
    own, nb = two_blobs()
    big = own | nb
    entries = [
        {"meta": meta_for(5, 33, 20, 50), "mask": big},
        # B covers the component but is much bigger than it
        {"meta": meta_for(5, 33, 20, 50), "mask": big},
    ]
    assert subtract_sibling_labels(entries) == []


def test_a_label_is_never_stripped_to_nothing():
    from fnt.usv.usv_detector.mad_migrate_composites import (
        subtract_sibling_labels)
    _own, nb = two_blobs()
    entries = [
        {"meta": meta_for(25, 33, 40, 50), "mask": nb},
        {"meta": meta_for(25, 33, 40, 50), "mask": nb},
    ]
    for ch in subtract_sibling_labels(entries):
        assert ch["own"].any()


def test_two_traces_of_one_call_are_reported_not_merged():
    """Which trace to keep is a judgement about the recording."""
    from fnt.usv.usv_detector.mad_migrate_composites import duplicate_pairs
    own, _ = two_blobs()
    almost = own.copy()
    almost[20:30, 11] = False               # one column narrower
    hits = duplicate_pairs([{"meta": meta_for(5, 12, 20, 30), "mask": own},
                            {"meta": meta_for(5, 12, 20, 30), "mask": almost}])
    assert len(hits) == 1
    assert hits[0][2] > 0.8


def test_the_two_passes_run_together_and_stay_idempotent():
    own, nb = two_blobs()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "training_data.h5")
        blank = np.zeros((H, W), np.uint8)
        # A's own box spans everything, so only the sibling pass can save it
        ms.td_save_example(p, blank, own | nb, meta_for(5, 33, 20, 50), "A")
        ms.td_save_example(p, blank, nb, meta_for(25, 33, 40, 50), "B")
        rep = migrate_store(p, dry_run=False, backup=False)
        assert rep.changed == 1
        got = {e["meta"]["id"]: e for e in ms.td_iter_examples(p)}
        assert np.array_equal(got["A"]["mask"] > 0, own)
        assert np.array_equal(got["A"]["neighbors"] > 0, nb)
        assert np.array_equal(got["B"]["mask"] > 0, nb)
        assert migrate_store(p, dry_run=False, backup=False).changed == 0


# ----------------------------------------------------------------------
# The rebuild is the only route from sidecar to training
# ----------------------------------------------------------------------
def test_the_rebuild_carries_neighbours_across():
    """``rebuild_training_store`` regenerates the project store from the
    ``.mad`` sidecars before every run, so anything it drops never reaches
    training. It dropped ``neighbors``.

    The symptom was visible in the live preview panels: a tile plainly
    containing several calls, all of them labelled, scored dice≈0 because its
    ground truth held only one — so the model was penalised for correctly
    finding calls the user had already confirmed. Measured on a real project,
    113 of 250 sidecar examples carried neighbours and the rebuilt store had
    zero.
    """
    from scipy.io import wavfile
    from fnt.usv.usv_detector.mad_examples import rebuild_training_store
    from fnt.usv.usv_detector import fnt_mask_store as _ms
    own, nb = two_blobs()
    with tempfile.TemporaryDirectory() as d:
        wav = os.path.join(d, "rec.wav")
        wavfile.write(wav, SR, np.zeros(SR // 10, np.int16))
        sidecar = _ms.masks_sibling_path(wav)
        ms.td_save_example(sidecar, np.zeros((H, W), np.uint8), own,
                           meta_for(5, 12, 20, 30), "ex1",
                           neighbors_patch=nb)
        out = os.path.join(d, "training_data")
        assert rebuild_training_store(out, [wav]) == 1
        got = list(ms.td_iter_examples(os.path.join(out, "training_data.h5")))[0]
    assert got["neighbors"] is not None, "the rebuild dropped neighbours"
    assert np.array_equal(got["neighbors"] > 0, nb)
    assert np.array_equal(got["mask"] > 0, own), "and it must not merge them"


def test_a_rebuilt_store_still_supervises_the_neighbour():
    """End to end: sidecar -> rebuild -> training target."""
    from scipy.io import wavfile
    from fnt.usv.usv_detector.mad_examples import rebuild_training_store
    from fnt.usv.usv_detector import fnt_mask_store as _ms
    own, nb = two_blobs()
    with tempfile.TemporaryDirectory() as d:
        wav = os.path.join(d, "rec.wav")
        wavfile.write(wav, SR, np.zeros(SR // 10, np.int16))
        ms.td_save_example(_ms.masks_sibling_path(wav),
                           np.zeros((H, W), np.uint8), own,
                           meta_for(5, 12, 20, 30), "ex1", neighbors_patch=nb)
        out = os.path.join(d, "training_data")
        rebuild_training_store(out, [wav])
        _s, t, _w = mx.collect_training_examples(
            out, tile_time_frames=W, tile_freq_bins=H, placements=1, seed=0)
    assert t[0][40:50, 25:33].min() > 0.5, \
        "the neighbour is background again after a rebuild"


# ----------------------------------------------------------------------
# What the user actually saw
# ----------------------------------------------------------------------
def test_the_reconstructed_annotation_covers_one_call_only():
    """The overlay rebuilds a call's box from the stored mask. With the
    composite that box spanned both calls — the merged detection on screen."""
    from fnt.usv.mad_pyqt import MADMainWindow
    own, nb = two_blobs()
    ann = MADMainWindow._annotation_from_example(
        {"meta": meta_for(5, 12, 20, 30, patch_t_off=100), "mask": own}, {})
    assert (ann["t0"], ann["t1"]) == (105, 112)
    assert (ann["f0"], ann["f1"]) == (20, 30)

    merged = MADMainWindow._annotation_from_example(
        {"meta": meta_for(5, 12, 20, 30, patch_t_off=100), "mask": own | nb}, {})
    assert (merged["t0"], merged["t1"]) == (105, 133), \
        "composite should have produced the over-wide box this test guards"
    assert merged["t1"] - merged["t0"] == 4 * (ann["t1"] - ann["t0"])


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
