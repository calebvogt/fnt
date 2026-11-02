"""Rejecting a call must not destroy it, and must teach the model something.

Reject used to hard-delete the confirmed call's example. That got the training
half right -- a rejected call has to stop training as a call -- but it threw
away the traced mask with it. Reopening the file then showed a filled bounding
box (all the CSV row carries), and re-accepting would have saved *that box* as
the label, teaching the model that every pixel in it is a call. So Reject was
silently irreversible across sessions, and un-rejecting was worse than lossy.

Now the example is demoted to ``kind='rejected'``: pixels kept, and its patch
supervised with an all-zero target so the refused call trains as a hard
negative. Delete still deletes.

That negative supervision is also the only thing in the training set that can
teach *shape*. With positives alone, "bright ⇒ call" fits the data perfectly;
the quiet margin around each patch only ever teaches "quiet ⇒ not a call".
Rejections are bright things that are not calls -- the counterexample that
makes shape matter.

Runs under pytest, or directly.
"""
import os
import tempfile

import numpy as np

from fnt.usv.usv_detector import mad_examples as MX
from fnt.usv.usv_detector.fnt_mask_store import (
    example_kind, td_iter_examples, td_read_example, td_save_example,
    td_set_kind)

SR, NFFT = 250_000, 1024


def _store():
    d = tempfile.mkdtemp(prefix="mad_rej_")
    return d, MX._store_path(d)


def _meta(eid, f_off=100, t_off=500):
    return {'id': eid, 'class': 'USV', 'source_wav': 'a.wav',
            'sample_rate': SR, 'nfft': NFFT, 'nperseg': 512, 'noverlap': 384,
            'patch_f_off': f_off, 'patch_t_off': t_off,
            'patch_t0_s': 0.0, 'patch_t1_s': 0.1, 'patch_t_frames': 40}


def _save(dirpath, eid):
    """A patch with a diagonal 'call' in it, so the mask has a real shape."""
    spec = np.full((32, 40), 0.1, dtype=np.float32)
    mask = np.zeros((32, 40), dtype=np.uint8)
    for c in range(6, 30):
        mask[10 + c // 3, c] = 1
        spec[10 + c // 3, c] = 0.95
    return MX.save_example(dirpath, spec, mask, _meta(eid), example_id=eid)


# ----------------------------------------------------------------------
# The store primitive
# ----------------------------------------------------------------------
def test_demoting_keeps_every_pixel():
    d, h5 = _store()
    _save(d, 'e1')
    before = td_read_example(h5, 'e1')['mask']
    assert td_set_kind(h5, 'e1', 'rejected')
    after = td_read_example(h5, 'e1')
    assert example_kind(after['meta']) == 'rejected'
    assert np.array_equal(after['mask'], before), "the mask changed"
    assert after['mask'].sum() > 0


def test_demoting_is_reversible():
    d, h5 = _store()
    _save(d, 'e1')
    original = td_read_example(h5, 'e1')['mask'].copy()
    td_set_kind(h5, 'e1', 'rejected')
    td_set_kind(h5, 'e1', 'label')
    back = td_read_example(h5, 'e1')
    assert example_kind(back['meta']) == 'label'
    assert np.array_equal(back['mask'], original)


def test_demoting_a_missing_example_reports_failure():
    _d, h5 = _store()
    assert td_set_kind(h5, 'nope', 'rejected') is False


# ----------------------------------------------------------------------
# Training supervision
# ----------------------------------------------------------------------
def _targets_for(dirpath):
    _specs, targets, weights = MX.collect_training_examples(
        dirpath, placements=1, seed=0)
    return targets, weights


def test_a_label_supervises_its_pixels_as_positive():
    d, _h5 = _store()
    _save(d, 'e1')
    targets, _w = _targets_for(d)
    assert targets.sum() > 0


def test_a_rejected_call_supervises_its_pixels_as_NEGATIVE():
    """The point of the change: bright pixels a human refused become explicit
    "not a call" supervision, rather than vanishing from the training set."""
    d, h5 = _store()
    _save(d, 'e1')
    td_set_kind(h5, 'e1', 'rejected')
    targets, weights = _targets_for(d)
    assert targets.sum() == 0, "a rejected call still trains as a call"
    assert weights.sum() > 0, "it must still be supervised, not merely dropped"


def test_a_rejected_call_is_not_simply_discarded():
    """Dropping it would leave the model unpenalised on its own false positive.
    The patch has to reach training with weight."""
    d, h5 = _store()
    _save(d, 'e1')
    _t_before, w_before = _targets_for(d)
    td_set_kind(h5, 'e1', 'rejected')
    _t_after, w_after = _targets_for(d)
    assert w_after.sum() == w_before.sum()


def test_negatives_and_rejections_are_treated_alike():
    d, h5 = _store()
    _save(d, 'a')
    _save(d, 'b')
    td_set_kind(h5, 'a', 'negative')
    td_set_kind(h5, 'b', 'rejected')
    targets, _w = _targets_for(d)
    assert targets.sum() == 0


# ----------------------------------------------------------------------
# What the overlay shows
# ----------------------------------------------------------------------
GRID = (513, 4000)


def test_a_rejected_call_is_not_shown_as_a_confirmed_label():
    d, h5 = _store()
    _save(d, 'e1')
    assert len(list(MX.iter_file_annotations(d, 'a.wav', GRID))) == 1
    td_set_kind(h5, 'e1', 'rejected')
    assert list(MX.iter_file_annotations(d, 'a.wav', GRID)) == []


def test_but_its_geometry_is_still_retrievable():
    """This is what turns the red bounding box back into a real outline."""
    d, h5 = _store()
    _save(d, 'e1')
    truth = next(iter(MX.iter_file_annotations(d, 'a.wav', GRID)))
    td_set_kind(h5, 'e1', 'rejected')
    got = list(MX.iter_file_rejected_annotations(d, 'a.wav', GRID))
    assert len(got) == 1
    assert np.array_equal(got[0]['mask'], truth['mask'])
    assert (got[0]['f0'], got[0]['t0']) == (truth['f0'], truth['t0'])
    # And it is a traced shape, not a filled box.
    m = got[0]['mask']
    assert 0 < m.sum() < m.size * 0.5, "geometry collapsed to a bounding box"


def test_a_rejected_call_does_not_pollute_the_confirmed_overlay():
    d, h5 = _store()
    _save(d, 'e1')
    td_set_kind(h5, 'e1', 'rejected')
    assert MX.reconstruct_file_mask(d, 'a.wav', GRID).sum() == 0


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
