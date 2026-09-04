"""Harmonic grouping: the contour test, and the two ways it used to over-merge.

Both failure modes here were found by running the rule over ch4's 1019 hand
labels before any of it reached the GUI, and both produced confident, wrong
answers rather than errors -- which is why they are pinned:

* **Chained relations.** Connected components over pairwise matches merge A-B
  and B-C into one call even when A and C have no harmonic relation. On a busy
  file that staples simultaneous calls from different animals together. The
  measured signature was a stack holding two members with the same harmonic
  number, which cannot happen physically.
* **Manufactured fundamentals.** Allowing f0 to be inferred below the lowest
  detected element invented fundamentals at 8-12 kHz for a third of all stacks,
  because a low enough f0 explains any set of frequencies as near-multiples of
  itself.

Runs under pytest, or directly, since the lab environments do not all carry
pytest.
"""
import os

import numpy as np

from fnt.usv.usv_detector.mad_harmonics import (
    Element, HarmonicConfig, contour_from_patch, group_calls, pair_relation)


def sweep(eid, f_start, f_end, t0=0.0, t1=0.03, n=24):
    """A linearly frequency-modulated element."""
    return Element(eid, np.linspace(t0, t1, n), np.linspace(f_start, f_end, n))


def flat(eid, f, t0=0.0, t1=0.03, n=24):
    return Element(eid, np.linspace(t0, t1, n), np.full(n, float(f)))


def _stack_of(res, eid):
    return res.calls[[c.call_id for c in res.calls].index(res.call_of[eid])]


# ----------------------------------------------------------------------
# The contour test
# ----------------------------------------------------------------------
def test_a_scaled_copy_of_the_contour_is_a_harmonic():
    res = group_calls([sweep('f0', 25_000, 30_000),
                       sweep('h2', 50_000, 60_000),
                       sweep('h3', 75_000, 90_000)])
    assert res.n_calls == 1
    call = res.calls[0]
    assert call.members == {'f0': 1, 'h2': 2, 'h3': 3}
    assert abs(call.f0_hz - 27_500) < 500
    assert call.fundamental_detected


def test_a_contour_that_merely_crosses_2x_is_not():
    """Both sit near a 2x ratio on average; only one keeps it across the sweep.

    This is the case a median- or peak-frequency comparison cannot see, and it
    is most of the available signal on frequency-modulated calls.
    """
    fundamental = sweep('f0', 25_000, 30_000)
    real = sweep('h2', 50_000, 60_000)          # ratio pinned at 2.0
    impostor = sweep('x', 60_000, 50_000)       # same mean, opposite sweep
    assert pair_relation(fundamental, real).ok
    assert not pair_relation(fundamental, impostor).ok


def test_elements_that_do_not_overlap_in_time_are_never_linked():
    """A harmonic is part of the same vocal event, so it co-occurs."""
    res = group_calls([sweep('a', 25_000, 25_000, t0=0.0, t1=0.03),
                       sweep('b', 50_000, 50_000, t0=1.0, t1=1.03)])
    assert res.n_calls == 2


def test_a_short_element_skips_the_shape_test_but_not_the_ratio():
    """Below a few samples the CV estimate means nothing, so only the mean
    ratio is asked for -- but it is still asked for."""
    cfg = HarmonicConfig(min_points_for_cv=5)
    assert pair_relation(flat('a', 25_000, n=3), flat('b', 50_000, n=3), cfg).ok
    assert not pair_relation(flat('a', 25_000, n=3),
                             flat('b', 61_000, n=3), cfg).ok


# ----------------------------------------------------------------------
# Over-merging
# ----------------------------------------------------------------------
def test_relations_are_not_inherited_through_a_chain():
    """b is 2x a and c is 2x b, but c is 4x a -- one call, not a chain of two.

    The point is that every member is tested against the fundamental itself, so
    membership can never be acquired second-hand.
    """
    res = group_calls([flat('a', 20_000), flat('b', 40_000), flat('c', 80_000)])
    assert res.n_calls == 1
    assert res.calls[0].members == {'a': 1, 'b': 2, 'c': 4}


def test_two_animals_calling_at_once_stay_two_calls():
    """20/40 kHz and 27/54 kHz overlap in time. Chaining used to merge them."""
    res = group_calls([flat('a1', 20_000), flat('a2', 40_000),
                       flat('b1', 27_000), flat('b2', 54_000)])
    assert res.n_calls == 2
    assert res.call_of['a1'] == res.call_of['a2']
    assert res.call_of['b1'] == res.call_of['b2']
    assert res.call_of['a1'] != res.call_of['b1']


def test_no_stack_ever_holds_two_of_the_same_harmonic():
    """Physically impossible, and the signature of the chaining bug."""
    res = group_calls([flat('f0', 25_000), flat('h2a', 50_000),
                       flat('h2b', 49_000), flat('h3', 75_000)])
    for call in res.calls:
        ns = list(call.members.values())
        assert len(ns) == len(set(ns)), call.members
    # The better-fitting claimant keeps the slot; the loser seeds its own call.
    assert res.harmonic_of['h2a'] == 2
    assert res.call_of['h2b'] != res.call_of['f0']


def test_a_stack_is_rechecked_against_its_finished_fundamental():
    """pair_relation compares over the span two elements *share*, which on a
    partial overlap can land on a different integer than the elements' overall
    frequencies support. The symptom was a 4th harmonic sitting below the 3rd.
    """
    res = group_calls([sweep('f0', 20_000, 22_000),
                       sweep('h2', 40_000, 44_000),
                       sweep('h3', 60_000, 66_000),
                       # overlaps only at its start, where its ratio flatters
                       sweep('odd', 57_000, 59_000, t0=0.0, t1=0.03)])
    for call in res.calls:
        f0 = call.f0_hz
        # every member must sit near n * f0 on its whole contour, not just on
        # the slice it happens to share with the fundamental
        for eid, n in call.members.items():
            src = {'f0': 21_000, 'h2': 42_000, 'h3': 63_000, 'odd': 58_000}[eid]
            assert abs(src / f0 - n) / n <= 0.06 + 1e-9, (eid, n, f0)


# ----------------------------------------------------------------------
# Manufactured fundamentals
# ----------------------------------------------------------------------
def test_no_fundamental_is_invented_by_default():
    """50 and 75 kHz *could* be H2 and H3 of a missing 25 kHz. By default the
    answer is two calls, because inference off measured better than on."""
    res = group_calls([flat('a', 50_000), flat('b', 75_000)])
    assert res.n_calls == 2
    assert all(c.fundamental_detected for c in res.calls)


def test_inference_can_be_turned_on_and_then_finds_it():
    cfg = HarmonicConfig(max_missing_fundamental=3, min_f0_hz=15_000)
    res = group_calls([flat('a', 50_000), flat('b', 75_000)], cfg)
    assert res.n_calls == 1
    call = res.calls[0]
    assert not call.fundamental_detected
    assert abs(call.f0_hz - 25_000) < 1000
    assert call.members == {'a': 2, 'b': 3}


def test_the_f0_floor_blocks_the_implausible_fit():
    """Without a floor, a low enough f0 explains anything. 11 kHz is below what
    a vole produces, so the fit must be refused rather than reported."""
    cfg = HarmonicConfig(max_missing_fundamental=3, min_f0_hz=15_000)
    res = group_calls([flat('a', 22_000), flat('b', 33_000)], cfg)
    assert res.n_calls == 2, "an 11 kHz fundamental should not be inferred"


# ----------------------------------------------------------------------
# Manual corrections
# ----------------------------------------------------------------------
def test_a_reviewer_can_pin_an_element_out_of_its_stack():
    els = [flat('f0', 25_000), flat('h2', 50_000)]
    assert group_calls(els).n_calls == 1
    res = group_calls(els, forced={'h2': None})
    assert res.n_calls == 2
    assert res.harmonic_of['h2'] == 1


def test_a_reviewer_can_move_an_element_to_another_fundamental():
    """h2 auto-groups with a; the reviewer says it belongs to b."""
    els = [flat('a', 25_000), flat('b', 24_000), flat('h2', 50_000)]
    res = group_calls(els, forced={'h2': 'b'})
    assert res.call_of['h2'] == 'b'
    assert res.harmonic_of['h2'] == 2
    assert res.call_of['a'] == 'a' and len(_stack_of(res, 'a').members) == 1


def test_a_pin_survives_a_relation_the_test_would_reject():
    """The reviewer is allowed to be right when the tolerances are wrong."""
    els = [flat('f0', 25_000), flat('weird', 63_000)]
    assert group_calls(els).n_calls == 2
    res = group_calls(els, forced={'weird': 'f0'})
    assert res.n_calls == 1
    assert res.call_of['weird'] == 'f0'


def test_a_pinned_fundamental_is_never_absorbed_as_a_harmonic():
    """Naming something a fundamental has to stick, or the correction is
    silently undone by the next re-run."""
    els = [flat('low', 12_500), flat('mid', 25_000), flat('up', 50_000)]
    res = group_calls(els, forced={'up': 'mid'})
    assert res.call_of['up'] == 'mid'
    assert res.call_of['mid'] == 'mid', "mid was absorbed by low"


def test_a_correction_cycle_does_not_hang():
    els = [flat('a', 25_000), flat('b', 50_000)]
    res = group_calls(els, forced={'a': 'b', 'b': 'a'})
    assert res.n_elements == 2


# ----------------------------------------------------------------------
# Contours from stored patches
# ----------------------------------------------------------------------
def test_a_contour_is_read_out_of_a_stored_patch():
    meta = {'sample_rate': 250_000, 'nfft': 1024, 'patch_f_off': 0,
            'patch_t_frames': 8, 'patch_t0_s': 1.0, 'patch_t1_s': 1.008,
            'id': 'e1'}
    mask = np.zeros((513, 8), dtype=bool)
    spec = np.zeros((513, 8), dtype=np.float32)
    # A rising track: bin 100 -> 107, i.e. ~24.4 -> ~26.1 kHz.
    for c in range(8):
        mask[100 + c, c] = True
        spec[100 + c, c] = 1.0
    e = contour_from_patch(mask, spec, meta)
    assert e is not None and e.times.size == 8
    assert abs(e.freqs[0] - 100 * 250_000 / 1024) < 1.0
    assert e.freqs[-1] > e.freqs[0]
    assert abs(e.t0 - 1.0005) < 1e-3


def test_an_empty_mask_yields_no_element():
    meta = {'sample_rate': 250_000, 'nfft': 1024, 'patch_t_frames': 4,
            'patch_t0_s': 0.0, 'patch_t1_s': 0.004}
    z = np.zeros((16, 4))
    assert contour_from_patch(z.astype(bool), z.astype(np.float32), meta) is None


def test_grouping_an_empty_list_is_not_an_error():
    res = group_calls([])
    assert res.n_calls == 0 and res.n_elements == 0


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
