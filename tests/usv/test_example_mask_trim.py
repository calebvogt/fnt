"""Separate USVs must not come back as one merged call.

A saved patch is a time-crop with context, so it routinely contains calls other
than the one being saved. Storing only the one made every other confirmed call
in the crop supervised as BACKGROUND, so the save path composites every
confirmed pixel in the window into the stored mask. That union is the right
training target.

It is not the call's identity. ``_examples_to_annotations`` derives each
detection's box from the mask pixels, so the union became the box: two USVs
that were labelled separately reloaded as a single detection spanning both,
with a frequency range and pixel count to match. Observed as rows at one
timestamp reading 20-25, 20-38, 20-48, 20-53, 20-71 kHz — each label's box
swollen by whichever neighbours shared its window.

The metadata still records what was actually drawn, so the loader intersects
with it.
"""
import numpy as np
import pytest

from fnt.usv.usv_detector.mad_examples import _examples_to_annotations

SR, NPERSEG, NOVERLAP, NFFT = 250000, 512, 384, 1024
HOP = NPERSEG - NOVERLAP
DT = HOP / SR
DF = (SR / 2.0) / (NFFT // 2)
N_FREQ = NFFT // 2 + 1


def _meta(f0, f1, t0, t1, patch_t_off=0, eid="ex1"):
    """Metadata describing the call actually drawn, in patch-local bins."""
    return {
        'id': eid, 'class': 'USV', 'source_wav': 'rec.wav',
        'sample_rate': SR, 'nperseg': NPERSEG, 'noverlap': NOVERLAP,
        'nfft': NFFT, 'patch_t_off': patch_t_off, 'patch_f_off': 0,
        'patch_t0_s': patch_t_off * DT,
        't_start_s': (patch_t_off + t0) * DT,
        't_stop_s': (patch_t_off + t1) * DT,
        'f_low_hz': f0 * DF, 'f_high_hz': f1 * DF,
    }


def _example(mask, meta):
    return {'mask': mask.astype(np.uint8), 'meta': meta}


def _one(mask, meta):
    anns = list(_examples_to_annotations([_example(mask, meta)], 'rec.wav',
                                         (N_FREQ, 100000)))
    assert len(anns) == 1
    return anns[0]


def test_a_neighbour_in_the_patch_does_not_widen_the_box():
    """The reported bug: one call's box swallowing the call above it."""
    W = 120
    mask = np.zeros((N_FREQ, W), dtype=np.uint8)
    mask[80:100, 20:60] = 1          # the call this example is about
    mask[200:260, 20:60] = 1         # a separate USV sharing the window
    ann = _one(mask, _meta(80, 100, 20, 60))
    assert ann['f0'] == 80 and ann['f1'] == 100
    assert not ann['mask'][:, :].shape[0] > 20      # only its own band
    assert ann['mask'].sum() == 20 * 40


def test_without_the_trim_the_box_would_span_both():
    """States the failure, so the regression is unmistakable."""
    W = 120
    mask = np.zeros((N_FREQ, W), dtype=bool)
    mask[80:100, 20:60] = True
    mask[200:260, 20:60] = True
    fs = np.where(mask.any(axis=1))[0]
    assert int(fs[0]) == 80 and int(fs[-1]) + 1 == 260   # the merged extent


def test_a_neighbour_separated_in_time_is_also_trimmed():
    W = 200
    mask = np.zeros((N_FREQ, W), dtype=np.uint8)
    mask[80:100, 10:40] = 1          # this call
    mask[80:100, 120:170] = 1        # a later call in the same crop
    ann = _one(mask, _meta(80, 100, 10, 40))
    assert ann['t1'] - ann['t0'] == 30
    assert ann['mask'].sum() == 20 * 30


def test_an_example_saved_before_compositing_is_unchanged():
    """Its mask already equals the call, so the trim is a no-op."""
    W = 80
    mask = np.zeros((N_FREQ, W), dtype=np.uint8)
    mask[50:70, 5:35] = 1
    ann = _one(mask, _meta(50, 70, 5, 35))
    assert (ann['f0'], ann['f1']) == (50, 70)
    assert ann['mask'].sum() == 20 * 30


def test_the_patch_offset_is_still_applied():
    """Trimming must not disturb the mapping into full-file coordinates."""
    W = 80
    mask = np.zeros((N_FREQ, W), dtype=np.uint8)
    mask[50:70, 5:35] = 1
    ann = _one(mask, _meta(50, 70, 5, 35, patch_t_off=4000))
    assert ann['t0'] == 4005 and ann['t1'] == 4035


def test_a_call_shape_is_preserved_not_squared_off():
    """The trim intersects; it must not replace the mask with its box."""
    W = 80
    mask = np.zeros((N_FREQ, W), dtype=np.uint8)
    for i in range(30):              # a diagonal sweep
        mask[50 + i // 2, 5 + i] = 1
    ann = _one(mask, _meta(50, 65, 5, 35))
    assert ann['mask'].sum() == 30          # still 30 pixels, not a filled box
    assert not ann['mask'].all()


def test_metadata_without_a_usable_box_falls_back_to_the_mask():
    """Older or partial metadata must not lose the call entirely."""
    W = 80
    mask = np.zeros((N_FREQ, W), dtype=np.uint8)
    mask[50:70, 5:35] = 1
    meta = _meta(50, 70, 5, 35)
    del meta['f_low_hz']                    # _bbox_from_meta gives up
    ann = _one(mask, meta)
    assert ann['mask'].sum() == 20 * 30


def test_a_box_that_misses_the_mask_keeps_the_mask():
    """Never return an empty detection because the metadata disagreed."""
    W = 80
    mask = np.zeros((N_FREQ, W), dtype=np.uint8)
    mask[50:70, 5:35] = 1
    ann = _one(mask, _meta(300, 320, 60, 75))   # box nowhere near the pixels
    assert ann['mask'].sum() == 20 * 30


def test_a_hard_negative_still_rebuilds_its_box_from_metadata():
    """The blank-mask path is untouched by the trim."""
    from fnt.usv.usv_detector.fnt_mask_store import REJECTED_KINDS
    W = 80
    mask = np.zeros((N_FREQ, W), dtype=np.uint8)     # deliberately empty
    meta = _meta(50, 70, 5, 35, eid="neg1")
    meta['kind'] = REJECTED_KINDS[0]
    anns = list(_examples_to_annotations([_example(mask, meta)], 'rec.wav',
                                         (N_FREQ, 100000),
                                         kinds=REJECTED_KINDS))
    assert len(anns) == 1
    assert anns[0]['mask'].any()
