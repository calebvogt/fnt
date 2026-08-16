"""The train/val split must never put one group on both sides.

This is the test that guards a rigor bug, not a crash: the split used to be a
random permutation over *tiles*, and tiles are not independent. One long call is
sliced into several, and neighbouring tiles overlap, so shuffling tiles put
near-duplicate pixels in train and val and validation Dice measured
memorization. A model that regresses here still trains fine and still reports a
number — it just reports a number that means nothing, which is worse.
"""
import numpy as np
import pytest

from fnt.usv.usv_detector.mad_training import grouped_split, _split_tiles


def _groups(spec):
    """[('rec_0.wav', 'rec_0_call_0'), ...] from {file: (n_calls, tiles)}."""
    out = []
    for wav, (n_calls, tiles) in spec.items():
        for c in range(n_calls):
            out.extend([(wav, f'{wav}_call_{c}')] * tiles)
    return out


# ----------------------------------------------------------------------
# grouped_split
# ----------------------------------------------------------------------
def test_no_group_straddles_the_split():
    keys = ['a'] * 10 + ['b'] * 10 + ['c'] * 10 + ['d'] * 10
    train_idx, val_idx, val_groups = grouped_split(keys, 0.25, seed=42)

    train_groups = {keys[i] for i in train_idx}
    held_out = {keys[i] for i in val_idx}
    assert not (train_groups & held_out)
    assert sorted(held_out) == sorted(val_groups)
    # Every tile is used exactly once.
    assert len(train_idx) + len(val_idx) == len(keys)
    assert set(train_idx).isdisjoint(set(val_idx))


def test_split_is_reproducible_for_a_seed():
    keys = ['a'] * 5 + ['b'] * 5 + ['c'] * 5 + ['d'] * 5
    first = grouped_split(keys, 0.25, seed=42)
    again = grouped_split(keys, 0.25, seed=42)
    assert first[2] == again[2]
    assert np.array_equal(first[0], again[0])
    assert np.array_equal(first[1], again[1])


@pytest.mark.parametrize("val_fraction", [0.0, 0.01, 0.5, 0.99, 1.0])
def test_neither_side_is_ever_empty(val_fraction):
    """Including val_fraction=0 (still hold one out) and 1.0 (still train)."""
    keys = ['a'] * 4 + ['b'] * 4 + ['c'] * 4
    train_idx, val_idx, _ = grouped_split(keys, val_fraction, seed=1)
    assert len(train_idx) > 0
    assert len(val_idx) > 0


def test_single_group_cannot_be_split():
    assert grouped_split(['only'] * 8, 0.2) is None
    assert grouped_split([], 0.2) is None


def test_uneven_groups_stay_whole():
    """One dominant recording must not be sliced to hit the ratio."""
    keys = ['big'] * 100 + ['s1', 's2', 's3']
    train_idx, val_idx, _ = grouped_split(keys, 0.2, seed=42)
    assert not ({keys[i] for i in train_idx} & {keys[i] for i in val_idx})


# ----------------------------------------------------------------------
# _split_tiles — picks the strongest level the labels support
# ----------------------------------------------------------------------
def test_multiple_recordings_split_at_file_level():
    groups = _groups({f'rec_{i}.wav': (5, 3) for i in range(4)})
    out = _split_tiles(groups, len(groups), 0.25)

    assert out['split_level'] == 'file'
    assert out['val_held_out'] is True
    assert not ({groups[i][0] for i in out['train_idx']}
                & {groups[i][0] for i in out['val_idx']})


def test_one_recording_falls_back_to_call_level_and_says_so():
    groups = _groups({'only.wav': (6, 2)})
    out = _split_tiles(groups, len(groups), 0.25)

    assert out['split_level'] == 'call'
    # A call never straddles the split...
    assert not ({groups[i][1] for i in out['train_idx']}
                & {groups[i][1] for i in out['val_idx']})
    # ...but the recording is shared, so this is NOT a held-out measurement.
    assert out['val_held_out'] is False


def test_one_call_falls_back_to_tiles_and_admits_the_overlap():
    groups = _groups({'only.wav': (1, 4)})
    out = _split_tiles(groups, len(groups), 0.25)

    assert out['split_level'] == 'tile'
    assert out['val_held_out'] is False
    assert len(out['train_idx']) and len(out['val_idx'])


def test_missing_provenance_falls_back_rather_than_crashing():
    out = _split_tiles([], 6, 0.25)
    assert out['split_level'] == 'tile'
    assert len(out['train_idx']) + len(out['val_idx']) == 6


# ----------------------------------------------------------------------
# The regression itself, stated as a measurement
# ----------------------------------------------------------------------
def test_grouped_split_eliminates_leakage_that_tile_shuffle_produced():
    """At representative scale the old scheme leaked heavily; this must not."""
    groups = _groups({f'rec_{f}.wav': (40, 1) for f in range(3)})
    # Make a third of the calls long enough to span several tiles.
    groups += _groups({f'rec_{f}.wav': (13, 2) for f in range(3)})
    n_total = len(groups)

    # Old behavior, reproduced: a plain permutation over tiles.
    rng = np.random.default_rng(42)
    idx = rng.permutation(n_total)
    n_val = max(1, int(n_total * 0.20))
    old_val, old_train = idx[:n_val], idx[n_val:]
    old_train_files = {groups[i][0] for i in old_train}
    leaked = sum(1 for i in old_val if groups[i][0] in old_train_files)
    assert leaked > 0, "sanity: the old scheme is expected to leak here"

    # New behavior: nothing in val shares a recording OR a call with train.
    out = _split_tiles(groups, n_total, 0.20)
    train_files = {groups[i][0] for i in out['train_idx']}
    train_calls = {groups[i][1] for i in out['train_idx']}
    assert not any(groups[i][0] in train_files for i in out['val_idx'])
    assert not any(groups[i][1] in train_calls for i in out['val_idx'])
