"""Tile assembly must carry provenance, or the split has nothing to group on.

``collect_training_examples`` flattens per-call patches into fixed-size tiles.
The group keys it returns alongside them are what makes a leak-free train/val
split possible, so they have to stay aligned with the tiles one-for-one — an
off-by-one here would silently mis-attribute tiles to the wrong recording.
"""
import numpy as np
import pytest

from fnt.usv.usv_detector.fnt_mask_store import td_save_example
from fnt.usv.usv_detector.mad_examples import (
    _store_path, collect_training_examples,
)

TILE_F, TILE_T = 64, 64


@pytest.fixture
def store(tmp_path):
    """Builder for a training-data dir: add(wav, call_id, n_tiles_wide)."""
    td_dir = tmp_path / "training_data"
    td_dir.mkdir()
    rng = np.random.default_rng(0)

    def add(wav: str, call_id: str, tiles_wide: int = 1):
        width = TILE_T * tiles_wide
        spec = rng.random((TILE_F, width)).astype(np.float32)
        mask = np.zeros((TILE_F, width), dtype=np.float32)
        mask[20:30, 5:20] = 1.0
        td_save_example(
            _store_path(str(td_dir)), spec, mask,
            {'class': 'USV', 'source_wav': wav,
             'patch_t_off': 0, 'patch_f_off': 0},
            call_id,
        )

    add.dir = str(td_dir)
    return add


def _collect(td_dir):
    return collect_training_examples(
        td_dir, tile_time_frames=TILE_T, tile_freq_bins=TILE_F,
        return_groups=True,
    )


def test_one_group_key_per_tile(store):
    store('a.wav', 'a_call0', tiles_wide=3)
    store('b.wav', 'b_call0', tiles_wide=1)

    specs, targets, weights, groups = _collect(store.dir)

    assert specs.shape[0] == len(groups) == 4
    assert specs.shape[1:] == (TILE_F, TILE_T)
    assert targets.shape == specs.shape == weights.shape


def test_a_wide_call_yields_several_tiles_all_tagged_to_it(store):
    store('a.wav', 'a_call0', tiles_wide=3)

    _, _, _, groups = _collect(store.dir)

    assert len(groups) == 3
    assert {g[0] for g in groups} == {'a.wav'}
    assert {g[1] for g in groups} == {'a_call0'}


def test_group_keys_separate_recordings_and_calls(store):
    store('a.wav', 'a_call0')
    store('a.wav', 'a_call1')
    store('b.wav', 'b_call0')

    _, _, _, groups = _collect(store.dir)

    assert {g[0] for g in groups} == {'a.wav', 'b.wav'}
    assert len({g[1] for g in groups}) == 3


def test_groups_are_opt_in_so_the_old_contract_still_holds(store):
    store('a.wav', 'a_call0')

    out = collect_training_examples(
        store.dir, tile_time_frames=TILE_T, tile_freq_bins=TILE_F)

    assert len(out) == 3  # specs, targets, weights — no groups


def test_empty_store_returns_empty_stacks_and_no_groups(tmp_path):
    empty = tmp_path / "training_data"
    empty.mkdir()

    specs, targets, weights, groups = _collect(str(empty))

    assert specs.shape[0] == 0
    assert groups == []
