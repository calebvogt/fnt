"""A call's confirmed neighbours come from the whole store, not a snapshot.

Each example records the confirmed calls in its patch window when it is saved.
A call confirmed LATER in the same bout is missing from that snapshot, so the
earlier example's window taught it as background. ``collect_training_examples``
now rebuilds every window from all the labels on the recording.
"""
import numpy as np

from fnt.usv.usv_detector.fnt_mask_store import td_save_example
from fnt.usv.usv_detector.mad_examples import (
    _store_path, collect_training_examples, confirmed_neighbours, iter_examples,
)

H, W = 64, 64
GRID = {'nperseg': 512, 'noverlap': 384, 'nfft': 1024, 'sample_rate': 250000}


def _save(td_dir, eid, *, wav='a.wav', t_off=0, cols=(5, 20), rows=(20, 30),
          kind=None, neighbors=None, grid=GRID):
    spec = np.full((H, W), 0.2, dtype=np.float32)
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[rows[0]:rows[1], cols[0]:cols[1]] = 1
    meta = {'class': 'USV', 'source_wav': wav, 'patch_t_off': t_off,
            'patch_f_off': 0, **grid}
    if kind:
        meta['kind'] = kind
    td_save_example(_store_path(td_dir), spec, mask, meta, eid,
                    neighbors_patch=neighbors)


def _td(tmp_path):
    d = tmp_path / 'training_data'
    d.mkdir()
    return str(d)


def _by_id(td_dir):
    exs = list(iter_examples(td_dir))
    nbs = confirmed_neighbours(exs)
    return {ex['meta']['id']: nb for ex, nb in zip(exs, nbs)}


def _tiles(td_dir):
    """{example id: (target, weight)} with one placement. Patches are exactly
    one tile wide, so the tile IS the patch -- no random offset to undo."""
    _s, targets, weights, groups = collect_training_examples(
        td_dir, tile_time_frames=W, tile_freq_bins=H, return_groups=True,
        placements=1)
    return {g[1]: (t, w) for g, t, w in zip(groups, targets, weights)}


def test_a_call_confirmed_later_is_in_the_earlier_calls_window(tmp_path):
    td = _td(tmp_path)
    # A saved first: its snapshot could not know about B.
    _save(td, 'A', t_off=0, cols=(5, 20))
    # B saved later, 30 frames on: its own pixels sit at absolute 35..50,
    # inside A's window (0..64).
    _save(td, 'B', t_off=30, cols=(5, 20))

    nb = _by_id(td)
    assert nb['A'] is not None
    assert nb['A'][20:30, 35:50].all()
    assert nb['A'].sum() == 10 * 15
    # ...and A, at absolute 5..20, is outside B's window (30..94).
    assert nb['B'] is None


def test_the_later_call_trains_as_a_call_not_background(tmp_path):
    td = _td(tmp_path)
    _save(td, 'A', t_off=0, cols=(5, 20))
    _save(td, 'B', t_off=30, cols=(5, 20))

    target, weight = _tiles(td)['A']
    assert target[20:30, 5:20].all(), "A's own call"
    assert target[20:30, 35:50].all(), "B, confirmed after A was saved"
    assert target.sum() == 2 * 10 * 15
    assert weight.all()


def test_a_label_saved_after_a_rejection_is_ignored_in_its_window(tmp_path):
    """A rejection's target is all background, so a confirmed call inside its
    window must drop out of the loss -- including one confirmed later."""
    td = _td(tmp_path)
    _save(td, 'R', t_off=0, cols=(5, 20), kind='negative')
    _save(td, 'B', t_off=30, cols=(5, 20))

    target, weight = _tiles(td)['R']
    assert target.sum() == 0
    assert not weight[20:30, 35:50].any(), "B is neither call nor background"
    assert weight[20:30, 5:20].all(), "the rejected shape is still taught"


def test_rejections_are_never_anyones_neighbour(tmp_path):
    td = _td(tmp_path)
    _save(td, 'A', t_off=0, cols=(5, 20))
    _save(td, 'R', t_off=30, cols=(5, 20), kind='negative')

    assert _by_id(td)['A'] is None
    target, _w = _tiles(td)['A']
    assert target.sum() == 10 * 15


def test_other_recordings_and_other_grids_do_not_count(tmp_path):
    td = _td(tmp_path)
    _save(td, 'A', wav='a.wav', t_off=0, cols=(5, 20))
    _save(td, 'X', wav='b.wav', t_off=30, cols=(5, 20))
    _save(td, 'Y', wav='a.wav', t_off=30, cols=(5, 20),
          grid={**GRID, 'noverlap': 256})

    assert _by_id(td)['A'] is None


def test_the_stored_snapshot_is_kept_alongside(tmp_path):
    """Union, not replacement: a neighbour the snapshot holds stays positive
    even when nothing else in the store accounts for it."""
    td = _td(tmp_path)
    snap = np.zeros((H, W), dtype=np.uint8)
    snap[40:45, 50:60] = 1
    _save(td, 'A', t_off=0, cols=(5, 20), neighbors=snap)
    _save(td, 'B', t_off=30, cols=(5, 20))

    target, _w = _tiles(td)['A']
    assert target[40:45, 50:60].all()
    assert target[20:30, 35:50].all()


def test_a_call_far_away_is_not_a_neighbour(tmp_path):
    td = _td(tmp_path)
    _save(td, 'A', t_off=0, cols=(5, 20))
    _save(td, 'B', t_off=500, cols=(5, 20))

    nb = _by_id(td)
    assert nb['A'] is None and nb['B'] is None
