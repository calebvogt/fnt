"""One accept, one file open.

Confirming a call used to open its ``.mad`` four separate times — list ids for
the undo snapshot, scan metadata for stale copies, stamp grid attrs, save the
example. Each open of an HDF5 file over SMB measured ~9 ms, so 36 ms of a 60 ms
accept was spent opening and closing the same file, which is what made review
feel sticky under a keystroke-per-call workflow.

Collapsing them is only worth anything if the writes stay correct, so that is
what these check: the stale-copy sweep, the write-once provenance, and the
pre-write id list the undo snapshot depends on.
"""
import json

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from fnt.usv.usv_detector.fnt_mask_store import (  # noqa: E402
    td_commit_example, td_count, td_iter_meta, td_list_ids, td_read_example,
)

GRID = dict(sample_rate=250000, nperseg=512, noverlap=384, nfft=1024,
            n_freq_bins=513, n_time_frames=1000, source_wav="rec.wav")


def _meta(blob_id=None, kind=None):
    m = {'source_wav': 'rec.wav', 'class': 'USV',
         't_start_s': 1.0, 't_stop_s': 1.08,
         'f_low_hz': 50000.0, 'f_high_hz': 70000.0}
    if blob_id is not None:
        m['blob_id'] = blob_id
    if kind is not None:
        m['kind'] = kind
    return m


def _commit(h5, ex_id, blob_id=None, replace=None, grid=GRID, kind=None):
    spec = np.random.default_rng(0).random((16, 20)).astype(np.float32)
    mask = np.zeros((16, 20), dtype=bool)
    mask[4:9, 5:12] = True
    return td_commit_example(h5, spec, mask, _meta(blob_id, kind), ex_id,
                             grid=grid, replace_blob_id=replace)


def test_a_commit_writes_the_example(tmp_path):
    h5 = str(tmp_path / "rec_FNT.mad")
    res = _commit(h5, "rec_lbl_7", blob_id=7)
    assert res['id'] == "rec_lbl_7"
    assert td_list_ids(h5) == ["rec_lbl_7"]
    ex = td_read_example(h5, "rec_lbl_7")
    assert ex['spec'].shape == (16, 20)
    assert ex['mask'].sum() == 5 * 7


def test_grid_attrs_are_stamped_without_a_second_open(tmp_path):
    h5 = str(tmp_path / "rec_FNT.mad")
    _commit(h5, "rec_lbl_1", blob_id=1)
    with h5py.File(h5, "r") as f:
        for k in ('sample_rate', 'nperseg', 'noverlap', 'nfft',
                  'n_freq_bins', 'n_time_frames'):
            assert int(f.attrs[k]) == GRID[k]
        assert f.attrs['source_wav'] == 'rec.wav'
        assert 'created' in f.attrs and 'fnt_version' in f.attrs


def test_created_is_write_once_but_updated_moves(tmp_path):
    """Provenance must survive later accepts on the same recording."""
    h5 = str(tmp_path / "rec_FNT.mad")
    _commit(h5, "a", blob_id=1)
    with h5py.File(h5, "r") as f:
        created, updated = f.attrs['created'], f.attrs['updated']
    _commit(h5, "b", blob_id=2)
    with h5py.File(h5, "r") as f:
        assert f.attrs['created'] == created
        assert f.attrs['updated'] >= updated


def test_re_accepting_a_detection_replaces_rather_than_stacks(tmp_path):
    """The triplication guard, now inside the same write."""
    h5 = str(tmp_path / "rec_FNT.mad")
    for _ in range(4):
        _commit(h5, "rec_lbl_9", blob_id=9, replace=9)
    assert td_count(h5) == 1
    assert sum(1 for m in td_iter_meta(h5) if str(m.get('blob_id')) == '9') == 1


def test_stale_random_id_copies_are_swept(tmp_path):
    """Copies minted before ids were deterministic still get collapsed."""
    h5 = str(tmp_path / "rec_FNT.mad")
    _commit(h5, "rec_abc123", blob_id=9)
    _commit(h5, "rec_def456", blob_id=9)
    assert td_count(h5) == 2
    res = _commit(h5, "rec_lbl_9", blob_id=9, replace=9)
    assert sorted(res['dropped']) == ["rec_abc123", "rec_def456"]
    assert td_list_ids(h5) == ["rec_lbl_9"]


def test_the_sweep_only_touches_the_same_detection(tmp_path):
    h5 = str(tmp_path / "rec_FNT.mad")
    _commit(h5, "rec_lbl_1", blob_id=1, replace=1)
    _commit(h5, "rec_lbl_2", blob_id=2, replace=2)
    _commit(h5, "rec_lbl_1", blob_id=1, replace=1)
    assert sorted(td_list_ids(h5)) == ["rec_lbl_1", "rec_lbl_2"]


def test_the_sweep_never_removes_a_rejection(tmp_path):
    """A hard negative for a detection is not a duplicate of its label."""
    h5 = str(tmp_path / "rec_FNT.mad")
    _commit(h5, "rec_neg_5", blob_id=5, kind="negative")
    _commit(h5, "rec_lbl_5", blob_id=5, replace=5)
    assert sorted(td_list_ids(h5)) == ["rec_lbl_5", "rec_neg_5"]


def test_ids_before_is_the_state_prior_to_this_write(tmp_path):
    """What the undo snapshot uses as its baseline."""
    h5 = str(tmp_path / "rec_FNT.mad")
    assert _commit(h5, "a", blob_id=1)['ids_before'] == []
    assert _commit(h5, "b", blob_id=2)['ids_before'] == ["a"]
    assert sorted(_commit(h5, "c", blob_id=3)['ids_before']) == ["a", "b"]


def test_ids_before_excludes_what_the_same_call_drops(tmp_path):
    """A swept duplicate was present beforehand, so it must be listed."""
    h5 = str(tmp_path / "rec_FNT.mad")
    _commit(h5, "rec_old", blob_id=4)
    res = _commit(h5, "rec_lbl_4", blob_id=4, replace=4)
    assert "rec_old" in res['ids_before']
    assert res['dropped'] == ["rec_old"]


def test_committing_without_a_grid_leaves_attrs_alone(tmp_path):
    h5 = str(tmp_path / "rec_FNT.mad")
    _commit(h5, "a", blob_id=1, grid=None)
    with h5py.File(h5, "r") as f:
        assert 'sample_rate' not in f.attrs
    assert td_list_ids(h5) == ["a"]


def test_an_auto_generated_id_is_derived_from_the_recording(tmp_path):
    h5 = str(tmp_path / "rec_FNT.mad")
    res = _commit(h5, None)
    assert res['id'].startswith("rec_")
    assert td_list_ids(h5) == [res['id']]


def test_meta_round_trips_with_the_id_recorded_in_it(tmp_path):
    h5 = str(tmp_path / "rec_FNT.mad")
    _commit(h5, "rec_lbl_3", blob_id=3)
    m = next(iter(td_iter_meta(h5)))
    assert m['id'] == "rec_lbl_3"
    assert m['blob_id'] == 3
    assert m['class'] == 'USV'


def test_a_float_spec_is_stored_as_uint8(tmp_path):
    """Same normalization as td_save_example — the store is one format."""
    h5 = str(tmp_path / "rec_FNT.mad")
    _commit(h5, "a", blob_id=1)
    with h5py.File(h5, "r") as f:
        assert f["examples/a/spec"].dtype == np.uint8
        assert f["examples/a/mask"].dtype == np.uint8
