"""Scoped reads and writes of prediction crops.

Reviewing changes one detection at a time, but the store only offered
whole-group operations: capturing an undo snapshot read every crop in the file
and restoring one rewrote every crop. On a real recording with 1,239 crops that
was 1.55 s of reading per Delete keystroke, and the cost scaled with the file
rather than with the edit.

The risk in narrowing them is silent collateral damage — a scoped write that
quietly drops the crops it did not name, or loses the attributes (score, model
provenance) that make a stored detection self-describing. Those are what these
tests pin.
"""
import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from fnt.usv.usv_detector.fnt_mask_store import (  # noqa: E402
    delete_pred_mask, list_pred_ids, read_all_pred_masks, read_pred_masks,
    write_pred_mask_subset, write_pred_masks,
)


@pytest.fixture
def store(tmp_path):
    """A file holding 20 crops, each with the attrs inference records."""
    h5 = str(tmp_path / "rec_FNT.mad")
    crops = []
    for i in range(20):
        rng = np.random.default_rng(i)
        crops.append({
            'blob_id': i,
            'mask': rng.random((6 + i % 4, 9 + i % 5)) > 0.5,
            'f_off': 100 + i, 't_off': 1000 * i,
            'score': round(0.5 + i / 100, 4),
            'model_name': 'weights', 'threshold': 0.6, 'min_blob_pixels': 100,
        })
    write_pred_masks(h5, crops)
    return h5, {str(c['blob_id']): c for c in crops}


def test_reading_a_subset_returns_only_what_was_asked_for(store):
    h5, _ = store
    got = read_pred_masks(h5, ['3', '7'])
    assert sorted(got) == ['3', '7']


def test_a_subset_read_matches_the_whole_read(store):
    h5, _ = store
    every = read_all_pred_masks(h5)
    for key, rec in read_pred_masks(h5, ['0', '11', '19']).items():
        assert np.array_equal(rec['mask'], every[key]['mask'])
        assert rec['f_off'] == every[key]['f_off']
        assert rec['t_off'] == every[key]['t_off']
        assert rec['score'] == every[key]['score']


def test_reading_missing_ids_skips_them_rather_than_raising(store):
    """A crop can legitimately be gone already, mid-operation."""
    h5, _ = store
    assert sorted(read_pred_masks(h5, ['4', 'nope', '999'])) == ['4']
    assert read_pred_masks(h5, []) == {}


def test_reading_from_a_file_that_does_not_exist_is_empty(tmp_path):
    assert read_pred_masks(str(tmp_path / "absent.mad"), ['1']) == {}


def test_a_scoped_write_leaves_every_other_crop_untouched(store):
    """The whole point: one edit must not rewrite the file."""
    h5, orig = store
    before = read_all_pred_masks(h5)
    new_mask = np.ones((3, 3), dtype=bool)
    write_pred_mask_subset(h5, [{'blob_id': 5, 'mask': new_mask,
                                 'f_off': 42, 't_off': 4242}])
    after = read_all_pred_masks(h5)
    assert sorted(after) == sorted(before)
    for key in before:
        if key == '5':
            continue
        assert np.array_equal(after[key]['mask'], before[key]['mask'])
        assert after[key]['score'] == before[key]['score']
    assert np.array_equal(after['5']['mask'], new_mask)
    assert after['5']['f_off'] == 42 and after['5']['t_off'] == 4242


def test_a_scoped_write_can_add_a_crop_that_was_not_there(store):
    h5, _ = store
    write_pred_mask_subset(h5, [{'blob_id': 99, 'mask': np.ones((2, 2), bool),
                                 'f_off': 1, 't_off': 2}])
    assert '99' in list_pred_ids(h5)
    assert len(list_pred_ids(h5)) == 21


def test_delete_then_scoped_restore_round_trips_exactly(store):
    """The undo path, end to end — including the attributes."""
    h5, _ = store
    ids = ['2', '9', '14']
    before = read_pred_masks(h5, ids)
    for i in ids:
        delete_pred_mask(h5, i)
    assert not set(ids) & set(list_pred_ids(h5))

    write_pred_mask_subset(h5, [
        {'blob_id': k, 'mask': c['mask'], 'f_off': c['f_off'],
         't_off': c['t_off'],
         **{a: v for a, v in c.items() if a not in ('mask', 'f_off', 't_off')}}
        for k, c in before.items()])

    after = read_pred_masks(h5, ids)
    for k in ids:
        assert np.array_equal(after[k]['mask'], before[k]['mask'])
        assert after[k]['f_off'] == before[k]['f_off']
        assert after[k]['t_off'] == before[k]['t_off']
        assert after[k]['score'] == before[k]['score']
        assert after[k]['model_name'] == before[k]['model_name']
    assert len(list_pred_ids(h5)) == 20


def test_the_cached_count_stays_in_step(store):
    """File lists read n_pred_blobs instead of opening every crop."""
    h5, _ = store
    with h5py.File(h5, "r") as f:
        assert int(f.attrs['n_pred_blobs']) == 20
    write_pred_mask_subset(h5, [{'blob_id': 77, 'mask': np.ones((2, 2), bool),
                                 'f_off': 0, 't_off': 0}])
    with h5py.File(h5, "r") as f:
        assert int(f.attrs['n_pred_blobs']) == 21


def test_writing_nothing_is_a_no_op(store):
    h5, _ = store
    before = read_all_pred_masks(h5)
    write_pred_mask_subset(h5, [])
    assert sorted(read_all_pred_masks(h5)) == sorted(before)


def test_a_scoped_write_creates_the_group_when_absent(tmp_path):
    h5 = str(tmp_path / "fresh_FNT.mad")
    write_pred_mask_subset(h5, [{'blob_id': 1, 'mask': np.ones((2, 2), bool),
                                 'f_off': 0, 't_off': 0}])
    assert list_pred_ids(h5) == ['1']


def test_masks_round_trip_as_boolean(store):
    """Stored as uint8, read back as bool — a mask is never a probability."""
    h5, _ = store
    rec = read_pred_masks(h5, ['1'])['1']
    assert rec['mask'].dtype == np.bool_


def test_td_delete_reports_whether_it_removed_anything(tmp_path):
    """Deleting the same example twice is one deletion, not two.

    The confirmed-mask gallery counted its own calls, so three clicks on one
    tile logged three deletions when only the first removed anything — and the
    user reasonably read that as three different masks having gone.
    """
    from fnt.usv.usv_detector.fnt_mask_store import td_delete, td_save_example

    h5 = str(tmp_path / "rec_FNT.mad")
    td_save_example(h5, np.zeros((4, 4), np.float32),
                    np.ones((4, 4), np.uint8),
                    {'source_wav': 'rec.wav', 'class': 'USV'}, "only")
    assert td_delete(h5, "only") is True
    assert td_delete(h5, "only") is False
    assert td_delete(h5, "never_existed") is False


def test_td_delete_on_a_missing_file_is_false_not_an_error(tmp_path):
    from fnt.usv.usv_detector.fnt_mask_store import td_delete
    assert td_delete(str(tmp_path / "absent_FNT.mad"), "x") is False
