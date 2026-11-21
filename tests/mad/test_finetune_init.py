"""Fine-tuning from a previous model (``--init-from``).

A human-in-the-loop round adds a few recordings to a model that already works.
Retraining from ImageNet each round spends hours relearning what the last run
knew, so a run can start from a checkpoint instead. Two things must hold:

* the parent's weights are actually what training starts from, and a parent
  trained on a different grid or intensity scale is refused, not silently used;
* with a file-level split the parent's held-out recordings stay held out. A
  fresh seeded split over the grown label set would hand some of the parent's
  TRAINING recordings to validation, and the child would be scored on data its
  weights had already fit.
"""
import json
import os

import numpy as np
import pytest

from fnt.usv.usv_detector.fnt_mask_store import td_save_example
from fnt.usv.usv_detector.mad_examples import _store_path
from fnt.usv.usv_detector.mad_training import (
    UNetTrainingConfig, _split_tiles, load_init_checkpoint, train_unet,
)

T = 64      # tile edge: tiny, but divisible by 32 for the U-Net


def _groups(spec):
    """[(recording, call)] per tile from {recording: n_tiles}."""
    out = []
    for rec, n in spec.items():
        out += [(rec, f"{rec}_c{i}") for i in range(n)]
    return out


# ----------------------------------------------------------------------
# The split
# ----------------------------------------------------------------------
def test_inherited_recordings_are_exactly_the_validation_set():
    groups = _groups({'a.wav': 4, 'b.wav': 4, 'c.wav': 4, 'new.wav': 4})
    pos = np.ones(len(groups), dtype=bool)
    s = _split_tiles(groups, len(groups), 0.2, positives=pos, mode='file',
                     fixed_val_groups=['b.wav'])
    assert s['inherited'] is True
    assert s['split_level'] == 'file' and s['val_held_out'] is True
    assert s['val_groups'] == ['b.wav']
    assert {groups[i][0] for i in s['val_idx']} == {'b.wav'}
    assert 'new.wav' in {groups[i][0] for i in s['train_idx']}, \
        "a new recording is training data, never validation"


def test_a_held_out_recording_no_longer_in_the_store_is_just_absent():
    groups = _groups({'a.wav': 4, 'b.wav': 4})
    pos = np.ones(len(groups), dtype=bool)
    s = _split_tiles(groups, len(groups), 0.2, positives=pos, mode='file',
                     fixed_val_groups=['b.wav', 'gone.wav'])
    assert s['val_groups'] == ['b.wav']
    assert s['n_val_groups'] == 1


def test_an_unusable_inherited_split_falls_back_and_says_so():
    """Inherited recordings with no calls would score on background only."""
    groups = _groups({'a.wav': 4, 'b.wav': 4, 'c.wav': 4})
    pos = np.array([g[0] != 'b.wav' for g in groups])
    s = _split_tiles(groups, len(groups), 0.34, positives=pos, mode='file',
                     fixed_val_groups=['b.wav'])
    assert not s.get('inherited')
    assert 'inherited' in s['degraded_from']


def test_a_call_level_run_ignores_the_parents_recordings():
    groups = _groups({'a.wav': 4, 'b.wav': 4})
    pos = np.ones(len(groups), dtype=bool)
    s = _split_tiles(groups, len(groups), 0.25, positives=pos, mode='call',
                     fixed_val_groups=['b.wav'])
    assert not s.get('inherited')
    assert s['split_level'] == 'call'


# ----------------------------------------------------------------------
# Training end to end, tiny and on CPU
# ----------------------------------------------------------------------
def _store(td_dir, recordings):
    rng = np.random.default_rng(0)
    for rec in recordings:
        for k in range(3):
            spec = rng.random((T, T)).astype(np.float32) * 0.2
            mask = np.zeros((T, T), dtype=np.float32)
            mask[20:30, 10 + 5 * k:30 + 5 * k] = 1.0
            spec[mask > 0] = 0.9
            td_save_example(
                _store_path(td_dir), spec, mask,
                {'class': 'USV', 'source_wav': rec, 'patch_t_off': 1000 * k,
                 'patch_f_off': 0}, f"{os.path.splitext(rec)[0]}_c{k}")


def _cfg(tmp_path, **kw):
    td = tmp_path / 'training_data'
    td.mkdir(exist_ok=True)
    base = dict(project_dir=str(tmp_path), training_data_dir=str(td),
                encoder_weights=None, n_epochs=1, batch_size=4,
                device='cpu', tile_time_frames=T, tile_freq_bins=T,
                split_mode='file', val_fraction=0.25, tile_placements=1,
                augment=False, max_examples_per_recording=0)
    base.update(kw)
    return UNetTrainingConfig(**base)


def test_a_fine_tune_starts_from_the_parent_and_keeps_its_held_out_recordings(tmp_path):
    cfg = _cfg(tmp_path, run_name='parent')
    _store(cfg.training_data_dir, ['a.wav', 'b.wav', 'c.wav', 'd.wav'])
    parent = train_unet(cfg)
    parent_val = parent['val_groups']
    assert parent['split_level'] == 'file' and parent_val

    # The label set grows by a recording, as a QC round does.
    _store(cfg.training_data_dir, ['e.wav'])
    child_cfg = _cfg(tmp_path, run_name='child', init_from=parent['run_dir'])
    child = train_unet(child_cfg)

    assert child['split_inherited'] is True
    assert child['val_groups'] == parent_val
    import torch
    ck = torch.load(child['model_path'], map_location='cpu', weights_only=False)
    assert os.path.normcase(ck['init_from']) == os.path.normcase(
        os.path.join(parent['run_dir'], 'weights.pt'))
    saved = json.load(open(os.path.join(child['run_dir'],
                                        'training_summary.json')))
    assert saved['config']['init_from'] == parent['run_dir']


def test_the_parents_weights_are_loaded(tmp_path):
    """Load succeeds key-for-key into the same architecture: a silently
    partial load would be a fresh model wearing the parent's name."""
    cfg = _cfg(tmp_path, run_name='parent')
    _store(cfg.training_data_dir, ['a.wav', 'b.wav', 'c.wav', 'd.wav'])
    parent = train_unet(cfg)
    ck = load_init_checkpoint(_cfg(tmp_path, init_from=parent['model_path']))
    from fnt.usv.usv_detector.mad_training import build_model
    m = build_model('unet', 'resnet18', None, in_channels=1, classes=1)
    res = m.load_state_dict(ck['state_dict'], strict=True)
    assert not res.missing_keys and not res.unexpected_keys


def test_a_parent_on_another_grid_is_refused(tmp_path):
    cfg = _cfg(tmp_path, run_name='parent')
    _store(cfg.training_data_dir, ['a.wav', 'b.wav', 'c.wav', 'd.wav'])
    parent = train_unet(cfg)
    with pytest.raises(RuntimeError, match="noverlap"):
        load_init_checkpoint(_cfg(tmp_path, init_from=parent['run_dir'],
                                  noverlap=256))


def test_a_missing_parent_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="no checkpoint"):
        load_init_checkpoint(_cfg(tmp_path, init_from=str(tmp_path / 'nope')))


def test_the_flag_reaches_the_config():
    from fnt.usv.usv_detector import cli
    a = cli.build_parser().parse_args(
        ["train", "--project", "p", "--init-from", "models/x"])
    cfg = UNetTrainingConfig(project_dir="p", **cli.training_overrides(a))
    assert cfg.init_from == "models/x"
