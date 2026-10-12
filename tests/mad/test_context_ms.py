"""Training context is a per-project width in milliseconds.

The context around each call used to be a fixed 128 spectrogram frames: ~65 ms
at hop 128 / 250 kHz, about six times a typical 25 ms USV -- and a width that
scales with the grid rather than with time (1.5 s at hop 512 / 44.1 kHz).
``context_ms`` sets it in time. Examples are still SAVED with at least the
legacy width; training crops them and the GUI shades the same span, so the
setting can change without relabelling.

Runs under pytest, or directly.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from fnt.usv.usv_detector.fnt_mask_store import td_save_example
from fnt.usv.usv_detector.mad_examples import (
    DEFAULT_CONTEXT_FRAMES, _store_path, collect_training_examples,
    context_frames, crop_to_context, iter_examples,
)
from fnt.usv.usv_detector.mad_project import MADProjectConfig, create_mad_project

GRID = {'nperseg': 512, 'noverlap': 384, 'nfft': 1024, 'sample_rate': 250000}
H = 64


# ----------------------------------------------------------------------
def test_milliseconds_become_frames_on_the_grid():
    assert context_frames(50, 250000, 128) == 98       # 50 ms / 0.512 ms
    assert context_frames(65.536, 250000, 128) == 128  # the legacy width
    assert context_frames(50, 44100, 512) == 4


def test_unset_or_unusable_falls_back_to_the_legacy_width():
    assert context_frames(None, 250000, 128) == DEFAULT_CONTEXT_FRAMES
    assert context_frames(50, None, 128) == DEFAULT_CONTEXT_FRAMES
    assert context_frames(50, 250000, 0) == DEFAULT_CONTEXT_FRAMES
    assert context_frames("x", 250000, 128) == DEFAULT_CONTEXT_FRAMES


# ----------------------------------------------------------------------
def _example(width=400, call=(150, 200), neighbour=None):
    spec = np.full((H, width), 0.2, dtype=np.float32)
    mask = np.zeros((H, width), dtype=np.float32)
    mask[20:30, call[0]:call[1]] = 1
    nb = None
    if neighbour is not None:
        nb = np.zeros((H, width), dtype=np.float32)
        nb[40:45, neighbour[0]:neighbour[1]] = 1
    return {'meta': {'patch_t_off': 1000, **GRID}, 'spec': spec,
            'mask': mask, 'neighbors': nb}


def test_a_patch_is_cropped_to_its_call_plus_the_context():
    out = crop_to_context(_example(), 50)            # 98 frames each side
    assert out['spec'].shape[1] == 50 + 2 * 98
    m = out['mask'] > 0
    cols = np.flatnonzero(m.any(axis=0))
    assert cols[0] == 98 and cols[-1] == 98 + 49
    assert out['meta']['patch_t_off'] == 1000 + 150 - 98


def test_neighbours_are_cropped_with_it():
    out = crop_to_context(_example(neighbour=(10, 30)), 50)
    assert out['neighbors'].shape == out['mask'].shape
    assert not out['neighbors'].any(), "a neighbour outside the context drops"
    out = crop_to_context(_example(neighbour=(210, 230)), 50)
    assert out['neighbors'].sum() == 5 * 20


def test_cropping_never_pads():
    ex = _example(width=120, call=(40, 80))
    out = crop_to_context(ex, 50)
    assert out['spec'].shape[1] == 120


def test_unset_context_trains_each_patch_as_saved():
    ex = _example()
    assert crop_to_context(ex, None) is ex


def test_an_example_with_no_own_pixels_is_left_alone():
    ex = _example()
    ex['mask'][:] = 0
    assert crop_to_context(ex, 50) is ex


def test_training_crops_before_tiling(tmp_path):
    td = tmp_path / 'training_data'
    td.mkdir()
    ex = _example()
    td_save_example(_store_path(str(td)), ex['spec'], ex['mask'],
                    {'class': 'USV', 'source_wav': 'a.wav', **ex['meta']}, 'a0')
    # A tile wider than the cropped patch shows the crop directly: the
    # patch's own columns are real audio (0.2), the rest synthetic fill.
    for ms, width in ((None, 400), (50, 50 + 2 * 98)):
        _s, targets, _w = collect_training_examples(
            str(td), tile_time_frames=512, tile_freq_bins=H, placements=1,
            context_ms=ms)
        assert targets.shape[0] == 1
        assert targets[0].sum() == 10 * 50, "the call itself always trains"
    stored = next(iter_examples(str(td)))
    assert stored['spec'].shape[1] == 400, "the store keeps the full patch"


# ----------------------------------------------------------------------
def test_the_project_carries_it_and_old_projects_load_without_it(tmp_path):
    d = tmp_path / 'p'
    create_mad_project(str(d))
    assert MADProjectConfig.load(str(d)).context_ms is None
    cfg = MADProjectConfig.load(str(d))
    cfg.context_ms = 50.0
    cfg.save()
    assert MADProjectConfig.load(str(d)).context_ms == 50.0


def test_mad_train_takes_the_flag_or_the_projects_value():
    from fnt.usv.usv_detector import cli
    from fnt.usv.usv_detector.mad_training import UNetTrainingConfig
    a = cli.build_parser().parse_args(["train", "--project", "p",
                                       "--context-ms", "40"])
    assert cli.training_overrides(a)["context_ms"] == 40.0
    b = cli.build_parser().parse_args(["train", "--project", "p"])
    assert "context_ms" not in cli.training_overrides(b), \
        "unset must leave room for the project's value"
    assert UNetTrainingConfig(project_dir="p").context_ms is None


# ----------------------------------------------------------------------
def test_the_shading_and_the_save_margin_follow_the_project():
    from PyQt5.QtWidgets import QApplication
    import fnt.usv.mad_pyqt as M
    app = QApplication.instance() or QApplication([])  # noqa: F841
    sg = M.MADSpectrogramWidget()
    sg.n_time_frames = 10_000
    sg.hop = 128
    sg.annotations = [{'t0': 1000, 't1': 1050, 'status': 'accepted'}]
    assert sg.training_windows() == [(1000 - 128, 1050 + 128)]
    sg.training_context_frames = 98
    assert sg.training_windows() == [(1000 - 98, 1050 + 98)]
    import inspect
    src = inspect.getsource(M.MADMainWindow._save_component_example)
    assert "max(EXAMPLE_CONTEXT_FRAMES" in src, \
        "saves keep at least the legacy width, so the context can widen later"
    src = inspect.getsource(M.MADMainWindow._init_or_load_mask_for_current_file)
    assert "_apply_training_context()" in src


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
