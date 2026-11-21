"""Training upgrades made before the anosmia full run (2026-10).

* Faint-call augmentation -- the model sees each call as if from across the
  arena. The old gain jitter (x0.9-1.1 of the whole normalized tile) moves call
  and background together and cannot make a call fainter relative to noise.
* A plateau LR schedule -- every earlier run trained at a fixed 1e-3.
* A per-recording cap -- so one 906-label recording can join a 1,181-label
  project without becoming nearly half of it.
* Provenance -- each detection names the run that produced it; the checkpoint
  file is always ``weights.pt``, so its stem named nothing.
* CLI/GUI parity -- ``mad train`` and a GUI ``--train`` launch take one set
  of flags with the GUI panel's defaults.

Runs under pytest, or directly.
"""
import argparse
import os

import numpy as np
import pytest

from fnt.usv.usv_detector.mad_examples import cap_examples_per_recording
from fnt.usv.usv_detector.mad_inference import model_label
from fnt.usv.usv_detector.mad_training import (
    UNetTrainingConfig, _AugmentedTileDataset, lr_plateau_patience)

DB_MIN, DB_MAX = -100.0, -20.0


def norm(db):
    return (np.asarray(db, dtype=np.float32) - DB_MIN) / (DB_MAX - DB_MIN)


def tile_with_call(call_db=-40.0, floor_db=-80.0, shape=(64, 96)):
    """A noise floor with one bright horizontal call across columns 30-60."""
    rng = np.random.default_rng(0)
    db = floor_db + rng.normal(0, 1.0, shape)
    t = np.zeros(shape, dtype=np.float32)
    t[30:33, 30:60] = 1.0
    db[30:33, 30:60] = call_db
    return norm(db), t


def ds(**kw):
    s, t = tile_with_call()
    base = dict(p_faint=1.0, faint_db_range=(6.0, 6.0),
                faint_min_snr_db=3.0, db_min=DB_MIN, db_max=DB_MAX)
    base.update(kw)
    return _AugmentedTileDataset(s[None], t[None], np.ones_like(t)[None],
                                 **base), s, t


def to_db(s):
    return DB_MIN + s.astype(np.float64) * (DB_MAX - DB_MIN)


# ------------------------------------------------------- faint augmentation
def test_faint_aug_lowers_the_call_by_about_the_requested_db():
    d, s, t = ds()
    out = d.attenuate_calls(s, t, att_db=10.0)
    before = to_db(s)[t > 0].mean()
    after = to_db(out)[t > 0].mean()
    # 40 dB above the floor, so the floor's own contribution is negligible:
    # the call drops by almost exactly the attenuation.
    assert 9.0 < before - after < 10.5, before - after


def test_faint_aug_leaves_the_background_alone():
    """Distance makes the call quieter, not the room."""
    d, s, t = ds()
    out = d.attenuate_calls(s, t, att_db=10.0)
    np.testing.assert_array_equal(out[t == 0], s[t == 0])


def test_faint_aug_never_drives_a_call_under_the_noise():
    """Attenuating a labelled call into the noise would teach the model to
    report calls that are not there."""
    d, s, t = ds(faint_min_snr_db=3.0)
    out = d.attenuate_calls(s, t, att_db=80.0)      # asks for far too much
    call = np.median(to_db(out)[t > 0])
    floor = np.median(to_db(out)[t == 0])
    assert call - floor >= 2.5, call - floor


def test_an_already_faint_call_is_left_as_it_is():
    s, t = tile_with_call(call_db=-77.0)            # 3 dB over the floor
    d = _AugmentedTileDataset(s[None], t[None], np.ones_like(t)[None],
                              p_faint=1.0, faint_db_range=(6.0, 12.0),
                              db_min=DB_MIN, db_max=DB_MAX)
    np.testing.assert_array_equal(d.attenuate_calls(s, t), s)


def test_a_tile_without_a_call_is_untouched():
    """Hard-negative tiles have an all-zero target; nothing to make faint."""
    s, t = tile_with_call()
    d = _AugmentedTileDataset(s[None], t[None], np.ones_like(t)[None],
                              p_faint=1.0, db_min=DB_MIN, db_max=DB_MAX)
    np.testing.assert_array_equal(d.attenuate_calls(s, np.zeros_like(t)), s)


def test_the_target_is_never_changed_by_augmentation():
    pytest.importorskip("torch")
    d, s, t = ds()
    for _ in range(5):
        _, tt, _ = d[0]
        np.testing.assert_array_equal(tt.numpy()[0], t)


def test_faint_aug_is_on_by_default_and_reported():
    cfg = UNetTrainingConfig(project_dir="x")
    assert cfg.augment_faint and cfg.faint_p > 0


# ---------------------------------------------------------------- LR schedule
def test_plateau_is_the_default_schedule():
    assert UNetTrainingConfig(project_dir="x").lr_schedule == "plateau"


def test_the_rate_drops_before_early_stopping_could_end_the_run():
    """With patience 4 the run stops after 4 stagnant epochs; a plateau
    patience of 8 would never get to act."""
    cfg = UNetTrainingConfig(project_dir="x", early_stop_patience=4,
                             lr_plateau_patience=8)
    assert lr_plateau_patience(cfg) == 2


def test_the_scheduler_halves_the_rate_on_a_stalled_val_loss():
    torch = pytest.importorskip("torch")
    from fnt.usv.usv_detector.mad_training import make_lr_scheduler
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.Adam([p], lr=1e-3)
    cfg = UNetTrainingConfig(project_dir="x", early_stop_patience=40,
                             lr_plateau_patience=2)
    sch = make_lr_scheduler(opt, cfg)
    for loss in (1.0, 0.9, 0.9, 0.9, 0.9):
        sch.step(loss)
    assert opt.param_groups[0]["lr"] == pytest.approx(5e-4)


def test_no_schedule_means_a_fixed_rate():
    pytest.importorskip("torch")
    from fnt.usv.usv_detector.mad_training import make_lr_scheduler
    assert make_lr_scheduler(None, UNetTrainingConfig(
        project_dir="x", lr_schedule="none")) is None


# ------------------------------------------------------- per-recording cap
def ex(rec, i, kind="label"):
    return {"meta": {"source_wav": rec, "id": f"{rec}_{i}", "kind": kind}}


def test_the_cap_trims_only_the_recording_over_it():
    exs = ([ex("big.wav", i) for i in range(900)]
           + [ex("small.wav", i) for i in range(12)])
    rep = {}
    kept = cap_examples_per_recording(exs, 150, seed=1, report=rep)
    n = lambda r: sum(1 for e in kept if e["meta"]["source_wav"] == r)
    assert n("big.wav") == 150 and n("small.wav") == 12
    assert rep == {"big.wav": {"label": (150, 900)}}


def test_labels_and_negatives_are_capped_separately():
    """Capping them together would let a recording's rejections crowd out its
    calls, or the reverse."""
    exs = ([ex("r.wav", i) for i in range(200)]
           + [ex("r.wav", i, "negative") for i in range(200, 400)])
    kept = cap_examples_per_recording(exs, 100, seed=0)
    kinds = [e["meta"]["kind"] for e in kept]
    assert kinds.count("label") == 100 and kinds.count("negative") == 100


def test_the_cap_is_reproducible():
    exs = [ex("r.wav", i) for i in range(500)]
    a = cap_examples_per_recording(exs, 50, seed=3)
    b = cap_examples_per_recording(exs, 50, seed=3)
    assert [e["meta"]["id"] for e in a] == [e["meta"]["id"] for e in b]


def test_zero_means_no_cap():
    exs = [ex("r.wav", i) for i in range(500)]
    assert len(cap_examples_per_recording(exs, 0)) == 500


def test_the_default_cap_spares_every_v5_recording_but_the_outlier():
    """v5's densest recording holds 141 labels; the T001 file holds 906."""
    cap = UNetTrainingConfig(project_dir="x").max_examples_per_recording
    assert 141 <= cap < 906


# --------------------------------------------------------------- provenance
def test_a_run_is_named_by_its_folder():
    p = os.path.join("models", "20261006_134117_unet_n=1012", "weights.pt")
    assert model_label(p) == "20261006_134117_unet_n=1012"


def test_a_specifically_named_checkpoint_keeps_its_name():
    assert model_label(os.path.join("x", "vole_v5_final.pt")) == "vole_v5_final"


def test_no_path_no_name():
    assert model_label("") == ""


# ------------------------------------------------------------ CLI / GUI parity
def test_cli_defaults_match_the_gui_panel():
    """'Train a new model' must mean the same thing either way it starts."""
    from fnt.usv.usv_detector import cli
    a = cli.build_parser().parse_args(["train", "--project", "p"])
    kw = cli.training_overrides(a)
    assert kw["n_epochs"] == 1000
    assert kw["early_stop_patience"] == 40
    assert kw["batch_size"] == 16
    assert kw["learning_rate"] == pytest.approx(1e-3)
    assert kw["split_mode"] == "call"


def test_every_training_flag_lands_on_a_real_config_field():
    from fnt.usv.usv_detector import cli
    a = cli.build_parser().parse_args(
        ["train", "--project", "p", "--split-mode", "file",
         "--max-per-recording", "200", "--no-faint-aug", "--lr-schedule",
         "none", "--faint-db", "2", "9"])
    cfg = UNetTrainingConfig(project_dir="p", **cli.training_overrides(a))
    assert cfg.split_mode == "file"
    assert cfg.max_examples_per_recording == 200
    assert cfg.augment_faint is False
    assert cfg.lr_schedule == "none"
    assert tuple(cfg.faint_db_range) == (2.0, 9.0)


def test_a_gui_launch_overrides_only_what_was_given():
    """An explicit --split-mode call must survive even though 'call' is the
    default -- comparing against defaults would drop it."""
    from fnt.usv.usv_detector import cli
    p = argparse.ArgumentParser()
    cli.add_training_args(p)
    cli.suppress_training_defaults(p)
    assert cli.training_overrides(p.parse_args([])) == {}
    got = cli.training_overrides(p.parse_args(["--split-mode", "call"]))
    assert got == {"split_mode": "call"}


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
    sys.exit(1 if fails else 0)
