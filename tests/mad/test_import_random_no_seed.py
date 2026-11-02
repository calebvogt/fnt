"""The import dialog's Random draw is random, not seeded.

The seed spinner bought reproducibility of a draw nobody reproduces: this
chooses which recordings to open and look at, and re-running an import to get
the same 70 files back was never a thing. Worse, its 12345 default made
"Random" quietly deterministic — every project sampling the same tree drew the
same recordings, which is the opposite of what the option is for.

``SampleSpec`` still accepts a seed. The CLI's ``--sample-seed`` is a scripting
surface where a repeatable corpus is reasonable, and an older project's
recorded seed has to read back.

Runs under pytest, or directly.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fnt.usv.usv_detector.mad_sampling import SampleSpec  # noqa: E402


# ------------------------------------------------------------------ the dialog
def test_the_dialog_has_no_seed_control():
    import fnt.usv.mad_pyqt as M
    src = __import__('inspect').getsource(M)
    assert "_spin_seed" not in src, "the seed spinner is still wired up"


def test_the_dialog_asks_for_an_unseeded_draw():
    """spec() must hand the sampler None, or the draw is deterministic."""
    import inspect

    from fnt.usv.mad_pyqt import MADImportSamplingDialog
    src = inspect.getsource(MADImportSamplingDialog.spec)
    assert 'SampleSpec' in src
    assert 'seed=None' in src


# ----------------------------------------------------------------- the sampler
def test_an_unseeded_spec_still_describes_itself():
    """describe() used to interpolate the seed unconditionally, which reads as
    'random (seed None)' once there is no seed."""
    s = SampleSpec(per="folder", n=10, spacing="random", seed=None)
    d = s.describe()
    assert "random" in d
    assert "None" not in d, d


def test_a_seeded_spec_still_says_which_seed():
    s = SampleSpec(per="folder", n=10, spacing="random", seed=12345)
    assert "12345" in s.describe()


def test_an_old_projects_recorded_seed_reads_back():
    """sample_history entries written before this change carry one."""
    s = SampleSpec.from_dict({"per": "folder", "n": 10, "spacing": "random",
                              "seed": 12345, "channel_mode": "spread"})
    assert s.seed == 12345
    assert s.to_dict()["seed"] == 12345


def test_an_unseeded_draw_round_trips_as_unseeded():
    s = SampleSpec(per="folder", n=10, spacing="random", seed=None)
    assert SampleSpec.from_dict(s.to_dict()).seed is None


def test_unseeded_draws_actually_differ():
    """The point of the change. Two draws from the same pool should not be the
    same 10 files; with the old default they always were.

    Sampling 10 of 200 twice collides by chance with probability well below
    1e-15, so this is not a flaky assertion.
    """
    from fnt.usv.usv_detector.mad_sampling import random_pick
    import random
    pool = [f"r{i:03d}.wav" for i in range(200)]
    a = random_pick(pool, 10, random.Random(None))
    b = random_pick(pool, 10, random.Random(None))
    assert a != b, "two unseeded draws came out identical"


def test_a_seeded_draw_still_repeats():
    from fnt.usv.usv_detector.mad_sampling import random_pick
    import random
    pool = [f"r{i:03d}.wav" for i in range(200)]
    assert (random_pick(pool, 10, random.Random(7))
            == random_pick(pool, 10, random.Random(7)))


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
    sys.exit(1 if fails else 0)
