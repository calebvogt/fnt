"""Training a brain wired from anatomy: the learner sees exactly the network
the simulation runs, and what it learns stays inside the spec's wiring."""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fnt.abma.core import anatomy, presets                      # noqa: E402
from fnt.abma.core.senses import SENSORS                        # noqa: E402
from fnt.abma.learn.anatomy_ppo import AnatomyHyper, AnatomyTrainer, TorchAnatomy   # noqa: E402
from fnt.abma.learn.nursery import Curriculum                   # noqa: E402


@pytest.fixture(scope="module")
def spec():
    return anatomy.load_spec("c57_male")


def test_the_learner_runs_the_same_network_as_the_simulation(spec):
    b = anatomy.AnatomicalBrain(spec, seed=3)
    net = TorchAnatomy(b)
    rng = np.random.default_rng(0)
    T, B = 10, 3
    obs = rng.uniform(0, 1, (T, B, len(SENSORS)))
    r = np.zeros((B, b.n))
    acts = np.zeros((T, B, len(b.out_units)))
    for t in range(T):
        noise = np.zeros((B, b.n))
        noise[:, b.out_units] = 0.3 * rng.standard_normal((B, len(b.out_units)))
        r, x = b.step(r, obs[t], rng, noise=noise, pre=True)
        acts[t] = x[:, b.out_units]
    f = lambda a: torch.as_tensor(a, dtype=torch.float32)       # noqa: E731
    mu, val, rT = net(f(obs), torch.zeros(B, b.n), f(acts))
    assert np.abs(rT.detach().numpy() - r).max() < 0.01
    assert mu.shape == (T, B, len(b.out_units)) and val.shape == (T, B)


def test_training_keeps_the_wiring_and_the_signs(tmp_path, spec):
    cur = Curriculum(species="House mouse", tick_s=0.25, steps=64, animals=(2,), arena_m=(1.0, 1.2))
    hp = AnatomyHyper(worlds=2, workers=1, epochs=1, minibatches=2, wiring_seed=1)
    tr = AnatomyTrainer("c57_male", hp, cur, str(tmp_path), name="t")
    before = tr.net.export()
    info = tr.step()
    after = tr.net.export()
    assert info["iteration"] == 1 and info["agent_steps"] == 64 * info["animals"] and np.isfinite(info["return"])
    assert not np.array_equal(before["W"], after["W"])                      # it learned something
    b = anatomy.AnatomicalBrain(spec, seed=1)
    W = after["W"]
    assert not W[(b.S == 0) & ~b.local_mask].any()                           # nothing wired that the spec does not wire
    assert (W * b.S >= 0).all()                                              # no projection changed sign
    assert not after["Win"][~b.in_mask].any()                                # senses still enter where they did
    path = tr.save()
    tr.close()

    w, meta = anatomy.load_weights(path)
    assert meta["anatomy"] == "c57_male" and meta["agent_steps"] == info["agent_steps"] and meta["wiring_seed"] == 1
    trained = anatomy.AnatomicalBrain(spec, seed=1, weights=w)
    assert trained.trained and np.allclose(trained.W0, W)
    # and it drives an animal through the ordinary engine
    cfg = presets.open_field_test()
    cfg.brain.enabled, cfg.brain.anatomy, cfg.brain.anatomy_weights = True, "c57_male", path
    rec = anatomy.record(cfg, 10.0)
    assert np.isfinite(rec["xy"]).all()
    # resuming carries the count on
    tr2 = AnatomyTrainer("c57_male", AnatomyHyper(worlds=2, workers=1, epochs=1, minibatches=2), cur,
                         str(tmp_path), name="t2", resume=path)
    assert tr2.agent_steps == info["agent_steps"] and tr2.hp.wiring_seed == 1
    tr2.close()


def test_weights_that_break_the_spec_are_refused(spec):
    b = anatomy.AnatomicalBrain(spec, seed=0)
    w = {"W": b.W0.copy(), "Win": b.Win.copy(), "bias": b.bias.copy()}
    w["W"][b.sl["MOE"], b.sl["TURN"]] = 0.3                                  # muscles do not project to the nose
    with pytest.raises(ValueError, match="wire cells the spec does not"):
        anatomy.AnatomicalBrain(spec, seed=0, weights=w)
    w = {"W": -b.W0, "Win": b.Win, "bias": b.bias}
    with pytest.raises(ValueError):
        anatomy.AnatomicalBrain(spec, seed=0, weights=w)


def test_weights_follow_a_sensor_list_that_has_grown(tmp_path, spec):
    b = anatomy.AnatomicalBrain(spec, seed=0)
    old = [s for s in SENSORS if not s.startswith("vno_")]
    keep = [SENSORS.index(s) for s in old]
    path = anatomy.save_weights(str(tmp_path / "old.npz"),
                                {"W": b.W0, "Win": b.Win[:, keep], "bias": b.bias, "log_std": np.zeros(10)},
                                {"anatomy": "c57_male", "sensors": old, "wiring_seed": 0})
    w, _ = anatomy.load_weights(path)
    assert w["Win"].shape[1] == len(SENSORS) and not w["Win"][:, SENSORS.index("vno_contact")].any()
    assert os.path.isfile(path)
