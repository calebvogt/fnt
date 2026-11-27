"""Connections that an animal's own experience changes (an anatomy spec's ``plasticity``).

The brain is fed sensor values directly; no arena. The prairie vole spec writes the odour signature the nose is
carrying into its first olfactory synapse while a mating holds oxytocin and dopamine high.
"""
import numpy as np
import pytest

from fnt.abma.core import anatomy
from fnt.abma.core.senses import SENSORS

COL = {s: i for i, s in enumerate(SENSORS)}
IDS = [COL[f"olf_id_{k}"] for k in range(12)]
A = np.zeros(12)
A[[0, 4, 9]] = 1.0
SHARES_ONE = np.zeros(12)
SHARES_ONE[[2, 4, 7]] = 1.0
SHARES_NONE = np.zeros(12)
SHARES_NONE[[1, 6, 11]] = 1.0


@pytest.fixture(scope="module")
def brain():
    return anatomy.AnatomicalBrain(anatomy.load_spec("prairie_vole"), seed=0)


def obs(n, sig=None, mating=0.0, level=0.8):
    o = np.zeros((n, len(SENSORS)))
    if sig is not None:
        o[:, IDS] = sig[None, :] * level
    o[:, COL["mating"]] = mating
    return o


def run(brain, r, fast, o, ticks, smell=None):
    n = len(r)
    senses = {"smell": np.ones(n) if smell is None else np.asarray(smell, float), "vno": np.ones(n)}
    alive = brain.cells_alive(np.arange(n), senses)
    o = brain.whole_senses(o * 1.0, senses)
    moe = [COL[c] for c in brain.spec["entry"]["MOE"] if c in COL]
    o[:, moe] *= (senses["smell"] > 0)[:, None]          # nothing reaches a nose with no epithelium
    rng = np.random.default_rng(0)
    for _ in range(ticks):
        r = brain.step(r, o, rng, alive=alive, fast=fast)
    return r


def paired(brain, mating, smell=None, n=2):
    r, fast = brain.initial_state(n), brain.initial_fast(n)
    r = run(brain, r, fast, obs(n), 60, smell)
    r = run(brain, r, fast, obs(n, A, mating), 400, smell)
    return r, fast


def match(brain, r, fast, sig, smell=None):
    f2 = [f.copy() for f in fast]
    r2 = run(brain, r.copy(), f2, obs(len(r)), 60, smell)
    r2 = run(brain, r2, f2, obs(len(r), sig), 40, smell)
    return brain.memory(r2, f2)


def test_the_vole_spec_has_a_plastic_projection_and_a_mating_reaches_the_brain(brain):
    assert [p["name"] for p in brain.plastic] == ["MOE>MOB", "MOE>MOB"]      # who the partner is; which cues came with a reward
    for area in ("PVN", "VTA"):
        assert (brain.Win[brain.sl[area], COL["mating"]] == 3.0).all()


def test_a_mating_opens_the_gate_and_nothing_else_does(brain):
    n = 2
    r = run(brain, brain.initial_state(n), None, obs(n, A), 200)
    assert brain.gate(brain.plastic[0], r, obs(n, A)).max() == 0.0
    r = run(brain, r, None, obs(n, A, 1.0), 200)
    assert brain.gate(brain.plastic[0], r, obs(n, A, 1.0)).min() > 0.05


def test_without_a_mating_nothing_is_written(brain):
    r, fast = paired(brain, mating=0.0)
    assert fast[0].max() == 0.0
    assert brain.memory(r, fast)[0].max() == 0.0


def test_a_mating_writes_the_odour_that_was_there(brain):
    r, fast = paired(brain, mating=1.0)
    stored, _ = brain.memory(r, fast)
    assert stored.min() > 0.5
    m_a = match(brain, r, fast, A)[1]
    m_one = match(brain, r, fast, SHARES_ONE)[1]
    m_none = match(brain, r, fast, SHARES_NONE)[1]
    m_nobody = match(brain, r, fast, None)[1]
    assert (m_a > m_one + 0.1).all() and (m_one > m_none).all()
    assert (m_a > 0.4).all() and (m_nobody < 0.05).all()


def test_the_budget_of_a_target_cell_is_kept(brain):
    _, fast = paired(brain, mating=1.0)
    assert fast[0].min() >= 0.0
    assert fast[0].sum(axis=2).max() <= brain.plastic[0]["cap"] + 1e-9


def test_with_no_epithelium_a_mating_writes_nothing(brain):
    r, fast = paired(brain, mating=1.0, smell=[0.0, 0.0])
    assert brain.memory(r, fast)[0].max() < 0.01


def test_losing_the_epithelium_afterwards_makes_the_memory_unreadable(brain):
    r, fast = paired(brain, mating=1.0)
    assert match(brain, r, fast, A, smell=[0.0, 0.0])[1].max() < 0.01


def test_a_spec_without_plasticity_runs_as_before():
    b = anatomy.AnatomicalBrain(anatomy.load_spec("c57_male"), seed=0)
    assert b.plastic == [] and b.initial_fast(3) == []
    r = b.step(b.initial_state(3), np.zeros((3, len(SENSORS))), np.random.default_rng(0), fast=[])
    assert r.shape == (3, b.n)


def test_the_trainers_twin_writes_the_same_weights_as_the_brain(brain):
    torch = pytest.importorskip("torch")
    from fnt.abma.learn.anatomy_ppo import TorchAnatomy

    net = TorchAnatomy(brain)
    # the twin keeps every wired weight a hair above zero, so compare it with a brain built from ITS weights
    brain = anatomy.AnatomicalBrain(brain.spec, seed=0, weights=net.export())
    n, T = 3, 80
    rng = np.random.default_rng(3)
    seq = np.stack([obs(n, A if t > 10 else None, 1.0 if 20 < t < 60 else 0.0) for t in range(T)])
    seq += 0.02 * rng.random(seq.shape)
    out = brain.out_units
    r, fast = brain.initial_state(n), brain.initial_fast(n)
    acts = np.zeros((T, n, len(out)))
    for t in range(T):                       # the brain, with a known drive on the muscles
        noise = np.zeros((n, brain.n))
        noise[:, out] = 0.3 * rng.standard_normal((n, len(out)))
        r, x = brain.step(r, seq[t], rng, noise=noise, pre=True, fast=fast)
        acts[t] = x[:, out]
    f32 = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)     # noqa: E731
    with torch.no_grad():
        _, _, r_t, fast_t = net(f32(seq), torch.zeros(n, brain.n), f32(acts), return_fast=True)
    assert fast[0].max() > 0.01
    np.testing.assert_allclose(r_t.numpy(), r, atol=2e-3)
    np.testing.assert_allclose(fast_t[0].numpy(), fast[0], atol=2e-3)


def test_methimazole_given_mid_run_kills_the_sensory_cells_and_their_written_synapses():
    from fnt.abma.core.assays import cohort
    from fnt.abma.core.config import ArenaConfig, ExperimentConfig, SocialParams, Treatment
    from fnt.abma.core.simulation import Simulation

    groups = [cohort("Prairie vole", "F", 1), cohort("Prairie vole", "M", 1)]
    groups[0].treatment = Treatment(drug="methimazole", dose=1.0, day_offset=0.002)      # about 3 minutes in
    groups[1].treatment = Treatment(drug="saline", dose=0.0, day_offset=0.002)
    cfg = ExperimentConfig(name="drug", arena=ArenaConfig(width=1.0, height=1.0, objects=[]), groups=groups,
                           dt=1.0, seed=3, days=1.0, n_trials=1, start_datetime="2026-06-15T12:00:00")
    cfg.social = SocialParams(enabled=True, bond_in_brain=True)
    cfg.brain.enabled, cfg.brain.anatomy = True, "prairie_vole"
    sim = Simulation(cfg, 0)
    pol = sim.policy
    moe = pol.brain.sl["MOE"]
    for t in range(60):
        sim.step(float(t), 1.0)
    assert (sim.smell == 1.0).all() and pol.alive[:, moe].all()
    pol.fast[0][:] = 0.01                         # as if both had something written
    for t in range(60, 400):
        sim.step(float(t), 1.0)
    treated = int(np.nonzero(sim.smell == 0.0)[0][0])
    spared = 1 - treated
    assert pol.alive[treated, moe].sum() == 0 and pol.alive[spared, moe].all()
    assert pol.fast[0][treated].max() == 0.0 and pol.fast[0][spared].max() > 0.0
    assert pol.r[treated, moe].max() == 0.0


def test_a_bonded_brain_is_stressed_by_a_stranger_close_by_and_not_by_its_partner():
    from fnt.abma.core.assays import cohort
    from fnt.abma.core.config import ArenaConfig, ExperimentConfig, SocialParams
    from fnt.abma.core.simulation import Simulation

    def stress_beside(who, smell=1.0, mated=1.0):
        cfg = ExperimentConfig(name="s", arena=ArenaConfig(width=3.0, height=1.0, objects=[]),
                               groups=[cohort("Prairie vole", "F", 1), cohort("Prairie vole", "M", 2)],
                               dt=1.0, seed=5, days=1.0, n_trials=1, start_datetime="2026-06-15T12:00:00")
        cfg.social = SocialParams(enabled=True, bond_in_brain=True, stranger_stress_h=600.0)
        cfg.brain.enabled, cfg.brain.anatomy = True, "prairie_vole"
        sim = Simulation(cfg, 0)
        sim.smell[0] = smell
        sim.agents[0].traits.smell_ability = float(smell)
        sim.mark_biology_dirty()
        far = np.array([2.8, 0.5])
        spot = {1: np.array([0.52, 0.5]), 2: far.copy()}
        for t in range(300):                       # paired with animal 1; animal 2 out of smelling range
            sim.P[0] = (0.5, 0.5)
            sim.P[1], sim.P[2] = spot[1], spot[2]
            sim.soc.mated[0] = mated
            sim.step(float(t), 1.0)
        stored = float(sim.policy.stored[0])
        sim.stress[:] = 0.0
        spot = {who: np.array([0.52, 0.5]), 3 - who: far.copy()}
        for t in range(300, 600):
            sim.P[0] = (0.5, 0.5)
            sim.P[1], sim.P[2] = spot[1], spot[2]
            sim.step(float(t), 1.0)
        return stored, float(sim.stress[0])

    stored, with_partner = stress_beside(1)
    _, with_stranger = stress_beside(2)
    assert stored > 0.5
    assert with_stranger > with_partner + 10.0
    stored, unbonded = stress_beside(2, mated=0.0)          # nothing stored: a stranger is nobody in particular
    assert stored < 0.05 and unbonded < with_stranger - 10.0
    stored, no_nose = stress_beside(2, smell=0.0)           # no epithelium: nothing stored, nothing smelled
    assert stored < 0.05 and no_nose < with_stranger - 10.0


def test_a_meal_when_hungry_strengthens_the_food_odour_and_nothing_else():
    """Incentive salience: dopamine alone writes the cues that came with the reward."""
    import copy
    spec = copy.deepcopy(anatomy.load_spec("prairie_vole"))
    assert [p.get("gate") for p in spec["plasticity"]] == [["OT", "DA"], ["DA"]]
    spec["plasticity"][1]["gate_threshold"] = 0.5          # an unsettled, untrained VTA does not reach the spec's 0.7
    b = anatomy.AnatomicalBrain(spec, seed=0)
    cue, who = b.plastic[1], b.plastic[0]
    names = spec["entry"]["MOE"]
    assert all(names[j].startswith(("olf_food", "olf_water", "olf_male", "olf_female", "olf_estrous"))
               for j in np.nonzero(cue["pre_mask"])[0])
    assert not (cue["pre_mask"] * who["pre_mask"]).any()   # the two memories are written on different cells

    def live(eating):
        n = 2
        o = np.zeros((n, len(SENSORS)))
        o[:, COL["olf_food"]], o[:, COL["hunger"]], o[:, COL["eating"]] = 1.0, 0.9, eating
        r, fast = b.initial_state(n), b.initial_fast(n)
        rng = np.random.default_rng(0)
        for _ in range(300):
            r = b.step(r, o, rng, fast=fast)
        return fast

    fast = live(1.0)
    food_cell = names.index("olf_food")
    assert fast[1][:, :, food_cell].sum() > 0.05           # the food odour's own cell was written on
    assert fast[0].max() == 0.0                            # no mating: the partner memory is untouched
    assert fast[1][:, :, np.nonzero(who["pre_mask"])[0]].max() == 0.0
    assert live(0.0)[1].max() == 0.0                       # the same odour without a meal: nothing


def test_eating_and_drinking_are_felt_in_proportion_to_the_need():
    from fnt.abma.core.assays import cohort
    from fnt.abma.core.config import ArenaConfig, ExperimentConfig, ResourceObject
    from fnt.abma.core.simulation import Simulation
    from fnt.abma.core import senses as sn

    cfg = ExperimentConfig(name="e", arena=ArenaConfig(width=1.0, height=1.0,
                                                       objects=[ResourceObject("food", 0.5, 0.5, 0.2)]),
                           groups=[cohort("Prairie vole", "F", 2)], dt=1.0, seed=1, days=1.0, n_trials=1,
                           start_datetime="2026-06-15T12:00:00")
    cfg.brain.enabled, cfg.brain.anatomy = True, "prairie_vole"
    sim = Simulation(cfg, 0)
    sim.P[:] = (0.5, 0.5)
    sim.hunger[:] = (80.0, 5.0)
    sim.policy._ensure(sim.n)
    obs, _ = sn.sense(sim, 0.0, sim.policy.memory, sim.policy.params)
    e = obs[:, COL["eating"]]
    assert obs[:, COL["at_food"]].min() == 1.0 and e[0] == pytest.approx(0.8, abs=0.02) and e[1] < 0.1
    assert obs[:, COL["drinking"]].max() == 0.0


def test_weights_raised_before_a_channel_existed_get_the_specs_entry_gain():
    spec = anatomy.load_spec("prairie_vole")
    b = anatomy.AnatomicalBrain(spec, seed=0)
    old = {"W": b.W0.copy(), "Win": b.Win.copy(), "bias": b.bias.copy()}
    old["Win"][:, COL["eating"]] = 0.0                      # as if raised before `eating` existed
    old["Win"][b.sl["VTA"], COL["mating"]] = 1.25           # a connection training has changed stays as it is
    b2 = anatomy.AnatomicalBrain(spec, seed=0, weights=old)
    assert (b2.Win[b2.sl["VTA"], COL["eating"]] == 3.0).all()
    assert (b2.Win[b2.sl["VTA"], COL["mating"]] == 1.25).all()
