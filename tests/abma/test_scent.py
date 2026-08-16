"""Scent marking: the field itself, and the structure that emerges from it."""
import itertools

import numpy as np
import pytest

from fnt.abma.core.config import (
    ExperimentConfig, ArenaConfig, ResourceObject, ScentParams,
    AgentGroup, Genotype, Treatment, TraitProfile,
)
from fnt.abma.core.scent import ScentField
from fnt.abma.core.simulation import Simulation

DAY = 86400.0


def _cfg(n=6, smell=1.0, ident=1.0, half_life_h=24.0, aggression=0.6,
         scent_rate=20.0, seed=1, size=2.5):
    """A dense-enough population that territories actually have to contest."""
    arena = ArenaConfig(width=size, height=size, objects=[
        ResourceObject("food", size / 2, size / 2, 0.2),
        ResourceObject("water", size / 4, size / 2, 0.2)])
    g = [AgentGroup("M", "prairie", "M", n, Genotype({}), Treatment(),
                    TraitProfile(aggression=aggression, smell_ability=smell,
                                 identity_signal=ident, scent_rate=scent_rate))]
    return ExperimentConfig(
        arena=arena, groups=g, seed=seed,
        scent=ScentParams(enabled=True, half_life_h=half_life_h))


def _range_overlap(days=2.0, dt=8.0, sample_after=0.75, grid=12, **kw):
    """Mean pairwise overlap of individual space use — LOW means territorial.

    Mean pairwise *distance* is the wrong instrument here: animals that have
    lost territorial structure roam more, which can push them further apart at
    any given instant even though their ranges overlap almost completely.
    Overlap of the realised occupancy distributions measures the thing marking
    is supposed to produce.
    """
    cfg = _cfg(**kw)
    size = cfg.arena.width
    sim = Simulation(cfg, 0)
    n = sim.n
    occ = np.zeros((n, grid, grid))
    for k in range(int(days * DAY / dt)):
        sim.step(k * dt, dt)
        if k % 20 == 0 and k * dt > sample_after * DAY:
            ci = np.clip((sim.P[:, 0] / size * grid).astype(int), 0, grid - 1)
            cj = np.clip((sim.P[:, 1] / size * grid).astype(int), 0, grid - 1)
            for a in range(n):
                occ[a, cj[a], ci[a]] += 1
    occ /= occ.sum(axis=(1, 2), keepdims=True) + 1e-12
    ov = [np.minimum(occ[i], occ[j]).sum()
          for i, j in itertools.combinations(range(n), 2)]
    return float(np.mean(ov)), sim


def _mean_overlap(seeds=(1, 2), **kw):
    return float(np.mean([_range_overlap(seed=s, **kw)[0] for s in seeds]))


# --------------------------------------------------------------------------- #
# The field
# --------------------------------------------------------------------------- #
def test_decay_follows_half_life():
    f = ScentField(2.0, 2.0, ScentParams(enabled=True, half_life_h=1.0))
    f.deposit(np.array([0]), np.array([[1.0, 1.0]]), 1.0, 1.0)
    r, c = f.cells_of(np.array([[1.0, 1.0]]))
    assert f.strength[r[0], c[0]] == pytest.approx(1.0)
    f.decay(3600.0)                       # exactly one half-life
    assert f.strength[r[0], c[0]] == pytest.approx(0.5, rel=1e-4)
    f.decay(3600.0)
    assert f.strength[r[0], c[0]] == pytest.approx(0.25, rel=1e-4)


def test_decay_clears_ownership_when_faded():
    f = ScentField(2.0, 2.0, ScentParams(enabled=True, half_life_h=0.01))
    P = np.tile([1.0, 1.0], (4, 1))
    f.deposit(np.array([3]), P, 1.0, 1.0)
    assert f.marked_fraction() > 0
    f.decay(3600.0)                       # many half-lives
    assert f.marked_fraction() == 0.0
    assert (f.owner == -1).all(), "faded marks must release the cell"


def test_mark_records_who_was_last_here():
    f = ScentField(2.0, 2.0, ScentParams(enabled=True))
    P = np.tile([1.0, 1.0], (8, 1))        # 8 agents, all on the same spot
    r, c = f.cells_of(P[:1])
    f.deposit(np.array([2]), P, 0.5, 1.0)
    assert f.owner[r[0], c[0]] == 2
    # a stronger mark from someone else takes the spot over
    f.deposit(np.array([5]), P, 0.9, 1.0)
    assert f.owner[r[0], c[0]] == 5
    # a weaker one does not
    f.deposit(np.array([7]), P, 0.2, 1.0)
    assert f.owner[r[0], c[0]] == 5


def test_anonymous_mark_carries_less_territorial_weight():
    """A MUP-KO animal still marks; the mark is just not attributable."""
    named = ScentField(2.0, 2.0, ScentParams(enabled=True))
    anon = ScentField(2.0, 2.0, ScentParams(enabled=True))
    p = np.array([[1.0, 1.0]])
    named.deposit(np.array([0]), p, 1.0, 1.0)     # full signature
    anon.deposit(np.array([0]), p, 1.0, 0.0)      # no signature
    reader = np.array([[1.15, 1.0]])
    idx = np.array([1])                            # a different animal
    _, fv_named, _, lvl_named = named.sample(reader, idx)
    _, fv_anon, _, lvl_anon = anon.sample(reader, idx)
    assert np.linalg.norm(fv_named) > np.linalg.norm(fv_anon) > 0, \
        "an unsigned mark should repel, but far less than a signed one"
    # presence is still detected equally — only attribution is lost
    assert lvl_named[0] == pytest.approx(lvl_anon[0])


def test_own_marks_attract_foreign_marks_repel():
    f = ScentField(2.0, 2.0, ScentParams(enabled=True))
    f.deposit(np.array([0]), np.array([[1.3, 1.0]]), 1.0, 1.0)
    reader = np.array([[1.0, 1.0]])
    own_vec, _, _, _ = f.sample(reader, np.array([0]))      # it is mine
    _, foreign_vec, _, _ = f.sample(reader, np.array([1]))  # it is theirs
    assert own_vec[0, 0] > 0, "should be pulled toward its own mark (+x)"
    assert foreign_vec[0, 0] < 0, "should be pushed away from a rival's (-x)"


# --------------------------------------------------------------------------- #
# Marking as a limited resource
# --------------------------------------------------------------------------- #
def test_marking_is_reserve_limited():
    slow = Simulation(_cfg(n=2, scent_rate=1.0), 0)
    fast = Simulation(_cfg(n=2, scent_rate=60.0), 0)
    for sim in (slow, fast):
        for k in range(int(0.5 * DAY / 8.0)):
            sim.step(k * 8.0, 8.0)
    assert fast.marks_made.sum() > slow.marks_made.sum() * 3, \
        "scent_rate must cap how much an animal can mark"
    assert (slow.scent_reserve >= 0).all()
    assert (fast.scent_reserve <= 1.0).all()


def test_no_marking_when_disabled():
    cfg = _cfg()
    cfg.scent = ScentParams(enabled=False)
    sim = Simulation(cfg, 0)
    assert sim.scent is None
    for k in range(300):
        sim.step(k * 8.0, 8.0)
    assert sim.territory_area().sum() == 0.0


# --------------------------------------------------------------------------- #
# Emergent structure — the reason the field exists
# --------------------------------------------------------------------------- #
def test_territory_is_emergent_not_prescribed():
    """Nothing sets a home-range size; marked areas still come out bounded."""
    _, sim = _range_overlap()
    area = sim.territory_area()
    arena = sim.cfg.arena.width * sim.cfg.arena.height
    assert (area > 0).all(), "every animal should hold some marked ground"
    assert area.sum() < arena * 0.6, \
        "marked patches should partition, not blanket, the arena"


def test_mark_persistence_structures_space():
    """The headline experiment: marks that fade fast cannot hold boundaries."""
    fleeting = _mean_overlap(half_life_h=2.0)
    persistent = _mean_overlap(half_life_h=168.0)
    assert fleeting > persistent * 3, (
        f"short-lived marks should leave ranges far more overlapping "
        f"(2 h {fleeting:.3f} vs 7 d {persistent:.3f})")


def test_identity_loss_degrades_territory():
    """MUP-KO: an unsigned mark can't hold a boundary a signed one can."""
    wt = _mean_overlap(ident=1.0)
    ko = _mean_overlap(ident=0.0)
    assert ko > wt, (f"unsigned marks should give more overlapping ranges "
                     f"(KO {ko:.3f} vs WT {wt:.3f})")


def test_anosmia_expands_overlap_without_dispersing():
    """Methimazole phenotype: keeps a home area (memory) but stops excluding."""
    intact_ov, intact = _range_overlap(smell=1.0)
    anosmic_ov, anosmic = _range_overlap(smell=0.0)
    assert anosmic_ov > intact_ov * 5, (
        f"without a nose, ranges should collapse into each other "
        f"(anosmic {anosmic_ov:.3f} vs intact {intact_ov:.3f})")
    assert anosmic.territory_area().mean() > intact.territory_area().mean(), \
        "anosmic animals should sprawl over more ground"
    # ...but they must not simply scatter: spatial memory still anchors them
    spread = np.linalg.norm(anosmic.P - anosmic.P.mean(0), axis=1).mean()
    assert spread < anosmic.cfg.arena.width * 0.75, \
        "anosmic animals should still be anchored, not dispersed"


# --------------------------------------------------------------------------- #
# Config plumbing
# --------------------------------------------------------------------------- #
def test_scent_params_roundtrip():
    cfg = _cfg(half_life_h=6.0)
    cfg.scent.anonymous_weight = 0.4
    cfg.scent.cell_size = 0.05
    d = cfg.to_dict()
    cfg2 = ExperimentConfig.from_dict(d)
    assert cfg2.to_dict() == d
    assert cfg2.scent.half_life_h == 6.0
    assert cfg2.scent.anonymous_weight == 0.4


def test_configs_without_scent_key_stay_legacy():
    """Old saved projects must reproduce exactly, not silently gain marking."""
    cfg = _cfg()
    d = cfg.to_dict()
    d.pop("scent")
    assert ExperimentConfig.from_dict(d).scent.enabled is False
