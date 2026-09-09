"""Animals in an enclosure must actually interact.

This exists because two failures got all the way to a finished run without
anything going red. Both were silent — no error, no warning, just a cohort that
had quietly stopped doing something:

  * **contests could not happen.** A fight required at least one animal to be
    ROAMING. Once animals settled successfully they rested ~84% of the time and
    only ~2% of same-sex contacts had anyone roaming, so a resident could not
    defend its patch and `dominance_<trial>.csv` had nothing in it.
  * **matings could not happen.** Social attraction only acts within
    ``perception_r``, which at 0.6 m is a quarter of a cage and effectively
    blind across 523 m². Opposite-sex pairs never came into contact at all.

Nothing in the suite asserted that either event was possible, so nothing
noticed. These tests are cheap floors, not calibration: they check that the
behaviour is *reachable*, not that any particular rate is correct.
"""
from __future__ import annotations

import numpy as np
import pytest

from fnt.abma.core.compose import design
from fnt.abma.core.presets import voleterra
from fnt.abma.core.simulation import Simulation


class EventLog:
    """Minimal stand-in for the CSV recorder — events only resolve when one
    is supplied, which is itself easy to forget and worth pinning."""

    def __init__(self):
        self.kinds: list[str] = []

    def record(self, elapsed, event, actor, target, x, y, value=1.0):
        self.kinds.append(event)

    def count(self, kind: str) -> int:
        return self.kinds.count(kind)


def _run(cfg, days=None) -> tuple[Simulation, EventLog]:
    days = cfg.days if days is None else days
    sim = Simulation(cfg, 0)
    log = EventLog()
    elapsed = 0.0
    for _ in range(int(days * 86400 / cfg.dt)):
        sim.step(elapsed, cfg.dt, events=log)
        elapsed += cfg.dt
    return sim, log


@pytest.fixture(scope="module")
def enclosure_run():
    cfg = design(preset="voleterra", males=6, females=6, days=2.0,
                 season="summer")
    return _run(cfg)


# --------------------------------------------------------------------------- #
# The floors
# --------------------------------------------------------------------------- #
def test_contests_happen_in_a_field_enclosure(enclosure_run):
    sim, log = enclosure_run
    assert log.count("fight") > 0, (
        "no contest resolved in two days — check the activity gate in "
        "_resolve_events")
    assert sim.fights_won.sum() == sim.fights_lost.sum()


def test_matings_happen_in_a_field_enclosure(enclosure_run):
    _, log = enclosure_run
    assert log.count("mating") > 0, (
        "no mating in two days — opposite-sex pairs are probably never "
        "coming into contact; check policy.perception_r against arena size")


def test_a_dominance_hierarchy_can_form(enclosure_run):
    sim, _ = enclosure_run
    contested = sim.fights_won + sim.fights_lost
    assert (contested > 0).sum() >= 2, "fewer than two animals ever contested"
    assert sim.fights_won.max() > 0 and sim.fights_lost.max() > 0


def _contest_rule(activities, steps=400):
    """Fights resolved for one touching same-sex pair held in given states.

    ``_resolve_events`` is called directly rather than through ``step``,
    because ``step`` recomputes ``activity`` from hunger and location before
    resolving events — so a state pinned beforehand is overwritten and the
    test would silently exercise whatever the engine relabelled it to.
    """
    cfg = design(preset="voleterra", males=4, females=0, days=0.1,
                 season="summer")
    sim = Simulation(cfg, 0)
    log = EventLog()
    dist = np.full((sim.n, sim.n), np.inf)
    dist[0, 1] = dist[1, 0] = 0.02            # in contact
    rec = np.zeros(sim.n)
    elapsed = 0.0
    for k in range(steps):
        sim.activity[0], sim.activity[1] = activities
        # The per-agent RNG is a pure function of (seed, uid, step, channel),
        # so the step counter has to advance or every "roll" returns the same
        # number and the contest either always or never fires.
        sim._step_k = k + 1
        sim._resolve_events(elapsed, cfg.dt, dist, rec, log)
        elapsed += cfg.dt * 60                # outrun the per-dyad cooldown
    return log.count("fight")


def test_a_settled_resident_can_be_contested_by_an_intruder():
    """The specific regression: resting resident vs a moving intruder."""
    assert _contest_rule((0, 2)) > 0, (
        "a roaming intruder never contested a settled resident")


def test_a_foraging_animal_can_also_be_contested():
    assert _contest_rule((0, 1)) > 0


def test_a_huddle_is_not_a_fight():
    """Both animals settled together is affiliation, and must never contest."""
    assert _contest_rule((0, 0)) == 0


def test_events_do_not_resolve_without_a_recorder():
    """`step(events=None)` skips events entirely — a real trap when measuring.

    An earlier parameter sweep reported zero fights for every setting because
    it never passed a recorder, and was silently measuring nothing.
    """
    cfg = design(preset="voleterra", males=6, females=6, days=0.5,
                 season="summer")
    sim = Simulation(cfg, 0)
    elapsed = 0.0
    for _ in range(int(0.5 * 86400 / cfg.dt)):
        sim.step(elapsed, cfg.dt)               # no events=
        elapsed += cfg.dt
    assert sim.fights_won.sum() == 0
    assert sim.matings.sum() == 0


# --------------------------------------------------------------------------- #
# The structure must survive the interaction
# --------------------------------------------------------------------------- #
def test_interacting_does_not_destroy_the_territorial_structure(enclosure_run):
    sim, _ = enclosure_run
    assert sim.territory_area().sum() > 10.0, "territory collapsed"
    d = np.linalg.norm(sim.P[:, None, :] - sim.P[None, :, :], axis=2)
    spacing = d[~np.eye(sim.n, dtype=bool)].mean()
    assert spacing > 3.0, f"cohort re-clumped to {spacing:.2f} m"


def test_the_cohort_stays_alive_and_fed(enclosure_run):
    sim, _ = enclosure_run
    assert sim.alive.all()
    assert sim.health.mean() > 50.0
    assert sim.energy.mean() > 25.0


def test_the_field_preset_widens_perception_but_not_the_cage_default():
    from fnt.abma.core.config import PolicyParams

    assert voleterra().policy.perception_r > PolicyParams().perception_r
    # k_social is deliberately unchanged: it buys contests at the cost of
    # territory and condition, and does nothing for opposite-sex encounters
    assert voleterra().policy.k_social == PolicyParams().k_social
