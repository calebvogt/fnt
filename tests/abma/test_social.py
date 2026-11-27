"""Rest, warmth and attachment (SocialParams, core/social.py).

These check the body, not behaviour: what each need does when an animal is
held in a given situation, and that smell enters only where it should.
"""
import numpy as np
import pytest

from fnt.abma.core.config import (ArenaConfig, ExperimentConfig, Hut, ResourceObject, ResourceZone, ScentParams,
                                  SocialParams)
from fnt.abma.core.compose import cohort
from fnt.abma.core.neural import discomfort
from fnt.abma.core.physiology import preset_params
from fnt.abma.core.policy import Policy
from fnt.abma.core.senses import SENSORS, PlaceMemory, SenseParams, index, sense
from fnt.abma.core.simulation import Simulation


class Still(Policy):
    """Every animal stays where it is put, or walks at a set fraction of its pace."""

    def __init__(self, speed=0.0):
        self.speed = speed
        self.memory = None

    def decide(self, sim, elapsed_s, dt):
        if self.memory is None:
            self.memory = PlaceMemory(sim.n)
        self.last_obs, perc = sense(sim, elapsed_s, self.memory, SenseParams())
        perc.update(speed_frac=np.full(sim.n, float(self.speed)), resting=np.full(sim.n, self.speed < 0.12),
                    drives={}, want_chew=np.zeros(sim.n, bool))
        return np.stack([np.cos(sim.H), np.sin(sim.H)], axis=1), perc


class Sink:
    def record(self, *a, **k):
        pass


def world(males=1, females=1, social=True, boxes=False, huts=False, smell=None, side=2.0, **sp):
    arena = ArenaConfig(width=side, height=side, objects=[ResourceObject("food", 0.2, 0.2, 0.1)])
    if boxes:
        arena.resource_zones = [ResourceZone(x=1.5, y=1.5, w=0.5, d=0.5, entrance="W", label="box")]
    if huts:
        arena.huts = [Hut(kind="dome", x=0.5, y=1.5, w=0.127)]
    groups = [g for g in (cohort("Prairie vole", "M", males) if males else None,
                          cohort("Prairie vole", "F", females) if females else None) if g is not None]
    for g in groups:
        g.dists = {}
        if smell is not None:
            g.traits.smell_ability = smell
    # animals are held in place here, so a coarse step loses nothing and the tests run in seconds
    cfg = ExperimentConfig(name="t", arena=arena, groups=groups, dt=10.0, days=1.0, n_trials=1, seed=3,
                           start_datetime="2026-06-01T12:00:00")
    cfg.physiology = preset_params("Standard")
    cfg.scent = ScentParams(enabled=False)
    cfg.social = SocialParams(enabled=social, **sp)
    return cfg


def run(sim, seconds, place=None, events=None):
    dt = sim.cfg.dt
    for k in range(int(seconds / dt)):
        if place is not None:
            sim.P[:] = place
        sim.step(k * dt, dt, events=events)
    if place is not None:
        sim.P[:] = place


APART = np.array([[0.5, 0.5], [1.5, 0.6]])
TOGETHER = np.array([[1.0, 1.0], [1.05, 1.0]])


def test_off_by_default_and_nothing_exists():
    cfg = world(social=False)
    assert ExperimentConfig().social.enabled is False
    sim = Simulation(cfg, 0, policy=Still())
    assert sim.soc is None
    run(sim, 60)
    obs = sim.policy.last_obs
    for name in ("olf_partner", "olf_stranger", "tired", "cold", "lonely", "touch", "door", "in_box"):
        assert not obs[:, index(name)].any()


def test_new_sensors_are_appended_after_the_old_ones():
    assert SENSORS.index("olf_partner") > SENSORS.index("vno_mup_11")
    assert len(set(SENSORS)) == len(SENSORS)


def test_tired_rises_when_up_and_falls_at_rest():
    sim = Simulation(world(), 0, policy=Still(speed=0.5))
    run(sim, 3600, place=APART)
    up = sim.soc.tired.copy()
    p = sim.cfg.social
    assert np.allclose(up, p.tired_rise_h + 0.5 * p.tired_move_h, atol=1.0)
    sim.policy.speed = 0.0
    run(sim, 3600, place=APART)
    assert np.allclose(up - sim.soc.tired, p.tired_fall_h, atol=1.0)


def test_cold_alone_shelter_and_huddle():
    p = SocialParams()
    alone = Simulation(world(), 0, policy=Still())
    run(alone, 3 * 3600, place=APART)
    assert np.allclose(alone.soc.cold, p.cold_alone, atol=1.0)

    huddle = Simulation(world(), 0, policy=Still())
    run(huddle, 3 * 3600, place=TOGETHER)
    assert np.allclose(huddle.soc.cold, p.cold_alone * (1 - p.huddle_warmth), atol=1.0)

    box = Simulation(world(boxes=True), 0, policy=Still())
    run(box, 3 * 3600, place=np.array([[1.5, 1.5], [0.5, 0.5]]))
    assert box.soc.cold[0] == pytest.approx(p.cold_alone * (1 - p.shelter_warmth), abs=1.0)
    assert box.soc.cold[1] == pytest.approx(p.cold_alone, abs=1.0)

    moving = Simulation(world(), 0, policy=Still(speed=1.0))
    run(moving, 3 * 3600, place=APART)
    assert (moving.soc.cold < 1.0).all()


def test_warmth_does_not_need_a_nose():
    a = Simulation(world(smell=0.0), 0, policy=Still())
    b = Simulation(world(smell=1.0), 0, policy=Still())
    for sim in (a, b):
        run(sim, 3600, place=TOGETHER)
    assert np.allclose(a.soc.cold, b.soc.cold)


def test_attachment_grows_in_contact_only_with_a_nose():
    intact = Simulation(world(smell=1.0), 0, policy=Still())
    run(intact, 6 * 3600, place=TOGETHER)
    assert (intact.soc.bond() > 0.3).all()
    assert list(intact.soc.partner()) == [1, 0]

    anosmic = Simulation(world(smell=0.0), 0, policy=Still())
    run(anosmic, 6 * 3600, place=TOGETHER)
    assert (anosmic.soc.A == 0).all()

    apart = Simulation(world(smell=1.0), 0, policy=Still())
    run(apart, 6 * 3600, place=APART)
    assert (apart.soc.A == 0).all()


def test_no_identity_signal_no_attachment_to_that_animal():
    cfg = world()
    cfg.groups[0].traits.identity_signal = 0.0          # the male's odour says nothing of who he is
    sim = Simulation(cfg, 0, policy=Still())
    run(sim, 6 * 3600, place=TOGETHER)
    male = int(np.nonzero(sim.sex_m > 0.5)[0][0])
    female = 1 - male
    assert sim.soc.A[female, male] == 0.0
    assert sim.soc.A[male, female] > 0.3


def test_which_nose_an_attachment_needs_is_a_setting():
    def bond(smell, vno, **sp):
        cfg = world(smell=smell, **sp)
        for g in cfg.groups:
            g.traits.vno_ability = vno
        sim = Simulation(cfg, 0, policy=Still())
        run(sim, 6 * 3600, place=TOGETHER)
        return sim.soc.bond().min()

    # main olfaction only (the default): the vomeronasal organ is not needed
    assert bond(1.0, 0.0) > 0.3 and bond(0.0, 1.0) == 0.0
    # the vomeronasal organ only: methimazole would leave it standing
    vno_only = dict(bond_needs_moe=False, bond_needs_vno=True)
    assert bond(0.0, 1.0, **vno_only) > 0.3 and bond(1.0, 0.0, **vno_only) == 0.0
    # both
    both = dict(bond_needs_moe=True, bond_needs_vno=True)
    assert bond(1.0, 1.0, **both) > 0.3 and bond(0.0, 1.0, **both) == 0.0 and bond(1.0, 0.0, **both) == 0.0


def test_an_attachment_read_by_the_vomeronasal_organ_works_in_contact_without_smell():
    def lonely_after(read_by_vno):
        sim = Simulation(world(smell=0.0, read_by_vno=read_by_vno), 0, policy=Still())
        run(sim, 10, place=TOGETHER)
        sim.soc.A[0, 1] = sim.soc.A[1, 0] = 1.0
        sim.soc.lonely[:] = 50.0
        run(sim, 1800, place=TOGETHER)
        return sim.soc.lonely.max(), sim.policy.last_obs[:, index("vno_partner")].min()

    lonely, signal = lonely_after(True)
    assert lonely < 1.0 and signal > 0.9
    lonely, signal = lonely_after(False)
    assert lonely > 50.0 and signal == 0.0


def test_an_animal_without_a_vomeronasal_organ_reads_nothing_with_it():
    cfg = world()
    cfg.scent = ScentParams(enabled=True)
    cfg.groups[0].traits.vno_ability = 0.0
    sim = Simulation(cfg, 0, policy=Still())
    sim.H[:] = 0.0
    place = np.array([[1.0, 1.0], [1.02, 1.0]])
    run(sim, 20, place=place)
    obs = sim.policy.last_obs
    without = int(np.nonzero(sim.vno == 0.0)[0][0])
    assert obs[without, index("vno_contact")] == 0.0


def test_a_mating_steps_attachment_up():
    sim = Simulation(world(), 0, policy=Still())
    run(sim, 10, place=APART)
    sim.soc.on_mating(sim, 0, 1)
    assert sim.soc.A[0, 1] == pytest.approx(sim.cfg.social.mate_bond)
    assert sim.soc.A[1, 0] == pytest.approx(sim.cfg.social.mate_bond)


def test_an_existing_attachment_holds_back_a_new_one():
    sim = Simulation(world(males=2, females=1), 0, policy=Still())
    run(sim, 10, place=np.array([[0.3, 0.3], [1.0, 1.0], [1.7, 1.7]]))
    f = int(np.nonzero(sim.sex_m <= 0.5)[0][0])
    m1, m2 = (int(i) for i in np.nonzero(sim.sex_m > 0.5)[0])
    sim.soc.A[f, m1] = 1.0
    sim.soc.on_mating(sim, f, m2)
    assert sim.soc.A[f, m2] == 0.0                      # no room left
    sim.soc.A[f, m1] = 0.5
    sim.soc.on_mating(sim, f, m2)
    assert sim.soc.A[f, m2] == pytest.approx(0.5 * sim.cfg.social.mate_bond)


def _bonded(smell=1.0, place=APART):
    sim = Simulation(world(smell=smell), 0, policy=Still())
    run(sim, 10, place=place)
    sim.soc.A[0, 1] = sim.soc.A[1, 0] = 1.0
    return sim


def test_separation_distress_rises_apart_and_falls_together():
    sim = _bonded()
    run(sim, 2 * 3600, place=APART)
    lonely = sim.soc.lonely.copy()
    assert np.allclose(lonely, 2 * sim.cfg.social.lonely_rise_h, rtol=0.1)
    run(sim, 1800, place=TOGETHER)
    assert (sim.soc.lonely < 1.0).all()


def test_an_unattached_animal_is_never_lonely():
    sim = Simulation(world(), 0, policy=Still())
    run(sim, 2 * 3600, place=APART)
    assert (sim.soc.lonely == 0).all()


def test_an_attached_animal_without_a_nose_is_not_relieved_by_its_partner():
    sim = _bonded(smell=0.0, place=TOGETHER)
    run(sim, 2 * 3600, place=TOGETHER)
    assert (sim.soc.lonely > 20).all()


def test_stranger_at_close_quarters_stresses_only_the_attached_and_only_with_a_nose():
    def stress_after(smell, attached):
        sim = Simulation(world(males=2, females=1, smell=smell), 0, policy=Still())
        place = np.array([[1.0, 1.0], [1.08, 1.0], [0.2, 1.8]])
        run(sim, 10, place=place)
        if attached:                                     # animal 0 is attached to the far one
            sim.soc.A[0, 2] = sim.soc.A[2, 0] = 1.0
        s0 = sim.stress[0]
        sim.cfg.social.bond_rate_h = 0.0                 # hold the attachments where they are
        run(sim, 1800, place=place)
        return sim.stress[0] - s0

    assert stress_after(1.0, True) > 20
    assert stress_after(1.0, False) <= 0.0
    assert stress_after(0.0, True) <= 0.0


def test_partner_and_stranger_odours_reach_the_senses_scaled_by_smell():
    for smell in (1.0, 0.0):
        sim = Simulation(world(males=2, females=1, smell=smell), 0, policy=Still())
        place = np.array([[1.0, 1.0], [1.3, 1.0], [1.0, 1.3]])
        run(sim, 10, place=place)
        sim.soc.A[0, 1] = 1.0
        sim.H[:] = 0.0                                   # animal 0 faces +x: its partner is ahead, the other to its left
        run(sim, 10, place=place)
        obs = sim.policy.last_obs[0]
        if smell == 0.0:
            assert obs[index("olf_partner")] == 0.0 and obs[index("olf_stranger")] == 0.0
        else:
            assert obs[index("olf_partner")] > 0.3
            assert obs[index("olf_partner_fwd")] > 0.5 and abs(obs[index("olf_partner_lat")]) < 0.2
            assert obs[index("olf_stranger_lat")] > 0.5


def test_touch_and_in_box_and_the_sight_of_a_doorway_need_no_nose():
    sim = Simulation(world(boxes=True, smell=0.0), 0, policy=Still())
    place = np.array([[0.9, 1.5], [0.95, 1.5]])           # outside the box, west of its doorway at x = 1.25
    sim.H[:] = 0.0
    run(sim, 20, place=place)
    obs = sim.policy.last_obs
    assert (obs[:, index("touch")] > 0).all()
    assert (obs[:, index("in_box")] == 0).all()
    assert (obs[:, index("door")] > 0.5).all()
    assert (obs[:, index("door_fwd")] > 0.9).all()
    run(sim, 20, place=np.array([[1.5, 1.5], [0.4, 0.4]]))
    assert sim.policy.last_obs[0, index("in_box")] == 1.0


def test_the_needs_enter_discomfort():
    sim = Simulation(world(), 0, policy=Still())
    run(sim, 10, place=APART)
    for bar in (sim.soc.tired, sim.soc.cold, sim.soc.lonely):
        bar[:] = 0.0
    base = discomfort(sim).copy()
    sim.soc.tired[:] = 100.0
    p = sim.cfg.social
    assert np.allclose(discomfort(sim) - base, p.w_tired, atol=1e-6)
    sim.soc.cold[:] = 100.0
    sim.soc.lonely[:] = 100.0
    assert np.allclose(discomfort(sim) - base, p.w_tired + p.w_cold + p.w_lonely, atol=1e-6)


def test_cold_costs_energy():
    warm = Simulation(world(cold_alone=0.0), 0, policy=Still())
    cold = Simulation(world(), 0, policy=Still())
    for sim in (warm, cold):
        sim.energy[:] = 80.0
        run(sim, 4 * 3600, place=APART)
    assert (cold.energy < warm.energy - 0.5).all()


def test_config_round_trip_keeps_social():
    cfg = world(cold_alone=55.0)
    back = ExperimentConfig.from_dict(cfg.to_dict())
    assert back.social.enabled and back.social.cold_alone == 55.0


def test_roster_growth_keeps_the_attachments():
    sim = _bonded()
    sim._add_agents(cohort("Prairie vole", "F", 1), 20.0, None)
    run(sim, 10)
    assert sim.soc.A.shape == (3, 3) and sim.soc.A[0, 1] > 0.9 and len(sim.soc.tired) == 3
