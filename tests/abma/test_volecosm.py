"""The VoleCosm preset, and the three things the engine learned for it:
resource zones with more than one doorway, zones that hold water, and a light
cycle that is not 12:12."""
import os

import numpy as np
import pytest

from fnt.abma.core import movie
from fnt.abma.core.brain import Brain
from fnt.abma.core.compose import design
from fnt.abma.core.config import (ExperimentConfig, ResourceZone, zone_doors,
                                  zone_sides)
from fnt.abma.core.neural import NeuralPolicy
from fnt.abma.core.presets import find_preset, volecosm
from fnt.abma.core.simulation import Simulation
from fnt.abma.run_headless import main as headless

W, H, Z = 2.81, 2.61, 0.51


# --------------------------------------------------------------------------- #
# The layout
# --------------------------------------------------------------------------- #
def test_the_floor_and_the_four_boxes_are_the_measured_ones():
    a = volecosm().arena
    assert (a.width, a.height) == (W, H)
    boxes = {z.label: z for z in a.resource_zones}
    assert sorted(boxes) == ["Z1", "Z2", "Z3", "Z4"]
    # numbered as in the tracking analysis: Z1 bottom-left, then clockwise
    centre = {"Z1": (Z / 2, Z / 2), "Z2": (Z / 2, H - Z / 2),
              "Z3": (W - Z / 2, H - Z / 2), "Z4": (W - Z / 2, Z / 2)}
    for tag, z in boxes.items():
        assert (z.x, z.y) == pytest.approx(centre[tag])
        assert (z.w, z.d) == (Z, Z)
        assert z.water, "every zone holds a water tower as well as chow"


def test_each_box_has_two_doorways_and_both_face_the_room():
    for z in volecosm().arena.resource_zones:
        doors = zone_doors(z)
        assert len(doors) == 2
        for x, y in doors:
            assert 0.2 < x < W - 0.2 and 0.2 < y < H - 0.2, \
                f"{z.label} has a doorway in a wall against the room wall"


def test_one_free_standing_tower_in_the_middle():
    towers = volecosm().arena.water_towers
    assert len(towers) == 1
    assert (towers[0].x, towers[0].y) == pytest.approx((W / 2, H / 2))


def test_the_standard_cohort_and_protocol():
    cfg = volecosm()
    assert sorted((g.sex, g.count) for g in cfg.groups) == [("F", 3), ("M", 3)]
    assert all(g.treatment.drug == "saline" and g.treatment.day_offset == -5
               for g in cfg.groups)
    assert cfg.days == 10 and cfg.day_length_hours == 14
    assert cfg.scent.enabled and cfg.physiology.enabled
    assert not cfg.sky.enabled, "an indoor room has a light timer, not a sun"


def test_it_is_found_by_name_and_does_not_shadow_the_others():
    assert "VoleCosm" in find_preset("volecosm").name
    assert "anosmia" in find_preset("anosmia").name
    assert "VoleTerra" in find_preset("voleterra").name


# --------------------------------------------------------------------------- #
# More than one doorway
# --------------------------------------------------------------------------- #
def test_entrance_may_name_several_walls():
    z = ResourceZone(x=1.0, y=1.0, w=0.5, d=0.4, entrance="ne")
    assert zone_sides(z) == ["N", "E"]
    assert zone_doors(z) == [(1.0, 1.2), (1.25, 1.0)]
    assert zone_sides(ResourceZone()) == ["E"], "old configs keep one doorway"
    assert zone_sides(ResourceZone(entrance="")) == ["E"]


def _crossings(sim, steps, dt):
    """Every time an animal goes into or out of a zone: which zone, the point
    on that zone's wall where the straight line of that step crossed it, and
    how long the step was."""
    cx, cy, hw, hd = sim.food_rects.T

    def inside(P):
        return ((np.abs(P[:, 0][:, None] - cx) <= hw)
                & (np.abs(P[:, 1][:, None] - cy) <= hd))

    def wall_point(a, b, j):
        """Where the step a -> b meets the edge of box j."""
        best = None
        for axis, edge in ((0, cx[j] - hw[j]), (0, cx[j] + hw[j]),
                           (1, cy[j] - hd[j]), (1, cy[j] + hd[j])):
            if a[axis] == b[axis]:
                continue
            t = (edge - a[axis]) / (b[axis] - a[axis])
            if not 0.0 <= t <= 1.0:
                continue
            pt = a + t * (b - a)
            other = 1 - axis
            lo = (cy[j] - hd[j], cx[j] - hw[j])[other == 0]
            hi = (cy[j] + hd[j], cx[j] + hw[j])[other == 0]
            if lo - 1e-9 <= pt[other] <= hi + 1e-9 and (best is None
                                                        or t < best[0]):
                best = (t, pt)
        return None if best is None else best[1]

    out = []
    was, prev = inside(sim.P), sim.P.copy()
    for k in range(steps):
        sim.step(k * dt, dt)
        now = inside(sim.P)
        for i, j in zip(*np.nonzero(was != now)):
            pt = wall_point(prev[i], sim.P[i], j)
            if pt is not None:
                out.append((j, pt, float(np.linalg.norm(sim.P[i] - prev[i]))))
        was, prev = now, sim.P.copy()
    return out


@pytest.mark.parametrize("brain", [False, True])
def test_nothing_passes_a_zone_wall_except_at_a_doorway(brain):
    """Six animals, an hour and a half, with either nervous system: every
    entry and every exit goes through one of that box's two doorways.

    Within one step an animal may slide along a wall and then turn through
    the gap, so the straight line of the step can cut the wall beside the
    doorway by up to that step's length. Half-second steps keep that to a few
    centimetres; a real leak anywhere else along the 51 cm wall would show."""
    cfg = volecosm()
    policy = NeuralPolicy(Brain.random(3)) if brain else None
    sim = Simulation(cfg, 0, policy=policy)
    if brain:                        # a random brain needs a start by a door
        sim.P[:] = [[0.35, 0.62], [0.62, 0.3], [0.3, 2.0],
                    [2.5, 2.0], [2.2, 0.3], [2.5, 0.62]]
    else:                            # the rule-based animal needs a reason
        sim.energy[:] = 35.0
        sim.hunger[:] = 65.0
        sim.thirst[:] = 50.0
    crossed = _crossings(sim, 10800, dt=0.5)
    assert len(crossed) >= 4, "the test saw too few crossings to mean anything"
    half_gap = cfg.arena.resource_zones[0].hole / 2
    assert max(step for _, _, step in crossed) < 0.10
    for j, where, step in crossed:
        doors = sim._zone_door_sets[j]
        off = np.linalg.norm(doors - where, axis=1).min()
        assert off <= half_gap + step + 1e-6, \
            f"crossed zone {j}'s wall {off * 100:.1f} cm from a doorway centre"


def test_an_animal_inside_leaves_by_the_nearer_doorway():
    sim = Simulation(volecosm(), 0)
    # Z1 has doorways at (0.255, 0.51) [north] and (0.51, 0.255) [east]
    P = np.array([[0.20, 0.45], [0.45, 0.20]])
    door = sim.exit_door(P, np.array([0, 0]))
    assert door[0] == pytest.approx((0.255, 0.51))
    assert door[1] == pytest.approx((0.51, 0.255))


# --------------------------------------------------------------------------- #
# Water inside a zone
# --------------------------------------------------------------------------- #
def test_water_in_a_box_is_drunk_inside_it_and_not_through_its_wall():
    sim = Simulation(volecosm(), 0)
    inside = np.array([[0.30, 0.30]])
    just_outside = np.array([[0.30, 0.56]])          # 5 cm beyond Z1's wall
    open_floor = np.array([[0.9, 0.7]])
    by_the_tower = np.array([[W / 2 + 0.2, H / 2]])
    assert sim._on_water(inside)[0]
    assert not sim._on_water(just_outside)[0]
    assert not sim._on_water(open_floor)[0]
    assert sim._on_water(by_the_tower)[0]


def test_a_thirsty_animal_in_a_box_drinks_and_does_not_leave_to_find_water():
    cfg = volecosm()
    sim = Simulation(cfg, 0)
    sim.P[:] = (0.25, 0.25)
    sim.thirst[:] = 80.0
    sim.energy[:] = 100.0
    sim.hunger[:] = 0.0
    inside = 0
    for k in range(300):                              # ten minutes
        sim.step(k * 2.0, 2.0)
        inside += sim._on_water()
    assert (inside / 300 > 0.9).all(), \
        "it hovered in the doorway instead of staying in to drink"
    assert sim.thirst.max() < 55, "ten minutes of drinking takes ~30 points"


def test_a_zone_without_water_is_unchanged():
    """Zones in older presets hold food only; an animal in one is not at
    water, and a thirsty one still heads for the doorway."""
    from fnt.abma.core.presets import liddell_echo_t1
    sim = Simulation(liddell_echo_t1(), 0)
    assert len(sim.water_rects) == 0 and not sim.zone_has_water.any()
    z = sim.cfg.arena.resource_zones[0]
    assert not sim._on_water(np.array([[z.x, z.y]]))[0]


# --------------------------------------------------------------------------- #
# The light cycle
# --------------------------------------------------------------------------- #
def test_a_14_10_light_cycle():
    sim = Simulation(volecosm(), 0)          # released 12:30, lights 07:00-21:00
    h = 3600.0
    assert sim._is_day(0.0)
    assert sim._is_day(8.0 * h), "20:30 is still light on a 14 h day"
    assert not sim._is_day(9.0 * h), "21:30 is dark"
    assert not sim._is_day(18.0 * h), "06:30 is still dark"
    assert sim._is_day(19.0 * h), "07:30 the lights are on"


def test_day_length_defaults_to_twelve_and_round_trips():
    cfg = volecosm()
    d = cfg.to_dict()
    again = ExperimentConfig.from_dict(d)
    assert again.day_length_hours == 14 and again.to_dict() == d
    z = again.arena.resource_zones[0]
    assert z.water and zone_sides(z) == ["N", "E"]
    d.pop("day_length_hours")
    for zd in d["arena"]["resource_zones"]:
        zd.pop("water")
    old = ExperimentConfig.from_dict(d)       # a config from before either
    assert old.day_length_hours == 12
    assert not old.arena.resource_zones[0].water


# --------------------------------------------------------------------------- #
# Using it
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("brain", [False, True])
def test_a_cohort_feeds_and_waters_itself(brain):
    """Released hungry and thirsty in the middle of the room, every animal
    finds its way into a box and is in better shape six hours later."""
    cfg = design(preset="volecosm")
    cfg.brain.enabled = brain
    sim = Simulation(cfg, 0)
    sim.energy[:] = 45.0
    sim.hunger[:] = 55.0
    sim.thirst[:] = 55.0
    for k in range(10800):                               # six hours
        sim.step(k * 2.0, 2.0)
    assert sim.alive.all()
    assert (sim.food_eaten_g > 0).all() and (sim.water_drunk_ml > 0).all()
    assert sim.energy.min() > 50 and sim.thirst.max() < 40


def test_the_one_line_experiment(capsys):
    """'3 males and 3 females in the VoleCosm for 5 days; 2 trials intact
    and 2 trials on methimazole.'"""
    rc = headless(["--preset", "volecosm", "--males", "3", "--females", "3",
                   "--days", "5", "--anosmia-study", "--doses", "0,1",
                   "--replicates", "2", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "2.81 x 2.61 m" in out and "3 F" in out and "3 M" in out
    assert "2 arms" in out and "2 replicate(s), 5 days" in out


def test_a_movie_draws_the_boxes(tmp_path):
    clip = movie.film(design(preset="volecosm"), seconds=16)
    path = movie.render([clip], str(tmp_path / "vc.gif"), dpi=50)
    assert os.path.getsize(path) > 0


def test_the_window_keeps_the_light_cycle_and_the_zones():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QApplication
    from fnt.abma.gui.abma_main_pyqt import ABMAWindow

    app = QApplication.instance() or QApplication([])   # noqa: F841
    win = ABMAWindow()
    win._load_config(volecosm())
    out = win._collect_config()
    assert out.day_length_hours == 14 and out.day_start_hour == 7
    zones = out.arena.resource_zones
    assert len(zones) == 4 and all(z.water for z in zones)
    assert zone_sides(zones[2]) == ["S", "W"]


# --------------------------------------------------------------------------- #
# A wall that meets the room's wall
# --------------------------------------------------------------------------- #
def test_a_box_against_the_room_wall_cannot_be_walked_around_outside():
    """A step may overshoot the room's boundary before it is bounced back. A
    box wall that ended exactly at the boundary could be passed on the
    outside: out past its end, back in on the other side. It must not be."""
    sim = Simulation(volecosm(), 0)
    # just outside Z3's south wall (y = 2.10), a hair from the east room wall
    old = np.array([[2.7987, 2.0993]])
    want = np.array([[2.8139, 2.1317]])              # overshoots x = 2.81
    sim.P = np.repeat(old, sim.n, axis=0)
    got = sim._resolve_obstacles(sim.P, np.repeat(want, sim.n, axis=0))
    sim.P = got
    sim._apply_boundary()
    assert (sim.P[:, 1] < 2.10).all(), "it ended up inside the box"
    # ...and the same from the inside, trying to leave by the corner
    old = np.array([[2.8090, 2.1010]])
    want = np.array([[2.8150, 2.0900]])
    sim.P = np.repeat(old, sim.n, axis=0)
    sim.P = sim._resolve_obstacles(sim.P, np.repeat(want, sim.n, axis=0))
    sim._apply_boundary()
    assert (sim.P[:, 1] > 2.10).all(), "it left the box through the corner"


def test_walls_along_the_room_wall_are_not_extended():
    """Only a wall that meets the boundary head-on is carried past it; one
    lying along the boundary stays as drawn, so the room's own wall still
    bounces animals rather than catching them."""
    from fnt.abma.core.simulation import _run_wall_past_boundary as run
    assert run((2.593, 2.10, 2.81, 2.10), W, H)[2] > 100     # head-on: extended
    assert run((2.593, 2.10, 2.81, 2.10), W, H)[:2] == (2.593, 2.10)
    assert run((2.81, 2.10, 2.81, 2.50), W, H) == (2.81, 2.10, 2.81, 2.50)
    assert run((1.0, 1.0, 1.5, 1.0), W, H) == (1.0, 1.0, 1.5, 1.0)
    assert run((2.30, 0.0, 2.30, 0.217), W, H)[1] < -100     # south end
