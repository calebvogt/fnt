"""Where founders are put down at t=0.

A release is part of an experimental design, not an implementation detail.
Twelve animals set down inside one body-length of each other in a 523 m²
enclosure do not disperse — social attraction holds the clump together, and the
run measures a scrum instead of a population. Measured before the fix: the
VoleTerra cohort used 16% of the enclosure and sat at a mean female-female
distance of 0.18 m.
"""
from __future__ import annotations

import numpy as np
import pytest

from fnt.abma.core.compose import design
from fnt.abma.core.presets import vole_anosmia
from fnt.abma.core.simulation import Simulation


def _spacing(cfg) -> float:
    sim = Simulation(cfg, 0)
    d = np.linalg.norm(sim.P[:, None, :] - sim.P[None, :, :], axis=2)
    return float(d[~np.eye(sim.n, dtype=bool)].mean())


def _positions(cfg) -> np.ndarray:
    return Simulation(cfg, 0).P.copy()


def test_a_cage_keeps_its_point_release():
    """The historical behaviour, which was never wrong at cage scale."""
    cfg = vole_anosmia()
    assert cfg.arena.width <= Simulation._POINT_RELEASE_MAX_M
    assert _spacing(cfg) < 0.5


def test_an_enclosure_disperses_its_founders():
    cfg = design(preset="voleterra", males=6, females=6, days=1)
    assert cfg.release_mode == "auto"
    assert _spacing(cfg) > 5.0


def test_point_release_can_still_be_asked_for():
    cfg = design(preset="voleterra", males=6, females=6, days=1)
    cfg.release_mode = "point"
    assert _spacing(cfg) < 1.0


def test_scatter_stays_inside_the_arena():
    cfg = design(preset="voleterra", males=8, females=8, days=1)
    cfg.release_mode = "scatter"
    P = _positions(cfg)
    assert (P[:, 0] > 0).all() and (P[:, 0] < cfg.arena.width).all()
    assert (P[:, 1] > 0).all() and (P[:, 1] < cfg.arena.height).all()


def test_nests_release_puts_animals_at_the_structures():
    cfg = design(preset="voleterra", males=4, females=4, days=1)
    cfg.release_mode = "nests"
    P = _positions(cfg)
    sites = np.array([[z.x, z.y] for z in cfg.arena.resource_zones])
    nearest = np.linalg.norm(P[:, None, :] - sites[None, :, :],
                             axis=2).min(axis=1)
    assert nearest.max() < 1.5, "an animal was not released near any structure"


def test_nests_release_falls_back_when_there_is_nowhere_to_nest():
    cfg = design(preset="voleterra", males=3, females=3, days=1)
    cfg.arena.resource_zones = []
    cfg.arena.objects = [o for o in cfg.arena.objects if o.kind != "nest"]
    cfg.release_mode = "nests"
    assert _spacing(cfg) > 5.0        # scattered rather than stacked at 0,0


def test_an_explicit_scatter_radius_is_honoured():
    cfg = design(preset="voleterra", males=6, females=6, days=1)
    cfg.release_mode = "point"
    cfg.release_scatter_m = 3.0
    wide = _spacing(cfg)
    cfg.release_scatter_m = 0.1
    assert wide > _spacing(cfg) * 3


@pytest.mark.parametrize("mode", ["auto", "point", "scatter", "nests"])
def test_every_mode_round_trips_through_json(mode, tmp_path):
    from fnt.abma.core.config import ExperimentConfig

    cfg = design(preset="voleterra", males=2, females=2, days=1)
    cfg.release_mode = mode
    cfg.release_scatter_m = 1.25
    path = tmp_path / "cfg.json"
    cfg.to_json(str(path))
    back = ExperimentConfig.from_json(str(path))
    assert back.release_mode == mode
    assert back.release_scatter_m == 1.25


def test_dispersing_the_release_is_what_lets_territory_form():
    """The measured consequence, at the scale where it matters.

    Not a claim that spacing alone fixes the enclosure — it does not, the
    metabolic and marking scale bugs mattered too — but the clump is a real
    and separable cause, so this pins it.
    """
    got = {}
    for mode in ("point", "auto"):
        cfg = design(preset="voleterra", males=6, females=6, days=0.75)
        cfg.release_mode = mode
        sim = Simulation(cfg, 0)
        elapsed = 0.0
        for _ in range(int(cfg.days * 86400 / cfg.dt)):
            sim.step(elapsed, cfg.dt)
            elapsed += cfg.dt
        d = np.linalg.norm(sim.P[:, None, :] - sim.P[None, :, :], axis=2)
        got[mode] = (float(d[~np.eye(sim.n, dtype=bool)].mean()),
                     float(sim.territory_area().sum()))
    assert got["auto"][0] > got["point"][0], "dispersal did not survive the run"
    assert got["auto"][1] > got["point"][1], "dispersal did not widen territory"
