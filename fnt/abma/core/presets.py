"""Arena / experiment presets — ready-made starting points a user can load.

Each preset is a factory returning a full :class:`ExperimentConfig`. They are the
quick on-ramps to common paradigms; the user then edits freely. Register new
paradigms by adding a factory and a :class:`Preset` entry to ``PRESETS``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .config import (
    ExperimentConfig, ArenaConfig, AgentGroup, Genotype, Treatment,
    TraitProfile, Zone, blank_experiment, default_vole_experiment,
)


@dataclass
class Preset:
    name: str
    description: str
    factory: Callable[[], ExperimentConfig]


def open_field_test() -> ExperimentConfig:
    """Standard Open Field Test: a 50x50 cm empty box, one subject, 10 minutes.

    The arena carries the canonical OFT centre zone (inner 50% by side length),
    so the run renders the centre/periphery distinction and the analysis reports
    centre-time — the classic thigmotaxis / anxiety readout.
    """
    w = h = 0.5
    center = Zone(name="center", x=0.25 * w, y=0.25 * h,
                  w=0.5 * w, h=0.5 * h, role="center")
    arena = ArenaConfig(width=w, height=h, boundary="reflective",
                        objects=[], zones=[center],
                        wall_height=0.5, wall_thickness=0.005)  # 50x50x50 cm box
    subject = AgentGroup(
        label="subject", species="mouse", sex="M", count=1,
        genotype=Genotype({}), treatment=Treatment("none", 0.0, 0.0),
        traits=TraitProfile(mass=25.0, home_range_r=0.28, exploration=0.6),
        dists={"mass": "N(25,2)"})
    return ExperimentConfig(
        name="open_field_test", arena=arena, groups=[subject],
        days=10.0 / 1440.0,          # 10 minutes
        dt=0.5, record_interval=1.0, n_trials=1,
        start_datetime="2025-01-01T12:00:00")


# Ordered registry shown in the GUI's Presets menu.
PRESETS: list[Preset] = [
    Preset("Blank experiment",
           "Empty arena and one generic agent type.",
           blank_experiment),
    Preset("Open Field Test (50×50 cm)",
           "Single subject, 10 min, centre/periphery zones.",
           open_field_test),
    Preset("Prairie vole — anosmia",
           "2.2 m enclosure, 4M/4F, saline vs methimazole.",
           default_vole_experiment),
]


def get_preset(name: str) -> ExperimentConfig:
    for p in PRESETS:
        if p.name == name:
            return p.factory()
    raise KeyError(name)
