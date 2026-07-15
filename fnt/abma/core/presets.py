"""Arena / experiment presets — ready-made starting points a user can load.

Each preset is a factory returning a full :class:`ExperimentConfig`. They are the
quick on-ramps to common paradigms; the user then edits freely. Register new
paradigms by adding a factory and a :class:`Preset` entry to ``PRESETS``.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Callable

from .config import (
    ExperimentConfig, ArenaConfig, AgentGroup, Genotype, Treatment,
    TraitProfile, Zone, Pole, AntennaBox, AntennaLayout, blank_experiment,
    default_vole_experiment,
)

FT = 0.3048   # feet -> metres
IN = 0.0254   # inches -> metres


@dataclass
class Preset:
    name: str
    description: str
    factory: Callable[[], ExperimentConfig]
    abbr: str = ""          # short tag used to suggest "Custom <abbr>_N" names
    blank: bool = False     # a build-from-scratch arena (shows the editor)
    builtin: bool = True


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


def voleterra() -> ExperimentConfig:
    """VoleTerra semi-natural enclosure: 75×75 ft, 3 ft walls, 25 support poles.

    A 5×5 grid of 7 ft, 6 in-diameter poles spans the full enclosure. The 16
    perimeter poles sit on the wall centreline (the walls physically connect
    through them); the inner 3×3 = 9 poles stand free inside the arena. Poles
    are rendered as vertical cylinders in the 2D/3D preview.
    """
    W = H = 75 * FT                          # 22.86 m
    sp = W / 4.0                             # 5 poles / side -> 4 gaps
    # perimeter poles (walls connect through them) are 6" dia; the inner 3×3
    # free-standing poles are 3" dia.
    def _pole_radius(i, j):
        perimeter = i in (0, 4) or j in (0, 4)
        return (3 * IN) if perimeter else (1.5 * IN)
    poles = [Pole(x=i * sp, y=j * sp, radius=_pole_radius(i, j), height=7 * FT,
                  label=f"pole_{i}{j}")
             for i in range(5) for j in range(5)]
    # UWB antenna boxes mounted ~6 ft up on the poles. Two selectable layouts:
    #   full 25-antenna (one per pole) and a 13-antenna checkerboard subset.
    def _box(p, label):
        return AntennaBox(x=p.x, y=p.y, z=6 * FT, w=8 * IN, d=4 * IN,
                          h=8 * IN, label=label)

    # 25-antenna: 1 = SW corner, increasing east along each row, then north row
    # by row (1-5 south … 21-25 north). Antennas 1-3 are gateways, tagged "(G)".
    full = []
    for p in poles:
        i = round(p.x / sp)          # 0=W … 4=E
        j = round(p.y / sp)          # 0=S … 4=N
        num = j * 5 + i + 1
        full.append(_box(p, f"{num} (G)" if num <= 3 else str(num)))

    # 13-antenna: checkerboard where (i+j) is even, numbered S→N then W→E.
    # Antennas 1-2 are gateways, tagged "(G)".
    pole_at = {(round(p.x / sp), round(p.y / sp)): p for p in poles}
    alt, k = [], 1
    for j in range(5):
        for i in range(5):
            if (i + j) % 2 == 0:
                alt.append(_box(pole_at[(i, j)], f"{k} (G)" if k <= 2 else str(k)))
                k += 1

    # optimal PoE daisy-chain wiring for each layout (gateways = "(G)" boxes)
    from .poe import solve_poe_wiring

    def _wire(boxes):
        gws = [(b.label, (b.x, b.y)) for b in boxes if "(G)" in b.label]
        leaves = [(b.label, (b.x, b.y)) for b in boxes if "(G)" not in b.label]
        cables, _ = solve_poe_wiring(gws, leaves, z=6 * FT)
        return cables

    arena = ArenaConfig(
        width=W, height=H, boundary="reflective", objects=[], zones=[],
        poles=poles, antennas=full,
        antenna_layouts=[AntennaLayout("25-antenna", full, _wire(full)),
                         AntennaLayout("13-antenna", alt, _wire(alt))],
        wall_height=3 * FT, wall_thickness=2 * IN, ground="grass",
        oriented=True)
    # a suggested starter cohort — adjust freely after loading
    voles = [
        AgentGroup(label="males", species="prairie", sex="M", count=4,
                   genotype=Genotype({}), treatment=Treatment("none", 0.0, 0.0),
                   traits=TraitProfile(mass=42.0, home_range_r=2.0,
                                       exploration=0.6, base_speed=0.12),
                   dists={"mass": "N(42,4)"}),
        AgentGroup(label="females", species="prairie", sex="F", count=4,
                   genotype=Genotype({}), treatment=Treatment("none", 0.0, 0.0),
                   traits=TraitProfile(mass=38.0, home_range_r=1.6,
                                       exploration=0.5, base_speed=0.11),
                   dists={"mass": "N(38,4)"}),
    ]
    return ExperimentConfig(
        name="voleterra", arena=arena, groups=voles,
        days=10.0, dt=2.0, record_interval=10.0, n_trials=1,
        start_datetime="2025-11-07T18:00:00")


# Ordered registry of built-in presets shown in the GUI picker.
PRESETS: list[Preset] = [
    Preset("Blank arena",
           "Start from scratch — set dimensions, boundary, and place objects.",
           blank_experiment, abbr="Arena", blank=True),
    Preset("Open Field Test (50×50 cm)",
           "Single subject, 10 min, centre/periphery zones.",
           open_field_test, abbr="OFT"),
    Preset("Prairie vole — anosmia",
           "2.2 m enclosure, 4M/4F, saline vs methimazole.",
           default_vole_experiment, abbr="Vole"),
    Preset("VoleTerra enclosure (75×75 ft)",
           "Semi-natural field enclosure: 3 ft walls, 25 support poles (5×5).",
           voleterra, abbr="VoleTerra"),
]


# --------------------------------------------------------------------------- #
# User-saved presets (persisted as JSON configs)
# --------------------------------------------------------------------------- #
def user_preset_dir() -> str:
    d = os.path.join(os.path.expanduser("~"), "ABMA_presets")
    os.makedirs(d, exist_ok=True)
    return d


def user_presets() -> list[Preset]:
    out = []
    for fp in sorted(glob.glob(os.path.join(user_preset_dir(), "*.json"))):
        name = os.path.splitext(os.path.basename(fp))[0]
        out.append(Preset(name, "Saved preset",
                          (lambda p=fp: ExperimentConfig.from_json(p)),
                          abbr=name, builtin=False))
    return out


def save_user_preset(name: str, config: ExperimentConfig) -> str:
    safe = "".join(c for c in name if c.isalnum() or c in " _-").strip() or "preset"
    path = os.path.join(user_preset_dir(), f"{safe}.json")
    config.to_json(path)
    return path


def suggest_preset_name(abbr: str) -> str:
    """'Custom <abbr>_N' where N is the next free ordinal among saved presets."""
    abbr = abbr or "Arena"
    existing = {p.name for p in user_presets()}
    i = 1
    while f"Custom {abbr}_{i}" in existing:
        i += 1
    return f"Custom {abbr}_{i}"


def all_presets() -> list[Preset]:
    return PRESETS + user_presets()


def get_preset(name: str) -> ExperimentConfig:
    for p in all_presets():
        if p.name == name:
            return p.factory()
    raise KeyError(name)
