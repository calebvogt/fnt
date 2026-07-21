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
    TraitProfile, Zone, Pole, AntennaBox, AntennaLayout, WaterTower,
    ResourceZone, Hut, GrassSpec, blank_experiment, default_vole_experiment,
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
    # named alternate configurations of the same enclosure (e.g. trial 1 / 2).
    # Empty = a single configuration (``factory``).
    configs: list = field(default_factory=list)   # [(name, factory), ...]

    def config_list(self):
        """Always returns [(name, factory), …] — one entry if none defined."""
        return list(self.configs) if self.configs else [(self.name, self.factory)]


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

    # Ground resources in the 16 inter-pole squares (4×4). Squares alternate
    # water / food like a checkerboard: (si+sj) even -> water tower, odd ->
    # resource zone (nesting material + grass). 8 of each.
    water, zones = [], []
    for si in range(4):
        for sj in range(4):
            cx, cy = (si + 0.5) * sp, (sj + 0.5) * sp
            if (si + sj) % 2 == 0:
                water.append(WaterTower(x=cx, y=cy, radius=3 * IN, height=8 * IN,
                                        label=f"W{si}{sj}"))
            else:
                # alternate the 3" entrance hole E / W for successive zones
                side = "E" if len(zones) % 2 == 0 else "W"
                zones.append(ResourceZone(x=cx, y=cy, w=30 * IN, d=20 * IN,
                                          h=17 * IN, entrance=side, hole=3 * IN,
                                          label=f"R{si}{sj}"))

    arena = ArenaConfig(
        width=W, height=H, boundary="reflective", objects=[], zones=[],
        poles=poles, water_towers=water, resource_zones=zones, antennas=full,
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


def voleterra_2026_t001() -> ExperimentConfig:
    """VoleTerra with the ground cover seen in the 2026_VT_T001 drone photo.

    MEASURED from the 2026-07-20 overhead drone photo with
    :mod:`fnt.abma.core.photo_cover` (homography onto the pen from its four
    inner wall corners, then Excess-Green classification on an 8x8 grid).
    Live green cover is only **6.7%** of the ground (0-48% by cell): a dry,
    semi-arid sward of straw thatch and bare soil with a few green tufts.

    Note the pattern is *banded, not a gradient*: green concentrates in a strip
    down the east edge and a patch left of centre, with a near-dead zone
    through the centre-east. North/south are effectively equal (6.5% vs 7.0%).
    """
    cfg = voleterra()
    cfg.name = "voleterra_2026_VT_T001"
    # Measured relative cover, rows S->N and cols W->E (arena orientation;
    # the photo is north-up so this is it flipped). 0.40 floor represents the
    # roughly uniform dead layer, which colour alone cannot separate from soil.
    cover = [
        [0.40, 0.41, 0.41, 0.45, 0.42, 0.41, 0.43, 0.65],
        [0.40, 0.52, 0.65, 0.49, 0.41, 0.40, 0.43, 0.55],
        [0.40, 0.45, 1.00, 0.55, 0.41, 0.41, 0.41, 0.55],
        [0.41, 0.45, 0.70, 0.44, 0.40, 0.40, 0.41, 0.74],
        [0.44, 0.49, 0.60, 0.42, 0.43, 0.40, 0.42, 0.78],
        [0.50, 0.43, 0.56, 0.44, 0.41, 0.41, 0.40, 0.51],
        [0.42, 0.42, 0.76, 0.41, 0.41, 0.41, 0.44, 0.82],
        [0.45, 0.42, 0.53, 0.42, 0.40, 0.41, 0.41, 0.60],
    ]
    # density is the peak (richest-patch) rate; the cover map and tuft
    # clumping scale it down. dry_fraction 0.88 from the 6.7% live-green share.
    cfg.arena.grass = GrassSpec(density=70.0, h_min=0.038, h_max=0.102,
                                dry_fraction=0.88, patchiness=0.75,
                                cover_map=cover)
    return cfg


def liddell_echo_t1() -> ExperimentConfig:
    """Liddell Echo enclosure (Cornell, mice) — trial 1 (T001) configuration.

    A 15 ft × 12 ft 10 in room (~9 ft ceiling; the room walls are not drawn so
    the interior stays visible, but the boundary is reflective). Eight
    50×50 cm resource zones: three against the north wall, three against the
    south wall, two free-standing in the middle. Eight bare Wiser UWB antennas
    wall-mounted high, two per wall. Red acrylic huts: 6×4×4" hollow tubes and
    5" half domes. One central water tower. Dimensions are approximate — only
    the room size was measured.
    """
    W = 15 * FT                       # 4.572 m  (east-west)
    H = (12 + 10 / 12.0) * FT         # 3.912 m  (north-south)
    ZS, ZH = 0.50, 0.50               # 50 × 50 × 50 cm resource zones
    HOLE = 2 * IN                     # mouse-sized doorway

    zones, huts = [], []

    def _stock(cx, cy, tag):
        """Each zone holds a tube hut + a dome hut (food is the zone's own)."""
        huts.append(Hut(kind="tube", x=cx - 0.10, y=cy + 0.10, w=6 * IN,
                        d=4 * IN, h=4 * IN, angle=0.0, label=f"tube_{tag}"))
        huts.append(Hut(kind="dome", x=cx + 0.11, y=cy - 0.10, w=5 * IN,
                        h=2.5 * IN, label=f"dome_{tag}"))

    for i, fx in enumerate((0.25, 0.50, 0.75)):
        # north row (shoved against the wall, doorway facing into the room)
        zones.append(ResourceZone(x=fx * W, y=H - ZS / 2, w=ZS, d=ZS, h=ZH,
                                  entrance="S", hole=HOLE, label=f"N{i+1}"))
        _stock(fx * W, H - ZS / 2, f"N{i+1}")
        # south row
        zones.append(ResourceZone(x=fx * W, y=ZS / 2, w=ZS, d=ZS, h=ZH,
                                  entrance="N", hole=HOLE, label=f"S{i+1}"))
        _stock(fx * W, ZS / 2, f"S{i+1}")
    # two free-standing middle zones, doorways facing outward (W / E)
    for fx, side, tag in ((0.36, "W", "M1"), (0.64, "E", "M2")):
        zones.append(ResourceZone(x=fx * W, y=H / 2, w=ZS, d=ZS, h=ZH,
                                  entrance=side, hole=HOLE, label=tag))
        _stock(fx * W, H / 2, tag)

    # free-standing tube huts: four in the open middle, four by the side walls
    for fx, fy in ((0.215, 0.30), (0.742, 0.30), (0.215, 0.667), (0.742, 0.667)):
        huts.append(Hut(kind="tube", x=fx * W, y=fy * H, w=6 * IN, d=4 * IN,
                        h=4 * IN, angle=0.0, label="tube_open"))
    for fx, fy in ((0.05, 0.07), (0.95, 0.07), (0.05, 0.93), (0.95, 0.93)):
        huts.append(Hut(kind="tube", x=fx * W, y=fy * H, w=6 * IN, d=4 * IN,
                        h=4 * IN, angle=90.0, label="tube_wall"))

    # eight bare Wiser antennas, two per wall, mounted high
    AZ, inset = 2.35, 0.05
    ants, k = [], 1
    for x, y in ((W / 3, inset), (2 * W / 3, inset),           # south wall
                 (W - inset, H / 3), (W - inset, 2 * H / 3),   # east wall
                 (2 * W / 3, H - inset), (W / 3, H - inset),   # north wall
                 (inset, 2 * H / 3), (inset, H / 3)):          # west wall
        ants.append(AntennaBox(x=x, y=y, z=AZ, w=0.05, d=0.05, h=0.10,
                               style="bare", label=str(k)))
        k += 1

    arena = ArenaConfig(
        width=W, height=H, boundary="reflective", objects=[], zones=[],
        water_towers=[WaterTower(x=W / 2, y=H / 2, radius=3 * IN,
                                 height=8 * IN, label="water")],
        resource_zones=zones, huts=huts, antennas=ants,
        antenna_layouts=[AntennaLayout("8-antenna (trial 1)", ants, [])],
        # room walls are not drawn (they obscure the view); the boundary is
        # still reflective, so agents bounce off the edges as if they were there
        wall_height=0.0, wall_thickness=4 * IN, ground="floor",
        oriented=False)

    mice = [
        AgentGroup(label="males", species="mouse", sex="M", count=2,
                   genotype=Genotype({}), treatment=Treatment("none", 0.0, 0.0),
                   traits=TraitProfile(mass=26.0, home_range_r=0.9,
                                       exploration=0.7, base_speed=0.14),
                   dists={"mass": "N(26,2)"}),
        AgentGroup(label="females", species="mouse", sex="F", count=2,
                   genotype=Genotype({}), treatment=Treatment("none", 0.0, 0.0),
                   traits=TraitProfile(mass=21.0, home_range_r=0.8,
                                       exploration=0.65, base_speed=0.13),
                   dists={"mass": "N(21,2)"}),
    ]
    return ExperimentConfig(
        name="liddell_echo_t1", arena=arena, groups=mice,
        days=3.0, dt=1.0, record_interval=5.0, n_trials=1,
        start_datetime="2025-01-01T18:00:00")


liddell_echo = liddell_echo_t1      # backwards-compatible alias


def liddell_echo_t2() -> ExperimentConfig:
    """Liddell Echo enclosure — trial 2 (T002) 'mesocosm layout'.

    Changes from T001: resource zones pulled off the walls (so mice can't sit
    behind them); extra acrylic housing added along the walls instead of in the
    open middle; more UWB antennas, several mounted on the middle resource
    zones and all lowered nearer the floor (to clear the metal duct in the NW
    corner); founders 4M + 4F plus 2M invaders.

    The slide planned 14 antennas, but the trial's Wiser config defines 13 and
    only 12 have a real height, so the 12 deployed anchors are used here at
    their surveyed positions. Resource zones use the true 50 cm dimensions at
    idealized positions — the config's zone polygons were traced by dropping a
    tag at each corner and average 38% oversized, so they are not used.
    """
    W = 15 * FT                       # 4.572 m  (east-west)
    H = (12 + 10 / 12.0) * FT         # 3.912 m  (north-south)
    ZS = ZH = 0.50                    # 50 × 50 × 50 cm resource zones
    HOLE = 2 * IN
    GAP = 0.45                        # clearance from the wall to the zone face

    zones, huts = [], []

    def _stock(cx, cy, tag):
        huts.append(Hut(kind="tube", x=cx - 0.10, y=cy + 0.10, w=6 * IN,
                        d=4 * IN, h=4 * IN, angle=0.0, label=f"tube_{tag}"))
        huts.append(Hut(kind="dome", x=cx + 0.11, y=cy - 0.10, w=5 * IN,
                        h=2.5 * IN, label=f"dome_{tag}"))

    y_n, y_s, y_m = H - (GAP + ZS / 2), GAP + ZS / 2, H / 2
    for i, fx in enumerate((0.20, 0.50, 0.80)):
        zones.append(ResourceZone(x=fx * W, y=y_n, w=ZS, d=ZS, h=ZH,
                                  entrance="S", hole=HOLE, label=f"N{i+1}"))
        _stock(fx * W, y_n, f"N{i+1}")
        zones.append(ResourceZone(x=fx * W, y=y_s, w=ZS, d=ZS, h=ZH,
                                  entrance="N", hole=HOLE, label=f"S{i+1}"))
        _stock(fx * W, y_s, f"S{i+1}")
    for fx, side, tag in ((0.33, "W", "M1"), (0.67, "E", "M2")):
        zones.append(ResourceZone(x=fx * W, y=y_m, w=ZS, d=ZS, h=ZH,
                                  entrance=side, hole=HOLE, label=tag))
        _stock(fx * W, y_m, tag)

    # extra acrylic housing along the walls (the open-middle tubes are gone)
    for x, y, ang in ((0.12, H - 0.12, 90.0), (0.49 * W, H - 0.10, 0.0),
                      (W - 0.12, H - 0.12, 90.0),
                      (0.10, 0.51 * H, 90.0), (W - 0.10, 0.51 * H, 90.0),
                      (0.12, 0.12, 90.0), (0.50 * W, 0.10, 0.0),
                      (W - 0.12, 0.12, 90.0)):
        huts.append(Hut(kind="tube", x=x, y=y, w=6 * IN, d=4 * IN, h=4 * IN,
                        angle=ang, label="tube_wall"))

    # Real anchor placements, read from the T002 Wiser configuration
    # (EchoConfiguration_2024.12.4.xml) and mapped onto the room through the
    # config's own "Arena" outline. 13 anchors are defined but one (shortid
    # 102) has z = 0 — never actually placed — so 12 were deployed. Heights
    # 0.43 / 0.65 / 0.75 m confirm the "lowered near ground level" change.
    ants = [AntennaBox(x=x, y=y, z=z, w=0.05, d=0.05, h=0.10,
                       style="bare", label=sid)
            for x, y, z, sid in (
                (2.477, 3.882, 0.430, "101"), (2.362, 0.812, 0.645, "3"),
                (1.434, 0.030, 0.430, "8"),   (0.052, 0.959, 0.430, "9"),
                (0.124, 2.210, 0.430, "11"),  (2.045, 3.071, 0.645, "12"),
                (3.110, 2.241, 0.753, "13"),  (1.653, 1.881, 0.753, "17"),
                (0.513, 3.882, 0.430, "14"),  (0.850, 0.757, 0.645, "18"),
                (4.542, 3.175, 0.430, "103"), (3.636, 0.256, 0.430, "105"))]

    arena = ArenaConfig(
        width=W, height=H, boundary="reflective", objects=[], zones=[],
        water_towers=[WaterTower(x=W / 2, y=H / 2, radius=3 * IN,
                                 height=8 * IN, label="water")],
        resource_zones=zones, huts=huts, antennas=ants,
        antenna_layouts=[AntennaLayout("12-antenna (trial 2, as deployed)",
                                       ants, [])],
        wall_height=0.0, wall_thickness=4 * IN, ground="floor", oriented=False)

    mice = [
        AgentGroup(label="founder males", species="mouse", sex="M", count=4,
                   genotype=Genotype({}), treatment=Treatment("none", 0.0, 0.0),
                   traits=TraitProfile(mass=26.0, home_range_r=0.9,
                                       exploration=0.7, base_speed=0.14),
                   dists={"mass": "N(26,2)"}),
        AgentGroup(label="founder females", species="mouse", sex="F", count=4,
                   genotype=Genotype({}), treatment=Treatment("none", 0.0, 0.0),
                   traits=TraitProfile(mass=21.0, home_range_r=0.8,
                                       exploration=0.65, base_speed=0.13),
                   dists={"mass": "N(21,2)"}),
        AgentGroup(label="invader males", species="mouse", sex="M", count=2,
                   genotype=Genotype({}), treatment=Treatment("none", 0.0, 0.0),
                   traits=TraitProfile(mass=25.0, home_range_r=1.2,
                                       exploration=0.85, base_speed=0.15),
                   dists={"mass": "N(25,2)"}),
    ]
    return ExperimentConfig(
        name="liddell_echo_t2", arena=arena, groups=mice,
        days=3.0, dt=1.0, record_interval=5.0, n_trials=1,
        start_datetime="2025-01-01T18:00:00")


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
           voleterra, abbr="VoleTerra",
           configs=[("Default — generic green turf", voleterra),
                    ("2026_VT_T001 — dry patchy sward (from drone photo)",
                     voleterra_2026_t001)]),
    Preset("Liddell Echo enclosure (15 × 12'10\")",
           "Indoor mouse room: 8 resource zones, acrylic huts; pick a trial config.",
           liddell_echo_t1, abbr="Liddell",
           configs=[("Trial 1 (T001) — 8 antennas, zones on the wall",
                     liddell_echo_t1),
                    ("Trial 2 (T002) — 14 antennas, zones pulled off the wall",
                     liddell_echo_t2)]),
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
