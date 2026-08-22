"""Config serialization must be lossless: from_dict(to_dict(cfg)) == cfg.

This is the test that guards against fields silently falling out of the
round-trip (the GUI once dropped PolicyParams exactly this way).
"""
from fnt.abma.core.config import (
    ExperimentConfig, ArenaConfig, ResourceObject, Zone, Pole, WaterTower,
    ResourceZone, Hut, AntennaBox, AntennaLayout, Cable, GrassSpec,
    AgentGroup, Genotype, Treatment, TraitProfile, Appearance,
    Intervention, Coupling, PolicyParams, default_vole_experiment,
)


def _maximal_config() -> ExperimentConfig:
    """A config exercising every nested structure and non-default value."""
    arena = ArenaConfig(
        width=3.3, height=4.4, units="m", boundary="wrap",
        objects=[ResourceObject("food", 1.0, 2.0, 0.25, 0.8, "chow_A"),
                 ResourceObject("nest", 0.5, 0.5, 0.1, label="nest_A")],
        zones=[Zone("center", 0.8, 0.8, 1.7, 2.8, "center")],
        poles=[Pole(1.5, 1.5, 0.08, 2.0, "pole_1")],
        water_towers=[WaterTower(2.0, 3.0, 0.09, 0.25, "tower_1")],
        resource_zones=[ResourceZone(1.0, 1.0, 0.7, 0.5, 0.4, "W", 0.08, "rz")],
        huts=[Hut("dome", 2.5, 2.5, 0.13, 0.1, 0.1, 0.006, 45.0, "hut_1")],
        antennas=[AntennaBox(0.1, 0.2, 1.8, style="bare", label="A1")],
        antenna_layouts=[AntennaLayout(
            "grid", [AntennaBox(0.3, 0.4, label="B1")],
            [Cable("B1", "B", [[0.3, 0.4, 1.8], [0.6, 0.8, 1.8]])])],
        wall_height=0.5, wall_thickness=0.01, ground="grass",
        grass=GrassSpec(60.0, 0.04, 0.12, 0.3, 0.5, [[0.1, 0.9], [0.5, 0.2]]),
        oriented=True,
    )
    groups = [AgentGroup(
        "exp_M", "prairie", "M", 6,
        Genotype({"OXTR": "KO", "AVPR1A": "HET"}),
        Treatment("methimazole", 0.7, 2.5),
        TraitProfile(aggression=0.8, boldness=0.3, mass=44.0, wander=0.9),
        Appearance("blob", "#4a90d9", 1.4),
        dists={"mass": "N(44,4)", "boldness": "U(0.2,0.6)"},
    )]
    return ExperimentConfig(
        name="maximal", arena=arena, groups=groups,
        interventions=[Intervention(3.5, "exp_M", "smell_ability",
                                    "scale", 0.0, "anosmia")],
        dynamics=[Coupling("movement", "energy", "rate", -1.1, "mass",
                           "source_high", 0.4, "locomotion cost")],
        days=7.5, dt=1.5, record_interval=20.0, n_trials=5, seed=42,
        day_start_hour=7.0, day_activity=0.6, night_activity=1.5,
        individual_variation=0.15, enable_mortality=True,
        energy_speed_coupling=0.4, rest_speed_factor=0.2,
        policy=PolicyParams(k_home=1.7, forage_releases_home=0.5,
                            k_resource=2.2, k_social=0.8, k_territory=3.0,
                            k_random=0.9, perception_r=0.8,
                            forage_threshold=0.4),
        trial_prefix="X", start_datetime="2026-03-01T09:30:00",
        parallel=True, n_workers=4,
    )


def test_maximal_roundtrip():
    cfg = _maximal_config()
    d1 = cfg.to_dict()
    cfg2 = ExperimentConfig.from_dict(d1)
    assert cfg2.to_dict() == d1, "round-trip lost or mutated fields"


def test_policy_survives_roundtrip():
    cfg = _maximal_config()
    cfg2 = ExperimentConfig.from_dict(cfg.to_dict())
    assert cfg2.policy.k_territory == 3.0
    assert cfg2.policy.forage_releases_home == 0.5


def test_default_vole_roundtrip():
    cfg = default_vole_experiment()
    assert ExperimentConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()
