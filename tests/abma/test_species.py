"""Species library: bodies, not behaviours."""
import pytest

from fnt.abma.core.config import AgentGroup, ExperimentConfig, TraitProfile
from fnt.abma.core.biology import genes_for_species
from fnt.abma.core.sampling import parse_spec
from fnt.abma.core.species import (
    SPECIES, get_species, by_name, apply_species, build_group,
)


def test_every_card_is_complete_and_parseable():
    assert len(SPECIES) >= 4
    for sp in SPECIES:
        assert sp.name and sp.latin and sp.summary
        assert sp.activity in ("nocturnal", "diurnal", "crepuscular")
        assert sp.body_length_cm > 0 and sp.base_speed > 0
        assert 0.0 <= sp.smell_ability <= 1.0
        assert 0.0 <= sp.identity_signal <= 1.0
        assert sp.scent_rate > 0
        for sex in ("M", "F"):
            parse_spec(sp.mass_g[sex])          # raises if malformed
        for trait, spec in sp.personality.items():
            assert hasattr(TraitProfile(), trait), f"unknown trait {trait}"
            parse_spec(spec)


def test_no_card_prescribes_a_home_range():
    """Space use is the experiment's result; a card must not preempt it."""
    for sp in SPECIES:
        assert "home_range_r" not in sp.personality
        g = build_group(sp, label="t")
        assert "home_range_r" not in g.dists, \
            f"{sp.name} prescribes a home range"


def test_apply_species_sets_body_facts():
    sp = get_species("house_mouse")
    g = AgentGroup(label="mice", sex="M")
    apply_species(g, sp)
    assert g.traits.body_length_cm == sp.body_length_cm
    assert g.traits.base_speed == sp.base_speed
    assert g.traits.scent_rate == sp.scent_rate
    assert g.traits.metabolism == sp.metabolism
    assert g.species == sp.name
    assert g.dists["mass"] == sp.mass_g["M"]


def test_mass_distribution_is_sex_specific():
    sp = get_species("prairie_vole")
    m = build_group(sp, "m", sex="M")
    f = build_group(sp, "f", sex="F")
    assert m.dists["mass"] != f.dists["mass"]
    assert parse_spec(m.dists["mass"]).mean > parse_spec(f.dists["mass"]).mean


def test_personality_can_be_preserved_when_swapping_body():
    sp_a, sp_b = get_species("prairie_vole"), get_species("house_mouse")
    g = build_group(sp_a, "t")
    g.dists["aggression"] = "U(0.9,1.0)"          # experimenter's own value
    apply_species(g, sp_b, personality=False)
    assert g.dists["aggression"] == "U(0.9,1.0)", \
        "swapping the body must not clobber tuned dispositions"
    assert g.traits.scent_rate == sp_b.scent_rate  # body did change


def test_body_length_drives_drawn_size():
    mouse = build_group(get_species("house_mouse"), "m")
    vole = build_group(get_species("prairie_vole"), "v")
    assert vole.appearance.size > mouse.appearance.size, \
        "a 13 cm vole should be drawn larger than an 8.5 cm mouse"


def test_species_group_roundtrips_through_config():
    groups = [build_group(sp, sp.key) for sp in SPECIES]
    cfg = ExperimentConfig(groups=groups)
    d = cfg.to_dict()
    assert ExperimentConfig.from_dict(d).to_dict() == d


@pytest.mark.parametrize("name,expected", [
    ("Prairie vole", "OXTR"), ("Meadow vole", "OXTR"),
    ("House mouse", "MUP"), ("Deer mouse", "MUP"),
])
def test_gene_lookup_matches_library_names(name, expected):
    assert expected in genes_for_species(name)


def test_gene_lookup_tolerates_free_text_and_unknowns():
    assert "OXTR" in genes_for_species("prairie")
    assert genes_for_species("Generic rodent") == []
    assert genes_for_species("") == []


def test_lookup_helpers():
    assert by_name("House mouse").key == "house_mouse"
    assert get_species("nope") is None
