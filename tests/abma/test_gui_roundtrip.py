"""GUI config round-trip: load → collect must preserve everything.

Runs the real ABMAWindow offscreen (no display needed). Guards against widget
plumbing dropping config fields — PolicyParams were once lost exactly here.
"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt5")


@pytest.fixture(scope="module")
def qapp():
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_policy_survives_gui_roundtrip(qapp):
    from fnt.abma.gui.abma_main_pyqt import ABMAWindow
    from fnt.abma.core.config import default_vole_experiment, PolicyParams

    cfg = default_vole_experiment()
    cfg.policy = PolicyParams(k_home=1.9, k_territory=3.5,
                              forage_releases_home=0.42, perception_r=0.9)
    win = ABMAWindow()
    win._load_config(cfg)
    out = win._collect_config()
    assert out.policy.k_home == 1.9
    assert out.policy.k_territory == 3.5
    assert out.policy.forage_releases_home == 0.42
    assert out.policy.perception_r == 0.9


def test_groups_and_scalars_survive_gui_roundtrip(qapp):
    from fnt.abma.gui.abma_main_pyqt import ABMAWindow
    from fnt.abma.core.config import default_vole_experiment

    cfg = default_vole_experiment()
    cfg.days = 6.5
    cfg.seed = 123
    win = ABMAWindow()
    win._load_config(cfg)
    out = win._collect_config()
    assert out.days == 6.5
    assert out.seed == 123
    assert [g.label for g in out.groups] == [g.label for g in cfg.groups]
    assert out.groups[0].dists == cfg.groups[0].dists


def test_scent_params_survive_gui_roundtrip(qapp):
    from fnt.abma.gui.abma_main_pyqt import ABMAWindow
    from fnt.abma.core.config import default_vole_experiment, ScentParams

    cfg = default_vole_experiment()
    cfg.scent = ScentParams(enabled=True, half_life_h=6.0, perception_r=0.75,
                            cell_size=0.05, anonymous_weight=0.4,
                            deposit_cost=0.2, counter_mark=3.0)
    win = ABMAWindow()
    win._load_config(cfg)
    out = win._collect_config()
    assert out.scent.enabled is True
    assert out.scent.half_life_h == 6.0
    assert out.scent.perception_r == 0.75
    assert out.scent.anonymous_weight == 0.4
    # knobs without widgets must ride through untouched
    assert out.scent.deposit_cost == 0.2
    assert out.scent.counter_mark == 3.0


def test_agent_builder_species_card_sets_body_not_home_range(qapp):
    from fnt.abma.gui.abma_main_pyqt import AgentBuilderDialog
    from fnt.abma.core.species import get_species

    dlg = AgentBuilderDialog()
    dlg.in_name.setText("wild_mice")
    dlg.in_sex.setCurrentText("M")
    dlg._on_species("house_mouse")
    sp = get_species("house_mouse")
    g = dlg.result_group()
    assert g.species == sp.name
    assert g.traits.body_length_cm == sp.body_length_cm
    assert g.traits.scent_rate == sp.scent_rate
    assert g.dists["mass"] == sp.mass_g["M"]
    assert "home_range_r" not in g.dists, \
        "the builder must not let a species prescribe space use"


def test_agent_builder_sex_switch_updates_mass_distribution(qapp):
    from fnt.abma.gui.abma_main_pyqt import AgentBuilderDialog
    from fnt.abma.core.species import get_species

    dlg = AgentBuilderDialog()
    dlg._on_species("prairie_vole")
    sp = get_species("prairie_vole")
    dlg.in_sex.setCurrentText("F")
    assert dlg._body_fields["mass"].text() == sp.mass_g["F"]


def test_protocol_survives_gui_roundtrip(qapp):
    from fnt.abma.gui.abma_main_pyqt import ABMAWindow
    from fnt.abma.core.config import (default_vole_experiment, ProtocolEvent,
                                      AgentGroup, ResourceObject)

    cfg = default_vole_experiment()
    cfg.protocol = [
        ProtocolEvent(at_day=3.0, kind="add_agents",
                      group=AgentGroup("late_M", "prairie", "M", 2)),
        ProtocolEvent(at_day=5.0, kind="remove_agents", target="saline_F",
                      count=2),
        ProtocolEvent(at_day=6.0, kind="add_resource",
                      object=ResourceObject("food", 0.5, 0.5, 0.2,
                                            label="chow_B")),
    ]
    win = ABMAWindow()
    win._load_config(cfg)
    out = win._collect_config()
    assert len(out.protocol) == 3
    assert out.protocol[0].group.label == "late_M"
    assert out.protocol[1].target == "saline_F"
    assert out.protocol[2].object.label == "chow_B"
    # timeline context offers the scheduled resource for later removal
    ctx = win._protocol_context()
    assert "chow_B" in ctx["resource_labels"]
    assert [g.label for g in ctx["groups"]] == ["saline_F", "saline_M"]
