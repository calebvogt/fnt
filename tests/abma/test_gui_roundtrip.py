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


def test_physiology_survives_gui_roundtrip(qapp):
    from fnt.abma.gui.abma_main_pyqt import ABMAWindow
    from fnt.abma.core.config import default_vole_experiment
    from fnt.abma.core.physiology import preset_params

    cfg = default_vole_experiment()
    cfg.physiology = preset_params("Arid (water-limited)")
    win = ABMAWindow()
    win._load_config(cfg)
    assert win.in_phys_preset.currentText() == "Arid (water-limited)"
    out = win._collect_config()
    assert out.physiology.enabled is True
    assert out.physiology.water_loss_ml_h == cfg.physiology.water_loss_ml_h
    assert out.physiology.bladder_ml_per_g == cfg.physiology.bladder_ml_per_g


def test_food_type_and_amount_survive_gui_roundtrip(qapp):
    from fnt.abma.gui.abma_main_pyqt import ABMAWindow
    from fnt.abma.core.config import default_vole_experiment

    cfg = default_vole_experiment()
    food = [o for o in cfg.arena.objects if o.kind == "food"][0]
    food.food_type = "high_fat"
    food.amount_g = 250.0
    win = ABMAWindow()
    win._load_config(cfg)
    out = win._collect_config()
    got = [o for o in out.arena.objects if o.kind == "food"][0]
    assert got.food_type == "high_fat"
    assert got.amount_g == 250.0
    # non-food rows must not acquire a stray finite pile
    water = [o for o in out.arena.objects if o.kind == "water"][0]
    assert water.amount_g == 0.0


def test_editing_a_rate_marks_the_preset_custom(qapp):
    from fnt.abma.gui.abma_main_pyqt import ABMAWindow
    from fnt.abma.core.config import default_vole_experiment
    from fnt.abma.core.physiology import preset_params

    cfg = default_vole_experiment()
    cfg.physiology = preset_params("Standard")
    win = ABMAWindow()
    win._load_config(cfg)
    assert win.in_phys_preset.currentText() == "Standard"
    win._phys_fields["basal_kj_h"].setValue(9.9)
    assert win.in_phys_preset.currentText() == "Custom"
    assert win._collect_config().physiology.basal_kj_h == 9.9


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


def test_olfaction_settings_survive_the_gui_round_trip(qapp):
    """The mechanistic nose must be reachable without hand-editing JSON."""
    from fnt.abma.gui.abma_main_pyqt import ABMAWindow
    from fnt.abma.core.config import OlfactionParams, default_vole_experiment

    qt_window = ABMAWindow()
    qt_window._load_config(default_vole_experiment())
    cfg = qt_window._collect_config()
    assert cfg.olfaction.enabled is False        # off by default

    qt_window.in_olf_on.setChecked(True)
    qt_window.in_olf_ch.setValue(32)
    qt_window.in_olf_sel.setValue(0.25)
    got = qt_window._collect_config()
    assert got.olfaction == OlfactionParams(
        enabled=True, n_channels=32,
        discrimination=qt_window.in_olf_disc.value(),
        confusion_threshold=qt_window.in_olf_thr.value(),
        ablation_selectivity=0.25)

    qt_window._load_config(got)
    assert qt_window._collect_config().olfaction == got.olfaction
    assert qt_window.in_olf_ch.isEnabled()


# --------------------------------------------------------------------------- #
# Nervous system: rules or a learned brain
# --------------------------------------------------------------------------- #
def _brain_file(tmp_path):
    from fnt.abma.core.brain import Brain
    return Brain.random(3).save(str(tmp_path / "test_brain.npz"))


def test_rules_stay_the_default_nervous_system(qapp):
    from fnt.abma.gui.abma_main_pyqt import ABMAWindow
    from fnt.abma.core.config import default_vole_experiment

    win = ABMAWindow()
    win._load_config(default_vole_experiment())
    assert win.in_brain.currentData() == ""
    assert win._collect_config().brain.enabled is False
    assert not win.in_brain_noise.isEnabled(), \
        "motor noise is a brain setting; it has no meaning for the rules"


def test_brain_choice_survives_the_gui_round_trip(qapp, tmp_path):
    """A config that names a brain must still name it after passing through
    the window — the way PolicyParams once did not."""
    from fnt.abma.gui.abma_main_pyqt import ABMAWindow
    from fnt.abma.core.config import (BrainParams, Stimulation,
                                      default_vole_experiment)

    path = _brain_file(tmp_path)
    cfg = default_vole_experiment()
    cfg.brain = BrainParams(enabled=True, name=path, stochastic=False,
                            noise_scale=0.7)
    cfg.stimulations = [Stimulation(at_day=1.0, unit=9, op="scale", value=0)]
    win = ABMAWindow()
    win._load_config(cfg)
    out = win._collect_config()
    assert out.brain == cfg.brain
    assert out.stimulations == cfg.stimulations, \
        "scheduled stimulation has no editor yet, so it must ride through"
    assert "untrained" in win.lbl_brain.text()
    assert win.in_brain_noise.isEnabled()
    # the live editor preview is driven by the same brain the run will use
    assert type(win._preview_sim.policy).__name__ == "NeuralPolicy"
    # ...and switching back to the rules really switches back
    win.in_brain.setCurrentIndex(0)
    assert win._collect_config().brain.enabled is False


def test_twin_dialog_offers_only_what_this_run_can_do(qapp, tmp_path):
    from fnt.abma.gui.twin_dialog import CHANGES, TwinDialog
    from fnt.abma.core.config import BrainParams, default_vole_experiment

    cfg = default_vole_experiment()
    animals = [(0, "F9001 · saline_F"), (1, "F9002 · saline_F")]
    dlg = TwinDialog(cfg, animals, selected=1)
    assert dlg.in_animal.currentData() == 1
    tw = dlg.tweak()
    assert (tw.kind, tw.name, tw.value, tw.target) == \
        ("state", "hunger", -30.0, "1")
    model = dlg.in_change.model()
    for k, (label, needs_brain, *_rest) in enumerate(CHANGES):
        assert model.item(k).isEnabled() == (not needs_brain), label

    cfg.brain = BrainParams(enabled=True, name=_brain_file(tmp_path))
    dlg = TwinDialog(cfg, animals)
    assert all(dlg.in_change.model().item(k).isEnabled()
               for k in range(len(CHANGES)))
    dlg.in_change.setCurrentIndex(
        [c[0] for c in CHANGES].index("One brain unit silenced"))
    dlg.in_unit.setValue(12)
    dlg.in_seconds.setValue(30)
    tw = dlg.tweak()
    assert (tw.kind, tw.name, tw.op, tw.value, tw.duration_s) == \
        ("unit", "rnn:12", "scale", 0.0, 30.0)
    assert not dlg.in_unit.isHidden() and dlg.in_amount.isHidden()


def test_twin_dialog_runs_and_reports(qapp):
    """End to end through the dialog's own worker, on a short horizon."""
    from PyQt5.QtCore import QEventLoop, QTimer
    from fnt.abma.gui.twin_dialog import TwinDialog
    from fnt.abma.core.config import default_vole_experiment

    cfg = default_vole_experiment()
    dlg = TwinDialog(cfg, [(0, "F9001 · saline_F")])
    dlg.in_fork.setValue(0.05)
    dlg.in_horizon.setValue(0.25)
    loop = QEventLoop()
    dlg._run()
    dlg.worker.finished.connect(loop.quit)
    QTimer.singleShot(60_000, loop.quit)
    loop.exec_()
    qapp.processEvents()
    assert dlg.result is not None, "the twin run did not finish"
    assert dlg.canvas is not None and dlg.btn_save.isEnabled()
    assert "F9001" in dlg.verdict.text()
