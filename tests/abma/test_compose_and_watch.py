"""Describing a run in one line, and watching it without clicking.

The workflow these support is: say what the experiment is, see it built, watch
it on screen, react. So the tests care about two things — that a loose
description resolves to the right config and refuses an ambiguous one, and that
the window can be brought up already running that config.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from fnt.abma.core.compose import (          # noqa: E402
    design, cohort, check, summary, find_species,
)
from fnt.abma.core.presets import find_preset, preset_names   # noqa: E402
from fnt.abma.run_headless import main       # noqa: E402


# --------------------------------------------------------------------------- #
# Resolving a description
# --------------------------------------------------------------------------- #
def test_preset_matches_a_fragment():
    assert "VoleTerra" in find_preset("voleterra").name
    assert "Open Field" in find_preset("open field").name


def test_an_ambiguous_preset_is_an_error_not_a_guess():
    with pytest.raises(ValueError, match="ambiguous"):
        find_preset("enclosure")            # VoleTerra and Liddell Echo both


def test_an_unknown_preset_lists_the_options():
    with pytest.raises(ValueError) as e:
        find_preset("mars colony")
    assert all(name in str(e.value) for name in preset_names()[:2])


def test_species_matches_a_fragment_but_not_ambiguously():
    assert find_species("prairie").key == "prairie_vole"
    assert find_species("Prairie vole").key == "prairie_vole"
    assert find_species("house_mouse").key == "house_mouse"
    with pytest.raises(ValueError, match="ambiguous"):
        find_species("vole")                # prairie and meadow both


# --------------------------------------------------------------------------- #
# Building the config
# --------------------------------------------------------------------------- #
def test_six_males_six_females_five_days_in_voleterra():
    cfg = design(preset="voleterra", males=6, females=6, days=5)
    assert cfg.total_agents() == 12
    assert [(g.sex, g.count) for g in cfg.groups] == [("M", 6), ("F", 6)]
    assert cfg.days == 5.0
    assert cfg.arena.width == pytest.approx(22.86, abs=0.01)   # 75 ft
    assert len(cfg.arena.poles) == 25          # the preset's world is intact
    assert cfg.scent.enabled and cfg.physiology.enabled


def test_bodies_come_from_the_species_card_and_vary():
    cfg = design(preset="voleterra", males=6, females=6)
    males, females = cfg.groups
    assert males.dists["mass"] == "N(45,4)"     # prairie vole card, male
    assert females.dists["mass"] == "N(38,3)"
    assert "aggression" in males.dists          # personality is a spread


def test_home_range_is_never_prescribed():
    """It is what the run produces; setting it would answer the question."""
    cfg = design(preset="voleterra", males=4, females=4)
    for g in cfg.groups:
        assert "home_range_r" not in g.dists


def test_the_preset_alone_is_a_valid_design():
    cfg = design(preset="voleterra")
    assert cfg.total_agents() == 8              # the preset's own cohort
    assert cfg.name == "voleterra"


def test_a_named_run_keeps_its_name():
    assert design(preset="voleterra", males=2, females=2,
                  name="pilot_A").name == "pilot_A"


def test_generated_names_say_what_ran():
    cfg = design(preset="voleterra", males=6, females=6, days=5)
    assert "12agents" in cfg.name and "5d" in cfg.name


def test_treated_and_knockout_cohorts_are_expressible():
    groups = [cohort("prairie", "M", 4, drug="methimazole", dose=1.0,
                     day_offset=-5.0),
              cohort("house mouse", "F", 4, genes={"MUP": "KO"})]
    cfg = design(preset="voleterra", groups=groups, days=2)
    assert cfg.groups[0].treatment.dose == 1.0
    assert cfg.groups[1].genotype.genes == {"MUP": "KO"}


def test_zero_and_zero_means_unspecified_not_empty():
    """``males``/``females`` default to 0, so 0 and 0 cannot mean "no animals".

    It keeps the preset's own cohort, which is what someone who only named a
    world wanted. An actually-empty population is caught by ``check``.
    """
    assert design(preset="voleterra", males=0, females=0).total_agents() == 8


def test_an_empty_population_is_refused():
    with pytest.raises(ValueError, match="no animals"):
        design(groups=[])


@pytest.mark.parametrize("field,value,message", [
    ("days", 0, "days must be"),
    ("n_trials", 0, "trials must be"),
    ("dt", 0, "outside the usable range"),
    ("dt", 120, "outside the usable range"),
])
def test_designs_that_cannot_run_are_refused_up_front(field, value, message):
    cfg = design(preset="voleterra", males=1, females=1)
    setattr(cfg, field, value)
    with pytest.raises(ValueError, match=message):
        check(cfg)


def test_a_record_interval_finer_than_the_timestep_is_refused():
    cfg = design(preset="voleterra", males=2, females=2)
    cfg.record_interval = cfg.dt / 2
    with pytest.raises(ValueError, match="finer than the timestep"):
        check(cfg)


def test_summary_says_what_will_actually_run():
    text = summary(design(preset="voleterra", males=6, females=6, days=5))
    for fragment in ("6 M Prairie vole", "6 F Prairie vole", "5 days",
                     "scent marking", "22.86"):
        assert fragment in text


# --------------------------------------------------------------------------- #
# Bringing it up on screen
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def qapp():
    from PyQt5.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_build_window_loads_the_design_and_arms_the_view(qapp, tmp_path):
    from fnt.abma.gui.launch import build_window

    cfg = design(preset="voleterra", males=3, females=3, days=0.02)
    win = build_window(cfg, out_dir=str(tmp_path))
    try:
        assert len(win.science.roster.cards) == 6
        assert win.in_outdir.text() == str(tmp_path)
        assert win.btn_scent.isChecked()          # territory map on
        assert win.science.selected_index() == 0  # something to show at once
    finally:
        win.close()


def test_watch_starts_the_run_without_a_confirmation_dialog(qapp, tmp_path):
    """A dialog would block a scripted launch forever with nobody to click it."""
    from fnt.abma.gui.launch import watch

    cfg = design(preset="voleterra", males=2, females=2, days=0.01)
    win = watch(cfg, out_dir=str(tmp_path), exec_=False)
    try:
        project_dir = win._resolve_run_dir(cfg, ask=False)
        win._start_run(cfg, project_dir)
        assert win._running
        assert win.worker is not None
        win.worker.cancel()
        win.worker.wait(20000)
    finally:
        win.close()


def test_the_cli_can_describe_a_run_without_running_it(capsys, tmp_path):
    cfg_path = tmp_path / "design.json"
    assert main(["--preset", "voleterra", "--males", "6", "--females", "6",
                 "--days", "5", "--dry-run",
                 "--save-config", str(cfg_path)]) == 0
    printed = capsys.readouterr().out
    assert "6 M Prairie vole" in printed and "5 days" in printed
    assert cfg_path.exists()

    from fnt.abma.core.config import ExperimentConfig
    assert ExperimentConfig.from_json(str(cfg_path)).total_agents() == 12


def test_the_cli_lists_presets(capsys):
    assert main(["--list-presets"]) == 0
    assert "VoleTerra" in capsys.readouterr().out


def test_the_cli_refuses_an_empty_description():
    with pytest.raises(SystemExit):
        main([])


def test_save_config_creates_the_folder_it_needs(tmp_path):
    """A described run usually names a folder that does not exist yet."""
    target = tmp_path / "brand" / "new" / "design.json"
    assert main(["--preset", "voleterra", "--males", "2", "--females", "2",
                 "--dry-run", "--save-config", str(target)]) == 0
    assert target.exists()
