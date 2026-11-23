"""The headless CLI, including the one-command anosmia study.

The flagship experiment has to be runnable on a cluster without the GUI, so
this checks the argument handling and that a study lands its tidy outputs.
"""
from __future__ import annotations

import os

import pytest

from fnt.abma.run_headless import main


def test_write_default_produces_a_loadable_config(tmp_path):
    from fnt.abma.core.config import ExperimentConfig

    path = tmp_path / "cfg.json"
    assert main(["--write-default", str(path)]) == 0
    cfg = ExperimentConfig.from_json(str(path))
    assert cfg.total_agents() == 8


def test_anosmia_study_runs_end_to_end_and_writes_tidy_results(tmp_path):
    out = tmp_path / "studies"
    assert main(["--anosmia-study", "--out", str(out), "--doses", "0,1.0",
                 "--replicates", "1", "--days", "0.02"]) == 0
    root = out / "methimazole_dose_response"
    results = root / "results"
    assert (results / "metrics_long.csv").exists()
    assert (results / "comparison.csv").exists()
    assert (root / "conditions" / "01_saline" / "data").is_dir()
    assert (root / "conditions" / "02_methimazole_1" / "data").is_dir()

    import pandas as pd
    long_df = pd.read_csv(results / "metrics_long.csv")
    assert set(long_df["condition"]) == {"saline", "methimazole_1"}
    assert "mean_dist_MM" in set(long_df["metric"])


def test_the_two_arms_are_paired_by_seed(tmp_path):
    """A paired design is the whole reason the arms are comparable."""
    from fnt.abma.core.study import anosmia_study

    study = anosmia_study(doses=(0.0, 1.0), replicates=3, days=0.02)
    assert study.seed_policy == "paired"
    assert study.config_for(0).seed == study.config_for(1).seed
    assert study.config_for(0).groups[0].treatment.drug == "saline"
    assert study.config_for(1).groups[0].treatment.dose == 1.0


def test_scalar_nose_flag_turns_the_model_off(tmp_path):
    from fnt.abma.core.study import anosmia_study

    assert anosmia_study(mechanistic_nose=True).base.olfaction.enabled
    assert not anosmia_study(mechanistic_nose=False).base.olfaction.enabled


@pytest.mark.parametrize("doses", ["1.0", "not,numbers"])
def test_bad_doses_are_rejected_rather_than_guessed(tmp_path, doses):
    with pytest.raises(SystemExit):
        main(["--anosmia-study", "--out", str(tmp_path), "--doses", doses,
              "--replicates", "1", "--days", "0.01"])


def test_config_without_a_subcommand_is_required():
    with pytest.raises(SystemExit):
        main([])
