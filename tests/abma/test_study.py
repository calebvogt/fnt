"""Study machinery: paired seeds and override paths."""
import numpy as np

from fnt.abma.core.config import default_vole_experiment
from fnt.abma.core.simulation import Simulation
from fnt.abma.core.study import Study, Condition, set_path, get_path


def test_paired_seeds_identical_when_no_override():
    base = default_vole_experiment()
    study = Study(name="s", base=base, replicates=2, seed_policy="paired",
                  conditions=[Condition("a"), Condition("b")])
    cfg_a = study.config_for(0)
    cfg_b = study.config_for(1)
    assert cfg_a.seed == cfg_b.seed
    sa = Simulation(cfg_a, trial_index=0)
    sb = Simulation(cfg_b, trial_index=0)
    for k in range(100):
        sa.step(k * 2.0, 2.0)
        sb.step(k * 2.0, 2.0)
    assert np.array_equal(sa.P, sb.P), \
        "paired replicate 0 must be identical across identical arms"


def test_independent_seeds_differ():
    base = default_vole_experiment()
    study = Study(name="s", base=base, seed_policy="independent",
                  conditions=[Condition("a"), Condition("b")])
    assert study.config_for(0).seed != study.config_for(1).seed


def test_override_path_star():
    base = default_vole_experiment()
    d = base.to_dict()
    set_path(d, "groups[*].traits.smell_ability", 0.0)
    assert all(g["traits"]["smell_ability"] == 0.0 for g in d["groups"])
    assert get_path(d, "groups[0].traits.smell_ability") == 0.0


def test_override_applied_in_condition():
    base = default_vole_experiment()
    study = Study(name="s", base=base, conditions=[
        Condition("anosmic", overrides={"groups[*].traits.smell_ability": 0.0})])
    cfg = study.config_for(0)
    assert all(g.traits.smell_ability == 0.0 for g in cfg.groups)
