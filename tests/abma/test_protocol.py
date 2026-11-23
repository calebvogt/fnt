"""Protocol timeline: timed add/remove of animals and resources."""
import numpy as np
import pandas as pd
import pytest

from fnt.abma.core.config import (
    ExperimentConfig, ArenaConfig, ResourceObject, ProtocolEvent,
    AgentGroup, Genotype, Treatment, TraitProfile,
)
from fnt.abma.core.simulation import Simulation

DAY = 86400.0


def _cfg(protocol=(), n_groups=1):
    arena = ArenaConfig(width=2.0, height=2.0, objects=[
        ResourceObject("food", 1.5, 1.5, 0.2, label="chow_A"),
        ResourceObject("water", 0.5, 1.5, 0.2, label="water_A")])
    groups = [AgentGroup(f"g{i}", "prairie", "MF"[i % 2], 4,
                         Genotype({}), Treatment(), TraitProfile())
              for i in range(n_groups)]
    return ExperimentConfig(arena=arena, groups=groups, seed=3,
                            protocol=list(protocol), days=2.0)


def _run_until(sim, start_day, end_day, dt=600.0):
    """Step sim from start_day to end_day of continuous simulated time."""
    t = start_day * DAY
    while t < end_day * DAY:
        sim.step(t, dt)
        t += dt


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #
def test_protocol_roundtrip():
    ev = [
        ProtocolEvent(at_day=1.0, kind="add_agents",
                      group=AgentGroup("late_M", "prairie", "M", 2),
                      label="introduce males"),
        ProtocolEvent(at_day=1.5, kind="remove_agents", target="g0", count=2),
        ProtocolEvent(at_day=0.5, kind="add_resource",
                      object=ResourceObject("food", 0.5, 0.5, 0.2,
                                            label="chow_B")),
        ProtocolEvent(at_day=1.2, kind="remove_resource", target="chow_A"),
    ]
    cfg = _cfg(ev)
    d = cfg.to_dict()
    cfg2 = ExperimentConfig.from_dict(d)
    assert cfg2.to_dict() == d
    assert cfg2.protocol[0].group.label == "late_M"
    assert cfg2.protocol[2].object.label == "chow_B"


# --------------------------------------------------------------------------- #
# Add agents
# --------------------------------------------------------------------------- #
def test_add_agents_mid_run():
    ev = ProtocolEvent(at_day=0.5, kind="add_agents",
                       group=AgentGroup("late_M", "prairie", "M", 3))
    sim = Simulation(_cfg([ev]), 0)
    assert sim.n == 4
    _run_until(sim, 0.0, 0.4)
    assert sim.n == 4, "added too early"
    _run_until(sim, 0.4, 0.6)     # crosses day 0.5
    assert sim.n == 7
    # every per-agent array grew together
    for arr in (sim.P, sim.home, sim.agent_rgba):
        assert len(arr) == 7
    for arr in (sim.H, sim.hunger, sim.alive, sim.aggr, sim.mass, sim.mass0,
                sim.sex_m, sim.agent_size, sim.agent_shape, sim.activity):
        assert len(arr) == 7
    # identities are unique and continue the sequence
    ids = [a.shortid for a in sim.agents]
    assert len(set(ids)) == 7
    assert [a.group for a in sim.agents[4:]] == ["late_M"] * 3
    # the newcomers move and stay in-bounds
    _run_until(sim, 0.6, 0.8)
    assert np.isfinite(sim.P).all()


def test_added_agents_recorded_in_frame():
    ev = ProtocolEvent(at_day=0.25, kind="add_agents",
                       group=AgentGroup("late_F", "prairie", "F", 2))
    sim = Simulation(_cfg([ev]), 0)
    _run_until(sim, 0.0, 0.5)
    fr = sim._frame(0.5 * DAY)
    assert len(fr["x"]) == 6
    assert len(fr["color"]) == 6


# --------------------------------------------------------------------------- #
# Remove agents
# --------------------------------------------------------------------------- #
def test_remove_agents_by_group():
    cfg = _cfg([ProtocolEvent(at_day=0.5, kind="remove_agents", target="g0")],
               n_groups=2)
    sim = Simulation(cfg, 0)
    assert sim.n == 8
    _run_until(sim, 0.0, 0.6)
    assert sim.n == 8, "removal must not delete rows (identity is preserved)"
    removed = [a for a in sim.agents if a.removed]
    assert len(removed) == 4
    assert all(a.group == "g0" for a in removed)
    assert (~sim.alive[[a.index for a in removed]]).all()
    # removed animals stop moving
    P0 = sim.P.copy()
    _run_until(sim, 0.6, 0.8)
    still = [a.index for a in removed]
    assert np.array_equal(sim.P[still], P0[still])


def test_remove_agents_count_limited():
    cfg = _cfg([ProtocolEvent(at_day=0.5, kind="remove_agents",
                              target="all", count=3)])
    sim = Simulation(cfg, 0)
    _run_until(sim, 0.0, 0.6)
    assert int((~sim.alive).sum()) == 3


# --------------------------------------------------------------------------- #
# Add / remove resources
# --------------------------------------------------------------------------- #
def test_add_and_remove_resource():
    ev = [ProtocolEvent(at_day=0.3, kind="add_resource",
                        object=ResourceObject("food", 0.5, 0.5, 0.2,
                                              label="chow_B")),
          ProtocolEvent(at_day=0.6, kind="remove_resource", target="chow_A")]
    sim = Simulation(_cfg(ev), 0)
    assert len(sim.food) == 1
    _run_until(sim, 0.0, 0.4)
    assert len(sim.food) == 2, "provisioned food did not appear"
    _run_until(sim, 0.4, 0.7)
    assert len(sim.food) == 1, "removed food is still present"
    # the remaining pile is chow_B at (0.5, 0.5)
    assert np.allclose(sim.food[0], [0.5, 0.5])


def test_remove_all_of_kind():
    sim = Simulation(_cfg([ProtocolEvent(at_day=0.2, kind="remove_resource",
                                         target="food")]), 0)
    _run_until(sim, 0.0, 0.3)
    assert len(sim.food) == 0
    assert len(sim.water) == 1, "water must be untouched"


def test_resource_events_do_not_mutate_shared_config():
    cfg = _cfg([ProtocolEvent(at_day=0.2, kind="remove_resource",
                              target="chow_A")])
    n_before = len(cfg.arena.objects)
    sim = Simulation(cfg, 0)
    _run_until(sim, 0.0, 0.3)
    assert len(cfg.arena.objects) == n_before, \
        "protocol must act on sim-local copies (config is shared across trials)"


# --------------------------------------------------------------------------- #
# End-to-end: output files stay schema-stable through roster changes
# --------------------------------------------------------------------------- #
def test_full_run_with_protocol(tmp_path):
    ev = [ProtocolEvent(at_day=0.25, kind="add_agents",
                        group=AgentGroup("late_M", "prairie", "M", 2)),
          ProtocolEvent(at_day=0.75, kind="remove_agents", target="g0",
                        count=2)]
    cfg = _cfg(ev)
    cfg.days = 1.0
    cfg.dt = 300.0
    cfg.record_interval = 600.0
    sim = Simulation(cfg, 0)
    res = sim.run(str(tmp_path))

    traj = pd.read_csv(res["trajectory"])
    ids_early = set(traj[traj["time_sec"] <= traj["time_sec"].min() + 3600]
                    ["sexid"])
    ids_all = set(traj["sexid"])
    assert len(ids_early) == 4
    assert len(ids_all) == 6, "introduced agents never hit the trajectory"

    ev_df = pd.read_csv(res["events"])
    assert (ev_df["event"] == "release").sum() == 2
    assert (ev_df["event"] == "removal").sum() == 2

    agents = pd.read_csv(res["agents"])
    assert len(agents) == 6, "agents table not refreshed after introduction"

    cond = pd.read_csv(res["condition"])
    assert "removed" in set(cond["status"]), \
        "removed animals should be distinguishable from dead ones"
