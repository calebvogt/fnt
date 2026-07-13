"""Streaming writers that emit ABMA output in FNT's canonical formats.

The trajectory file matches, column-for-column, the schema produced by FNT's UWB
preprocessing (``uwb_<trial>_processed.csv``)::

    Trial, Species, sex, sexid, shortid, Date, Day, Timestamp, time_sec,
    location_x, location_y, smoothed_x, smoothed_y, Meso1Start

Because the schema is identical, simulated trials are a drop-in substitute for
real tracking data: proximity detection, daily edgelists, GBI, and network
metrics all run without modification.
"""
from __future__ import annotations

import csv
import os
from datetime import datetime, timedelta, timezone

TRAJ_HEADER = [
    "Trial", "Species", "sex", "sexid", "shortid", "Date", "Day",
    "Timestamp", "time_sec", "location_x", "location_y",
    "smoothed_x", "smoothed_y", "Meso1Start",
]

EVENT_HEADER = [
    "Trial", "Day", "Timestamp", "time_sec", "event", "actor", "target",
    "actor_sex", "target_sex", "location_x", "location_y", "value",
]


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + \
        f"{dt.microsecond // 1000:03d}Z"


class TrajectoryRecorder:
    """Streams one row per agent per record-tick to a processed-format CSV."""

    def __init__(self, path: str, trial_id: str, start_dt: datetime, agents):
        self.path = path
        self.trial_id = trial_id
        self.start_dt = start_dt if start_dt.tzinfo else start_dt.replace(tzinfo=timezone.utc)
        self.meso_start = _iso_z(self.start_dt)
        self.agents = agents  # list[AgentMeta]
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._fh = open(path, "w", newline="")
        self._w = csv.writer(self._fh)
        self._w.writerow(TRAJ_HEADER)

    def record(self, elapsed_s: float, x, y):
        """Write a sample for every living agent at simulation time ``elapsed_s``."""
        ts = self.start_dt + timedelta(seconds=elapsed_s)
        iso = _iso_z(ts)
        epoch = int(ts.timestamp())
        day = int(elapsed_s // 86400) + 1
        date = ts.strftime("%Y-%m-%d")
        rows = []
        for i, a in enumerate(self.agents):
            if not a.alive:
                continue
            xi = round(float(x[i]), 6)
            yi = round(float(y[i]), 6)
            rows.append([
                self.trial_id, a.species, a.sex, a.sexid, a.shortid,
                date, day, iso, epoch, xi, yi, xi, yi, self.meso_start,
            ])
        self._w.writerows(rows)

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None


class EventRecorder:
    """Streams social events (mating, aggression) to a tidy CSV."""

    def __init__(self, path: str, trial_id: str, start_dt: datetime):
        self.trial_id = trial_id
        self.start_dt = start_dt if start_dt.tzinfo else start_dt.replace(tzinfo=timezone.utc)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._fh = open(path, "w", newline="")
        self._w = csv.writer(self._fh)
        self._w.writerow(EVENT_HEADER)

    def record(self, elapsed_s, event, actor, target, x, y, value=1.0):
        ts = self.start_dt + timedelta(seconds=elapsed_s)
        day = int(elapsed_s // 86400) + 1
        self._w.writerow([
            self.trial_id, day, _iso_z(ts), int(ts.timestamp()), event,
            actor.sexid, target.sexid if target else "",
            actor.sex, target.sex if target else "",
            round(float(x), 6), round(float(y), 6), value,
        ])

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None


CONDITION_HEADER = [
    "Trial", "Species", "sex", "sexid", "shortid", "Date", "Day", "Timestamp",
    "time_sec", "health", "energy", "hunger", "thirst", "stress", "mass",
    "status",
]


class ConditionRecorder:
    """Streams each agent's condition (0–100 bars, mass in g) over time."""

    def __init__(self, path: str, trial_id: str, start_dt: datetime, agents):
        self.trial_id = trial_id
        self.start_dt = start_dt if start_dt.tzinfo else start_dt.replace(tzinfo=timezone.utc)
        self.agents = agents
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._fh = open(path, "w", newline="")
        self._w = csv.writer(self._fh)
        self._w.writerow(CONDITION_HEADER)

    def record(self, elapsed_s, health, energy, hunger, thirst, stress, mass,
               anosmic):
        ts = self.start_dt + timedelta(seconds=elapsed_s)
        iso = _iso_z(ts)
        epoch = int(ts.timestamp())
        day = int(elapsed_s // 86400) + 1
        date = ts.strftime("%Y-%m-%d")
        rows = []
        for i, a in enumerate(self.agents):
            if not a.alive:
                status = "dead"
            elif anosmic[i]:
                status = "anosmic"
            else:
                status = "ok"
            rows.append([
                self.trial_id, a.species, a.sex, a.sexid, a.shortid, date, day,
                iso, epoch,
                round(float(health[i]) * 100, 1), round(float(energy[i]) * 100, 1),
                round(float(hunger[i]) * 100, 1), round(float(thirst[i]) * 100, 1),
                round(float(stress[i]) * 100, 1), round(float(mass[i]), 2), status,
            ])
        self._w.writerows(rows)

    def close(self):
        if self._fh:
            self._fh.close()
            self._fh = None


def write_agents_table(path: str, agents) -> None:
    """Write per-agent metadata (id, sex, genotype, treatment, traits)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cols = [
        "sexid", "shortid", "species", "sex", "group", "genotype",
        "drug", "dose", "aggression", "boldness", "sociability",
        "exploration", "smell_ability", "identity_signal",
        "base_speed", "home_range_r", "mass_g", "metabolism", "home_x", "home_y",
    ]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for a in agents:
            t = a.traits
            geno = ";".join(f"{k}:{v}" for k, v in (a.genotype.genes or {}).items()) or "WT"
            w.writerow([
                a.sexid, a.shortid, a.species, a.sex, a.group, geno,
                a.treatment.drug, a.treatment.dose,
                round(t.aggression, 3), round(t.boldness, 3),
                round(t.sociability, 3), round(t.exploration, 3),
                round(t.smell_ability, 3), round(t.identity_signal, 3),
                round(t.base_speed, 4), round(t.home_range_r, 3),
                round(t.mass, 2), round(t.metabolism, 3),
                round(a.home[0], 3), round(a.home[1], 3),
            ])


def parse_start(dt_str: str) -> datetime:
    """Parse an ISO-ish start datetime string; assume UTC if naive."""
    s = dt_str.replace("Z", "")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        dt = datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
