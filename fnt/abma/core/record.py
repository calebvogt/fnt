"""The run record — one self-describing archive you can replay and inspect.

What this is for
----------------
ABMA already writes CSVs, and those stay exactly as they are: they are the
contract with FNT's real UWB pipeline and nothing here touches them. But a CSV
of positions cannot answer the question an in-silico experiment is actually
for — *why* did this animal go there. The engine knows: the policy computes a
named set of competing drives every step and then throws the decomposition away
when it sums them.

This module keeps it. Each recorded frame stores, per animal, its position and
heading, its condition bars, the magnitude of every drive acting on it, what it
was smelling, and what it was doing. A finished run can then be reopened and
scrubbed: pick an animal, watch its trajectory, and read the motivation
underneath it.

Self-describing
---------------
The archive ships its own field names and schema version, so a record written
today is still readable when fields are added later: a reader looks up columns
by name rather than by position. Append to :data:`VALUE_FIELDS` or
:data:`STATE_FIELDS`; never renumber or reorder them.

Bounded by construction
-----------------------
A 10-day run at a 5-minute frame interval is ~2900 frames — a couple of MB for
a small cohort. A long run with a large cohort is not. Rather than truncate
(which loses the end of the experiment, usually the interesting part) the
archive *decimates*: when it fills, it drops every other frame and halves its
sampling rate, so the record always spans the whole run and simply gets coarser.
``frame_interval_s`` on the archive reports the rate actually in force.
"""
from __future__ import annotations

import json
import os

import numpy as np

#: Bumped when the meaning of an existing field changes. Adding a field to the
#: end of either list does NOT need a bump — readers look fields up by name.
RECORD_SCHEMA_VERSION = 1

#: Continuous per-agent, per-frame quantities (stored float32).
VALUE_FIELDS: list[str] = [
    # where it is and where it is pointing
    "x", "y", "heading",
    # what it is trying to do: the named drives whose sum is the desired
    # heading, as magnitudes. This is the part a plain trajectory loses.
    "drive_scent_home", "drive_memory", "drive_home", "drive_resource",
    "drive_social", "drive_territory", "drive_wander",
    "desired_x", "desired_y",
    # what it is sensing
    "scent_own", "scent_foreign", "recognition_mean", "detection_mean",
    "need_food", "need_water", "neighbours",
    # how it is doing
    "health", "energy", "hunger", "thirst", "stress", "mass", "bladder",
    # what it has accumulated
    "dist_today", "territory_m2",
    # the sward it is standing in, and the trail work it has done
    "grass_cm", "grass_speed_factor", "chewing", "grass_cut_cm",
    # the sky overhead (constant across the cohort, stored per agent so the
    # record stays one rectangular table rather than two)
    "sun_elevation", "moon_elevation", "moon_illumination", "daylight",
    "night_light",
]

#: Small integer per-agent, per-frame quantities (stored int16/int32).
STATE_FIELDS: list[str] = [
    "activity",       # 0 rest 1 forage 2 roam 3 flee 4 mate 5 dead
    "alive",
    "fights_won", "fights_lost", "matings", "marks_made",
]

#: Per-frame scalars shared by the whole cohort.
FRAME_FIELDS: list[str] = ["elapsed", "day", "hour", "is_day"]

#: default archive ceiling; see the module docstring on decimation
DEFAULT_CAP_BYTES = 256 * 1024 * 1024

_VALUE_INDEX = {name: i for i, name in enumerate(VALUE_FIELDS)}
_STATE_INDEX = {name: i for i, name in enumerate(STATE_FIELDS)}


def layout() -> dict:
    """The archive's self-description, written alongside the data."""
    return {
        "schema_version": RECORD_SCHEMA_VERSION,
        "value_fields": list(VALUE_FIELDS),
        "state_fields": list(STATE_FIELDS),
        "frame_fields": list(FRAME_FIELDS),
        "value_dtype": "float32",
        "state_dtype": "int32",
    }


class RunRecord:
    """A growable, memory-bounded archive of per-agent frames.

    ``append(frame)`` takes the same dict :meth:`Simulation._frame` already
    emits to the live view, so recording is a tee on an existing signal rather
    than a second traversal of the simulation.
    """

    def __init__(self, trial_id: str = "", n_agents: int = 0,
                 frame_interval_s: float = 300.0,
                 cap_bytes: int = DEFAULT_CAP_BYTES, agents: list | None = None):
        self.trial_id = trial_id
        self.frame_interval_s = float(frame_interval_s)
        self.cap_bytes = int(cap_bytes)
        #: static per-agent identity, as `Simulation.agent_static()` returns it
        self.agents = list(agents or [])
        self._n_agents = max(1, int(n_agents) or len(self.agents) or 1)
        self._cap_frames = max(
            8, self.cap_bytes // max(1, self._row_bytes(self._n_agents)))
        self._values = np.zeros((0, self._n_agents, len(VALUE_FIELDS)),
                                np.float32)
        self._states = np.zeros((0, self._n_agents, len(STATE_FIELDS)),
                                np.int32)
        self._frames = np.zeros((0, len(FRAME_FIELDS)), np.float64)
        self._n = 0
        #: how many source frames each stored frame now stands for (doubles on
        #: every decimation) — the honest sampling rate of this archive
        self.decimation = 1
        self._since_kept = 0

    # ---- sizing --------------------------------------------------------- #
    @staticmethod
    def _row_bytes(n_agents: int) -> int:
        return n_agents * (len(VALUE_FIELDS) * 4 + len(STATE_FIELDS) * 4) + 32

    def __len__(self) -> int:
        return self._n

    @property
    def n_agents(self) -> int:
        return self._n_agents

    def nbytes(self) -> int:
        return int(self._values.nbytes + self._states.nbytes
                   + self._frames.nbytes)

    def sample_interval_s(self) -> float:
        """Seconds between stored frames, after any decimation."""
        return self.frame_interval_s * self.decimation

    # ---- writing -------------------------------------------------------- #
    def append(self, frame: dict) -> None:
        """Store one frame, decimating first if the archive is full."""
        self._since_kept += 1
        if self._since_kept < self.decimation:
            return
        self._since_kept = 0
        if self._n >= self._cap_frames:
            self._decimate()
        n_in = len(frame.get("x", ()))
        if n_in > self._n_agents:
            self._grow_agents(n_in)
        self._ensure_capacity()

        vals = self._values[self._n]
        vals[:] = 0.0
        states = self._states[self._n]
        states[:] = 0

        def put(name, value):
            if value is None:
                return
            arr = np.asarray(value, np.float32).ravel()
            if arr.size == 1:
                # a cohort-wide scalar (the sun's elevation, say) is stored on
                # every row, so the archive stays one rectangular table and a
                # reader never has to know which fields are per-animal
                vals[:, _VALUE_INDEX[name]] = arr[0]
            else:
                vals[:len(arr), _VALUE_INDEX[name]] = arr

        def put_state(name, value):
            if value is None:
                return
            arr = np.asarray(value).astype(np.int32).ravel()
            if arr.size == 1:
                states[:, _STATE_INDEX[name]] = arr[0]
            else:
                states[:len(arr), _STATE_INDEX[name]] = arr

        for name in VALUE_FIELDS:
            if name in frame:
                put(name, frame[name])
        for name in STATE_FIELDS:
            if name in frame:
                put_state(name, frame[name])

        self._frames[self._n] = [
            float(frame.get("elapsed", 0.0)), float(frame.get("day", 0)),
            float(frame.get("hour", 0.0)), float(bool(frame.get("is_day", 1))),
        ]
        self._n += 1

    def _ensure_capacity(self) -> None:
        if self._n < len(self._values):
            return
        grow = max(64, len(self._values))
        pad_v = np.zeros((grow, self._n_agents, len(VALUE_FIELDS)), np.float32)
        pad_s = np.zeros((grow, self._n_agents, len(STATE_FIELDS)), np.int32)
        pad_f = np.zeros((grow, len(FRAME_FIELDS)), np.float64)
        self._values = np.concatenate([self._values, pad_v])
        self._states = np.concatenate([self._states, pad_s])
        self._frames = np.concatenate([self._frames, pad_f])

    def _grow_agents(self, n_agents: int) -> None:
        """Widen the agent axis when a protocol event adds animals mid-run."""
        extra = n_agents - self._n_agents
        self._values = np.concatenate(
            [self._values,
             np.zeros((len(self._values), extra, len(VALUE_FIELDS)),
                      np.float32)], axis=1)
        self._states = np.concatenate(
            [self._states,
             np.zeros((len(self._states), extra, len(STATE_FIELDS)),
                      np.int32)], axis=1)
        self._n_agents = n_agents
        self._cap_frames = max(8, self.cap_bytes
                               // max(1, self._row_bytes(n_agents)))

    def _decimate(self) -> None:
        """Halve the stored sampling rate, keeping the run's full span.

        Thinned from the *end* backwards, so the most recent frame is always
        one of the survivors: an archive that quietly dropped its newest data
        every time it filled would go stale exactly when a long run got
        interesting.
        """
        keep = np.arange(self._n - 1, -1, -2)[::-1]
        kept = len(keep)
        self._values[:kept] = self._values[keep]
        self._states[:kept] = self._states[keep]
        self._frames[:kept] = self._frames[keep]
        self._n = kept
        self.decimation *= 2

    # ---- reading -------------------------------------------------------- #
    def frame(self, index: int) -> dict:
        """One stored frame, in the same shape the live view consumes."""
        i = int(np.clip(index, 0, max(0, self._n - 1)))
        vals, states = self._values[i], self._states[i]
        out = {name: vals[:, k].astype(np.float64)
               for name, k in _VALUE_INDEX.items()}
        out.update({name: states[:, k] for name, k in _STATE_INDEX.items()})
        elapsed, day, hour, is_day = self._frames[i]
        out.update(trial=self.trial_id, elapsed=float(elapsed), day=int(day),
                   hour=float(hour), is_day=bool(is_day), index=i)
        return out

    def view_frame(self, index: int) -> dict:
        """A stored frame plus the display fields the arena views expect.

        Colour, size and sex are properties of the *animal*, not of the moment,
        so they live in the archive's agent table rather than being stored
        again for every frame. Rebuilding them here is what lets a finished run
        be reopened and replayed through the same view code that drew it live.
        """
        out = self.frame(index)
        n = self._n_agents
        sex_m = np.zeros(n)
        colour = np.zeros((n, 4))
        for meta in self.agents:
            i = int(meta.get("index", -1))
            if not (0 <= i < n):
                continue
            male = str(meta.get("sex", "M")) == "M"
            sex_m[i] = 1.0 if male else 0.0
            colour[i] = ((0.29, 0.56, 0.85, 1.0) if male
                         else (0.88, 0.33, 0.60, 1.0))
        out.update(sex_m=sex_m, color=colour, size=np.ones(n),
                   shape=np.zeros(n, int),
                   anosmic=out["detection_mean"] < 0.5,
                   estrus=np.zeros(n, bool))
        return out

    def series(self, agent: int, field: str, start: int = 0,
               stop: int | None = None) -> np.ndarray:
        """One agent's history of one field — the trace the GUI plots."""
        stop = self._n if stop is None else min(stop, self._n)
        if field in _VALUE_INDEX:
            return self._values[start:stop, agent,
                                _VALUE_INDEX[field]].astype(np.float64)
        if field in _STATE_INDEX:
            return self._states[start:stop, agent,
                                _STATE_INDEX[field]].astype(np.float64)
        raise KeyError(f"unknown record field {field!r}")

    def times(self) -> np.ndarray:
        """Elapsed seconds for every stored frame."""
        return self._frames[:self._n, 0].copy()

    # ---- persistence ---------------------------------------------------- #
    def save(self, path: str) -> str:
        """Write the archive plus its layout to a single ``.npz``."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez_compressed(
            path,
            values=self._values[:self._n], states=self._states[:self._n],
            frames=self._frames[:self._n],
            meta=np.array(json.dumps({
                "layout": layout(), "trial_id": self.trial_id,
                "frame_interval_s": self.frame_interval_s,
                "decimation": self.decimation, "agents": self.agents,
            })))
        return path

    @staticmethod
    def load(path: str) -> "RunRecord":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            values, states, frames = z["values"], z["states"], z["frames"]
        got = meta.get("layout", {})
        version = int(got.get("schema_version", 0))
        if version > RECORD_SCHEMA_VERSION:
            raise ValueError(
                f"record schema v{version} is newer than this build "
                f"(v{RECORD_SCHEMA_VERSION}); update FNT to read it")
        rec = RunRecord(trial_id=meta.get("trial_id", ""),
                        n_agents=values.shape[1] if values.size else 1,
                        frame_interval_s=meta.get("frame_interval_s", 300.0),
                        agents=meta.get("agents", []))
        rec.decimation = int(meta.get("decimation", 1))
        # Remap by NAME so an archive written before a field existed still
        # loads: unknown columns are dropped, missing ones stay zero.
        rec._values = np.zeros((len(frames), values.shape[1],
                                len(VALUE_FIELDS)), np.float32)
        rec._states = np.zeros((len(frames), states.shape[1],
                                len(STATE_FIELDS)), np.int32)
        for src, name in enumerate(got.get("value_fields", [])):
            if name in _VALUE_INDEX:
                rec._values[:, :, _VALUE_INDEX[name]] = values[:, :, src]
        for src, name in enumerate(got.get("state_fields", [])):
            if name in _STATE_INDEX:
                rec._states[:, :, _STATE_INDEX[name]] = states[:, :, src]
        rec._frames = frames.astype(np.float64)
        rec._n = len(frames)
        rec._n_agents = rec._values.shape[1]
        # A loaded archive must never decimate itself just because it is bigger
        # than the default in-memory ceiling — it is already on disk at this
        # resolution, and thinning it on open would silently lose samples.
        rec._cap_frames = max(rec._cap_frames, rec._n)
        return rec
