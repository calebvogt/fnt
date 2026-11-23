"""Where every number in a run came from, written next to the run.

A simulator that lets you drag a slider until the result looks right is not a
research tool, it is a drawing program. What separates the two is whether the
finished output can still tell you which of its inputs were *measured*, which
came from the literature, which were estimated, and which were simply tuned
until the model behaved. This module writes that down.

Two things get recorded beside every run:

**A content hash of the exact config.** Canonical JSON, SHA-256. It answers
"did this data really come from that config" without trusting a filename, and
it makes an accidental edit to a saved config detectable rather than silent.

**A source class per parameter**, drawn from :data:`fnt.abma.core.project.SOURCES`:

  ``measured``    from the user's own animals
  ``literature``  a published value for this species
  ``estimated``   a defensible order-of-magnitude guess
  ``free``        a weight with no empirical referent, tuned for behaviour
  ``default``     never touched by the experimenter

The classes below are ABMA's honest self-assessment of its own defaults. Body
facts on a species card are ``literature``; the metabolic constants are
``estimated``; every ``k_*`` in the movement policy and every readout constant
in the olfactory model is ``free``. A user who has measured a value in their own
colony overrides the class through the project's provenance map, and that
override is what gets written out.

The point is not to make free parameters go away — a model of behaviour needs
some — but to make a reader able to see, from the run folder alone, exactly how
much of the result rests on them.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

from .project import SOURCES

#: Schema version of the manifest itself.
MANIFEST_VERSION = 1
MANIFEST_NAME = "provenance.json"

#: Source class for each config sub-tree, by dotted path prefix. Longest
#: matching prefix wins, so a specific key overrides its section.
DEFAULT_SOURCES: dict[str, str] = {
    # --- what an experimenter actually sets up --------------------------- #
    "arena": "measured",            # the enclosure is a real, measured place
    "days": "measured",
    "n_trials": "measured",
    "groups": "measured",           # cohort composition is the design
    "protocol": "measured",
    "interventions": "measured",

    # --- biology taken from the literature ------------------------------- #
    "groups.traits.mass": "literature",
    "groups.traits.body_length_cm": "literature",
    "groups.traits.base_speed": "literature",
    "groups.traits.metabolism": "literature",
    "groups.traits.scent_rate": "literature",

    # --- the metabolic model: defensible, not measured here --------------- #
    "physiology": "estimated",
    "dynamics": "estimated",

    # --- weights with no empirical referent ------------------------------- #
    "policy": "free",
    "scent.perception_r": "estimated",
    "scent.half_life_h": "literature",   # mark persistence is measurable
    "scent.deposit_cost": "free",
    "scent.mark_strength": "free",
    "scent.counter_mark": "free",
    "scent.anonymous_weight": "free",
    "scent.cell_size": "default",
    "olfaction.n_channels": "free",
    "olfaction.discrimination": "free",
    "olfaction.confusion_threshold": "free",
    "olfaction.ablation_selectivity": "estimated",

    # --- bookkeeping ------------------------------------------------------ #
    "seed": "default",
    "dt": "default",
    "record_interval": "default",
    "individual_variation": "estimated",
    "energy_speed_coupling": "free",
    "rest_speed_factor": "free",
}


def canonical_json(config) -> str:
    """Stable JSON for hashing: sorted keys, fixed separators."""
    payload = config.to_dict() if hasattr(config, "to_dict") else config
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=str)


def config_hash(config) -> str:
    """SHA-256 of the canonical config — the run's fingerprint."""
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def source_of(path: str, overrides: dict[str, str] | None = None) -> str:
    """Source class for a dotted config path; longest matching prefix wins."""
    table = dict(DEFAULT_SOURCES)
    table.update(overrides or {})
    best, best_len = "default", -1
    for prefix, kind in table.items():
        if (path == prefix or path.startswith(prefix + ".")) \
                and len(prefix) > best_len:
            best, best_len = kind, len(prefix)
    return best


def _walk(node: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Flatten a config dict to (dotted path, leaf value) pairs.

    List indices are dropped from the path: ten agent groups share one
    provenance answer, because they are ten instances of the same kind of
    number, not ten different kinds.
    """
    out: list[tuple[str, Any]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            out.extend(_walk(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(node, (list, tuple)):
        for item in node:
            out.extend(_walk(item, prefix))
    elif prefix:
        out.append((prefix, node))
    return out


def summarize(config, overrides: dict[str, str] | None = None) -> dict:
    """Count parameters by source class, and list the free ones by name.

    The free list is the part worth reading: it is exactly the set of numbers
    a sceptical reader is entitled to ask about.
    """
    counts = {kind: 0 for kind in SOURCES}
    free: set[str] = set()
    seen: set[str] = set()
    for path, _value in _walk(config.to_dict() if hasattr(config, "to_dict")
                              else config):
        if path in seen:
            continue
        seen.add(path)
        kind = source_of(path, overrides)
        counts[kind] = counts.get(kind, 0) + 1
        if kind == "free":
            free.add(path)
    return {"counts": counts, "free_parameters": sorted(free),
            "n_parameters": len(seen)}


def manifest(config, overrides: dict[str, str] | None = None,
             extra: dict | None = None) -> dict:
    """The full provenance record for one run."""
    from .record import RECORD_SCHEMA_VERSION

    try:
        from fnt import __version__ as fnt_version
    except Exception:                     # a source checkout may not expose it
        fnt_version = "unknown"

    got = {
        "manifest_version": MANIFEST_VERSION,
        "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fnt_version": str(fnt_version),
        "config_sha256": config_hash(config),
        "config_schema_version": getattr(config, "schema_version", None),
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "seed": getattr(config, "seed", None),
        "mechanisms": {
            "scent_marking": bool(getattr(getattr(config, "scent", None),
                                          "enabled", False)),
            "physiology": bool(getattr(getattr(config, "physiology", None),
                                       "enabled", False)),
            "mechanistic_olfaction": bool(
                getattr(getattr(config, "olfaction", None), "enabled", False)),
            "mortality": bool(getattr(config, "enable_mortality", False)),
        },
        "parameter_sources": summarize(config, overrides),
        "source_overrides": dict(overrides or {}),
    }
    got.update(extra or {})
    return got


def write_manifest(directory: str, config,
                   overrides: dict[str, str] | None = None,
                   extra: dict | None = None) -> str:
    """Write ``provenance.json`` into ``directory`` and return its path."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, MANIFEST_NAME)
    with open(path, "w") as fh:
        json.dump(manifest(config, overrides, extra), fh, indent=2)
    return path


def verify(directory: str, config) -> tuple[bool, str]:
    """Check a run folder's manifest against ``config``.

    Returns ``(ok, message)``. A mismatch means the config beside the data is
    not the config that produced it — the one failure mode a run folder cannot
    otherwise detect.
    """
    path = os.path.join(directory, MANIFEST_NAME)
    if not os.path.exists(path):
        return False, "no provenance.json in this run folder"
    with open(path) as fh:
        got = json.load(fh)
    want = config_hash(config)
    if got.get("config_sha256") != want:
        return False, ("config does not match the manifest: data was produced "
                       f"by {str(got.get('config_sha256'))[:12]}, this config "
                       f"hashes to {want[:12]}")
    return True, "config matches the run manifest"
