"""Describe a run the way you'd say it out loud, get an ExperimentConfig.

"VoleTerra, six males and six females, five days" is a complete experimental
description, but turning it into a config previously meant knowing which preset
factory to call, that a species card is stamped onto a group with
``apply_species``, that mass comes from a distribution spec rather than a
number, and that ``home_range_r`` must be left alone when scent marking is on.
That is a lot of internals to remember correctly for something you can say in
one sentence, and it is exactly the kind of setup that gets quietly wrong.

This module is that sentence::

    cfg = design(preset="voleterra", males=6, females=6, days=5)

Everything it does is something you could do by hand; the value is that it does
it the same way every time, refuses ambiguous input instead of guessing, and
leaves the parts an experiment is supposed to discover alone.

What it deliberately does not do
--------------------------------
It never sets home-range size, territory radius or anything else a run is meant
to *produce* — see :mod:`fnt.abma.core.species`. A cohort's bodies come from the
species card and its personality from that card's suggested distributions, so
founders vary the way a real cohort does rather than being identical clones.
"""
from __future__ import annotations

import copy
from datetime import datetime

from .config import AgentGroup, ExperimentConfig, Genotype, Treatment
from .sky import day_length_hours, season_of, season_start
from .species import Species, by_name, build_group, species_names


def find_species(query: str) -> Species:
    """Resolve a species by name, tolerantly. Raises with the options listed.

    Accepts the display name ("Prairie vole"), the key ("prairie_vole"), or any
    unambiguous fragment ("prairie", "deer"). Ambiguity is an error rather than
    a silent pick: "vole" matches three cards and guessing which one an
    experiment meant is not this function's business.
    """
    from .species import SPECIES

    q = (query or "").strip().lower().replace("_", " ")
    if not q:
        raise ValueError(f"no species given; choose one of {species_names()}")
    exact = by_name(query) or next(
        (s for s in SPECIES if s.key.lower() == query.strip().lower()), None)
    if exact is not None:
        return exact
    hits = [s for s in SPECIES
            if q in s.name.lower() or q in s.key.lower().replace("_", " ")]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise ValueError(
            f"no species matches {query!r}; choose one of {species_names()}")
    raise ValueError(
        f"{query!r} is ambiguous — it matches "
        f"{[s.name for s in hits]}. Say which one.")


def cohort(species, sex: str, count: int, label: str = "",
           drug: str = "none", dose: float = 0.0, day_offset: float = 0.0,
           genes: dict | None = None) -> AgentGroup:
    """One agent type: N animals of a species and sex, optionally treated.

    Bodies come from the species card and personality from its suggested
    distributions, so founders vary rather than being clones. ``genes`` is a
    ``{"OXTR": "KO"}``-style mapping; ``drug``/``dose``/``day_offset`` deliver a
    treatment (a negative offset means "dosed before release", as in the
    wet-lab protocol).
    """
    sp = species if isinstance(species, Species) else find_species(species)
    if sex not in ("M", "F"):
        raise ValueError(f"sex must be 'M' or 'F', got {sex!r}")
    if int(count) < 0:
        raise ValueError(f"count must be >= 0, got {count}")
    name = label or ("males" if sex == "M" else "females")
    group = build_group(sp, name, sex=sex, count=int(count))
    group.genotype = Genotype(dict(genes or {}))
    group.treatment = Treatment(drug=drug, dose=float(dose),
                                day_offset=float(day_offset))
    return group


def design(preset: str | None = None, base: ExperimentConfig | None = None,
           males: int = 0, females: int = 0, species: str = "Prairie vole",
           days: float | None = None, trials: int | None = None,
           seed: int | None = None, name: str | None = None,
           groups: list[AgentGroup] | None = None,
           record_interval: float | None = None,
           dt: float | None = None, season: str | None = None,
           year: int = 2026) -> ExperimentConfig:
    """Build a runnable config from a plain description of the experiment.

    ``preset`` names the world (matched loosely — "voleterra" finds the
    75x75 ft enclosure); ``base`` supplies one directly instead. ``males`` and
    ``females`` replace the preset's suggested cohort with your own, keeping
    that preset's arena, mechanisms and protocol intact. Pass ``groups`` to
    specify the population fully (mixed species, treated arms, knockouts).

    ``season`` sets the release date — "winter", "spring", "summer", "fall" —
    which at a site with a real latitude means day length, solar angle and
    grass growth all change together. That is the point of asking for a season
    rather than a date: outdoors those things are not independent.

    Anything left as ``None`` keeps the preset's value, so
    ``design(preset="voleterra")`` is just the preset.
    """
    from .presets import find_preset

    if base is not None and preset is not None:
        raise ValueError("give either preset= or base=, not both")
    cfg = (copy.deepcopy(base) if base is not None
           else find_preset(preset).factory() if preset is not None
           else ExperimentConfig())

    if groups is not None:
        cfg.groups = list(groups)
    elif males or females:
        sp = find_species(species)
        cfg.groups = [g for g in (
            cohort(sp, "M", males) if males else None,
            cohort(sp, "F", females) if females else None) if g is not None]

    if days is not None:
        cfg.days = float(days)
    if trials is not None:
        cfg.n_trials = int(trials)
    if seed is not None:
        cfg.seed = int(seed)
    if dt is not None:
        cfg.dt = float(dt)
    if record_interval is not None:
        cfg.record_interval = float(record_interval)
    if season:
        cfg.start_datetime = season_start(season, year)
        if not name:
            cfg.name = f"{cfg.name}_{season.strip().lower()}"
    if name:
        cfg.name = name
    elif males or females or groups:
        # a name that says what was actually run, so run folders are legible
        cfg.name = f"{cfg.name}_{cfg.total_agents()}agents_{cfg.days:g}d"

    check(cfg)
    return cfg


def check(cfg: ExperimentConfig) -> ExperimentConfig:
    """Refuse a config that cannot produce a meaningful run.

    These are the mistakes that otherwise surface as an empty CSV or a run
    that takes a week — worth catching before anything is written to disk.
    """
    if cfg.total_agents() == 0:
        raise ValueError("no animals: give males=/females= or groups=")
    if cfg.days <= 0:
        raise ValueError(f"days must be > 0, got {cfg.days}")
    if cfg.n_trials < 1:
        raise ValueError(f"trials must be >= 1, got {cfg.n_trials}")
    if cfg.dt <= 0 or cfg.dt > 60:
        raise ValueError(
            f"dt of {cfg.dt}s is outside the usable range (0, 60]; the "
            f"movement model is calibrated near 2 s")
    if cfg.record_interval < cfg.dt:
        raise ValueError(
            f"record_interval ({cfg.record_interval}s) is finer than the "
            f"timestep ({cfg.dt}s), so samples would repeat")
    area = cfg.arena.width * cfg.arena.height
    if area <= 0:
        raise ValueError("arena has no area")
    return cfg


def summary(cfg: ExperimentConfig) -> str:
    """One human-readable paragraph describing what this config will run."""
    per_group = ", ".join(
        f"{g.count} {g.sex} {g.species}"
        + (f" [{g.treatment.drug} {g.treatment.dose:g}]"
           if g.treatment.drug not in ("none", "saline", "") else "")
        for g in cfg.groups)
    steps = int(cfg.days * 86400 / cfg.dt)
    mech = [n for n, on in (
        ("scent marking", cfg.scent.enabled),
        ("energy/water budget", cfg.physiology.enabled),
        ("mechanistic olfaction", cfg.olfaction.enabled),
        ("living sward", cfg.sward.enabled),
        ("sun & moon", cfg.sky.enabled),
        ("mortality", cfg.enable_mortality)) if on]
    lines = [
        f"{cfg.name}: {cfg.arena.width:.2f} x {cfg.arena.height:.2f} m "
        f"({cfg.arena.ground}) · {per_group} · {cfg.days:g} days · "
        f"{cfg.n_trials} replicate(s) · dt {cfg.dt:g}s "
        f"({steps:,} steps/trial) · sample every {cfg.record_interval:g}s",
        f"mechanisms: {', '.join(mech) if mech else 'none'}",
        f"release: {cfg.release_mode}",
    ]
    if cfg.sky.enabled:
        when = datetime.fromisoformat(cfg.start_datetime)
        lines.append(
            f"site: {cfg.sky.latitude:.3f}N {cfg.sky.longitude:.3f}E · "
            f"{when:%d %b %Y %H:%M} ({season_of(when)}) · "
            f"day length {day_length_hours(when, cfg.sky):.1f} h")
    if cfg.sward.enabled:
        lines.append(
            f"sward: {cfg.sward.initial_min_cm:g}-"
            f"{cfg.sward.initial_max_cm:g} cm at release, regrowing "
            f"{cfg.sward.regrowth_cm_per_day:g} cm/day; clipping is "
            f"{cfg.sward.chew_rate_multiplier:g}x walking")
    return "\n".join(lines)
