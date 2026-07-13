"""Genetics and pharmacology: how genotype and treatment reshape traits.

Both systems resolve to concrete modifications of a :class:`TraitProfile` at the
moment an agent is instantiated. Keeping them here (rather than baked into the
agent) makes it trivial to add new genes/drugs and to document *mechanism*.

Effect encoding
---------------
Each effect is ``(trait_name, op, value)`` where ``op`` is:
    'set'    -> trait = value
    'scale'  -> trait = trait * value
    'add'    -> trait = trait + value
Effects are clamped to sensible ranges afterwards.
"""
from __future__ import annotations

from copy import deepcopy

from .config import TraitProfile, Genotype, Treatment


# --------------------------------------------------------------------------- #
# Gene registry
# --------------------------------------------------------------------------- #
# Per gene, per status, a list of (trait, op, value) effects.
GENE_EFFECTS: dict[str, dict[str, list[tuple[str, str, float]]]] = {
    # Major Urinary Proteins — individual identity signalling (house mouse).
    # Knockout animals emit no personal scent signature.
    "MUP": {
        "KO":  [("identity_signal", "set", 0.0)],
        "HET": [("identity_signal", "scale", 0.5)],
    },
    # Oxytocin receptor — social bonding / affiliation (voles).
    "OXTR": {
        "KO":  [("sociability", "scale", 0.4)],
        "HET": [("sociability", "scale", 0.7)],
    },
    # Vasopressin 1a receptor — pair bonding, territorial aggression (voles).
    "AVPR1A": {
        "KO":  [("aggression", "scale", 0.6), ("home_range_r", "scale", 0.8)],
        "HET": [("aggression", "scale", 0.85)],
    },
}

# Which genes are meaningful for which species (for GUI menus).
SPECIES_GENES: dict[str, list[str]] = {
    "prairie": ["OXTR", "AVPR1A"],
    "meadow": ["OXTR", "AVPR1A"],
    "mouse": ["MUP"],
    "house mouse": ["MUP"],
}


# --------------------------------------------------------------------------- #
# Drug registry
# --------------------------------------------------------------------------- #
# Each drug maps to a function of dose -> list of (trait, op, value) effects.
def _methimazole(dose: float) -> list[tuple[str, str, float]]:
    # Ablates olfactory epithelium -> anosmia scaled by dose.
    # dose 1.0 -> smell_ability 0 (fully anosmic); 0.5 -> half sensitivity.
    return [("smell_ability", "scale", max(0.0, 1.0 - dose))]


DRUG_EFFECTS = {
    "none": lambda dose: [],
    "saline": lambda dose: [],  # vehicle control — no trait effect
    "methimazole": _methimazole,
}


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #
_TRAIT_RANGES = {
    "aggression": (0.0, 2.0),
    "boldness": (0.0, 2.0),
    "sociability": (0.0, 2.0),
    "exploration": (0.0, 2.0),
    "smell_ability": (0.0, 1.0),
    "identity_signal": (0.0, 1.0),
    "base_speed": (0.0, 1.0),
    "home_range_r": (0.05, 5.0),
    "mass": (3.0, 500.0),
    "metabolism": (0.2, 5.0),
    "turn_rate": (0.0, 3.0),
    "wander": (0.0, 3.0),
}


def _apply(traits: TraitProfile, effects: list[tuple[str, str, float]]) -> None:
    for trait, op, value in effects:
        if not hasattr(traits, trait):
            continue
        cur = getattr(traits, trait)
        if op == "set":
            cur = value
        elif op == "scale":
            cur = cur * value
        elif op == "add":
            cur = cur + value
        lo, hi = _TRAIT_RANGES.get(trait, (float("-inf"), float("inf")))
        setattr(traits, trait, min(hi, max(lo, cur)))


def resolve_traits(base: TraitProfile, genotype: Genotype,
                   treatment: Treatment, drug_active: bool | None = None) -> TraitProfile:
    """Return a new TraitProfile with genetics and treatment applied.

    ``drug_active`` controls whether the pharmacological effect is included:
      * ``None`` (default): include only if the treatment is already in effect at
        release (``day_offset <= 0``) — the steady-state / pre-release case.
      * ``True`` / ``False``: force the drug on/off. The simulation uses this to
        build a pre-onset profile (False) and a post-onset profile (True) for
        treatments delivered *after* release (``day_offset > 0``).
    """
    traits = deepcopy(base)
    # Genetics first (developmental), then pharmacology (acute/state).
    for gene, status in (genotype.genes or {}).items():
        status = (status or "WT").upper()
        if status in ("WT", ""):
            continue
        effects = GENE_EFFECTS.get(gene, {}).get(status, [])
        _apply(traits, effects)

    if treatment and treatment.drug not in ("none", None):
        active = (treatment.day_offset <= 0.0) if drug_active is None else drug_active
        if active:
            effects = DRUG_EFFECTS.get(treatment.drug, lambda d: [])(treatment.dose)
            _apply(traits, effects)
    return traits


def apply_drug(traits: TraitProfile, treatment: Treatment) -> None:
    """Apply only a treatment's pharmacological effect to ``traits`` in place."""
    if treatment and treatment.drug not in ("none", None):
        effects = DRUG_EFFECTS.get(treatment.drug, lambda d: [])(treatment.dose)
        _apply(traits, effects)


# trait name -> the live simulation array attribute it maps to
TRAIT_TO_ARRAY = {
    "aggression": "aggr",
    "boldness": "bold",
    "sociability": "social",
    "exploration": "explore",
    "smell_ability": "smell",
    "identity_signal": "identity",
    "base_speed": "speed",
    "home_range_r": "home_r",
    "mass": "mass",
    "metabolism": "metabolism",
    "turn_rate": "turn_rate",
    "wander": "wander",
}
