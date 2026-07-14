"""Attribute distribution specs — seed agent attributes from domain knowledge.

An attribute can be a fixed value or a distribution the user parameterises from
what they know about the animal, e.g. adult male prairie-vole mass. Each founder
in a group then samples its own value, so a cohort varies realistically instead
of being identical clones.

Supported cell/string syntax
----------------------------
    33            fixed value
    N(33,3)       normal, mean 33, sd 3
    N(33,3)[25,42]  normal truncated to [25, 42]
    U(25,42)      uniform on [25, 42]

``parse_spec`` returns an :class:`AttrSpec`; ``.sample(rng)`` draws one value and
``.mean`` gives a representative scalar (for display / fallback).
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class AttrSpec:
    kind: str            # 'fixed' | 'normal' | 'uniform'
    a: float             # fixed value | normal mean | uniform low
    b: float = 0.0       # normal sd | uniform high
    lo: float | None = None
    hi: float | None = None

    @property
    def mean(self) -> float:
        if self.kind == "uniform":
            return 0.5 * (self.a + self.b)
        return self.a

    def sample(self, rng) -> float:
        if self.kind == "fixed":
            v = self.a
        elif self.kind == "uniform":
            v = rng.uniform(self.a, self.b)
        else:  # normal
            v = rng.normal(self.a, self.b)
        if self.lo is not None:
            v = max(self.lo, v)
        if self.hi is not None:
            v = min(self.hi, v)
        return float(v)

    def to_str(self) -> str:
        if self.kind == "fixed":
            return _fmt(self.a)
        if self.kind == "uniform":
            return f"U({_fmt(self.a)},{_fmt(self.b)})"
        s = f"N({_fmt(self.a)},{_fmt(self.b)})"
        if self.lo is not None or self.hi is not None:
            s += f"[{_fmt(self.lo if self.lo is not None else '')}," \
                 f"{_fmt(self.hi if self.hi is not None else '')}]"
        return s


_NORMAL = re.compile(
    r"^N\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)"
    r"(?:\[\s*([-\d.]*)\s*,\s*([-\d.]*)\s*\])?$", re.I)
_UNIFORM = re.compile(r"^U\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)$", re.I)


def parse_spec(s) -> AttrSpec:
    """Parse a spec string (or number) into an :class:`AttrSpec`."""
    if isinstance(s, (int, float)):
        return AttrSpec("fixed", float(s))
    s = str(s).strip()
    m = _NORMAL.match(s)
    if m:
        lo = float(m.group(3)) if m.group(3) else None
        hi = float(m.group(4)) if m.group(4) else None
        return AttrSpec("normal", float(m.group(1)), float(m.group(2)), lo, hi)
    m = _UNIFORM.match(s)
    if m:
        return AttrSpec("uniform", float(m.group(1)), float(m.group(2)))
    return AttrSpec("fixed", float(s))  # raises ValueError if not a number


def _fmt(v) -> str:
    if v == "" or v is None:
        return ""
    f = float(v)
    return str(int(f)) if f == int(f) else f"{f:g}"
