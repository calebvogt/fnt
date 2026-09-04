"""Group detected acoustic elements into calls by harmonic relationship.

MAD's model answers one question -- *are these pixels a vole-produced tonal
element?* -- and it answers it from a single tile. Whether a given element is a
fundamental or somebody's harmonic is a different question entirely: it depends
on whether another element exists at a simple frequency ratio *at the same
instant*, which is evidence the model's receptive field does not contain. So it
is answered here instead, after detection, where the whole element list is
visible.

That split is not a workaround, it is the correct factoring, and it buys three
things. Labeling stays decidable ("is this biological?" rather than "is this a
harmonic?", which two people will answer differently at low SNR). The rule stays
auditable and revisable -- changing your mind costs a re-run, not a relabel and
a retrain. And both numbers survive: element counts and call counts, where
deciding at label time would have destroyed one of them.

**The test.** Harmonics are produced simultaneously with their fundamental and
are *scaled copies* of its frequency contour. So for two elements overlapping in
time we take the ratio of their frequency tracks pointwise across the shared
span and ask two things of it: is its mean near an integer, and is it *steady*?
The second question is the one that does the work. If a call sweeps 25->30 kHz
its second harmonic sweeps 50->60 and the ratio sits pinned at 2.0 the whole
way, whereas two unrelated calls that happen to pass through 25 and 50 kHz at
one instant have a ratio that drifts. Comparing single summary frequencies --
medians, peaks -- throws that discriminator away, and on frequency-modulated
calls it is most of the available signal.

**The fundamental may not be detected.** If it is faint or below the recording
band you can see 50 and 75 kHz and nothing else; their ratio is 1.5, which
matches no harmonic relation. So a stack is not assumed to be rooted at its
lowest member: a common f0 is fitted to the whole group, and whether the
fundamental was actually *detected* falls out as "is there a member with n=1"
-- a reported fact rather than an assumption.

Anything that does not fit stays its own call. Subharmonics, biphonation and
step calls produce non-integer ratios, and forcing them into a stack would be
worse than leaving them alone.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "HarmonicConfig", "Element", "Call", "GroupingResult", "Relation",
    "contour_from_patch", "contour_from_mask", "pair_relation", "group_calls",
]


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class HarmonicConfig:
    """Tolerances for the harmonic test.

    Defaults are a starting point, not a measurement -- calibrate them against
    hand-labeled files before trusting them on a new recording setup.
    """

    #: Minimum temporal overlap, as a fraction of the SHORTER element. Harmonics
    #: are the same vocal event as their fundamental, so they co-occur; partial
    #: detections still pass because the ratio is only ever computed over the
    #: shared span.
    min_time_overlap: float = 0.5
    #: How far the mean frequency ratio may sit from a whole number, relatively.
    ratio_tol: float = 0.06
    #: Maximum coefficient of variation of the ratio across the shared span.
    #: This is the contour-shape test; loosening it is what lets unrelated
    #: co-occurring calls get stapled together.
    ratio_cv_max: float = 0.04
    #: Highest harmonic considered. Kept low on purpose: raising it widens the
    #: net faster than it finds real harmonics, because the grid of allowed
    #: frequencies (n * f0, each with a tolerance that grows with n) gets dense
    #: enough to swallow unrelated calls.
    max_harmonic: int = 4
    #: Below this many shared contour samples the CV estimate is too noisy to
    #: mean anything, so only the mean-ratio test is applied. Short calls (a few
    #: milliseconds) land here routinely.
    min_points_for_cv: int = 5
    #: How far below the lowest detected element a fundamental may be inferred.
    #: **1 means no inference**, which is the default for a measured reason: on
    #: hand-labeled data, allowing it produced fundamentals at 8-12 kHz for a
    #: third of all stacks -- well below anything a vole produces -- because a
    #: low enough f0 can explain any set of frequencies as near-multiples of
    #: itself. Turn it up only with ``min_f0_hz`` set to something defensible.
    max_missing_fundamental: int = 1
    #: Floor on an *inferred* fundamental. Guards the failure above; ignored
    #: when the fundamental was actually detected.
    min_f0_hz: float = 15000.0

    def __post_init__(self):
        if not 0.0 <= self.min_time_overlap <= 1.0:
            raise ValueError("min_time_overlap must be a fraction in [0, 1]")
        if self.max_harmonic < 2:
            raise ValueError("max_harmonic must be at least 2")
        if self.max_missing_fundamental < 1:
            raise ValueError("max_missing_fundamental must be at least 1")


# ----------------------------------------------------------------------
# Elements
# ----------------------------------------------------------------------
@dataclass
class Element:
    """One detected acoustic element, with its frequency contour.

    ``times`` and ``freqs`` are equal-length and time-ordered; ``freqs`` is the
    intensity-weighted centre frequency at each time sample, which tracks a
    frequency-modulated call rather than flattening it to one number.
    """

    id: str
    times: np.ndarray            # seconds, ascending
    freqs: np.ndarray            # Hz
    meta: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=float).ravel()
        self.freqs = np.asarray(self.freqs, dtype=float).ravel()
        if self.times.size != self.freqs.size:
            raise ValueError("times and freqs must be the same length")
        if self.times.size and np.any(np.diff(self.times) < 0):
            order = np.argsort(self.times)
            self.times, self.freqs = self.times[order], self.freqs[order]

    @property
    def t0(self) -> float:
        return float(self.times[0]) if self.times.size else math.nan

    @property
    def t1(self) -> float:
        return float(self.times[-1]) if self.times.size else math.nan

    @property
    def duration(self) -> float:
        return self.t1 - self.t0 if self.times.size else 0.0

    @property
    def mean_freq(self) -> float:
        return float(np.mean(self.freqs)) if self.freqs.size else math.nan

    def freq_at(self, t) -> np.ndarray:
        """The contour resampled onto ``t`` (clamped outside its own span)."""
        if self.times.size == 1:
            return np.full(np.shape(t), self.freqs[0], dtype=float)
        return np.interp(t, self.times, self.freqs)

    @classmethod
    def from_endpoints(cls, id, t0, t1, f_start, f_end, meta=None) -> "Element":
        """A straight-line contour from a detection's start/end frequencies.

        The fallback for callers that have summary columns but no stored mask
        (the per-wav CSV, say). It captures the sweep direction, which is most
        of what the shape test needs, but not curvature -- prefer
        :func:`contour_from_patch` whenever the mask is available.
        """
        return cls(id=str(id), times=np.array([float(t0), float(t1)]),
                   freqs=np.array([float(f_start), float(f_end)]),
                   meta=dict(meta or {}))


def contour_from_mask(mask, *, f_bin_offset: int, t_frame_offset: int,
                      sample_rate: float, nfft: int, hop: int,
                      id: str = "", spec=None,
                      meta: Optional[Dict] = None) -> Optional[Element]:
    """Build an Element from a mask crop positioned on the spectrogram grid.

    The live-review path: an on-screen annotation carries its mask and its
    position in spec-pixel coordinates, but the full-file spectrogram is never
    held in memory (it is recomputed per visible window), so ``spec`` is
    usually absent and the contour falls back to the mask's own centroid per
    column. For a narrowband tonal call the two agree closely; pass ``spec``
    when you have it and the centroid becomes energy-weighted.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2 or mask.size == 0 or sample_rate <= 0 or nfft <= 0:
        return None
    on = mask > 0
    cols = np.flatnonzero(on.any(axis=0))
    if cols.size == 0:
        return None
    hz_per_bin = float(sample_rate) / float(nfft)
    dt = float(hop) / float(sample_rate)
    bins = (np.arange(mask.shape[0], dtype=float) + f_bin_offset) * hz_per_bin
    spec = np.asarray(spec, dtype=float) if spec is not None else None
    if spec is not None and spec.shape != mask.shape:
        spec = None

    times, freqs = [], []
    for c in cols:
        rows = np.flatnonzero(on[:, c])
        if spec is None:
            f = float(bins[rows].mean())
        else:
            wt = spec[rows, c]
            wt = wt - wt.min()
            total = wt.sum()
            f = (float(np.dot(wt, bins[rows]) / total) if total > 0
                 else float(bins[rows].mean()))
        times.append((t_frame_offset + c + 0.5) * dt)
        freqs.append(f)
    return Element(id=str(id), times=np.array(times), freqs=np.array(freqs),
                   meta=dict(meta or {}))


def contour_from_patch(mask, spec, meta: Dict, *,
                       min_bins: int = 1) -> Optional[Element]:
    """Build an Element from one stored training example.

    ``mask``/``spec`` are the ``(n_freq, n_time)`` arrays saved beside the
    audio; ``meta`` is the example's metadata (sample rate, nfft, patch
    offsets). The contour is the intensity-weighted centroid of the masked
    pixels in each time column, so a column's energy distribution decides its
    frequency rather than the mask's midpoint.
    """
    mask = np.asarray(mask)
    spec = np.asarray(spec, dtype=float)
    if mask.ndim != 2 or mask.shape != spec.shape or mask.size == 0:
        return None
    sr = float(meta.get("sample_rate") or 0.0)
    nfft = int(meta.get("nfft") or 0)
    if sr <= 0 or nfft <= 0:
        return None

    f_off = int(meta.get("patch_f_off") or 0)
    hz_per_bin = sr / float(nfft)
    n_frames = int(meta.get("patch_t_frames") or mask.shape[1]) or mask.shape[1]
    pt0 = float(meta.get("patch_t0_s") or 0.0)
    pt1 = float(meta.get("patch_t1_s") or 0.0)
    dt = (pt1 - pt0) / n_frames if pt1 > pt0 and n_frames else 0.0

    on = mask > 0
    cols = np.flatnonzero(on.any(axis=0))
    if cols.size == 0:
        return None
    bins = (np.arange(mask.shape[0], dtype=float) + f_off) * hz_per_bin

    times, freqs = [], []
    for c in cols:
        rows = np.flatnonzero(on[:, c])
        if rows.size < min_bins:
            continue
        # Weight by energy above the column's floor; fall back to the plain
        # centroid where the patch is flat, so a valid column is never dropped.
        wt = spec[rows, c]
        wt = wt - wt.min()
        total = wt.sum()
        f = (float(np.dot(wt, bins[rows]) / total) if total > 0
             else float(bins[rows].mean()))
        times.append(pt0 + (c + 0.5) * dt)
        freqs.append(f)
    if not times:
        return None
    return Element(id=str(meta.get("id") or ""), times=np.array(times),
                   freqs=np.array(freqs), meta=dict(meta))


# ----------------------------------------------------------------------
# The pairwise test
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Relation:
    """The measured relationship between a lower and an upper element."""

    n: Optional[int]          # integer ratio, if one fits
    ratio: float              # mean of the pointwise ratio
    cv: float                 # its coefficient of variation (nan if too short)
    overlap: float            # shared time, as a fraction of the shorter one
    n_points: int

    @property
    def ok(self) -> bool:
        return self.n is not None


def pair_relation(lower: Element, upper: Element,
                  cfg: HarmonicConfig = HarmonicConfig()) -> Relation:
    """Measure whether ``upper`` is a harmonic of ``lower``.

    The ratio is computed pointwise over the shared time span and judged on two
    counts: its mean must sit near a whole number, and it must be steady. See
    the module docstring for why the steadiness test carries the weight.
    """
    lo_t0, lo_t1 = lower.t0, lower.t1
    up_t0, up_t1 = upper.t0, upper.t1
    start, stop = max(lo_t0, up_t0), min(lo_t1, up_t1)
    shorter = min(lower.duration, upper.duration)
    if not np.isfinite(start) or not np.isfinite(stop) or stop < start:
        return Relation(None, math.nan, math.nan, 0.0, 0)
    # A zero-duration element (one contour sample) still counts as overlapping
    # if it falls inside the other's span.
    overlap = 1.0 if shorter <= 0 else (stop - start) / shorter
    if overlap < cfg.min_time_overlap:
        return Relation(None, math.nan, math.nan, overlap, 0)

    # Sample the shared span on whichever element's grid is finer, so the
    # comparison is made at the resolution actually recorded.
    grids = [t[(t >= start) & (t <= stop)] for t in (lower.times, upper.times)]
    grid = max(grids, key=lambda g: g.size)
    if grid.size == 0:
        grid = np.array([(start + stop) / 2.0])
    f_lo = lower.freq_at(grid)
    f_up = upper.freq_at(grid)
    good = f_lo > 0
    if not good.any():
        return Relation(None, math.nan, math.nan, overlap, 0)
    r = f_up[good] / f_lo[good]
    mean = float(np.mean(r))
    cv = float(np.std(r) / mean) if mean > 0 and r.size > 1 else math.nan
    n_points = int(r.size)

    n = int(round(mean))
    fits = (2 <= n <= cfg.max_harmonic
            and abs(mean - n) / n <= cfg.ratio_tol)
    if fits and n_points >= cfg.min_points_for_cv and np.isfinite(cv):
        fits = cv <= cfg.ratio_cv_max
    return Relation(n if fits else None, mean, cv, overlap, n_points)


# ----------------------------------------------------------------------
# Grouping
# ----------------------------------------------------------------------
@dataclass
class Call:
    """One vocalization: a fundamental plus whichever harmonics were detected."""

    call_id: str
    f0_hz: float
    #: element id -> harmonic number (1 = the fundamental itself)
    members: Dict[str, int]
    #: True when a member actually sits at n=1; False means f0 was inferred
    #: from the harmonics alone because the fundamental was not detected.
    fundamental_detected: bool

    @property
    def n_elements(self) -> int:
        return len(self.members)

    @property
    def harmonics(self) -> List[int]:
        return sorted(self.members.values())


@dataclass
class GroupingResult:
    calls: List[Call]
    #: element id -> call_id
    call_of: Dict[str, str]
    #: element id -> harmonic number
    harmonic_of: Dict[str, int]
    #: every relation that was accepted, as (lower_id, upper_id, n). This is
    #: what the review overlay draws its links from.
    links: List[Tuple[str, str, int]] = field(default_factory=list)

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    @property
    def n_elements(self) -> int:
        return len(self.call_of)


def _resolve_forced(forced: Optional[Dict[str, Optional[str]]],
                    by_id: Dict[str, Element]):
    """Normalise manual corrections into (pinned_out, root -> children, child ->
    root), following chains so pinning A to B to C lands A on C."""
    pinned_out: set = set()
    pin_to: Dict[str, str] = {}
    for child, root in (forced or {}).items():
        if child not in by_id:
            continue
        if root is None:
            pinned_out.add(child)
        elif root in by_id and root != child:
            pin_to[child] = root

    def resolve(c: str) -> str:
        seen = {c}
        while c in pin_to:
            c = pin_to[c]
            if c in seen:            # a cycle the user built by hand
                break
            seen.add(c)
        return c

    child_root = {c: resolve(c) for c in pin_to}
    children: Dict[str, List[str]] = {}
    for child, root in child_root.items():
        if root != child:
            children.setdefault(root, []).append(child)
    return pinned_out, children, child_root


def _fit_f0(members: Dict[str, int], by_id: Dict[str, Element]) -> float:
    """The fundamental implied by a stack: the median of each member's
    frequency divided by its harmonic number, so every member gets a vote and
    one bad contour cannot drag it."""
    votes = [by_id[eid].mean_freq / n for eid, n in members.items() if n > 0]
    return float(np.median(votes)) if votes else math.nan


def _prune_stack(members: Dict[str, int], pinned: set,
                 by_id: Dict[str, Element],
                 cfg: HarmonicConfig) -> Tuple[Dict[str, int], List[str]]:
    """Drop members that fit the pairwise test but not the finished stack.

    The two are not the same check. :func:`pair_relation` measures the ratio
    over the span two elements *share*, which is the right place to compare
    contours -- but on a partial overlap between strongly modulated calls, that
    local ratio can land on a different integer than the elements' overall
    frequencies support. The symptom is a stack whose 4th harmonic sits below
    its 3rd. So once a stack is assembled, every member is re-checked against
    the fitted fundamental using its whole contour, worst offender first, and
    whatever cannot be reconciled is released to seed a call of its own.
    """
    members = dict(members)
    dropped: List[str] = []
    while len(members) > 1:
        f0 = _fit_f0(members, by_id)
        if not (f0 > 0):
            break
        worst, worst_err = None, 0.0
        for eid, n in members.items():
            if eid in pinned:
                continue
            err = abs(by_id[eid].mean_freq / f0 - n) / n
            if err > worst_err:
                worst, worst_err = eid, err
        if worst is None or worst_err <= cfg.ratio_tol:
            break
        del members[worst]
        dropped.append(worst)
    return members, dropped


def _infer_missing_fundamentals(calls: List["Call"], by_id: Dict[str, Element],
                                cfg: HarmonicConfig) -> List["Call"]:
    """Merge single-element calls that share an undetected fundamental.

    Off by default (``max_missing_fundamental == 1``), and deliberately so:
    measured against hand-labeled data, inference of this kind invented
    fundamentals at 8-12 kHz for a third of all stacks, because a low enough f0
    explains any set of frequencies as near-multiples of itself. ``min_f0_hz``
    is the guard that makes it usable at all -- validate on your own labels
    before turning this on.
    """
    if cfg.max_missing_fundamental <= 1:
        return calls
    solo = [c for c in calls if c.n_elements == 1]
    rest = [c for c in calls if c.n_elements > 1]
    solo.sort(key=lambda c: by_id[c.call_id].mean_freq)
    merged: Dict[str, Dict[str, int]] = {}
    taken: set = set()
    for i, low in enumerate(solo):
        if low.call_id in taken:
            continue
        e_low = by_id[low.call_id]
        for q in range(2, cfg.max_missing_fundamental + 1):
            f0 = e_low.mean_freq / q
            if f0 < cfg.min_f0_hz:
                continue
            group = {low.call_id: q}
            for other in solo[i + 1:]:
                if other.call_id in taken:
                    continue
                e_up = by_id[other.call_id]
                rel = pair_relation(e_low, e_up, cfg)
                if rel.overlap < cfg.min_time_overlap:
                    continue
                raw = e_up.mean_freq / f0
                n = int(round(raw))
                if (n <= q or n > cfg.max_harmonic
                        or abs(raw - n) / n > cfg.ratio_tol
                        or n in group.values()):
                    continue
                group[other.call_id] = n
            if len(group) > 1:
                merged[low.call_id] = group
                taken.update(group)
                break
    out = list(rest)
    for c in solo:
        if c.call_id in merged:
            members = merged[c.call_id]
            out.append(Call(call_id=c.call_id, f0_hz=_fit_f0(members, by_id),
                            members=members, fundamental_detected=False))
        elif c.call_id not in taken:
            out.append(c)
    return out


def group_calls(elements: Sequence[Element],
                cfg: HarmonicConfig = HarmonicConfig(),
                *, forced: Optional[Dict[str, Optional[str]]] = None,
                ) -> GroupingResult:
    """Group elements into calls by harmonic relationship.

    Stacks are built by **anchoring on a candidate fundamental**, not by taking
    connected components over pairwise matches. Every member is tested against
    the fundamental itself, so a relation is never inherited through a chain.
    Component-style grouping looks equivalent and is not: on a busy file it
    merges A with B and B with C into one call even where A and C have no
    harmonic relation at all, and simultaneous calls from different animals get
    stapled together. The measured signature of that failure is a stack
    containing two members with the *same* harmonic number, which cannot happen
    physically -- so it is also forbidden here directly.

    ``forced`` carries manual corrections: ``{element_id: other_element_id}``
    pins an element into that element's stack, and ``{element_id: None}`` pins
    it out into a call of its own. Corrections outrank the automatic test in
    both directions, so a reviewer's judgment survives a later re-run under
    different tolerances.
    """
    elements = [e for e in elements if e.times.size and e.mean_freq > 0]
    by_id = {e.id: e for e in elements}
    pinned_out, forced_children, child_root = _resolve_forced(forced, by_id)

    order = sorted(elements, key=lambda e: (e.mean_freq, e.t0, e.id))
    used: set = set()
    calls: List[Call] = []
    links: List[Tuple[str, str, int]] = []
    # An element the user named as a fundamental has to stay a seed; letting it
    # be absorbed as somebody's harmonic would silently undo the correction.
    protected = set(forced_children)

    for seed in order:
        if seed.id in used:
            continue
        used.add(seed.id)
        members = {seed.id: 1}
        # One element per harmonic number: keep the best-fitting claimant and
        # leave the others to seed calls of their own.
        best: Dict[int, Tuple[float, str, Relation]] = {}
        for cand in order:
            if cand.id in used or cand.id == seed.id or cand.id in pinned_out:
                continue
            if cand.id in protected:
                continue
            pinned = child_root.get(cand.id)
            if pinned is not None and pinned != seed.id:
                continue            # spoken for by a different fundamental
            if (cand.mean_freq / seed.mean_freq
                    > cfg.max_harmonic * (1 + cfg.ratio_tol)):
                break               # frequency-sorted: nothing further to find
            rel = pair_relation(seed, cand, cfg)
            if not rel.ok:
                continue
            err = abs(rel.ratio - rel.n) / rel.n
            prev = best.get(rel.n)
            if prev is None or err < prev[0]:
                best[rel.n] = (err, cand.id, rel)

        # Manual pins join regardless of what the ratio test thinks, and win
        # their harmonic slot outright.
        for child_id in forced_children.get(seed.id, ()):
            if child_id in used or child_id not in by_id:
                continue
            child = by_id[child_id]
            lo, up = ((seed, child) if seed.mean_freq <= child.mean_freq
                      else (child, seed))
            rel = pair_relation(lo, up, cfg)
            n = (rel.n if rel.ok
                 else max(2, int(round(child.mean_freq / seed.mean_freq))))
            best[n] = (-1.0, child_id, rel)      # -1 sorts ahead of any auto fit

        for n, (_err, eid, _rel) in best.items():
            members[eid] = n
            used.add(eid)
        # The seed anchors the stack, and a manual pin is the reviewer's call --
        # neither is the algorithm's to second-guess.
        pinned = {seed.id} | set(forced_children.get(seed.id, ()))
        members, dropped = _prune_stack(members, pinned, by_id, cfg)
        for eid in dropped:
            used.discard(eid)        # released to seed a call of its own
        for eid, n in members.items():
            if eid != seed.id:
                links.append((seed.id, eid, n))
        calls.append(Call(call_id=seed.id, f0_hz=_fit_f0(members, by_id),
                          members=members, fundamental_detected=True))

    calls = _infer_missing_fundamentals(calls, by_id, cfg)
    call_of: Dict[str, str] = {}
    harmonic_of: Dict[str, int] = {}
    for c in calls:
        for eid, n in c.members.items():
            call_of[eid] = c.call_id
            harmonic_of[eid] = n
    calls.sort(key=lambda c: (by_id[c.call_id].t0, c.f0_hz))
    return GroupingResult(calls=calls, call_of=call_of,
                          harmonic_of=harmonic_of, links=links)
