"""Pick a representative subset of recordings from a folder tree.

A 24/7 multi-mic set is far too large to label, or even to run a detector over:
the 2021_8x8 USV tree is 33,996 ten-minute wavs, and inference on this hardware
runs at roughly 3x realtime, so analyzing all of it is measured in weeks. What
makes the work tractable is not faster hardware but a *subset that tiles the
whole series* — a few files from every trial, every microphone, and every part
of every recording day. Twenty per trial is an overnight run instead of a month,
and it teaches a detector far more than twenty consecutive files from one trial.

Two ideas do the work here:

**Channel is part of the grouping, not part of the pool.** In these sets the
microphone is encoded in the filename (``..._ch3_T0000042.wav``), so a plain
sort is channel-major and time-minor: every ch1 file precedes every ch2 file.
Striding over that flat list only *looks* right when the channels happen to
hold equal counts — the moment one is short (T014 has 610 and 548 against a
usual 4x~700) the picks skew to whichever channel sorts first. Grouping by
(folder, channel) and striding inside each group makes the good behaviour
deliberate rather than accidental.

**Channels are near-duplicates of each other.** ch1-ch4 at the same sequence
number are one moment heard by four microphones, so four channels is not four
times the diversity — it is roughly one moment with four signal-to-noise ratios.
That is worth *something* (a call that is obvious on one mic and marginal on
another is exactly the hard case worth labelling) but it is not worth four times
the labelling effort, which is why the default spreads a per-folder budget
across the channels instead of multiplying by them.

Everything here is pure: paths in, paths out, no filesystem access beyond what
the caller already did. The GUI import dialog and ``mad analyze --sample-*``
both call it, so a headless run samples exactly what the GUI would have.
"""
from __future__ import annotations

import os
import random
import re
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "CHANNEL_RE", "parse_channel", "natural_key", "group_paths",
    "allocate", "stride_pick", "sample_paths", "prepare", "draw",
    "SampleSpec", "SampleResult", "Prepared",
]

#: Microphone token in a recording's filename: ``_ch3_``, ``_ch12T``, ``ch1T``.
#:
#: Anchored to a separator (or the start) on the left and to a separator, a ``T``
#: timestamp marker, or the end on the right, so it matches the channel and not
#: the ``ch`` inside a word — a bare ``ch\d+`` would happily claim the ``ch4`` in
#: a genotype like ``Arch4``. Verified against all 33,996 names in the 2021_8x8
#: tree: 33,996 parsed, 0 failures, and the per-(trial, channel) counts match a
#: manual tally including T014's lopsided 610/548.
CHANNEL_RE = re.compile(r"(?:^|[_-])ch(\d+)(?=[_T.\-]|$)", re.IGNORECASE)

_NUM_RE = re.compile(r"(\d+)")


def parse_channel(path: str) -> Optional[str]:
    """``'ch3'`` for a recording whose name carries a channel, else ``None``.

    Normalized to lowercase without zero padding so ``CH03`` and ``ch3`` are one
    group. Returns ``None`` rather than guessing when there is no channel token,
    which is what makes single-mic sets fall through to plain folder grouping.
    """
    m = CHANNEL_RE.search(os.path.splitext(os.path.basename(path))[0])
    return f"ch{int(m.group(1))}" if m else None


def natural_key(path: str):
    """Sort key placing ``file_9`` before ``file_10``.

    Sequence numbers in these sets are zero-padded, so a plain sort is already
    correct — but only by convention, and a single unpadded run of files would
    silently scramble the time order that striding depends on. Splitting digits
    out costs nothing and removes the assumption.
    """
    base = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in _NUM_RE.split(base)]


def group_paths(paths: Iterable[str],
                by_channel: bool = True) -> "OrderedDict[Tuple[str, ...], List[str]]":
    """Bucket recordings by containing folder, and by channel when asked.

    The key is ``(folder,)`` or ``(folder, channel)``; a file in a channel-aware
    grouping whose name has no channel token gets ``''`` for the channel, so it
    forms its own group instead of being dropped or lumped in with ch1.

    Groups come back in sorted key order and each group's paths in natural
    order, because striding is only meaningful over a stable sequence.
    """
    buckets: Dict[Tuple[str, ...], List[str]] = {}
    for p in paths:
        folder = os.path.dirname(os.path.abspath(p))
        key = (folder, parse_channel(p) or "") if by_channel else (folder,)
        buckets.setdefault(key, []).append(p)
    return OrderedDict(
        (k, sorted(buckets[k], key=natural_key)) for k in sorted(buckets))


def allocate(sizes: Sequence[int], total: int) -> List[int]:
    """Split ``total`` picks over groups of the given sizes.

    Largest-remainder apportionment (the Hamilton method), capped by each
    group's own size and with the freed remainder handed back to groups that can
    still absorb it. Proportional rather than equal because the groups are not
    equal: T005 holds 238 files per channel and T001 holds 847, and giving them
    the same budget would oversample the short trial's timeline four times as
    densely as the long one.

    Ties break toward the larger group and then toward the earlier one, so the
    result depends only on the inputs — never on dict ordering.

    One departure from strict proportionality: a non-empty group is never left
    with nothing while some other group holds more than one. Pure apportionment
    hands a 3-file folder beside a 1000-file folder exactly zero picks, which is
    arithmetically right and useless here — the whole reason for sampling across
    trials is that every strain should appear at all. When there are fewer picks
    than groups the floor is simply unreachable and proportionality wins.
    """
    n = len(sizes)
    if n == 0 or total <= 0:
        return [0] * n
    capacity = sum(sizes)
    if total >= capacity:
        return list(sizes)

    quota = [total * s / capacity for s in sizes]
    take = [min(int(q), sizes[i]) for i, q in enumerate(quota)]
    # Hand out what rounding left over, best remainder first, skipping groups
    # already at capacity. Looping because granting one seat can only ever
    # exhaust a group, never create new room.
    while sum(take) < total:
        candidates = [i for i in range(n) if take[i] < sizes[i]]
        if not candidates:
            break
        candidates.sort(key=lambda i: (-(quota[i] - take[i]), -sizes[i], i))
        take[candidates[0]] += 1

    # Represent every non-empty group, funding it from whoever holds most.
    if total >= sum(1 for s in sizes if s > 0):
        while True:
            starved = [i for i in range(n) if sizes[i] > 0 and take[i] == 0]
            if not starved:
                break
            donor = max(range(n), key=lambda i: (take[i], sizes[i]))
            if take[donor] <= 1:
                break            # nobody can spare a pick without starving
            take[donor] -= 1
            take[min(starved)] += 1
    return take


def stride_pick(items: Sequence[str], k: int) -> List[str]:
    """``k`` items spread evenly over ``items``, endpoints included.

    Evenly spaced rather than random because the point is coverage: a random
    draw of 5 from a six-day recording can legitimately return five files from
    Tuesday afternoon, which is the exact failure this whole module exists to
    avoid. It is also reproducible with no seed to record.

    A single pick comes from the middle, not the start — the first file of a
    trial is the least representative one in it (animals still settling, rig
    just started).
    """
    n = len(items)
    if k <= 0 or n == 0:
        return []
    if k >= n:
        return list(items)
    if k == 1:
        return [items[n // 2]]
    return [items[round(i * (n - 1) / (k - 1))] for i in range(k)]


def random_pick(items: Sequence[str], k: int, rng: random.Random) -> List[str]:
    """``k`` items drawn without replacement, returned in the original order."""
    n = len(items)
    if k <= 0 or n == 0:
        return []
    if k >= n:
        return list(items)
    idx = sorted(rng.sample(range(n), k))
    return [items[i] for i in idx]


class SampleSpec:
    """What to draw. Recorded into the project so a draw can be reproduced.

    ``per``
        ``'all'`` takes everything; ``'folder'`` draws ``n`` from each folder;
        ``'total'`` draws ``n`` across the whole tree, apportioned by size.
    ``channel_mode``
        ``'spread'`` splits a folder's budget across its microphones (the
        default, and the reason this module exists); ``'only'`` keeps just the
        channels in ``channels``; ``'pool'`` ignores channels entirely, which is
        right for single-mic sets and wrong for these.
    ``spacing``
        ``'stride'`` for even coverage, ``'random'`` with ``seed`` when an
        unbiased draw actually matters.
    """

    def __init__(self, per: str = "folder", n: int = 20,
                 spacing: str = "stride", seed: Optional[int] = None,
                 channel_mode: str = "spread",
                 channels: Sequence[str] = ()):
        if per not in ("all", "folder", "total"):
            raise ValueError(f"per must be all/folder/total, got {per!r}")
        if spacing not in ("stride", "random"):
            raise ValueError(f"spacing must be stride/random, got {spacing!r}")
        if channel_mode not in ("spread", "only", "pool"):
            raise ValueError(
                f"channel_mode must be spread/only/pool, got {channel_mode!r}")
        self.per = per
        self.n = int(n)
        self.spacing = spacing
        self.seed = seed
        self.channel_mode = channel_mode
        self.channels = tuple(channels)

    def to_dict(self) -> dict:
        return {"per": self.per, "n": self.n, "spacing": self.spacing,
                "seed": self.seed, "channel_mode": self.channel_mode,
                "channels": list(self.channels)}

    @classmethod
    def from_dict(cls, d: dict) -> "SampleSpec":
        return cls(per=d.get("per", "folder"), n=int(d.get("n", 20)),
                   spacing=d.get("spacing", "stride"), seed=d.get("seed"),
                   channel_mode=d.get("channel_mode", "spread"),
                   channels=d.get("channels") or ())

    def describe(self) -> str:
        if self.per == "all":
            return "all recordings"
        where = "per folder" if self.per == "folder" else "in total"
        how = ("evenly spaced" if self.spacing == "stride"
               else f"random (seed {self.seed})")
        chan = {"spread": "spread across channels",
                "pool": "channels pooled",
                "only": "channels " + ", ".join(self.channels or ("—",)),
                }[self.channel_mode]
        return f"{self.n} {where}, {how}, {chan}"


class SampleResult:
    """The drawn paths plus the per-group breakdown the dialog previews."""

    def __init__(self, paths: List[str], rows: List[dict],
                 n_candidates: int, n_excluded: int, spec: SampleSpec):
        self.paths = paths
        self.rows = rows                  # folder / channel / available / picked
        self.n_candidates = n_candidates  # after channel filtering and exclusion
        self.n_excluded = n_excluded
        self.spec = spec

    def __len__(self) -> int:
        return len(self.paths)


class Prepared:
    """Recordings filtered and grouped, ready for any number of draws.

    Split out from :func:`sample_paths` because the two halves have very
    different costs and very different reasons to change. Grouping 33,996 paths
    — a ``dirname`` and a channel parse each, then a natural-order sort — takes
    0.89 s of the 1.06 s a full draw costs, and depends only on the channel
    policy and the exclusion set. The draw itself is milliseconds.

    The import dialog re-runs on every keystroke in its count box, so redoing
    the grouping each time made typing a two-digit number visibly sluggish. It
    prepares once per channel policy and redraws freely.
    """

    def __init__(self, groups, n_candidates: int, n_excluded: int):
        self.groups = groups
        self.keys = list(groups)
        self.sizes = [len(groups[k]) for k in self.keys]
        self.n_candidates = n_candidates
        self.n_excluded = n_excluded


def prepare(paths: Iterable[str], channel_mode: str = "spread",
            channels: Sequence[str] = (),
            exclude: Iterable[str] = ()) -> Prepared:
    """Filter and group ``paths`` for repeated draws under one channel policy.

    ``exclude`` (recordings already in the project) is applied *before* the
    draw, not after, so asking for twenty more genuinely yields twenty more,
    evenly spread over what is left rather than over the original series with
    holes punched in it. That is what makes a label / correct / retrain loop
    work: each pass tiles the remaining data instead of re-offering the files
    already reviewed.
    """
    skip = {os.path.normcase(os.path.abspath(p)) for p in exclude}
    want = {c.lower() for c in channels}

    candidates: List[str] = []
    n_excluded = 0
    for p in paths:
        if skip and os.path.normcase(os.path.abspath(p)) in skip:
            n_excluded += 1
            continue
        if channel_mode == "only" and want:
            ch = parse_channel(p)
            if ch is None or ch.lower() not in want:
                continue
        candidates.append(p)

    # 'only' has already filtered to the wanted mics, so the remaining files are
    # grouped by folder alone — otherwise a one-channel draw would be split
    # across groups that no longer differ in any way that matters.
    groups = group_paths(candidates, by_channel=(channel_mode == "spread"))
    return Prepared(groups, len(candidates), n_excluded)


def draw(prep: Prepared, spec: SampleSpec) -> SampleResult:
    """Take a sample from an already-grouped :class:`Prepared`."""
    groups, keys, sizes = prep.groups, prep.keys, prep.sizes

    if spec.per == "all":
        take = list(sizes)
    elif spec.per == "total":
        take = allocate(sizes, spec.n)
    else:
        # Per *folder*, which is not the same as per group once channels split a
        # folder into four: the budget is the folder's, apportioned across its
        # own microphones. Asking for 20 with 4 channels means 5 each, not 80.
        take = [0] * len(keys)
        folders: Dict[str, List[int]] = {}
        for i, k in enumerate(keys):
            folders.setdefault(k[0], []).append(i)
        for idxs in folders.values():
            for j, share in zip(idxs, allocate([sizes[i] for i in idxs], spec.n)):
                take[j] = share

    rng = random.Random(spec.seed) if spec.spacing == "random" else None
    picked: List[str] = []
    rows: List[dict] = []
    for k, size, k_take in zip(keys, sizes, take):
        items = groups[k]
        got = (stride_pick(items, k_take) if rng is None
               else random_pick(items, k_take, rng))
        picked.extend(got)
        rows.append({"folder": k[0], "channel": (k[1] if len(k) > 1 else ""),
                     "available": size, "picked": len(got)})
    picked.sort(key=lambda p: (os.path.dirname(os.path.abspath(p)),
                               natural_key(p)))
    return SampleResult(picked, rows, prep.n_candidates, prep.n_excluded, spec)


def sample_paths(paths: Iterable[str], spec: SampleSpec,
                 exclude: Iterable[str] = ()) -> SampleResult:
    """Draw a subset of ``paths`` according to ``spec`` — prepare, then draw."""
    return draw(prepare(paths, spec.channel_mode, spec.channels, exclude), spec)
