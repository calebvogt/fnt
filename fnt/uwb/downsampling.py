"""Random downsampling of a tag's reads, before filtering and smoothing.

Tags flashed with different rate pairs - 10/1, 4/1, 1/1 - report the same
movement with very different numbers of reads, and everything downstream of
the read is sensitive to that. The thresholds compare consecutive reads, and
Savitzky-Golay smooths over a fixed 31 of them: about 3-4 s of a 10/1 tag's
track but 30-40 s of a 1/1 tag's. So distance, speed and chases come out
lower for an animal on a slow tag because of the tag, not the animal (on the
VoleCosm anosmia trials, by about 70 % for distance and 87 % for chases).
Downsampling every tag to the same ceiling before any of that runs gives
every animal the same input.

Each tag's time line is cut into bins of 1/hz seconds, counted from the Unix
epoch, and one read is kept per bin. Which one is random, but reproducible:
the read with the smallest seeded hash of (tag, timestamp) wins. The pick
therefore depends on nothing but the read itself, so the preview, which
processes the trial in time slices, keeps exactly the reads the export keeps,
and re-running an export with the same seed gives the same file. A tag that
already reports at or below the ceiling loses only the rare second read that
lands in a bin it already has a read in.

Nothing here imports Qt, so it is testable without a GUI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DOWNSAMPLING_NONE = "None (all reads)"
DOWNSAMPLING_RANDOM = "Random"
DOWNSAMPLING_METHODS = (DOWNSAMPLING_NONE, DOWNSAMPLING_RANDOM)
DEFAULT_DOWNSAMPLING_HZ = 1.0
DEFAULT_DOWNSAMPLING_SEED = 0

_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)


def _mix64(x):
    """splitmix64's finaliser: a well-spread 64-bit hash of each element."""
    x = np.array(x, dtype=np.uint64, copy=True)
    with np.errstate(over="ignore"):
        x ^= x >> np.uint64(30)
        x *= _M1
        x ^= x >> np.uint64(27)
        x *= _M2
        x ^= x >> np.uint64(31)
    return x


def _timestamps_ms(data, time_col):
    """Epoch milliseconds of every read, from ``time_col`` or 'Timestamp'."""
    if time_col in data.columns:
        return data[time_col].to_numpy(dtype="int64")
    ts = data["Timestamp"]
    origin = pd.Timestamp(0, tz=ts.dt.tz) if ts.dt.tz is not None else pd.Timestamp(0)
    return ((ts - origin) // pd.Timedelta(milliseconds=1)).to_numpy(dtype="int64")


def downsample_keep(timestamp_ms, shortid, hz, seed=DEFAULT_DOWNSAMPLING_SEED):
    """Boolean mask, True for the reads that random downsampling keeps.

    ``timestamp_ms`` and ``shortid`` are equal-length arrays (any order). At
    most one read is kept per tag per 1/hz-second bin.
    """
    ms = np.asarray(timestamp_ms, dtype=np.int64)
    tag = np.asarray(shortid, dtype=np.int64)
    n = len(ms)
    keep = np.zeros(n, dtype=bool)
    if n == 0:
        return keep
    bins = np.floor(ms * (float(hz) / 1000.0)).astype(np.int64)
    with np.errstate(over="ignore"):
        salt = _mix64(tag.astype(np.uint64) * _GOLDEN
                      + np.uint64(int(seed) & 0xFFFFFFFFFFFFFFFF))
    key = _mix64(ms.astype(np.uint64) ^ salt)
    # Sorted by tag, then bin, then hash: the first row of each (tag, bin)
    # run is the read that bin keeps.
    order = np.lexsort((key, bins, tag))
    first = np.ones(n, dtype=bool)
    first[1:] = ((tag[order][1:] != tag[order][:-1])
                 | (bins[order][1:] != bins[order][:-1]))
    keep[order[first]] = True
    return keep


def downsample_reads(data, hz, seed=DEFAULT_DOWNSAMPLING_SEED,
                     time_col="timestamp", tag_col="shortid"):
    """``data`` with at most one read per tag per 1/hz-second bin.

    Rows keep their original order and index. Time comes from ``time_col``
    (epoch ms) when present, otherwise from the 'Timestamp' datetimes; a frame
    without ``tag_col`` is treated as one tag. ``hz`` of None or <= 0 returns
    ``data`` unchanged.
    """
    if not hz or hz <= 0 or len(data) == 0:
        return data
    ms = _timestamps_ms(data, time_col)
    tag = (data[tag_col].to_numpy(dtype="int64") if tag_col in data.columns
           else np.zeros(len(data), dtype=np.int64))
    return data[downsample_keep(ms, tag, hz, seed)]
