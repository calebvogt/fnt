"""Lost frames in videos that carry their own timestamps (mp4, mkv, mov, ...).

A recorder that drops frames, or a file with damaged stretches, leaves a jump in the
video's timestamps. FFmpeg's ``fps`` filter (Video PreProcessing's encode) keeps the
video at its real length across the jump, but fills it with copies of the last frame
before it: a still picture that a tracker reads as an animal sitting still. This
module finds the jumps from the packet timestamps (no decoding) so the encode can draw
those frames black instead, and lists them in the same frames CSV as damaged AVIs
(:func:`fnt.videoProcessing.avi_recovery.write_frames_csv`).

AVIs are not covered here: their timestamps are frame counts, so a lost frame leaves
no jump (see :mod:`fnt.videoProcessing.avi_recovery`).
"""
import subprocess
from dataclasses import dataclass, field

import numpy as np

# A jump counts as lost frames when it is at least this many frame intervals long, so
# the jitter of a variable-frame-rate camera (intervals +/- half a frame) is not a gap.
MIN_GAP_INTERVALS = 2.5
# More gaps than this are merged where they are close, to keep the filter short.
MAX_GAPS = 400


@dataclass
class FrameGaps:
    """The lost stretches of one video, in source seconds."""
    first_s: float                   # timestamp of the first frame
    last_s: float                    # timestamp of the last frame
    interval_s: float                # the video's own frame interval (median)
    gaps: list = field(default_factory=list)   # (last frame before, first frame after), s
    frames: int = 0                  # frames with a timestamp

    @property
    def lost_s(self):
        return sum(b - a - self.interval_s for a, b in self.gaps)


def frame_times(path, timeout=600):
    """Presentation timestamps (s) of the first video stream's packets, sorted, or
    None when FFprobe cannot read them. Reads the packets, decodes nothing."""
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "packet=pts_time", "-of", "csv=p=0", path],
                           capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None
    if r.returncode != 0:
        return None
    t = []
    for ln in r.stdout.split():
        ln = ln.strip().rstrip(",")
        try:
            t.append(float(ln))
        except ValueError:
            continue                 # N/A: a packet without a timestamp
    if len(t) < 3:
        return None
    return np.unique(np.asarray(t))


def find_gaps(times, min_intervals=MIN_GAP_INTERVALS):
    """The jumps in ``times`` (sorted s) of at least ``min_intervals`` frame intervals."""
    d = np.diff(times)
    iv = float(np.median(d))
    if iv <= 0:
        return None
    k = np.nonzero(d >= min_intervals * iv)[0]
    gaps = [(float(times[i]), float(times[i + 1])) for i in k]
    if len(gaps) > MAX_GAPS:                     # merge the closest until few enough
        gaps = _merge(gaps, MAX_GAPS)
    return FrameGaps(float(times[0]), float(times[-1]), iv, gaps, len(times))


def _merge(gaps, n):
    gaps = sorted(gaps)
    while len(gaps) > n:
        j = min(range(len(gaps) - 1), key=lambda i: gaps[i + 1][0] - gaps[i][1])
        gaps[j:j + 2] = [(gaps[j][0], gaps[j + 1][1])]
    return gaps


def probe(path):
    """:class:`FrameGaps` of a video, or None when its timestamps can't be read."""
    t = frame_times(path)
    return find_gaps(t) if t is not None else None


def _windows(fg, out_fps):
    """Per gap, the output-time window whose frames are black: from half a source
    interval after the last real frame to half an output frame before the next."""
    return [(a + 0.5 * fg.interval_s, b - 0.5 / out_fps) for a, b in fg.gaps]


def blackout_filter(fg, out_fps):
    """A ``drawbox`` filter that paints the gap frames black (goes right after the
    ``fps`` filter, where ``t`` is still the source's time), or None if no gaps."""
    if not fg or not fg.gaps:
        return None
    expr = "+".join(f"between(t,{a:.4f},{b:.4f})" for a, b in _windows(fg, out_fps))
    return f"drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable='{expr}'"


def frame_rows(fg, out_fps):
    """``(first_frame, last_frame, 'recovered'|'black', 'timestamps')`` stretches of the
    encode (frame k at the source time first_s + k / out_fps), and the frame count."""
    n = int(round((fg.last_s - fg.first_s) * out_fps)) + 1
    black = np.zeros(n, bool)
    t = fg.first_s + np.arange(n) / out_fps
    for a, b in _windows(fg, out_fps):
        black |= (t >= a - 1e-6) & (t <= b + 1e-6)
    rows, start = [], 0
    for k in range(1, n + 1):
        if k == n or black[k] != black[start]:
            rows.append((start, k - 1, "black" if black[start] else "recovered", "timestamps"))
            start = k
    return rows, n, int(black.sum())
