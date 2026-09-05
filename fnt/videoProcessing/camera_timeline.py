"""Wall-clock timeline reconstruction for NVR/DVR camera segment folders.

This is the synchronisation engine behind the Camera Grid tool. It is
deliberately free of Qt and of any I/O beyond ffprobe, so it can be unit
tested directly.

THE PROBLEM
-----------
An NVR records continuously and splits each camera's footage into many files.
Those splits are size-based, not time-based, so one camera may write 21 files a
day where another writes 10, and no two cameras' file boundaries line up. To
show several cameras side by side "as if watching a live feed", every camera has
to be placed on a shared wall-clock timeline -- and a camera that dropped out
must hold its place with black rather than sliding its later footage earlier.

WHAT THE FILENAMES ACTUALLY MEAN
--------------------------------
Files look like::

    VoleCosm_Camera3_{GUID}_20260224000000(007)_processed.mp4

The 14-digit stamp is a DAY BUCKET (midnight), not that file's start time --
every file recorded that day repeats it. ``(NNN)`` is the split index within the
day, and the file with NO index is the day's FIRST segment. Correct ordering is
therefore ``(day, seq)`` with the un-suffixed file sorting as seq 0.

HOW A SEGMENT'S TRUE TIME IS RECOVERED
--------------------------------------
Verified against ten days of five cameras: for a complete day a camera's segment
durations sum to 86400s within about two seconds. So::

    segment_start = day_bucket + sum(durations of earlier segments that day)

That is exact and needs no OCR. It was cross-checked against the cameras' own
burnt-in clocks, which agreed to within two seconds.

HOW DROPOUTS ARE FOUND
----------------------
A dropout leaves a HOLE IN THE SEQUENCE NUMBERS -- the NVR keeps counting, so a
missing ``(006)`` is a missing segment. Its length is the day's shortfall
(expected seconds minus the durations actually present), which lets the gap be
reinserted at exactly the right position so that everything after it keeps its
true wall-clock time.

A day can also come up short with no missing sequence number (observed once:
about 19 minutes lost with all files present). That loss is real but cannot be
localised, so it is recorded at the end of the day and flagged ``unlocated`` --
the caller should surface it, since footage after it may be shifted.
"""

import datetime
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

DAY_SECONDS = 86400

# <stamp>(<seq>) with an optional processing suffix; the seq group is absent for
# the first segment of a day.
SEGMENT_RE = re.compile(
    r'_(?P<stamp>\d{14})(?:\((?P<seq>\d+)\))?(?:_[A-Za-z0-9]+)?\.(?P<ext>mp4|avi|mkv|mov)$',
    re.IGNORECASE)

VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov")

#: How a recorder names and organises its files. The default understands
#: ViewTron/Hikvision-style ``<stamp>(<seq>)`` naming, where the stamp is an
#: offload-window bucket and the sequence counts splits within it -- which is
#: what makes both wall-clock anchoring and dropout detection possible.
#: "alphabetical" is the fallback for a recorder we do not model: files are
#: simply chained in name order, so playback is continuous but a dropout cannot
#: be detected and cells will drift if footage is missing.
PROFILES = ("viewtron", "alphabetical")
DEFAULT_PROFILE = "viewtron"


def _no_window():
    """Keep ffprobe from flashing a console window on Windows."""
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}
    return {}


class Segment:
    """One recorded file, once its place on the wall clock is known."""

    __slots__ = ("path", "day", "seq", "duration", "start", "end")

    def __init__(self, path, day, seq, duration=None):
        self.path = path
        self.day = day              # 'YYYYMMDDHHMMSS' day-bucket stamp
        self.seq = seq              # split index within that day (0 = first)
        self.duration = duration    # seconds, from ffprobe
        self.start = None           # datetime, filled in by build_timeline
        self.end = None

    def __repr__(self):
        return (f"Segment({os.path.basename(self.path)!r}, seq={self.seq}, "
                f"start={self.start}, dur={self.duration})")


class Gap:
    """A stretch of wall-clock time for which a camera has no footage."""

    __slots__ = ("start", "duration", "seq", "unlocated")

    def __init__(self, start, duration, seq=None, unlocated=False):
        self.start = start
        self.duration = duration
        self.seq = seq              # the sequence number that is missing, if known
        # True when a day is short but no sequence number is missing: we know
        # footage was lost but not where, so later times may be unreliable.
        self.unlocated = unlocated

    @property
    def end(self):
        return self.start + datetime.timedelta(seconds=self.duration)

    def __repr__(self):
        kind = "unlocated" if self.unlocated else f"seq {self.seq}"
        return f"Gap({self.start}, {self.duration:.0f}s, {kind})"


def probe_duration(path):
    """Duration of `path` in seconds, or None if ffprobe can't read it."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, timeout=120, **_no_window())
        return float(r.stdout.strip())
    except (ValueError, OSError, subprocess.SubprocessError):
        return None


def parse_segment_name(filename):
    """(day_stamp, seq) for an NVR segment filename, or None if it isn't one."""
    m = SEGMENT_RE.search(filename)
    if not m:
        return None
    return m.group("stamp"), int(m.group("seq") or 0)


def scan_folder(folder, durations=True, workers=16, progress=None):
    """Discover segments in `folder` and probe their durations.

    Only files whose names carry the ``<stamp>(<seq>)`` structure are used;
    anything else in the folder is ignored, so a stray export or a notes file
    can't corrupt the timeline.
    """
    found = []
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    for name in names:
        if not name.lower().endswith(VIDEO_EXTS):
            continue
        parsed = parse_segment_name(name)
        if parsed is None:
            continue
        day, seq = parsed
        found.append(Segment(os.path.join(folder, name), day, seq))

    found.sort(key=lambda s: (s.day, s.seq))
    if not durations:
        return found

    done = [0]

    def probe(seg):
        seg.duration = probe_duration(seg.path)
        done[0] += 1
        if progress and done[0] % 10 == 0:
            progress(done[0], len(found))
        return seg

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(probe, found))
    return [s for s in found if s.duration]


#: A camera identifier embedded in the filename, e.g. "Camera3" or "cam07".
#: This is the recorder's own label for the device, so it stays correct whether
#: the files sit in per-camera subfolders or loose in one trial folder.
CAMERA_TOKEN_RE = re.compile(r'(?i)(?:^|[-_\s])((?:camera|cam)[-_\s]?\d+)')


def camera_token(filename):
    """The Camera<N> label inside a filename, normalised, or None."""
    m = CAMERA_TOKEN_RE.search(os.path.basename(filename))
    if not m:
        return None
    return re.sub(r'[-_\s]', '', m.group(1))


def discover_cameras(root, max_depth=1):
    """Group the segment files under `root` into cameras.

    Handles the layouts that actually turn up in trial folders:
      - one subfolder per camera (cam1/, cam2/, ...)
      - every camera's files loose in a single trial folder
      - a mixture, including empty leftover subfolders

    Files are grouped by the Camera<N> label in their own name, falling back to
    the containing folder when a recorder does not embed one. That is what lets
    a single trial folder be pointed at directly, instead of the footage having
    to be sorted into subfolders by hand first.

    Returns {camera_name: [paths]}, ordered by camera name.
    """
    groups = {}
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath[len(root):].count(os.sep)
        if depth >= max_depth:
            dirnames[:] = []          # don't descend further
        for name in filenames:
            if not name.lower().endswith(VIDEO_EXTS):
                continue
            if parse_segment_name(name) is None:
                continue
            key = camera_token(name)
            if key is None:
                folder = os.path.basename(dirpath)
                key = folder if folder != os.path.basename(root) else "camera"
            groups.setdefault(key, []).append(os.path.join(dirpath, name))
    return {k: sorted(v) for k, v in sorted(groups.items())}


def scan_paths(paths, workers=16, progress=None):
    """Build Segments (with durations) from an explicit list of files."""
    found = []
    for path in paths:
        parsed = parse_segment_name(os.path.basename(path))
        if parsed:
            found.append(Segment(path, parsed[0], parsed[1]))
    found.sort(key=lambda s: (s.day, s.seq))

    done = [0]

    def probe(seg):
        seg.duration = probe_duration(seg.path)
        done[0] += 1
        if progress and done[0] % 10 == 0:
            progress(done[0], len(found))
        return seg

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(probe, found))
    return [s for s in found if s.duration]


def _expected_bucket_seconds(stamp, next_stamp=None):
    """How long the bucket starting at `stamp` should run.

    Measured from the NEXT bucket's stamp rather than assumed to end at
    midnight. Offload windows are a choice the operator makes, not a property
    of the recorder: midnight-to-midnight is one convention, but a trial
    offloaded 06:00-to-06:00 is equally valid, and judging its buckets against
    a calendar day would invent hours of phantom "gap". Consecutive stamps
    state the real spacing whatever convention was used.

    Falls back to the remainder of the calendar day when there is no following
    bucket to measure against.
    """
    dt = datetime.datetime.strptime(stamp, "%Y%m%d%H%M%S")
    if next_stamp:
        nxt = datetime.datetime.strptime(next_stamp, "%Y%m%d%H%M%S")
        span = (nxt - dt).total_seconds()
        if span > 0:
            return span
    into_day = dt.hour * 3600 + dt.minute * 60 + dt.second
    return DAY_SECONDS - into_day


def build_timeline(segments, tolerance=60.0, profile=DEFAULT_PROFILE):
    """Place every segment on the wall clock and describe the gaps between them.

    `tolerance` is how many seconds a day may fall short before it's treated as
    lost footage rather than rounding.

    Returns (segments, gaps) with `start`/`end` populated on each segment.
    """
    if profile == "alphabetical":
        return _chain_in_order(segments)

    by_day = {}
    for s in segments:
        by_day.setdefault(s.day, []).append(s)

    days = sorted(by_day)
    placed, gaps = [], []

    for i, day in enumerate(days):
        segs = sorted(by_day[day], key=lambda s: s.seq)
        base = datetime.datetime.strptime(day, "%Y%m%d%H%M%S")
        total = sum(s.duration for s in segs)
        seqs = [s.seq for s in segs]
        missing = sorted(set(range(0, max(seqs) + 1)) - set(seqs))

        # The final day ends whenever recording stopped, so its length tells us
        # nothing -- we can't infer a gap length from a shortfall there.
        is_last = (i == len(days) - 1)
        expected = _expected_bucket_seconds(
            day, days[i + 1] if not is_last else None)
        deficit = 0.0 if is_last else max(0.0, expected - total)

        if missing:
            if deficit > tolerance:
                per_gap = deficit / len(missing)
            else:
                # Missing file but no measurable shortfall (or a final day):
                # fall back to this day's typical segment length.
                per_gap = total / len(segs) if segs else 0.0
        else:
            per_gap = 0.0

        cursor = 0.0
        pending = list(missing)
        for s in segs:
            while pending and pending[0] < s.seq:
                if per_gap > 0:
                    gaps.append(Gap(base + datetime.timedelta(seconds=cursor),
                                    per_gap, seq=pending[0]))
                    cursor += per_gap
                pending.pop(0)
            s.start = base + datetime.timedelta(seconds=cursor)
            s.end = s.start + datetime.timedelta(seconds=s.duration)
            placed.append(s)
            cursor += s.duration

        # Short day with every sequence number present: footage was lost but we
        # can't say where, so record it at the end and flag it.
        if not is_last:
            unexplained = expected - cursor
            if unexplained > tolerance:
                gaps.append(Gap(base + datetime.timedelta(seconds=cursor),
                                unexplained, unlocated=True))

    return placed, gaps


def _chain_in_order(segments):
    """Lay segments end to end in filename order, inventing no gaps.

    For recorders whose naming we do not model: playback is continuous and in
    the right order, but nothing tells us when footage is MISSING, so cameras
    can drift apart if any recorder dropped out. Reported as such in the UI.
    """
    ordered = sorted(segments, key=lambda s: (s.day, s.seq, s.path))
    if not ordered:
        return [], []
    base = datetime.datetime.strptime(ordered[0].day, "%Y%m%d%H%M%S")
    cursor = 0.0
    for seg in ordered:
        seg.start = base + datetime.timedelta(seconds=cursor)
        seg.end = seg.start + datetime.timedelta(seconds=seg.duration)
        cursor += seg.duration
    return ordered, []


class CameraTrack:
    """One camera's segments arranged on the wall clock."""

    def __init__(self, name, folder, segments=None, gaps=None):
        self.name = name
        self.folder = folder
        self.segments = segments or []
        self.gaps = gaps or []
        # Manual sync correction, in seconds. Reconstructing time from the day
        # stamp plus accumulated duration lands each camera within about a
        # second, but every camera carries its own small, stable offset from
        # that anchor -- measured at a ~1s spread across one trial, steady to
        # within 0.1s over a full day.
        #
        # SIGN: positive shifts this camera's footage LATER, so its burnt-in
        # clock advances. That matches the calibration dialog's ">" button. A
        # camera whose clock reads ahead of the others is therefore nudged
        # NEGATIVE to bring it back into line. Keep locate() and collect_runs()
        # agreeing with this, or the buttons run backwards.
        self.clock_offset = 0.0

    def copy(self):
        """A copy that keeps its own clock_offset.

        Segments are shared: they are not mutated after the timeline is built,
        and only the offset needs to be independent so a queued job holds the
        calibration it was added with.
        """
        clone = CameraTrack(self.name, self.folder, self.segments, self.gaps)
        clone.clock_offset = self.clock_offset
        return clone

    @classmethod
    def from_paths(cls, name, paths, folder=None, progress=None,
                   profile=DEFAULT_PROFILE):
        """Build a track from an explicit file list (used by auto-detection)."""
        segs = scan_paths(paths, progress=progress)
        placed, gaps = build_timeline(segs, profile=profile)
        return cls(name, folder or os.path.dirname(paths[0]) if paths else "",
                   placed, gaps)

    @classmethod
    def from_folder(cls, folder, name=None, progress=None,
                    profile=DEFAULT_PROFILE):
        segs = scan_folder(folder, progress=progress)
        placed, gaps = build_timeline(segs, profile=profile)
        return cls(name or infer_camera_name(folder, placed),
                   folder, placed, gaps)

    @property
    def start(self):
        return self.segments[0].start if self.segments else None

    @property
    def end(self):
        return self.segments[-1].end if self.segments else None

    @property
    def footage_seconds(self):
        return sum(s.duration for s in self.segments)

    @property
    def span_seconds(self):
        if not self.segments:
            return 0.0
        return (self.end - self.start).total_seconds()

    @property
    def gap_seconds(self):
        return sum(g.duration for g in self.gaps)

    def locate(self, when):
        """(path, offset_seconds) for wall-clock `when`, or (None, None).

        (None, None) means this camera has no footage at that moment and the
        caller should render black.
        """
        # Positive offset = show later footage, so look further along this
        # camera's own timeline.
        if self.clock_offset:
            when = when + datetime.timedelta(seconds=self.clock_offset)
        lo, hi = 0, len(self.segments) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            s = self.segments[mid]
            if when < s.start:
                hi = mid - 1
            elif when >= s.end:
                lo = mid + 1
            else:
                return s.path, (when - s.start).total_seconds()
        return None, None

    def coverage(self, start, end, buckets=800):
        """Fraction of footage present in each of `buckets` slices of the window.

        Used to draw the coverage strip in the UI so dropouts are visible before
        committing to a long encode.
        """
        total = (end - start).total_seconds()
        if total <= 0:
            return []
        width = total / buckets
        out = []
        for i in range(buckets):
            b0 = start + datetime.timedelta(seconds=i * width)
            b1 = b0 + datetime.timedelta(seconds=width)
            covered = 0.0
            for s in self.segments:
                if s.end <= b0 or s.start >= b1:
                    continue
                covered += (min(s.end, b1) - max(s.start, b0)).total_seconds()
                if covered >= width:
                    break
            out.append(min(1.0, covered / width) if width else 0.0)
        return out


def infer_camera_name(folder, segments=None):
    """Best-effort display name for a camera folder.

    Uses the folder name when it looks specific (``cam3``), otherwise pulls a
    ``Camera<N>`` token out of a filename -- which is what makes flat trial
    folders (no ``cam*`` subdirectories) work.
    """
    base = os.path.basename(os.path.normpath(folder))
    if re.fullmatch(r'(?i)cam\w*\d+', base):
        return base
    for seg in (segments or [])[:5]:
        m = re.search(r'(?i)(camera\s*\d+|cam\s*\d+)', os.path.basename(seg.path))
        if m:
            return m.group(1).replace(" ", "")
    return base


class TrialTimeline:
    """Several CameraTracks sharing one wall-clock window."""

    def __init__(self, tracks):
        self.tracks = [t for t in tracks if t.segments]

    @property
    def start(self):
        return min((t.start for t in self.tracks), default=None)

    @property
    def end(self):
        return max((t.end for t in self.tracks), default=None)

    @property
    def duration_seconds(self):
        if not self.tracks:
            return 0.0
        return (self.end - self.start).total_seconds()

    def chunks(self, mode="daily"):
        """Split the trial window into output chunks.

        "daily"      -- one chunk per calendar day, so a chunk's name matches
                        the date burnt into the footage.
        "continuous" -- a single chunk covering the whole trial.
        "both"       -- the daily chunks; the full-trial file is produced by
                        joining those afterwards rather than encoding again.
        """
        if not self.tracks:
            return []
        start, end = self.start, self.end
        if mode == "continuous":
            return [(start, end)]

        out = []
        cursor = start
        while cursor < end:
            midnight = datetime.datetime.combine(
                cursor.date() + datetime.timedelta(days=1), datetime.time.min)
            out.append((cursor, min(midnight, end)))
            cursor = midnight
        return out
