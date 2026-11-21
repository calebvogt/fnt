"""Device schedules: the text a FED3 running firmware v3 executes.

A schedule is what FNT uploads and what the device keeps on its SD card. This
module is the contract between them: the text format, the limits the firmware
enforces, the canonical form that makes the crc meaningful, and the reference
semantics of conditions and windows. The firmware is written to agree with
this file, and the fake device in the tests is built on it.

See ``docs/fed3/design.md`` (Part 2) for the format itself.
"""
from __future__ import annotations

import re
import time
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

MAX_PHASES = 256
MAX_WINDOWS = 8          # distinct time ranges per phase
MAX_UNTILS = 6
MAX_BYTES = 64 * 1024
# Longest directive. Uploaded as "S:<line>\n", which keeps every upload line
# inside the firmware's 128-byte input buffer.
MAX_LINE = 100
MAX_NAME = 24            # phase names
MAX_CUE_ACTIONS = 3

SECONDS_PER_DAY = 86400


class ScheduleError(ValueError):
    """A schedule line the device would refuse."""

    def __init__(self, line_no, message):
        super().__init__(f"line {line_no}: {message}")
        self.line_no = line_no
        self.message = message


# ---------------------------------------------------------------------------
# Rule keys
# ---------------------------------------------------------------------------

def _enum(*choices):
    def parse(text):
        if text not in choices:
            raise ValueError(f"must be one of: {', '.join(choices)}")
        return text
    return parse


def _int(lo, hi):
    def parse(text):
        if not re.fullmatch(r"\d+", text):
            raise ValueError("must be a whole number")
        value = int(text)
        if not lo <= value <= hi:
            raise ValueError(f"must be between {lo} and {hi}")
        return str(value)
    return parse


def _cue(text):
    return parse_cue(text)


@dataclass(frozen=True)
class Key:
    name: str
    default: str
    parse: object
    doc: str = ""


# Canonical order. Compiled SET lines list keys in exactly this order, which is
# what makes the same protocol always produce the same text and the same crc.
KEYS = (
    Key("active", "left", _enum("left", "right", "both", "none")),
    Key("count", "shared", _enum("shared", "per_port")),
    Key("ratio", "1", _int(1, 999)),
    Key("ratio_kind", "fixed", _enum("fixed", "random", "progressive")),
    Key("timeout", "0", _int(0, 3600)),
    Key("min_poke_ms", "0", _int(0, 5000)),
    Key("reward", "pellet", _enum("pellet", "none")),
    Key("quota", "0", _int(0, 9999)),
    Key("quota_period", "0", _int(0, 7 * SECONDS_PER_DAY)),
    Key("free_feed", "0", _enum("0", "1")),
    Key("refill_delay", "5", _int(0, 3600)),
    Key("cue.reward", "none", _cue),
    Key("cue.counted", "none", _cue),
    Key("cue.inactive", "none", _cue),
    Key("cue.timeout", "none", _cue),
    Key("cue.depleted", "none", _cue),
)
KEY_BY_NAME = {key.name: key for key in KEYS}
KEY_ORDER = tuple(key.name for key in KEYS)
DEFAULTS = {key.name: key.default for key in KEYS}
SECONDS_KEYS = ("timeout", "quota_period", "refill_delay")


def parse_value(key, text):
    """Validate one ``key=value`` and return the value in canonical form."""
    if key not in KEY_BY_NAME:
        raise ValueError(f"unknown key '{key}'")
    try:
        return KEY_BY_NAME[key].parse(text)
    except ValueError as exc:
        raise ValueError(f"{key}: {exc}") from None


# ---------------------------------------------------------------------------
# Cues
# ---------------------------------------------------------------------------

FLASH_TARGETS = ("bar", "poke_l", "poke_r", "pokes", "poke")
# Which LEDs each target can light, to catch two flashes fighting over one LED.
# "poke" is whichever poke triggered the cue, so it may be either.
_FLASH_OUTPUTS = {
    "bar": {"bar"},
    "poke_l": {"L"},
    "poke_r": {"R"},
    "pokes": {"L", "R"},
    "poke": {"L", "R"},
}


def _cue_int(action, name, text, lo, hi):
    if not re.fullmatch(r"\d+", text or ""):
        raise ValueError(f"{action}: {name} must be a whole number")
    value = int(text)
    if not lo <= value <= hi:
        raise ValueError(f"{action}: {name} must be between {lo} and {hi}")
    return value


def parse_cue(text):
    """Validate a cue (``none`` or up to 3 actions joined by ``+``)."""
    if text == "none":
        return text
    actions = text.split("+")
    if len(actions) > MAX_CUE_ACTIONS:
        raise ValueError(f"has {len(actions)} actions; a cue plays at most "
                         f"{MAX_CUE_ACTIONS}")
    sounds = 0
    pulses = 0
    lit = set()
    canonical = []
    for action in actions:
        kind, *args = action.split(":")
        if kind == "click":
            if args:
                raise ValueError("click takes no parameters")
            sounds += 1
            canonical.append("click")
        elif kind == "tone":
            if len(args) != 2:
                raise ValueError("tone is tone:<hz>:<ms>")
            hz = _cue_int("tone", "hz", args[0], 20, 20000)
            ms = _cue_int("tone", "ms", args[1], 1, 10000)
            sounds += 1
            canonical.append(f"tone:{hz}:{ms}")
        elif kind == "noise":
            if len(args) != 1:
                raise ValueError("noise is noise:<ms>")
            ms = _cue_int("noise", "ms", args[0], 1, 10000)
            sounds += 1
            canonical.append(f"noise:{ms}")
        elif kind == "flash":
            if len(args) != 6:
                raise ValueError("flash is flash:<target>:<r>:<g>:<b>:<w>:<ms>")
            target = args[0]
            if target not in FLASH_TARGETS:
                raise ValueError(f"flash target must be one of: "
                                 f"{', '.join(FLASH_TARGETS)}")
            rgbw = [_cue_int("flash", "colour", c, 0, 255) for c in args[1:5]]
            ms = _cue_int("flash", "ms", args[5], 1, 60000)
            if lit & _FLASH_OUTPUTS[target]:
                raise ValueError("two flashes light the same LEDs")
            lit |= _FLASH_OUTPUTS[target]
            canonical.append(f"flash:{target}:{':'.join(map(str, rgbw))}:{ms}")
        elif kind == "bnc":
            if len(args) != 3:
                raise ValueError("bnc is bnc:<width_ms>:<hz>:<count>")
            width = _cue_int("bnc", "width", args[0], 1, 1000)
            hz = _cue_int("bnc", "hz", args[1], 1, 500)
            count = _cue_int("bnc", "count", args[2], 1, 1000)
            if width * hz >= 1000:
                raise ValueError("bnc: the pulse width must be shorter than "
                                 "the period (1000/hz ms)")
            pulses += 1
            canonical.append(f"bnc:{width}:{hz}:{count}")
        else:
            raise ValueError(f"unknown cue action '{kind}'")
    if sounds > 1:
        raise ValueError("plays two sounds at once; the buzzer plays one at a time")
    if pulses > 1:
        raise ValueError("has two BNC pulse trains; there is one BNC output")
    return "+".join(canonical)


# ---------------------------------------------------------------------------
# Times
# ---------------------------------------------------------------------------

def format_clock(minutes):
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def parse_clock(text):
    """``HH:MM`` to minutes after midnight."""
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", text)
    if not match or int(match[1]) > 23 or int(match[2]) > 59:
        raise ValueError(f"'{text}' is not a clock time (HH:MM, 00:00-23:59)")
    return int(match[1]) * 60 + int(match[2])


def format_seconds(seconds):
    """3600 -> '1h', 90 -> '1m30s', 0 -> '0s'."""
    if seconds == 0:
        return "0s"
    out = []
    for unit, size in (("d", SECONDS_PER_DAY), ("h", 3600), ("m", 60), ("s", 1)):
        if seconds >= size:
            out.append(f"{seconds // size}{unit}")
            seconds %= size
    return "".join(out)


def plural(count, noun):
    """``1 phase``, ``3 phases``."""
    return f"{count} {noun}{'' if count == 1 else 's'}"


def standard_timezone():
    """The PC's local standard time: its UTC offset with daylight saving ignored.

    Devices, schedules and FNT's displays all run on this, so a clock time in a
    protocol stays put against a vivarium light cycle that ignores DST.
    """
    return timezone(timedelta(seconds=-time.timezone))


# ---------------------------------------------------------------------------
# Conditions and windows
# ---------------------------------------------------------------------------

UNTIL_KINDS = ("after", "at", "rewards", "pokes", "goto")
SIDES = ("left", "right", "any")


@dataclass(frozen=True)
class Until:
    """One ``UNTIL`` line."""

    kind: str
    seconds: int = 0
    clock: int = 0          # minutes after midnight
    day: int = 0            # 0: the next occurrence
    count: int = 0
    side: str = "any"

    def text(self):
        if self.kind == "after":
            return f"after={self.seconds}"
        if self.kind == "at":
            day = f" day={self.day}" if self.day else ""
            return f"at={format_clock(self.clock)}{day}"
        if self.kind == "rewards":
            return f"rewards={self.count}"
        if self.kind == "pokes":
            return f"pokes={self.count} side={self.side}"
        return "goto"

    @property
    def is_time(self):
        return self.kind in ("after", "at")


def parse_until(text):
    tokens = text.split()
    if not tokens:
        raise ValueError("UNTIL needs a condition")
    if tokens == ["goto"]:
        return Until("goto")
    fields = {}
    for token in tokens:
        key, sep, value = token.partition("=")
        if not sep or key in fields:
            raise ValueError(f"bad condition '{text}'")
        fields[key] = value
    if set(fields) == {"after"}:
        seconds = int(_int(1, 366 * SECONDS_PER_DAY)(fields["after"]))
        return Until("after", seconds=seconds)
    if set(fields) in ({"at"}, {"at", "day"}):
        day = int(_int(1, 366)(fields["day"])) if "day" in fields else 0
        return Until("at", clock=parse_clock(fields["at"]), day=day)
    if set(fields) == {"rewards"}:
        return Until("rewards", count=int(_int(1, 99999)(fields["rewards"])))
    if set(fields) == {"pokes", "side"}:
        return Until("pokes", count=int(_int(1, 999999)(fields["pokes"])),
                     side=_enum(*SIDES)(fields["side"]))
    raise ValueError(f"bad condition '{text}'")


def due_time(until, phase_start, t0):
    """When a time condition comes due, or None for count conditions.

    This is the device's rule, and catch-up depends on it being exact:

    - ``after``: that long after the phase started.
    - ``at`` with no day: the first such clock time *strictly after* the phase
      started, so a phase entered at 19:00 runs until 19:00 the next day.
    - ``at`` with a day: that clock time on day *n* of the run, where day 1 is
      t0's calendar day. If that moment is already past when the phase starts,
      the phase ends immediately, which is reported as its start time.

    Datetimes must be timezone-aware and in the devices' standard time.
    """
    if until.kind == "after":
        return phase_start + timedelta(seconds=until.seconds)
    if until.kind != "at":
        return None
    hour, minute = divmod(until.clock, 60)
    if until.day:
        day = t0.date() + timedelta(days=until.day - 1)
        due = datetime(day.year, day.month, day.day, hour, minute,
                       tzinfo=t0.tzinfo)
        return max(due, phase_start)
    due = phase_start.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if due <= phase_start:
        due += timedelta(days=1)
    return due


@dataclass
class Window:
    start: int              # minutes after midnight
    end: int
    settings: dict = field(default_factory=dict)

    @property
    def range_text(self):
        return f"{format_clock(self.start)}-{format_clock(self.end)}"

    def active_at(self, moment):
        minute = moment.hour * 60 + moment.minute
        if self.start < self.end:
            return self.start <= minute < self.end
        return minute >= self.start or minute < self.end


def windows_overlap(a, b):
    """Whether two daily windows share any minute."""
    def spans(w):
        if w.start < w.end:
            return [(w.start, w.end)]
        return [(w.start, 1440), (0, w.end)]
    return any(s1 < e2 and s2 < e1
               for s1, e1 in spans(a) for s2, e2 in spans(b))


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

@dataclass
class Phase:
    name: str
    untils: list = field(default_factory=list)
    rule: dict = field(default_factory=lambda: dict(DEFAULTS))
    windows: list = field(default_factory=list)

    def rule_at(self, moment):
        """The rule in force at a moment, with any daily window applied."""
        rule = dict(self.rule)
        for window in self.windows:
            if window.active_at(moment):
                rule.update(window.settings)
        return rule


def _pack(prefix, tokens):
    """Spread ``tokens`` over as few ``prefix ...`` lines as fit MAX_LINE."""
    lines, current = [], []
    for token in tokens:
        if current and len(f"{prefix} {' '.join(current + [token])}") > MAX_LINE:
            lines.append(f"{prefix} {' '.join(current)}")
            current = []
        current.append(token)
    if current:
        lines.append(f"{prefix} {' '.join(current)}")
    return lines


def _tokens(settings, skip_defaults):
    return [f"{key}={settings[key]}" for key in KEY_ORDER
            if key in settings
            and not (skip_defaults and settings[key] == DEFAULTS[key])]


@dataclass
class Schedule:
    name: str
    phases: list = field(default_factory=list)

    def lines(self):
        out = [f"NAME {self.name}"]
        for phase in self.phases:
            out.append(f"PHASE {phase.name}")
            out.extend(f"UNTIL {until.text()}" for until in phase.untils)
            out.extend(_pack("SET", _tokens(phase.rule, skip_defaults=True)))
            for window in phase.windows:
                out.extend(_pack(f"WINDOW {window.range_text}",
                                 _tokens(window.settings, skip_defaults=False)))
        return out

    def text(self):
        return "".join(line + "\n" for line in self.lines())

    def crc(self):
        return zlib.crc32(self.text().encode("ascii"))

    def problems(self):
        """Every limit the device would enforce, as readable messages."""
        out = []
        if not 1 <= len(self.phases) <= MAX_PHASES:
            out.append(f"a schedule has 1-{MAX_PHASES} phases; this one has "
                       f"{len(self.phases)}")
        for index, phase in enumerate(self.phases):
            label = f"phase {index} ('{phase.name}')"
            problem = name_problem(phase.name, MAX_NAME, phase=True)
            if problem:
                out.append(f"{label}: name {problem}")
            if len(phase.untils) > MAX_UNTILS:
                out.append(f"{label}: {len(phase.untils)} end conditions; the "
                           f"limit is {MAX_UNTILS}")
            if len(phase.windows) > MAX_WINDOWS:
                out.append(f"{label}: {len(phase.windows)} daily windows; the "
                           f"limit is {MAX_WINDOWS}")
        for line in self.lines():
            if len(line) > MAX_LINE:
                out.append(f"line too long for the device ({len(line)} > "
                           f"{MAX_LINE}): {line[:40]}...")
        size = len(self.text().encode("ascii", "replace"))
        if size > MAX_BYTES:
            out.append(f"schedule is {size} bytes; the limit is {MAX_BYTES}")
        return out


def name_problem(name, limit, phase=False):
    if not name or name != name.strip():
        return "must not be empty or start or end with a space"
    if phase and "," in name:
        return "must not contain a comma (it is written into CSV rows)"
    if len(name) > limit:
        return f"is {len(name)} characters; the limit is {limit}"
    if not all(" " <= ch <= "~" for ch in name):
        return "must be plain ASCII (the device display and SD card use it)"
    return None


class ScheduleBuilder:
    """Accept a schedule one line at a time, as the device does during upload.

    ``add`` raises ValueError for a line the device would refuse, so the error
    can be reported against that line as it arrives. ``finish`` checks what can
    only be checked at the end.
    """

    def __init__(self):
        self.schedule = None
        self.phase = None
        self.lines = []
        self.size = 0
        self._seen_keys = set()

    def add(self, line):
        if len(line) > MAX_LINE:
            raise ValueError("line too long")
        self.size += len(line) + 1
        if self.size > MAX_BYTES:
            raise ValueError(f"schedule larger than {MAX_BYTES} bytes")
        directive, _, rest = line.partition(" ")
        if self.schedule is None:
            if directive != "NAME" or not rest:
                raise ValueError("a schedule starts with NAME")
            self.schedule = Schedule(rest)
        elif directive == "PHASE":
            problem = name_problem(rest, MAX_NAME, phase=True)
            if problem:
                raise ValueError(f"phase name {problem}")
            if len(self.schedule.phases) == MAX_PHASES:
                raise ValueError(f"more than {MAX_PHASES} phases")
            self.phase = Phase(rest)
            self.schedule.phases.append(self.phase)
            self._seen_keys = set()
        elif self.phase is None:
            raise ValueError(f"{directive} before the first PHASE")
        elif directive == "UNTIL":
            if len(self.phase.untils) == MAX_UNTILS:
                raise ValueError(f"more than {MAX_UNTILS} end conditions")
            self.phase.untils.append(parse_until(rest))
        elif directive == "SET":
            for key, value in _parse_settings(rest).items():
                if key in self._seen_keys:
                    raise ValueError(f"{key} is set twice")
                self._seen_keys.add(key)
                self.phase.rule[key] = value
        elif directive == "WINDOW":
            self._window(rest)
        else:
            raise ValueError(f"unknown directive '{directive}'")
        self.lines.append(line)

    def _window(self, rest):
        span, _, settings = rest.partition(" ")
        start, sep, end = span.partition("-")
        if not sep:
            raise ValueError("WINDOW is WINDOW HH:MM-HH:MM key=value ...")
        window = Window(parse_clock(start), parse_clock(end))
        if window.start == window.end:
            raise ValueError("a window must not start and end at the same time")
        parsed = _parse_settings(settings)
        existing = next((w for w in self.phase.windows
                         if (w.start, w.end) == (window.start, window.end)), None)
        if existing is None:
            if any(windows_overlap(window, w) for w in self.phase.windows):
                raise ValueError("windows overlap")
            if len(self.phase.windows) == MAX_WINDOWS:
                raise ValueError(f"more than {MAX_WINDOWS} windows")
            self.phase.windows.append(window)
            existing = window
        existing.settings.update(parsed)

    def finish(self):
        if self.schedule is None:
            raise ValueError("empty schedule")
        if not self.schedule.phases:
            raise ValueError("a schedule needs at least one phase")
        return self.schedule

    def crc(self):
        return zlib.crc32("".join(line + "\n" for line in self.lines).encode("ascii"))


def parse_schedule(text):
    """Parse schedule text exactly as the device does, raising ScheduleError."""
    builder = ScheduleBuilder()
    line_no = 0
    for line_no, line in enumerate(text.split("\n"), 1):
        if not line:
            continue
        try:
            builder.add(line)
        except ValueError as exc:
            raise ScheduleError(line_no, str(exc)) from None
    try:
        return builder.finish()
    except ValueError as exc:
        raise ScheduleError(max(line_no, 1), str(exc)) from None


def _parse_settings(text):
    settings = {}
    for token in text.split():
        key, sep, value = token.partition("=")
        if not sep:
            raise ValueError(f"'{token}' is not key=value")
        settings[key] = parse_value(key, value)
    if not settings:
        raise ValueError("no settings given")
    return settings


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def summarize(rule):
    """A short description of a rule, for cards, previews and the display.

    ``FR5 · both · TO30s · Q30/1h``, ``PR · left``, ``free feed``.
    """
    free = rule["free_feed"] == "1"
    if rule["active"] == "none":
        return "free feed" if free else "no active poke"
    kind = rule["ratio_kind"]
    parts = ["PR" if kind == "progressive"
             else f"{'RR' if kind == 'random' else 'FR'}{rule['ratio']}"]
    if rule["reward"] == "none":
        parts[0] += " no reward"
    side = rule["active"]
    if side == "both" and rule["count"] == "per_port":
        side = "both, per port"
    parts.append(side)
    if rule["timeout"] != "0":
        parts.append(f"TO{format_seconds(int(rule['timeout']))}")
    if rule["min_poke_ms"] != "0":
        parts.append(f"min {rule['min_poke_ms']}ms")
    if rule["quota"] != "0":
        period = int(rule["quota_period"])
        parts.append(f"Q{rule['quota']}/{format_seconds(period) if period else 'phase'}")
    if free:
        parts.append("+ free feed")
    return " · ".join(parts)
