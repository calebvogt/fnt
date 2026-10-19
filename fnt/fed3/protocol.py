"""Protocol files: an experiment written down, checked and compiled.

A protocol is a YAML file (``docs/fed3/design.md``, Part 4). This module
parses it with line numbers, reports every problem it can find before anything
reaches a device, and compiles it into one :class:`~.schedule.Schedule` per
device. No Qt: the CLI, the editor, the run engine and the tests all share it.

    protocol = load(text)                 # raises ProtocolError
    compiled = compile_protocol(protocol, seed=1234)
    compiled.schedules["P1"].text()
    preview(compiled, t0)
"""
from __future__ import annotations

import difflib
import random
import re
from dataclasses import dataclass, field
from datetime import timedelta

import yaml

from . import schedule as sch

FORMAT = "fnt-fed3-protocol/1"
MAX_PROTOCOL_NAME = 60
DEVICE_NAME = re.compile(r"[A-Za-z0-9_-]{1,16}")
BUILTIN_COLORS = {
    "white": (0, 0, 0, 40),
    "red": (30, 0, 0, 0),
    "green": (0, 30, 0, 0),
    "blue": (0, 0, 30, 0),
}
RESERVED_CUES = ("none", "click")

TOP_KEYS = ("format", "name", "description", "devices", "colors", "cues",
            "rules", "phases", "checks")
PHASE_KEYS = ("name", "until", "all", "devices", "rotate", "daily")
ROTATE_KEYS = ("every", "at", "mode", "assign")
WINDOW_KEYS = ("from", "to", "all", "devices")
CHECK_KEYS = ("no_pellets",)
CUE_SLOTS = ("reward", "counted", "inactive", "timeout", "depleted")
CUE_PARAMS = {
    "tone": ("hz", "ms"),
    "noise": ("ms",),
    "flash": ("target", "color", "ms"),
    "bnc": ("width", "hz", "count"),
}
RULE_KEYS = tuple(k for k in sch.KEY_ORDER if not k.startswith("cue."))


@dataclass
class Problem:
    message: str
    line: int | None = None
    level: str = "error"

    def format(self, path="protocol"):
        where = f"{path}:{self.line}" if self.line else path
        return f"{where}: {self.level}: {self.message}"


class ProtocolError(Exception):
    def __init__(self, problems):
        self.problems = problems
        errors = [p for p in problems if p.level == "error"]
        super().__init__("\n".join(p.format() for p in errors))


# ---------------------------------------------------------------------------
# YAML with line numbers
# ---------------------------------------------------------------------------

class _Map(dict):
    """A YAML mapping that remembers where it and each of its keys were."""

    def __init__(self):
        super().__init__()
        self.line = None
        self.key_lines = {}

    def line_of(self, key):
        return self.key_lines.get(key, self.line)


class _Seq(list):
    line = None


class _Loader(yaml.SafeLoader):
    def __init__(self, stream):
        super().__init__(stream)
        self.duplicates = []


def _construct_map(loader, node):
    loader.flatten_mapping(node)
    mapping = _Map()
    mapping.line = node.start_mark.line + 1
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        line = key_node.start_mark.line + 1
        if key in mapping:
            loader.duplicates.append((line, key))
        mapping[key] = loader.construct_object(value_node, deep=True)
        mapping.key_lines[key] = line
    return mapping


def _construct_seq(loader, node):
    seq = _Seq(loader.construct_sequence(node, deep=True))
    seq.line = node.start_mark.line + 1
    return seq


_Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_map)
_Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_SEQUENCE_TAG, _construct_seq)


# ---------------------------------------------------------------------------
# The parsed protocol
# ---------------------------------------------------------------------------

@dataclass
class Condition:
    """One way a phase can end, as written in the protocol."""

    kind: str                   # after | at | rewards | pokes | manual
    seconds: int = 0
    clock: int = 0
    day: int = 0
    count: int = 0
    side: str = "any"
    devices: object = None      # None: this device; "all" | "any" | tuple
    line: int | None = None

    @property
    def needs_fnt(self):
        return self.kind == "manual" or self.devices is not None

    def until(self):
        """The device-side condition; only for conditions that don't need FNT."""
        return sch.Until(self.kind, seconds=self.seconds, clock=self.clock,
                         day=self.day, count=self.count, side=self.side)

    def describe(self):
        if self.kind == "manual":
            return "Advance phase pressed"
        if self.kind == "after":
            return f"after {sch.format_seconds(self.seconds)}"
        if self.kind == "at":
            day = f" on day {self.day}" if self.day else ""
            return f"at {sch.format_clock(self.clock)}{day}"
        noun = "rewards" if self.kind == "rewards" else (
            "pokes" if self.side == "any" else f"{self.side} pokes")
        if self.devices is None:
            return f"{self.count} {noun}"
        who = (f"{self.devices} devices" if isinstance(self.devices, str)
               else ", ".join(self.devices))
        return f"{self.count} {noun} ({who})"


@dataclass
class Rotation:
    every: int
    at: int | None
    mode: str
    assign: dict
    line: int | None = None


@dataclass
class DailyWindow:
    start: int
    end: int
    all: dict
    devices: dict
    line: int | None = None


@dataclass
class PhaseSpec:
    name: str
    until: list
    all: dict
    devices: dict
    rotate: Rotation | None
    daily: list
    line: int | None = None

    @property
    def needs_fnt(self):
        return any(c.needs_fnt for c in self.until)


@dataclass
class Protocol:
    name: str
    description: str
    devices: list
    phases: list
    checks: dict
    rule_settings: dict = field(default_factory=dict)    # named rule -> settings
    warnings: list = field(default_factory=list)

    @property
    def needs_seed(self):
        return any(p.rotate and p.rotate.mode == "shuffle" for p in self.phases)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

class _Parser:
    def __init__(self):
        self.problems = []
        self.colors = dict(BUILTIN_COLORS)
        self.cues = {}
        self.rule_specs = {}
        self.rules = {}
        self.devices = []

    # -- reporting ------------------------------------------------------

    def error(self, line, message):
        self.problems.append(Problem(message, line))

    def warn(self, line, message):
        self.problems.append(Problem(message, line, "warning"))

    def unknown_keys(self, mapping, allowed, where):
        for key in mapping:
            if key not in allowed:
                hint = difflib.get_close_matches(str(key), allowed, n=1)
                suffix = f"; did you mean '{hint[0]}'?" if hint else ""
                self.error(mapping.line_of(key), f"unknown key '{key}' in {where}{suffix}")

    def expect_map(self, value, line, what):
        if isinstance(value, _Map):
            return value
        self.error(line, f"{what} must be a mapping (key: value)")
        return None

    # -- scalars --------------------------------------------------------

    def duration(self, value, line, what, allow_int=False):
        if isinstance(value, int) and not isinstance(value, bool) and allow_int:
            return value
        if isinstance(value, str):
            match = re.fullmatch(r"(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?",
                                 value.strip())
            if match and any(match.groups()):
                d, h, m, s = (int(g or 0) for g in match.groups())
                return d * 86400 + h * 3600 + m * 60 + s
        unit = " (or a whole number of seconds)" if allow_int else ""
        self.error(line, f"{what} must be a duration such as 90s, 15m, 48h, "
                         f"10d or 1d12h{unit}, not {value!r}")
        return None

    def clock(self, value, line, what):
        if isinstance(value, int) and not isinstance(value, bool):
            self.error(line, f"{what}: write clock times in quotes, e.g. "
                             f"\"19:00\"; unquoted, YAML reads them as numbers")
            return None
        try:
            return sch.parse_clock(str(value))
        except ValueError as exc:
            self.error(line, f"{what}: {exc}")
            return None

    def whole(self, value, line, what, lo, hi):
        if isinstance(value, int) and not isinstance(value, bool) and lo <= value <= hi:
            return value
        self.error(line, f"{what} must be a whole number from {lo} to {hi}, "
                         f"not {value!r}")
        return None

    # -- colours and cues -----------------------------------------------

    def color(self, value, line):
        if isinstance(value, str):
            if value in self.colors:
                return self.colors[value]
            hint = difflib.get_close_matches(value, list(self.colors), n=1)
            suffix = f"; did you mean '{hint[0]}'?" if hint else ""
            self.error(line, f"unknown colour '{value}'{suffix}")
            return None
        if (isinstance(value, list) and len(value) == 4
                and all(isinstance(c, int) and not isinstance(c, bool)
                        and 0 <= c <= 255 for c in value)):
            return tuple(value)
        self.error(line, "a colour is a name or [r, g, b, w] with each 0-255")
        return None

    def cue_action(self, item, line):
        if item == "click":
            return "click"
        if not (isinstance(item, _Map) and len(item) == 1):
            if isinstance(item, str) and item in self.cues:
                self.error(line, f"'{item}' is a named cue; use it on its own, "
                                 "not inside a list of actions")
            else:
                self.error(line, "a cue action is click, or one of "
                                 "tone/flash/noise/bnc with its parameters")
            return None
        (kind, params), = item.items()
        line = item.line_of(kind)
        if kind not in CUE_PARAMS:
            self.error(line, f"unknown cue action '{kind}'; use one of: "
                             f"click, {', '.join(CUE_PARAMS)}")
            return None
        params = self.expect_map(params, line, f"{kind}")
        if params is None:
            return None
        self.unknown_keys(params, CUE_PARAMS[kind], kind)
        missing = [p for p in CUE_PARAMS[kind] if p not in params]
        if missing:
            self.error(line, f"{kind} needs {', '.join(missing)}")
            return None
        values = []
        for name in CUE_PARAMS[kind]:
            value = params[name]
            if name == "target":
                values.append(str(value))
            elif name == "color":
                rgbw = self.color(value, params.line_of(name))
                if rgbw is None:
                    return None
                values.extend(rgbw)
            elif isinstance(value, int) and not isinstance(value, bool):
                values.append(value)
            else:
                self.error(params.line_of(name), f"{kind}: {name} must be a "
                                                 f"whole number")
                return None
        return ":".join([kind, *map(str, values)])

    def cue(self, value, line):
        """A cue as written, to its device text."""
        if value == "none":
            text = "none"
        elif value == "click":
            text = "click"
        elif isinstance(value, str):
            if value not in self.cues:
                hint = difflib.get_close_matches(value, list(self.cues), n=1)
                suffix = f"; did you mean '{hint[0]}'?" if hint else ""
                self.error(line, f"unknown cue '{value}'{suffix}")
                return None
            return self.cues[value]
        elif isinstance(value, list) and value:
            actions = [self.cue_action(item, line) for item in value]
            if None in actions:
                return None
            text = "+".join(actions)
        else:
            self.error(line, "a cue is none, click, a named cue, or a list of "
                             "actions")
            return None
        try:
            return sch.parse_cue(text)
        except ValueError as exc:
            self.error(line, f"cue {exc}")
            return None

    # -- rules ----------------------------------------------------------

    def rule_value(self, key, value, line):
        if key == "free_feed":
            if isinstance(value, bool) or value in (0, 1):
                text = "1" if value else "0"
            else:
                self.error(line, "free_feed is true or false")
                return None
        elif key in sch.SECONDS_KEYS:
            seconds = self.duration(value, line, key, allow_int=True)
            if seconds is None:
                return None
            text = str(seconds)
        elif isinstance(value, bool):
            self.error(line, f"{key} can't be true/false")
            return None
        else:
            text = str(value)
        try:
            return sch.parse_value(key, text)
        except ValueError as exc:
            self.error(line, str(exc))
            return None

    def named_rule(self, name, line, stack=()):
        if not isinstance(name, str) or name not in self.rule_specs:
            known = list(self.rule_specs)
            hint = difflib.get_close_matches(str(name), known, n=1)
            suffix = f"; did you mean '{hint[0]}'?" if hint else ""
            self.error(line, f"unknown rule '{name}'{suffix}")
            return {}
        if name in stack:
            self.error(line, f"rules use each other in a loop: "
                             f"{' -> '.join(stack + (name,))}")
            return {}
        if name not in self.rules:
            self.rules[name] = self.rule(self.rule_specs[name], line, stack + (name,))
        return dict(self.rules[name])

    def rule(self, value, line, stack=()):
        """A rule as written (a name or a mapping), to partial device settings."""
        if isinstance(value, str):
            return self.named_rule(value, line, stack)
        mapping = self.expect_map(value, line, "a rule")
        if mapping is None:
            return {}
        settings = {}
        if "use" in mapping:
            settings.update(self.named_rule(mapping["use"], mapping.line_of("use"),
                                            stack))
        for key, item in mapping.items():
            key_line = mapping.line_of(key)
            if key == "use":
                continue
            if key == "cue":
                cues = self.expect_map(item, key_line, "cue")
                if cues is None:
                    continue
                self.unknown_keys(cues, CUE_SLOTS, "cue")
                for slot, cue in cues.items():
                    if slot in CUE_SLOTS:
                        text = self.cue(cue, cues.line_of(slot))
                        if text is not None:
                            settings[f"cue.{slot}"] = text
            elif key in RULE_KEYS:
                text = self.rule_value(key, item, key_line)
                if text is not None:
                    settings[key] = text
            elif key in ("light", "lights"):
                self.error(key_line, "there are no steady lights; flash them "
                                     "from a cue, e.g. cue: {reward: [flash: "
                                     "{target: bar, color: blue, ms: 500}]}")
            else:
                hint = difflib.get_close_matches(str(key), RULE_KEYS + ("cue", "use"), n=1)
                suffix = f"; did you mean '{hint[0]}'?" if hint else ""
                self.error(key_line, f"unknown rule key '{key}'{suffix}")
        return settings

    def device_rules(self, value, line, where):
        mapping = self.expect_map(value, line, f"{where}.devices")
        if mapping is None:
            return {}
        out = {}
        for name, rule in mapping.items():
            if name not in self.devices:
                self.error(mapping.line_of(name), f"'{name}' is not in devices")
                continue
            out[name] = self.rule(rule, mapping.line_of(name))
        return out

    # -- phases ---------------------------------------------------------

    def condition(self, value, line):
        mapping = self.expect_map(value, line, "an until condition")
        if mapping is None:
            return None
        line = mapping.line
        keys = set(mapping)
        devices = None
        if "devices" in mapping:
            devices = self.who(mapping["devices"], mapping.line_of("devices"))
            keys.discard("devices")
            if not keys & {"rewards", "pokes"}:
                self.error(line, "devices only applies to rewards and pokes")
        if keys == {"after"}:
            seconds = self.duration(mapping["after"], line, "after")
            if seconds is None:
                return None
            if not 1 <= seconds <= 366 * sch.SECONDS_PER_DAY:
                self.error(line, "after must be between 1s and 366d")
                return None
            return Condition("after", seconds=seconds, line=line)
        if keys in ({"at"}, {"at", "day"}):
            clock = self.clock(mapping["at"], line, "at")
            day = (self.whole(mapping["day"], line, "day", 1, 366)
                   if "day" in mapping else 0)
            if clock is None or day is None:
                return None
            return Condition("at", clock=clock, day=day, line=line)
        if keys == {"rewards"}:
            count = self.whole(mapping["rewards"], line, "rewards", 1, 99999)
            if count is None:
                return None
            return Condition("rewards", count=count, devices=devices, line=line)
        if keys in ({"pokes"}, {"pokes", "side"}):
            count = self.whole(mapping["pokes"], line, "pokes", 1, 999999)
            side = mapping.get("side", "any")
            if side not in sch.SIDES:
                self.error(line, f"side must be one of: {', '.join(sch.SIDES)}")
                return None
            if count is None:
                return None
            return Condition("pokes", count=count, side=side, devices=devices,
                             line=line)
        if keys == {"manual"}:
            if mapping["manual"] is not True:
                self.error(line, "manual: true is the only form")
                return None
            return Condition("manual", line=line)
        if keys == {"pellets"}:
            self.error(line, "conditions count rewards (a met ratio), not "
                             "pellets; use rewards")
            return None
        self.error(line, "an until condition is one of: after, at (with optional "
                         "day), rewards, pokes (with optional side), manual")
        return None

    def who(self, value, line):
        if value in ("all", "any"):
            return value
        if (isinstance(value, list) and value
                and all(isinstance(v, str) for v in value)):
            unknown = [v for v in value if v not in self.devices]
            if unknown:
                self.error(line, f"not in devices: {', '.join(unknown)}")
            return tuple(value)
        self.error(line, "devices is all, any, or a list of device names")
        return None

    def rotation(self, value, line):
        mapping = self.expect_map(value, line, "rotate")
        if mapping is None:
            return None
        self.unknown_keys(mapping, ROTATE_KEYS, "rotate")
        if "every" not in mapping or "assign" not in mapping:
            self.error(mapping.line, "rotate needs every and assign")
            return None
        every = self.duration(mapping["every"], mapping.line_of("every"), "every")
        if every is not None and every < 60:
            self.error(mapping.line_of("every"), "every must be at least 1m")
            every = None
        at = None
        if "at" in mapping:
            at = self.clock(mapping["at"], mapping.line_of("at"), "rotate.at")
            if every is not None and every != sch.SECONDS_PER_DAY:
                self.error(mapping.line_of("at"), "rotate.at only works with "
                                                  "every: 1d")
        mode = mapping.get("mode", "cycle")
        if mode not in ("cycle", "shuffle"):
            self.error(mapping.line_of("mode"), "mode is cycle or shuffle")
        assign = self.expect_map(mapping["assign"], mapping.line_of("assign"),
                                 "rotate.assign")
        if assign is None or every is None:
            return None
        missing = [d for d in self.devices if d not in assign]
        if missing:
            self.error(assign.line, f"assign gives every device a rule; missing: "
                                    f"{', '.join(missing)}")
        for name, rule_name in assign.items():
            if name not in self.devices:
                self.error(assign.line_of(name), f"'{name}' is not in devices")
            elif not isinstance(rule_name, str):
                self.error(assign.line_of(name), "assign maps each device to the "
                                                 "name of a rule")
            else:
                self.named_rule(rule_name, assign.line_of(name))
        return Rotation(every, at, mode, dict(assign), mapping.line)

    def window(self, value, line):
        mapping = self.expect_map(value, line, "a daily window")
        if mapping is None:
            return None
        self.unknown_keys(mapping, WINDOW_KEYS, "a daily window")
        if "from" not in mapping or "to" not in mapping:
            self.error(mapping.line, "a daily window needs from and to")
            return None
        start = self.clock(mapping["from"], mapping.line_of("from"), "from")
        end = self.clock(mapping["to"], mapping.line_of("to"), "to")
        if start is None or end is None:
            return None
        if start == end:
            self.error(mapping.line, "a window must not start and end at the "
                                     "same time")
            return None
        all_rule = self.rule(mapping["all"], mapping.line_of("all")) \
            if "all" in mapping else {}
        devices = self.device_rules(mapping["devices"], mapping.line_of("devices"),
                                    "window") if "devices" in mapping else {}
        if not all_rule and not any(devices.values()):
            self.error(mapping.line, "a daily window must change something "
                                     "(all or devices)")
        return DailyWindow(start, end, all_rule, devices, mapping.line)

    def phase(self, value, line, index):
        mapping = self.expect_map(value, line, "a phase")
        if mapping is None:
            return None
        self.unknown_keys(mapping, PHASE_KEYS, "a phase")
        name = mapping.get("name")
        if not isinstance(name, str):
            self.error(mapping.line, f"phase {index + 1} needs a name")
            name = f"Phase {index + 1}"
        else:
            problem = sch.name_problem(name, sch.MAX_NAME, phase=True)
            if problem:
                self.error(mapping.line_of("name"), f"phase name {problem}")
        until = []
        if "until" in mapping:
            raw = mapping["until"]
            items = raw if isinstance(raw, list) else [raw]
            if not items:
                self.error(mapping.line_of("until"), "until is empty")
            for item in items:
                condition = self.condition(item, mapping.line_of("until"))
                if condition:
                    until.append(condition)
        all_rule = self.rule(mapping["all"], mapping.line_of("all")) \
            if "all" in mapping else {}
        devices = self.device_rules(mapping["devices"], mapping.line_of("devices"),
                                    "phase") if "devices" in mapping else {}
        rotate = self.rotation(mapping["rotate"], mapping.line_of("rotate")) \
            if "rotate" in mapping else None
        daily = []
        if "daily" in mapping:
            raw = mapping["daily"]
            if not isinstance(raw, list):
                self.error(mapping.line_of("daily"), "daily is a list of windows")
            else:
                daily = [w for w in (self.window(item, mapping.line_of("daily"))
                                     for item in raw) if w]
            for i, a in enumerate(daily):
                for b in daily[i + 1:]:
                    if sch.windows_overlap(sch.Window(a.start, a.end),
                                           sch.Window(b.start, b.end)):
                        self.error(b.line, "daily windows overlap")
            if len(daily) > sch.MAX_WINDOWS:
                self.error(mapping.line_of("daily"), f"at most {sch.MAX_WINDOWS} "
                                                     f"daily windows per phase")
        phase = PhaseSpec(name, until, all_rule, devices, rotate, daily, mapping.line)
        if rotate:
            self.check_rotation_length(phase, mapping)
        return phase

    def check_rotation_length(self, phase, mapping):
        line = mapping.line_of("until")
        rotate = phase.rotate
        if len(phase.until) != 1 or phase.until[0].kind != "after":
            self.error(line, "a rotating phase lasts a whole number of periods: "
                             "give it exactly one until: {after: ...}")
            return
        after = phase.until[0].seconds
        if after % rotate.every:
            self.error(line, f"until after {sch.format_seconds(after)} is not a "
                             f"whole number of {sch.format_seconds(rotate.every)} "
                             f"periods")

    # -- the whole file -------------------------------------------------

    def parse(self, text):
        try:
            loader = _Loader(text)
            try:
                root = loader.get_single_data()
            finally:
                loader.dispose()
        except yaml.YAMLError as exc:
            mark = getattr(exc, "problem_mark", None)
            self.error(mark.line + 1 if mark else None,
                       f"not valid YAML: {getattr(exc, 'problem', exc)}")
            return None
        for line, key in loader.duplicates:
            self.error(line, f"'{key}' appears twice")
        root = self.expect_map(root, 1, "a protocol file")
        if root is None:
            return None
        self.unknown_keys(root, TOP_KEYS, "the protocol")

        if root.get("format") != FORMAT:
            self.error(root.line_of("format"), f"the first line should be "
                                               f"format: {FORMAT}")
        name = root.get("name")
        if not isinstance(name, str):
            self.error(root.line, "the protocol needs a name")
            name = "Unnamed"
        else:
            problem = sch.name_problem(name, MAX_PROTOCOL_NAME)
            if problem:
                self.error(root.line_of("name"), f"name {problem}")
        description = root.get("description") or ""

        devices = root.get("devices")
        if (not isinstance(devices, list) or not devices
                or not all(isinstance(d, str) for d in devices)):
            self.error(root.line_of("devices"), "devices is a list of names, "
                                                "e.g. [A, B]")
            devices = []
        for device in devices:
            if not DEVICE_NAME.fullmatch(device):
                self.error(root.line_of("devices"), f"device name '{device}' "
                           "must be 1-16 letters, digits, - or _")
        if len(set(devices)) != len(devices):
            self.error(root.line_of("devices"), "device names must be unique")
        self.devices = devices

        if "colors" in root:
            colors = self.expect_map(root["colors"], root.line_of("colors"), "colors")
            for cname, value in (colors or {}).items():
                rgbw = self.color(value, colors.line_of(cname))
                if rgbw:
                    self.colors[cname] = rgbw
        if "cues" in root:
            cues = self.expect_map(root["cues"], root.line_of("cues"), "cues")
            for cname, value in (cues or {}).items():
                if cname in RESERVED_CUES:
                    self.error(cues.line_of(cname), f"'{cname}' is built in; "
                                                    "choose another name")
                    continue
                text = self.cue(value, cues.line_of(cname))
                if text is not None:
                    self.cues[cname] = text
        if "rules" in root:
            rules = self.expect_map(root["rules"], root.line_of("rules"), "rules")
            self.rule_specs = dict(rules or {})
            for rname in self.rule_specs:
                self.named_rule(rname, rules.line_of(rname))

        phases = []
        raw_phases = root.get("phases")
        if not isinstance(raw_phases, list) or not raw_phases:
            self.error(root.line_of("phases"), "phases is a list with at least "
                                               "one phase")
        else:
            for index, item in enumerate(raw_phases):
                phase = self.phase(item, root.line_of("phases"), index)
                if phase:
                    phases.append(phase)

        seen = {}
        for index, phase in enumerate(phases):
            if phase.name in seen:
                self.error(phase.line, f"two phases are called '{phase.name}'")
            seen[phase.name] = index
            last = index == len(phases) - 1
            if last and phase.until:
                self.error(phase.line, f"'{phase.name}' is the last phase, which "
                                       "runs until Stop; remove its until")
            if not last and not phase.until:
                self.error(phase.line, f"'{phase.name}' never ends, so the phases "
                                       "after it can't run; give it an until")
            if phase.needs_fnt and not any(not c.needs_fnt for c in phase.until):
                self.warn(phase.line, f"'{phase.name}' only ends through FNT; if "
                                      "the PC is down it waits. Add a device-local "
                                      "fallback such as {after: 4d}")

        checks = {}
        if "checks" in root:
            raw = self.expect_map(root["checks"], root.line_of("checks"), "checks")
            if raw is not None:
                self.unknown_keys(raw, CHECK_KEYS, "checks")
                if "no_pellets" in raw:
                    checks["no_pellets"] = self.duration(
                        raw["no_pellets"], raw.line_of("no_pellets"), "no_pellets")

        return Protocol(name, str(description).strip(), devices, phases, checks)


def load(text):
    """Parse and check a protocol. Raises ProtocolError listing every error."""
    parser = _Parser()
    protocol = parser.parse(text)
    errors = [p for p in parser.problems if p.level == "error"]
    if errors or protocol is None:
        raise ProtocolError(sorted(parser.problems, key=lambda p: p.line or 0))
    protocol.warnings = [p for p in parser.problems if p.level == "warning"]
    protocol.rule_settings = dict(parser.rules)
    return protocol


# ---------------------------------------------------------------------------
# Compiling
# ---------------------------------------------------------------------------

@dataclass
class CompiledPhase:
    """One phase as every device sees it (devices differ only in their rules)."""

    name: str
    source: int                 # index into protocol.phases
    period: int | None          # rotation period, counted from 0
    untils: list                # the device's UNTIL lines
    fnt: list                   # conditions FNT evaluates
    labels: dict                # device -> rotation assignment, when rotating


@dataclass
class Compiled:
    protocol: Protocol
    seed: int | None
    phases: list
    schedules: dict             # device name -> Schedule


def _untils(conditions):
    untils = []
    for condition in conditions:
        if condition.needs_fnt:
            if sch.Until("goto") not in untils:
                untils.append(sch.Until("goto"))
        else:
            untils.append(condition.until())
    return untils


def _assignments(protocol, index, rotate, seed):
    """Each period's device -> rule name mapping for one rotating phase."""
    devices = protocol.devices
    periods = protocol.phases[index].until[0].seconds // rotate.every
    first = [rotate.assign[d] for d in devices]
    rng = random.Random(f"{seed}:{index}")
    out = []
    for period in range(periods):
        if period == 0:
            values = first
        elif rotate.mode == "cycle":
            shift = period % len(devices)
            values = first[shift:] + first[:shift]
        else:
            values = list(first)
            rng.shuffle(values)
        out.append(dict(zip(devices, values)))
    return out


def compile_protocol(protocol, seed=None):
    """Compile a checked protocol into one schedule per device.

    ``seed`` is required when a phase shuffles its rotation; it is stored with
    the run so the same sequence can be rebuilt after a crash.
    """
    if protocol.needs_seed and seed is None:
        raise ProtocolError([Problem("this protocol shuffles a rotation, so "
                                     "compiling it needs a seed")])
    phases = []
    rules_per_device = {d: [] for d in protocol.devices}
    windows_per_device = {d: [] for d in protocol.devices}

    for index, spec in enumerate(protocol.phases):
        windows = spec.daily
        if spec.rotate:
            assignments = _assignments(protocol, index, spec.rotate, seed)
            untils = ([sch.Until("at", clock=spec.rotate.at)] if spec.rotate.at
                      is not None else [sch.Until("after", seconds=spec.rotate.every)])
            for period, assignment in enumerate(assignments):
                phases.append(CompiledPhase(f"{spec.name} {period + 1}", index,
                                            period, untils, [], assignment))
                for device in protocol.devices:
                    rule = dict(sch.DEFAULTS)
                    rule.update(spec.all)
                    rule.update(protocol.rule_settings[assignment[device]])
                    rule.update(spec.devices.get(device, {}))
                    rules_per_device[device].append(rule)
                    windows_per_device[device].append(windows)
        else:
            phases.append(CompiledPhase(spec.name, index, None, _untils(spec.until),
                                        [c for c in spec.until if c.needs_fnt], {}))
            for device in protocol.devices:
                rule = dict(sch.DEFAULTS)
                rule.update(spec.all)
                rule.update(spec.devices.get(device, {}))
                rules_per_device[device].append(rule)
                windows_per_device[device].append(windows)

    schedules = {}
    problems = []
    for device in protocol.devices:
        sched_phases = []
        for phase, rule, windows in zip(phases, rules_per_device[device],
                                        windows_per_device[device]):
            device_windows = []
            for window in windows:
                settings = dict(window.all)
                settings.update(window.devices.get(device, {}))
                if settings:
                    device_windows.append(sch.Window(window.start, window.end,
                                                     settings))
            sched_phases.append(sch.Phase(phase.name, list(phase.untils), rule,
                                          device_windows))
        schedule = sch.Schedule(protocol.name, sched_phases)
        problems += [Problem(f"{device}: {message}") for message in schedule.problems()]
        schedules[device] = schedule
    if problems:
        raise ProtocolError(problems)
    return Compiled(protocol, seed, phases, schedules)


def check(text):
    """Every problem in a protocol, errors and warnings, plus the compiled form.

    Returns ``(compiled_or_None, problems)``. Compiles with a placeholder seed so
    device limits are checked too; a run compiles again with its own seed.
    """
    try:
        protocol = load(text)
    except ProtocolError as exc:
        return None, exc.problems
    try:
        compiled = compile_protocol(protocol, seed=0)
    except ProtocolError as exc:
        return None, protocol.warnings + exc.problems
    return compiled, protocol.warnings


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

@dataclass
class PreviewPhase:
    index: int
    name: str
    start: object               # datetime, or None when it depends on counts
    latest_start: object        # an upper bound when start is None
    end: object
    latest_end: object
    ends: str                   # how it ends, in words
    needs_fnt: bool
    rules: dict                 # device -> summary
    windows: dict               # device -> [(range text, summary)]
    labels: dict


def preview(compiled, t0):
    """The timeline from ``t0``: when each phase starts and ends, where knowable.

    A phase that ends on a count has no known end; the phases after it show the
    latest they can start, if a time fallback bounds it.
    """
    protocol = compiled.protocol
    rows = []
    start, latest_start = t0, t0
    for index, phase in enumerate(compiled.phases):
        spec = protocol.phases[phase.source]
        if phase.period is None:
            conditions = spec.until
        else:
            conditions = [Condition(u.kind, seconds=u.seconds, clock=u.clock)
                          for u in phase.untils]
        times, latest_times, words = [], [], []
        for condition in conditions:
            words.append(condition.describe())
            if condition.kind in ("after", "at"):
                until = condition.until()
                if start is not None:
                    times.append(sch.due_time(until, start, t0))
                # A later start never makes a time condition due sooner, so the
                # latest start gives the latest end.
                if latest_start is not None:
                    latest_times.append(sch.due_time(until, latest_start, t0))
        counts = [c for c in conditions if c.kind not in ("after", "at")]
        end = min(times) if times and not counts else None
        latest_end = min(latest_times) if latest_times else None
        if end is not None:
            latest_end = end
        rules = {d: sch.summarize(compiled.schedules[d].phases[index].rule)
                 for d in protocol.devices}
        windows = {d: [(w.range_text,
                        sch.summarize({**compiled.schedules[d].phases[index].rule,
                                       **w.settings}))
                       for w in compiled.schedules[d].phases[index].windows]
                   for d in protocol.devices}
        last = index == len(compiled.phases) - 1
        ends = "when the run is stopped" if last else " or ".join(words)
        rows.append(PreviewPhase(index, phase.name, start, latest_start, end,
                                 latest_end, ends,
                                 any(c.needs_fnt for c in conditions), rules,
                                 windows, dict(phase.labels)))
        start, latest_start = end, latest_end
    return rows


def preview_text(compiled, t0, days=None):
    """The preview as aligned text, for the CLI."""
    def when(moment):
        return moment.strftime("%a %d %b %H:%M")

    rows = preview(compiled, t0)
    zone = t0.strftime("%Z") or t0.strftime("UTC%z")
    out = [f"{compiled.protocol.name}: {len(rows)} phases, starting "
           f"{when(t0)} ({zone})"]
    horizon = t0 + timedelta(days=days) if days else None
    for row in rows:
        if horizon and row.start is not None and row.start > horizon:
            out.append(f"... (stopping at the {days}-day mark)")
            break
        if row.start is not None:
            starts = when(row.start)
        elif row.latest_start is not None:
            starts = f"by {when(row.latest_start)}"
        else:
            starts = "after the previous phase"
        fnt = "  [needs FNT]" if row.needs_fnt else ""
        out.append(f"\n{row.index:>3}  {row.name}{fnt}")
        out.append(f"     starts  {starts}")
        out.append(f"     ends    {row.ends}")
        rules = set(row.rules.values())
        for device in compiled.protocol.devices:
            if len(rules) == 1 and not row.labels and device != compiled.protocol.devices[0]:
                break
            who = "all" if len(rules) == 1 and not row.labels else device
            label = f" ({row.labels[device]})" if device in row.labels else ""
            out.append(f"     {who:<8}{label} {row.rules[device]}")
            for span, summary in row.windows[device]:
                out.append(f"     {'':<8} {span}: {summary}")
    return "\n".join(out)
