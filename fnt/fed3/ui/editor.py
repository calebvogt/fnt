"""The protocol editor: the YAML file, checked as you type, with its timeline.

The file is the protocol, so the editor edits the file. A form covers the
common settings of a phase without typing YAML: **Add phase** writes a new
one, **Edit phase** reads one back into the form and rewrites it, and
**Remove phase** deletes one. With several FED3s the form also sets each one's
active poke, ratio and reward, written as the phase's ``per_device``. **Devices**
rewrites the file's list of FED3s, and templates take that list. Everything
else (rotation, daily windows, named rules) is written in the text, with
problems reported against their line as you go; the form says so rather than
touch a phase it can't show in full.
"""
import json
import re
from dataclasses import dataclass
from datetime import datetime

import yaml
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QTextCursor
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QGridLayout, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMenu, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSpinBox,
    QSplitter, QVBoxLayout, QWidget,
)

from .. import protocol as proto
from .. import schedule as sch
from . import theme

# ---------------------------------------------------------------------------
# One phase as the form sees it
# ---------------------------------------------------------------------------

ENDS = [
    ("after", "after a time"),
    ("at", "at a clock time"),
    ("rewards", "after N rewards"),
    ("pokes", "after N pokes"),
    ("all_rewards", "when every device reaches N rewards"),
    ("manual", "when Advance phase is pressed"),
    ("never", "never (the last phase)"),
]
END_AMOUNTS = {"after": "24h", "at": "19:00", "rewards": "100", "pokes": "500",
               "all_rewards": "100"}
FLASH_TARGETS = ["poke", "bar", "pokes", "poke_l", "poke_r"]
COLOURS = list(proto.BUILTIN_COLORS)
CLICK_SLOTS = ("counted", "timeout", "depleted")
ACTIVE = ["both", "left", "right", "none"]
RATIO_KINDS = ["fixed", "random", "progressive"]
# What the form sets for one FED3 (the phase's `only`); the rest is in the text.
ONLY_KEYS = ("active", "ratio_kind", "ratio", "reward")

# What a phase means when it doesn't say: the rule defaults (design.md, Part 1).
RULE_DEFAULTS = {
    "active": "left", "ratio_kind": "fixed", "ratio": 1, "timeout": 0, "quota": 0,
    "quota_period": "1h", "free_feed": False, "reward": "pellet",
    "tone": False, "tone_hz": 4000, "tone_ms": 200,
    "flash": False, "flash_target": "poke", "flash_color": "blue", "flash_ms": 500,
    "bnc": False, "bnc_width": 25, "bnc_hz": 20, "bnc_count": 20,
    "click_counted": False, "click_timeout": False, "click_depleted": False,
}
# Where a new phase from the form starts.
NEW_PHASE = dict(RULE_DEFAULTS, name="New phase", ends="after", amount="24h",
                 fallback="7d", day=0, side="any", active="both")

# The classic FED3 reward cue, the protocol's built-in `classic`: a 4 kHz tone
# and the blue light bar.
CLASSIC_CUE = {"tone": True, "tone_hz": 4000, "tone_ms": 200, "flash": True,
               "flash_target": "bar", "flash_color": "blue", "flash_ms": 1000}


def _plain(text):
    """A YAML scalar for ``text``: bare when YAML reads it back as the same string."""
    try:
        bare = (text == text.strip() and ": " not in text and " #" not in text
                and yaml.safe_load(text) == text)
    except yaml.YAMLError:
        bare = False
    return text if bare else json.dumps(text)


def _duration_text(value):
    """A duration as written in a protocol (``30s``), from YAML's reading of it."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return sch.format_seconds(value)
    if isinstance(value, str) and re.fullmatch(r"(?:\d+d)?(?:\d+h)?(?:\d+m)?(?:\d+s)?", value.strip()):
        return value.strip() or None
    return None


def _seconds(text):
    match = re.fullmatch(r"(?:(\d+)d)?(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", text)
    d, h, m, s = (int(g or 0) for g in match.groups())
    return d * 86400 + h * 3600 + m * 60 + s


REWARD_ENDS = ("rewards", "all_rewards")
EACH_ENDS = ("rewards", "pokes")         # each device on its own
CURRENT_KEYS = proto.KEY_NAMES[proto.FORMAT]


def syntax_of(text):
    """How a phase in ``text`` is written: its format's key names, and whether
    it has several devices (so an end on counts says whose)."""
    try:
        doc = yaml.load(text, Loader=proto._Loader)   # noqa: S506 - a SafeLoader
    except yaml.YAMLError:
        doc = None
    doc = doc if isinstance(doc, dict) else {}
    fmt = doc.get("format")
    keys = proto.KEY_NAMES.get(fmt, CURRENT_KEYS) if isinstance(fmt, str) else CURRENT_KEYS
    devices = doc.get("devices")
    return keys, isinstance(devices, list) and len(devices) > 1


def can_earn(s):
    """Whether a poke can earn a reward under these settings."""
    return not s["free_feed"] and s["active"] != "none"


def every_poke_earns(s):
    """Whether every counted poke is a reward, on every FED3: FR1 throughout."""
    return all(rule.get("ratio_kind", s["ratio_kind"]) == "fixed"
               and rule.get("ratio", s["ratio"]) == 1
               for rule in [{}, *s.get("only", {}).values()])


def coherent(s):
    """The settings with everything that can't apply taken out.

    Free feeding keeps a pellet in the well, so no poke is active and nothing
    is earned; with nothing earned the ratio, timeout, quota, reward and its
    cues mean nothing, and neither do per-device changes to them. A FED3's
    change that matches the shared rule is no change. A click needs the thing
    it marks: a counted poke that isn't yet a reward, a timeout, a quota.
    """
    s = dict(s)
    if s["free_feed"]:
        s["active"] = "none"
    if not can_earn(s):
        s.update(ratio_kind="fixed", ratio=1, timeout=0, quota=0, reward="pellet",
                 tone=False, flash=False, bnc=False,
                 click_counted=False, click_timeout=False, click_depleted=False)
    if s["ratio_kind"] == "progressive":
        s["ratio"] = 1
    only = {}
    for name, rule in (s.get("only") or {}).items() if can_earn(s) else ():
        rule = {k: v for k, v in rule.items() if k in ONLY_KEYS and v != s[k]}
        if rule.get("ratio_kind", s["ratio_kind"]) == "progressive":
            rule.pop("ratio", None)
        if rule:
            only[name] = rule
    s["only"] = only
    if every_poke_earns(s):
        s["click_counted"] = False          # every counted poke is a reward
    if not s["timeout"]:
        s["click_timeout"] = False
    if not s["quota"]:
        s["click_depleted"] = False
    return s


def ratio_text(kind, ratio):
    """A ratio as the file writes it: ``FR5``, ``RR3`` or ``PR``."""
    return "PR" if kind == "progressive" else f"{'RR' if kind == 'random' else 'FR'}{ratio}"


def _cue_actions(s):
    """The reward cue's actions, as the file writes them."""
    actions = []
    if s["tone"]:
        actions.append(f"tone: {{hz: {s['tone_hz']}, ms: {s['tone_ms']}}}")
    if s["flash"]:
        actions.append(f"flash: {{target: {s['flash_target']}, color: {s['flash_color']}, "
                       f"ms: {s['flash_ms']}}}")
    if s["bnc"]:
        actions.append(f"bnc: {{width: {s['bnc_width']}, hz: {s['bnc_hz']}, "
                       f"count: {s['bnc_count']}}}")
    return actions


def phase_yaml(s, indent=2, keys=CURRENT_KEYS, several=False, base=None):
    """One phase list item, from the form's settings. ``keys`` are the key
    names of the file's format; ``several`` says the protocol has more than one
    device, so an end on counts is written as each device's own. ``base`` is
    what the phase starts from (the file's every_phase): only what differs
    from it is written, apart from the active poke."""
    s = coherent(s)
    base = base or RULE_DEFAULTS
    pad = " " * indent
    lines = [f"{pad}- name: {_plain(s['name'].strip() or 'New phase')}"]
    amount = str(s.get("amount", "")).strip().strip('"\'')
    day = f", day: {s['day']}" if s.get("day") else ""
    side = f", side: {s['side']}" if s.get("side", "any") != "any" else ""
    each = ", of: each" if several and keys["of"] == "of" else ""     # format 1 had no each
    until = {
        "after": f"{{after: {_plain(amount)}}}",
        "at": f"{{at: {json.dumps(amount)}{day}}}",
        "rewards": f"{{rewards: {amount}{each}}}",
        "pokes": f"{{pokes: {amount}{side}{each}}}",
        "manual": "{manual: true}",
    }.get(s["ends"])
    if s["ends"] == "all_rewards":
        fallback = str(s.get("fallback", "")).strip()
        every = f"{{rewards: {amount}, {keys['of']}: all}}"
        until = (f"[{every}, {{after: {_plain(fallback)}}}]" if fallback else every)
    if until:
        lines.append(f"{pad}  until: {until}")
    rule, cues = [f"active: {s['active']}"], []
    if s["free_feed"] != base["free_feed"]:
        rule.append(f"free_feed: {int(s['free_feed'])}")
    if can_earn(s):         # otherwise nothing is earned, and the rest doesn't matter
        ratio = ratio_text(s["ratio_kind"], s["ratio"])
        if ratio != ratio_text(base["ratio_kind"], base["ratio"]):
            rule.append(f"ratio: {ratio}")
        if s["timeout"] != base["timeout"]:
            rule.append(f"timeout: {sch.format_seconds(s['timeout'])}")
        if s["quota"] != base["quota"] or (s["quota"] and s["quota_period"] != base["quota_period"]):
            rule.append(f"quota: {s['quota']}")
            if s["quota"]:
                rule.append(f"quota_period: {_plain(str(s['quota_period']).strip() or '1h')}")
        if s["reward"] != base["reward"]:
            rule.append(f"reward: {s['reward']}")
        actions = _cue_actions(s)
        if actions != _cue_actions(base):
            if actions == _cue_actions(dict(RULE_DEFAULTS, **CLASSIC_CUE)):
                cues.append("reward: classic")
            elif actions:
                cues.append("reward:")
                cues += [f"  - {action}" for action in actions]
            else:
                cues.append("reward: none")
        cues += [f"{slot}: {'[click]' if s[f'click_{slot}'] else 'none'}"
                 for slot in CLICK_SLOTS if s[f"click_{slot}"] != base[f"click_{slot}"]]
    lines.append(f"{pad}  {keys['rule']}:")
    lines += [f"{pad}    {line}" for line in rule]
    if cues:
        lines.append(f"{pad}    cue:")
        lines += [f"{pad}      {line}" for line in cues]
    # FED3s given the same change share one key: "FED002, FED003: {ratio: RR3}".
    groups = {}
    for name, change in s["only"].items():
        written = []
        if "active" in change:
            written.append(f"active: {change['active']}")
        if {"ratio_kind", "ratio"} & set(change):
            written.append("ratio: " + ratio_text(change.get("ratio_kind", s["ratio_kind"]),
                                                  change.get("ratio", s["ratio"])))
        if "reward" in change:
            written.append(f"reward: {change['reward']}")
        groups.setdefault(", ".join(written), []).append(name)
    if groups:
        lines.append(f"{pad}  {keys['only']}:")
        lines += [f"{pad}    {', '.join(names)}: {{{written}}}"
                  for written, names in groups.items()]
    return "\n".join(lines) + "\n"


def every_phase_of(text):
    """What each phase starts from, as form settings, and what of the file's
    every_phase the form can't show (then the form can't write phases for it)."""
    base, missing = dict(RULE_DEFAULTS), []
    try:
        doc = yaml.load(text, Loader=proto._Loader)   # noqa: S506 - a SafeLoader
    except yaml.YAMLError:
        return base, missing
    rule = doc.get("every_phase") if isinstance(doc, dict) else None
    if isinstance(rule, dict):
        _read_rule(rule, base, missing)
    elif rule is not None:
        missing.append(f"every_phase: {_flow(rule)}")
    return base, [f"the protocol's every_phase ({m})" for m in missing]


def settings_from_phase(mapping, keys=CURRENT_KEYS, devices=None, base=None):
    """Read a phase back into the form: ``(settings, what the form can't show)``.

    ``mapping`` is the phase as YAML read it, written with the key names
    ``keys``; ``devices`` is the file's list of FED3s, and ``base`` what each
    phase starts from (see every_phase_of). The second value lists what the
    form would lose; the phase can be edited in the form only when it's empty.
    """
    s = dict(base or RULE_DEFAULTS, ends="never", amount="", fallback="", day=0,
             side="any", only={})
    missing = []
    if not isinstance(mapping, dict):
        return s, ["it isn't written as a mapping"]
    s["name"] = str(mapping.get("name", ""))
    for key, what in (("rotate", "a rotation"), ("daily", "daily windows")):
        if key in mapping:
            missing.append(what)
    _read_until(mapping.get("until"), s, missing, keys["of"])
    rule = mapping.get(keys["rule"], {})
    if isinstance(rule, str):
        missing.append(f"the named rule '{rule}'")
    elif isinstance(rule, dict):
        _read_rule(rule, s, missing)
    else:
        missing.append("its rule")
    if keys["only"] in mapping:
        _read_only(mapping[keys["only"]], s, missing, devices or [])
    return s, missing


def _read_only(only, s, missing, devices):
    """Per-device changes the form can show: ``ONLY_KEYS``, for FED3s in ``devices``."""
    if not isinstance(only, dict):
        missing.append("its per-device rules")
        return
    if not can_earn(s):
        missing.append("per-device rules in a phase where no poke is active")
        return
    for key, rule in only.items():
        names = [str(n).strip() for n in
                 (key if isinstance(key, (list, tuple)) else str(key).split(","))]
        who = ", ".join(names)
        if not isinstance(rule, dict) or set(rule) - set(ONLY_KEYS):
            missing.append(f"the rule for {who}, {_flow(rule)}")
            continue
        outside = [name for name in names if name not in devices]
        if outside:
            missing.append(f"a rule for {', '.join(outside)}, which isn't in devices")
            continue
        read, problems = dict(RULE_DEFAULTS), []
        _read_rule(rule, read, problems)
        if problems:
            missing.append(f"the rule for {who}, {_flow(rule)}")
            continue
        written = set(rule) | ({"ratio_kind"} if isinstance(rule.get("ratio"), str) else set())
        for name in names:
            s["only"][name] = {k: read[k] for k in written}


def _read_until(until, s, missing, scope="of"):
    """``scope`` is the key saying whose counts end the phase."""
    if until is None:
        return
    conditions = list(until) if isinstance(until, list) else [until]
    if (len(conditions) == 2 and all(isinstance(c, dict) for c in conditions)
            and set(conditions[0]) == {"rewards", scope} and conditions[0][scope] == "all"
            and set(conditions[1]) == {"after"}):
        s.update(ends="all_rewards", amount=str(conditions[0]["rewards"]),
                 fallback=str(conditions[1]["after"]))
        return
    if len(conditions) != 1 or not isinstance(conditions[0], dict):
        missing.append("an until with several conditions")
        return
    condition = conditions[0]
    if condition.get(scope) == "each" and set(condition) & {"rewards", "pokes"}:
        condition = {k: v for k, v in condition.items() if k != scope}   # as the form writes it
    keys = set(condition)
    if keys == {"rewards", scope} and condition[scope] == "all":
        s.update(ends="all_rewards", amount=str(condition["rewards"]), fallback="")
    elif keys == {"pokes", "side"} and condition["side"] in ("any", "left", "right"):
        s.update(ends="pokes", amount=str(condition["pokes"]), side=condition["side"])
    elif keys == {"at", "day"} and isinstance(condition["day"], int):
        s.update(ends="at", amount=str(condition["at"]), day=condition["day"])
    elif len(keys) == 1 and next(iter(keys)) in ("after", "at", "rewards", "pokes", "manual"):
        kind = next(iter(keys))
        s.update(ends=kind, amount="" if kind == "manual" else str(condition[kind]))
    else:
        missing.append(f"the end condition {_flow(condition)}")


def _read_rule(rule, s, missing):
    for key, value in rule.items():
        shorthand = proto.RATIO_SHORTHAND.fullmatch(value.strip()) \
            if key == "ratio" and isinstance(value, str) else None
        if key in ("active", "ratio_kind", "reward") and isinstance(value, str):
            s[key] = value
        elif shorthand:
            s["ratio_kind"] = proto.RATIO_KIND_OF[(shorthand[1] or "PR").upper()]
            if shorthand[2]:
                s["ratio"] = int(shorthand[2])
        elif key in ("ratio", "quota") and isinstance(value, int) and not isinstance(value, bool):
            s[key] = value
        elif key == "timeout" and _duration_text(value) is not None:
            s[key] = _seconds(_duration_text(value))
        elif key == "quota_period" and _duration_text(value) is not None:
            s[key] = _duration_text(value)
        elif key == "free_feed" and (isinstance(value, bool) or value in (0, 1)):
            s[key] = bool(value)
        elif key == "cue" and isinstance(value, dict):
            _read_cues(value, s, missing)
        else:
            missing.append(f"{key}: {_flow(value)}")
    if s["active"] not in ("both", "left", "right", "none"):
        missing.append(f"active: {s['active']}")
    if s["ratio_kind"] not in ("fixed", "random", "progressive"):
        missing.append(f"ratio_kind: {s['ratio_kind']}")
    if s["reward"] not in ("pellet", "none"):
        missing.append(f"reward: {s['reward']}")


def _read_cues(cues, s, missing):
    """A phase's cues replace the ones it started from, slot by slot."""
    for slot, cue in cues.items():
        if slot == "reward" and cue in ("none", "classic"):
            s.update(tone=False, flash=False, bnc=False)
            if cue == "classic":
                s.update(CLASSIC_CUE)
        elif slot in CLICK_SLOTS and cue in ("none", "click", ["click"]):
            s[f"click_{slot}"] = cue != "none"
        elif slot == "reward" and isinstance(cue, list) and _read_reward_cue(cue, s):
            continue
        elif cue != "none":
            missing.append(f"the {slot} cue {_flow(cue)}")


def _read_reward_cue(actions, s):
    """The reward cue's tone, flash and BNC train; False if it has anything else."""
    s.update(tone=False, flash=False, bnc=False)
    seen = set()
    for action in actions:
        if not (isinstance(action, dict) and len(action) == 1):
            return False
        (kind, params), = action.items()
        if kind in seen or not isinstance(params, dict):
            return False
        seen.add(kind)
        try:
            if kind == "tone" and set(params) == {"hz", "ms"}:
                s.update(tone=True, tone_hz=int(params["hz"]), tone_ms=int(params["ms"]))
            elif (kind == "flash" and set(params) == {"target", "color", "ms"}
                  and params["target"] in FLASH_TARGETS and params["color"] in COLOURS):
                s.update(flash=True, flash_target=params["target"],
                         flash_color=params["color"], flash_ms=int(params["ms"]))
            elif kind == "bnc" and set(params) == {"width", "hz", "count"}:
                s.update(bnc=True, bnc_width=int(params["width"]), bnc_hz=int(params["hz"]),
                         bnc_count=int(params["count"]))
            else:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _plain_data(value):
    """The loader's line-numbered mappings and lists as plain ones, for dumping."""
    if isinstance(value, dict):
        return {k: _plain_data(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain_data(v) for v in value]
    return value


def _flow(value):
    """A value as one line of YAML, to name it in a message."""
    text = yaml.safe_dump(_plain_data(value), default_flow_style=True, width=1000)
    return text.strip().removesuffix("...").strip()


# ---------------------------------------------------------------------------
# The list of FED3s
# ---------------------------------------------------------------------------

def devices_of(text):
    """The FED3s ``text`` lists in ``devices``, or [] when it can't be read."""
    try:
        doc = yaml.load(text, Loader=proto._Loader)   # noqa: S506 - a SafeLoader
    except yaml.YAMLError:
        return []
    devices = doc.get("devices") if isinstance(doc, dict) else None
    return [str(d) for d in devices] if isinstance(devices, list) else []


def parse_devices(text):
    """FED3 IDs as someone types them: ``1-4``, ``FED001, FED007`` or ``2 5 9``.

    Returns ``(names, problem)``: the names in the order typed, without
    repeats, or ``([], why not)``.
    """
    names = []
    for part in re.split(r"[,\s]+", re.sub(r"\s*-\s*", "-", text.strip())):
        if not part:
            continue
        match = re.fullmatch(r"(?:FED)?(\d{1,3})(?:-(?:FED)?(\d{1,3}))?", part, re.IGNORECASE)
        if not match:
            return [], f"'{part}' isn't a FED3 ID: write 7, FED007 or a range such as 1-4"
        lo, hi = int(match[1]), int(match[2] or match[1])
        if not 1 <= lo <= hi <= 999:
            return [], f"'{part}' isn't an ID or range of IDs from 1 to 999"
        names += [f"FED{n:03d}" for n in range(lo, hi + 1) if f"FED{n:03d}" not in names]
    return (names, None) if names else ([], "list at least one FED3")


def device_span(names):
    """``FED001–FED004, FED007``: the names, with runs of consecutive IDs shortened."""
    runs = []
    for name in names:
        number = int(name[3:]) if proto.DEVICE_NAME.fullmatch(name) else None
        if runs and number is not None and runs[-1][1] is not None \
                and number == runs[-1][1] + 1:
            runs[-1][1] = number
            runs[-1][2] = name
        else:
            runs.append([name, number, name])
    return ", ".join(first if first == last else
                     f"{first}{', ' if int(last[3:]) - int(first[3:]) == 1 else '–'}{last}"
                     for first, _, last in runs)


def set_devices(text, names):
    """``text`` with its ``devices`` list replaced by ``names``, on one line.
    A comment after a one-line list is kept."""
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    new = f"devices: [{', '.join(names)}]"
    start = next((i for i, line in enumerate(lines) if re.match(r"devices\s*:", line)), None)
    if start is None:
        at = next((i for i, line in enumerate(lines)
                   if re.match(r"(colors|cues|rules|phases|checks)\s*:", line)), len(lines))
        lines.insert(at, new + "\n")
        return "".join(lines)
    comment = re.search(r"\]\s*(#.*)$", lines[start].rstrip("\n"))
    end = start + 1
    while end < len(lines) and lines[end].strip() and lines[end].startswith((" ", "\t", "-")):
        end += 1                                        # a block list, "- FED001" per line
    lines[start:end] = [new + (f"   {comment[1]}" if comment else "") + "\n"]
    return "".join(lines)


def with_devices(text, names):
    """A template for ``names``: its one FED3 becomes these. A template that
    names several FED3s of its own (it gives them different rules) keeps them."""
    if names and devices_of(text) == ["FED001"]:
        return set_devices(text, names)
    return text


# ---------------------------------------------------------------------------
# Where each phase is in the text
# ---------------------------------------------------------------------------

@dataclass
class PhaseBlock:
    name: str
    start: int          # first line, from 0
    end: int            # one past the last line that belongs to it
    indent: int         # spaces before its "-"
    mapping: dict


def phase_blocks(text):
    """The phases' lines in ``text``, or ``None`` when they can't be found:
    the YAML doesn't parse, or the phases are written on one line."""
    try:
        doc = yaml.load(text, Loader=proto._Loader)   # noqa: S506 - a SafeLoader
    except yaml.YAMLError:
        return None
    if not isinstance(doc, dict):
        return None
    phases = doc.get("phases")
    if phases is None:
        return []
    if not isinstance(phases, list) or not all(isinstance(p, dict) for p in phases):
        return None
    lines = text.splitlines()
    starts = [p.line - 1 for p in phases]
    if len(set(starts)) != len(starts):
        return None
    blocks = []
    for n, (phase, start) in enumerate(zip(phases, starts)):
        stripped = lines[start].lstrip()
        if not stripped.startswith("-"):
            return None
        indent = len(lines[start]) - len(stripped)
        if n + 1 < len(starts):
            end = starts[n + 1]
        else:
            end = next((i for i in range(start + 1, len(lines))
                        if lines[i][:1].strip() and not lines[i].startswith(("#", "-"))),
                       len(lines))
        # Blank lines and comments at the end belong to what follows.
        while end - 1 > start and (not lines[end - 1].strip()
                                   or lines[end - 1].lstrip().startswith("#")):
            end -= 1
        blocks.append(PhaseBlock(str(phase.get("name", f"Phase {n + 1}")), start, end,
                                 indent, phase))
    return blocks


def insert_phase(text, snippet_for, before=None):
    """``text`` with a phase added before phase ``before`` (or at the end).
    ``snippet_for(indent)`` gives the phase's YAML."""
    blocks = phase_blocks(text) or []
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    indent = blocks[0].indent if blocks else 2
    if before is not None and before < len(blocks):
        at = blocks[before].start
    elif blocks:
        at = blocks[-1].end
    else:
        header = next((i for i, line in enumerate(lines) if line.startswith("phases:")), None)
        if header is None:
            lines.append("phases:\n")
            at = len(lines)
        else:
            if lines[header].strip() != "phases:":       # "phases: []"
                lines[header] = "phases:\n"
            at = header + 1
    lines.insert(at, snippet_for(indent))
    return "".join(lines)


def replace_phase(text, index, snippet):
    block = phase_blocks(text)[index]
    lines = text.splitlines(keepends=True)
    lines[block.start:block.end] = [snippet]
    return "".join(lines)


def remove_phase(text, index):
    block = phase_blocks(text)[index]
    lines = text.splitlines(keepends=True)
    del lines[block.start:block.end]
    # Don't leave two blank lines where the phase was.
    if (block.start < len(lines) and block.start > 0 and not lines[block.start].strip()
            and not lines[block.start - 1].strip()):
        del lines[block.start]
    return "".join(lines)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

def _classic(name, phase, explain, **settings):
    s = dict(RULE_DEFAULTS, name=phase, ends="never", amount="", fallback="", **settings)
    return (f"format: {proto.FORMAT}\n"
            f"name: {_plain(name)}\n"
            f"description: >\n"
            + "".join(f"  {line}\n" for line in explain.splitlines())
            + "devices: [FED001]\n"
              "phases:\n"
            + phase_yaml(s))


# The modes of the classic FED3 firmware (and FNT's FRTO and RR), each as a
# one-phase protocol. Pokes on the other side are always logged as inactive.
CLASSIC = {
    "Free feeding": _classic(
        "Free feeding", "Free feeding",
        "Classic FED3 Free feeding: a pellet is kept in the well, refilled 5 s\n"
        "after each one is taken. Pokes are logged.",
        active="none", free_feed=True),
    "FR1": _classic(
        "FR1", "FR1",
        "Classic FED3 FR1: each left poke earns a pellet, with the classic\n"
        "4 kHz tone and blue light.",
        **CLASSIC_CUE),
    "FR3": _classic(
        "FR3", "FR3",
        "Classic FED3 FR3: every third left poke earns a pellet, with the\n"
        "classic tone and light.",
        ratio=3, **CLASSIC_CUE),
    "FR5": _classic(
        "FR5", "FR5",
        "Classic FED3 FR5: every fifth left poke earns a pellet, with the\n"
        "classic tone and light.",
        ratio=5, **CLASSIC_CUE),
    "Progressive ratio": _classic(
        "Progressive ratio", "PR",
        "Classic FED3 Progressive Ratio on the left poke: the pokes needed rise\n"
        "1, 2, 4, 6, 9, 12, 15, 20... Each poke clicks; a reward plays the\n"
        "classic tone and light.",
        ratio_kind="progressive", click_counted=True, **CLASSIC_CUE),
    "Extinction": _classic(
        "Extinction", "Extinction",
        "Classic FED3 Extinction: left pokes play the classic tone and light\n"
        "but deliver nothing.",
        reward="none", **CLASSIC_CUE),
    "FR1 reversed": _classic(
        "FR1 reversed", "FR1 right",
        "Classic FED3 FR1 (reversed): each right poke earns a pellet. For FR3 or\n"
        "FR5 reversed, change the ratio.",
        active="right", **CLASSIC_CUE),
    "Progressive ratio reversed": _classic(
        "Progressive ratio reversed", "PR right",
        "Classic FED3 PR (reversed): Progressive Ratio on the right poke.",
        active="right", ratio_kind="progressive", click_counted=True, **CLASSIC_CUE),
    "Optogenetic stimulation": _classic(
        "Optogenetic stimulation", "Opto",
        "Classic FED3 Optogenetic stimulation: each left poke plays the classic\n"
        "tone and light and sends 20 BNC pulses at 20 Hz (25 ms each), and\n"
        "delivers no pellet.",
        reward="none", bnc=True, **CLASSIC_CUE),
    "Optogenetic stimulation reversed": _classic(
        "Optogenetic stimulation reversed", "Opto right",
        "Classic FED3 Optogenetic stimulation (reversed): as Opto, on the right\n"
        "poke.",
        active="right", reward="none", bnc=True, **CLASSIC_CUE),
    "Timed feeding": """\
format: fnt-fed3-protocol/3
name: Timed feeding 19:00-21:00
description: >
  Classic FED3 Timed feeding: free feeding between two clock times each
  day, nothing outside them. Change the window's from and to.
devices: [FED001]
phases:
  - name: Restricted
    rule: {active: none}
    daily:
      - {from: "19:00", to: "21:00", rule: {free_feed: 1}}
""",
    "FR with timeout": _classic(
        "FR1 with a 30 s timeout", "FRTO",
        "FNT's FR with timeout: FR1 on the left poke, then 30 s in which pokes\n"
        "only click and don't count.",
        timeout=30, click_timeout=True, **CLASSIC_CUE),
    "Random ratio": _classic(
        "Random ratio 3", "RR3",
        "FNT's Random Ratio: each left poke earns a pellet with probability 1/3.\n"
        "Pokes that don't, click.",
        ratio_kind="random", ratio=3, click_counted=True, **CLASSIC_CUE),
}
# Shown in the menu, but it can't be made: the v3 lights are off unless a cue
# flashes them.
UNSUPPORTED = {"Light tracking": "Light tracking needs a poke light that stays on, "
                                 "which firmware v3 doesn't have."}

EXAMPLES = {
    "FR1, both pokes, with a cue": """\
format: fnt-fed3-protocol/3
name: FR1 both pokes
devices: [FED001]
phases:
  - name: FR1
    rule:
      active: both
      cue:
        reward:
          - tone: {hz: 4000, ms: 200}
          - flash: {target: poke, color: blue, ms: 500}
""",
    "Step up the ratio by rewards": """\
format: fnt-fed3-protocol/3
name: FR1 to FR5, both pokes
devices: [FED001]

# Every phase starts from this, so each only says what it changes.
every_phase:
  active: both
  cue:
    reward:
      - tone: {hz: 4000, ms: 200}
      - flash: {target: poke, color: blue, ms: 500}

phases:
  - name: FR1
    until: {rewards: 50, of: each}
    rule: {ratio: FR1}
  - name: FR3
    until: {rewards: 100, of: each}
    rule: {ratio: FR3}
  - name: FR5
    rule: {ratio: FR5}
""",
    "Four-patch competition": """\
format: fnt-fed3-protocol/3
name: Four-patch competition
description: >
  Two rich and two poor patches that each refill hourly. Which patches are
  rich is reshuffled every day at lights-off.
devices: [FED001, FED002, FED003, FED004]

cues:
  reward:
    - tone: {hz: 4000, ms: 200}
    - flash: {target: bar, color: green, ms: 1000}

rules:
  patch:
    active: both
    cue: {reward: reward, depleted: [click]}
  rich: {use: patch, ratio: FR1, quota: 30, quota_period: 1h}
  poor: {use: patch, ratio: FR5, quota: 10, quota_period: 1h}

phases:
  - name: Habituation
    until: {at: "19:00", day: 3}
    rule: {active: none, free_feed: 1}

  - name: Training
    until: [{rewards: 100, of: all}, {after: 4d}]
    rule: patch

  - name: Competition
    until: {after: 14d}
    rotate:
      every: 1d
      at: "19:00"
      mode: shuffle
      assign:
        FED001, FED002: rich
        FED003, FED004: poor

  - name: Recovery
    rule: {active: none, free_feed: 1}

checks:
  no_pellets: 8h
""",
}

TEMPLATES = {**EXAMPLES, **CLASSIC}
DEFAULT_TEMPLATE = "FR1, both pokes, with a cue"


# ---------------------------------------------------------------------------
# The form
# ---------------------------------------------------------------------------

def _spin(lo, hi, value, suffix=""):
    spin = QSpinBox()
    spin.setRange(lo, hi)
    spin.setValue(value)
    if suffix:
        spin.setSuffix(suffix)
    return spin


def _row(*widgets):
    row = QHBoxLayout()
    for widget in widgets:
        if isinstance(widget, str):
            widget = QLabel(widget)
        row.addWidget(widget)
    row.addStretch()
    return row


class PhaseDialog(QDialog):
    """The common settings of one phase, written out as YAML.

    ``places`` (phase names) adds a choice of where a new phase goes; editing
    passes ``settings`` read from the phase instead. With several ``devices``
    a table sets what differs for each FED3. ``base`` is what each phase starts
    from (the file's every_phase): a new phase starts there, and only what
    differs from it is written.
    """

    def __init__(self, parent=None, settings=None, places=None, note="", keys=CURRENT_KEYS,
                 several=False, devices=None, base=None):
        super().__init__(parent)
        editing = settings is not None
        self.setWindowTitle("Edit a phase" if editing else "Add a phase")
        self.places = places or []
        self.devices = list(devices or [])
        self.keys, self.several = keys, several or len(self.devices) > 1
        self.base = base or RULE_DEFAULTS
        inherited = {k: v for k, v in self.base.items() if v != RULE_DEFAULTS[k]}
        s = {**NEW_PHASE, **inherited, **(settings or {})}
        if not editing and self.places:
            s["ends"], s["amount"] = "never", ""     # it goes last, and the last runs until Stop
        form = QFormLayout(self)
        self.name = QLineEdit(s["name"])
        form.addRow("Name", self.name)
        self.place = QComboBox()
        if not editing:
            self.place.addItem("at the end")
            for name in self.places:
                self.place.addItem(f"before {name}")
            form.addRow("Place", self.place)
            self.place.currentIndexChanged.connect(self._place_changed)

        self.ends = QComboBox()
        for key, label in ENDS:
            if self.several and key in EACH_ENDS:
                label += " (each device on its own)"
            self.ends.addItem(label, key)
        self.amount = QLineEdit()
        self.fallback_label = QLabel("or after")
        self.fallback = QLineEdit(s.get("fallback", "7d"))
        self.fallback.setMaximumWidth(70)
        self.fallback.setToolTip("Ends on its own after this long even if FNT isn't running; "
                                 "leave empty for no limit")
        self.day = _spin(0, 366, 0)
        self.day.setPrefix("on day ")
        self.day.setSpecialValueText("the next one")
        self.day.setToolTip("Day 1 is the day the run starts")
        self.side = QComboBox()
        self.side.addItems(["any", "left", "right"])
        self.side.setToolTip("Which poke counts")
        form.addRow("Ends", _row(self.ends, self.amount, self.day, self.side,
                                 self.fallback_label, self.fallback))

        self.active = QComboBox()
        self.active.addItems(ACTIVE)
        form.addRow("Active poke", self.active)
        self.kind = QComboBox()
        self.kind.addItems(RATIO_KINDS)
        self.ratio = _spin(1, 999, 1)
        form.addRow("Ratio", _row(self.kind, "ratio", self.ratio))
        self.timeout = _spin(0, 3600, 0, " s")
        form.addRow("Timeout after a reward", self.timeout)
        self.quota = _spin(0, 9999, 0)
        self.quota.setSpecialValueText("unlimited")
        self.quota_period = QLineEdit()
        self.quota_period.setMaximumWidth(70)
        form.addRow("Quota", _row(self.quota, "rewards per", self.quota_period))
        self.reward = QComboBox()
        self.reward.addItem("deliver a pellet", "pellet")
        self.reward.addItem("no pellet (extinction, opto)", "none")
        form.addRow("A reward", self.reward)
        self.free = QCheckBox("keep a pellet in the well")
        form.addRow("Free feeding", self.free)

        self.tone = QCheckBox("tone")
        self.tone_hz = _spin(20, 20000, 4000, " Hz")
        self.tone_ms = _spin(1, 10000, 200, " ms")
        form.addRow("On a reward", _row(self.tone, self.tone_hz, self.tone_ms))
        self.flash = QCheckBox("flash")
        self.flash_target = QComboBox()
        self.flash_target.addItems(FLASH_TARGETS)
        self.flash_color = QComboBox()
        self.flash_color.addItems(COLOURS)
        self.flash_ms = _spin(1, 60000, 500, " ms")
        form.addRow("", _row(self.flash, self.flash_target, self.flash_color, self.flash_ms))
        self.bnc = QCheckBox("BNC pulses")
        self.bnc_width = _spin(1, 10000, 25, " ms")
        self.bnc_hz = _spin(1, 1000, 20, " Hz")
        self.bnc_count = _spin(1, 10000, 20, " pulses")
        self.bnc.setToolTip("A pulse train on the BNC output: for opto, external cues or sync")
        form.addRow("", _row(self.bnc, self.bnc_width, self.bnc_hz, self.bnc_count))
        self.clicks = {slot: QCheckBox(label) for slot, label in (
            ("counted", "pokes that don't earn a reward yet"),
            ("timeout", "pokes during the timeout"),
            ("depleted", "pokes after the quota is used"))}
        form.addRow("Click on", _row(*self.clicks.values()))

        # One row per FED3: "same" keeps the setting above, anything else is
        # written for that FED3 under the phase's `only`.
        self.per_device = {}
        if len(self.devices) > 1 or (s.get("only") and self.devices):
            grid = QGridLayout()
            grid.setContentsMargins(0, 0, 0, 0)
            for col, title in enumerate(("", "Active poke", "Ratio", "", "A reward", "Runs as")):
                header = QLabel(title)
                header.setObjectName("muted")
                grid.addWidget(header, 0, col)
            for row, name in enumerate(self.devices, start=1):
                widgets = {
                    "active": self._same_combo([(a, a) for a in ACTIVE]),
                    "ratio_kind": self._same_combo([(k, k) for k in RATIO_KINDS]),
                    "ratio": _spin(0, 999, 0),
                    "reward": self._same_combo([("pellet", "pellet"), ("no pellet", "none")]),
                }
                widgets["ratio"].setSpecialValueText("same")
                summary = QLabel()
                summary.setObjectName("muted")
                for col, widget in enumerate((QLabel(name), *widgets.values(), summary)):
                    grid.addWidget(widget, row, col)
                self.per_device[name] = (widgets, summary)
            grid.setColumnStretch(5, 1)
            table = QWidget()
            table.setLayout(grid)
            scroll = QScrollArea()
            scroll.setWidget(table)
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.NoFrame)
            scroll.setMinimumHeight(min(table.sizeHint().height() + 4, 260))   # up to ~8 rows
            form.addRow("Per device", scroll)
            self.per_device_note = QLabel(
                "'same' uses the settings above; anything else applies to that FED3 alone "
                f"(the file lists it under {keys['only']}).")
            self.per_device_note.setObjectName("muted")
            self.per_device_note.setWordWrap(True)
            form.addRow("", self.per_device_note)

        self.hint = QLabel(note)
        self.hint.setObjectName("muted")
        self.hint.setWordWrap(True)
        form.addRow(self.hint)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Save phase" if editing else "Add phase")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self._note = note

        self._active_before = None      # the poke to go back to when free feeding is unticked
        self.ends.currentIndexChanged.connect(self._ends_changed)
        self.kind.currentIndexChanged.connect(self._enable)
        self.active.currentIndexChanged.connect(self._enable)
        self.free.toggled.connect(self._free_toggled)
        self.ratio.valueChanged.connect(self._enable)
        self.timeout.valueChanged.connect(self._enable)
        for box in (self.tone, self.flash, self.bnc):
            box.toggled.connect(self._enable)
        self.quota.valueChanged.connect(self._enable)
        for widgets, _ in self.per_device.values():
            for key, widget in widgets.items():
                (widget.valueChanged if key == "ratio" else
                 widget.currentIndexChanged).connect(self._enable)
        self._load(s)

    @staticmethod
    def _same_combo(choices):
        """A choice for one FED3, starting with "same": the shared setting."""
        box = QComboBox()
        box.addItem("same", None)
        for label, value in choices:
            box.addItem(label, value)
        return box

    def _device_change(self, name):
        """What the table changes for one FED3, as keys of ``ONLY_KEYS``."""
        widgets, _ = self.per_device[name]
        change = {key: widget.currentData() for key, widget in widgets.items()
                  if key != "ratio" and widget.currentData() is not None}
        if widgets["ratio"].value():
            change["ratio"] = widgets["ratio"].value()
        return change

    def _load(self, s):
        self.ends.blockSignals(True)
        self.ends.setCurrentIndex([key for key, _ in ENDS].index(s["ends"]))
        self.ends.blockSignals(False)
        self.amount.setText(str(s.get("amount") or END_AMOUNTS.get(s["ends"], "")))
        self.fallback.setText(str(s.get("fallback", "")))
        self.day.setValue(s.get("day", 0))
        self.side.setCurrentText(s.get("side", "any"))
        self.active.setCurrentText(s["active"])
        self.kind.setCurrentText(s["ratio_kind"])
        self.ratio.setValue(s["ratio"])
        self.timeout.setValue(s["timeout"])
        self.quota.setValue(s["quota"])
        self.quota_period.setText(str(s["quota_period"]))
        self.reward.setCurrentIndex(self.reward.findData(s["reward"]))
        self.free.setChecked(s["free_feed"])
        self.tone.setChecked(s["tone"])
        self.tone_hz.setValue(s["tone_hz"])
        self.tone_ms.setValue(s["tone_ms"])
        self.flash.setChecked(s["flash"])
        self.flash_target.setCurrentText(s["flash_target"])
        self.flash_color.setCurrentText(s["flash_color"])
        self.flash_ms.setValue(s["flash_ms"])
        self.bnc.setChecked(s["bnc"])
        self.bnc_width.setValue(s["bnc_width"])
        self.bnc_hz.setValue(s["bnc_hz"])
        self.bnc_count.setValue(s["bnc_count"])
        for slot, box in self.clicks.items():
            box.setChecked(s[f"click_{slot}"])
        for name, (widgets, _) in self.per_device.items():
            change = (s.get("only") or {}).get(name, {})
            for key, widget in widgets.items():
                if key == "ratio":
                    widget.setValue(change.get("ratio", 0))
                else:
                    widget.setCurrentIndex(max(widget.findData(change.get(key)), 0))
        self._enable()
        self._update_hint()

    def _ends_changed(self):
        self.amount.setText(END_AMOUNTS.get(self.ends.currentData(), ""))
        self._enable()

    def _place_changed(self):
        # Only the last phase runs until Stop; every other one needs an end.
        last = self.place.currentIndex() == 0
        if last != (self.ends.currentData() == "never"):
            self.ends.setCurrentIndex([key for key, _ in ENDS].index("never" if last else "after"))
        self._update_hint()

    def _free_toggled(self, free):
        self.active.blockSignals(True)
        if free:
            self._active_before = self.active.currentText()
            self.active.setCurrentText("none")
        elif self._active_before:
            self.active.setCurrentText(self._active_before)
        self.active.blockSignals(False)
        self._enable()

    def _enable(self):
        """Switch off whatever can't apply (see coherent), so the form can't
        describe a phase that contradicts itself."""
        free = self.free.isChecked()
        if free and self.active.currentText() != "none":
            self.active.blockSignals(True)
            self.active.setCurrentText("none")
            self.active.blockSignals(False)
        earn = can_earn({"free_feed": free, "active": self.active.currentText()})
        self.active.setEnabled(not free)

        # A phase that ends on rewards needs a poke that earns them.
        model = self.ends.model()
        for n, (key, _) in enumerate(ENDS):
            model.item(n).setEnabled(earn or key not in REWARD_ENDS)
        if self.ends.currentData() in REWARD_ENDS and not earn:
            self.ends.setCurrentIndex([key for key, _ in ENDS].index("after"))   # re-enters
            return
        ends = self.ends.currentData()
        self.amount.setEnabled(ends in END_AMOUNTS)
        for widget in (self.fallback_label, self.fallback):
            widget.setVisible(ends == "all_rewards")
        self.day.setVisible(ends == "at")
        self.side.setVisible(ends == "pokes")

        progressive = self.kind.currentText() == "progressive"
        for widget in (self.kind, self.timeout, self.quota, self.reward,
                       self.tone, self.flash, self.bnc):
            widget.setEnabled(earn)
        self.ratio.setEnabled(earn and not progressive)
        self.quota_period.setEnabled(earn and self.quota.value() > 0)
        self.tone_hz.setEnabled(earn and self.tone.isChecked())
        self.tone_ms.setEnabled(earn and self.tone.isChecked())
        for widget in (self.flash_target, self.flash_color, self.flash_ms):
            widget.setEnabled(earn and self.flash.isChecked())
        for widget in (self.bnc_width, self.bnc_hz, self.bnc_count):
            widget.setEnabled(earn and self.bnc.isChecked())
        shared = {"ratio_kind": self.kind.currentText(), "ratio": self.ratio.value(),
                  "only": {name: self._device_change(name) for name in self.per_device}}
        for name, (widgets, summary) in self.per_device.items():
            change = shared["only"][name]
            for widget in widgets.values():
                widget.setEnabled(earn)
            widgets["ratio"].setEnabled(
                earn and change.get("ratio_kind", shared["ratio_kind"]) != "progressive")
            summary.setText(self._summary(change) if earn else "")
        every_poke = every_poke_earns(shared)
        self.clicks["counted"].setEnabled(earn and not every_poke)
        self.clicks["timeout"].setEnabled(earn and self.timeout.value() > 0)
        self.clicks["depleted"].setEnabled(earn and self.quota.value() > 0)
        self._update_hint()

    def _summary(self, change):
        """What one FED3 runs, as the cards and timeline say it: ``RR2 · both``."""
        s = dict(self.settings(), **change)
        period = _duration_text(s["quota_period"])
        rule = dict(sch.DEFAULTS, active=s["active"], ratio_kind=s["ratio_kind"],
                    ratio=str(s["ratio"]), reward=s["reward"], timeout=str(s["timeout"]),
                    quota=str(s["quota"]), free_feed="1" if s["free_feed"] else "0",
                    quota_period=str(_seconds(period)) if period
                    else sch.DEFAULTS["quota_period"])
        return sch.summarize(rule)

    def _update_hint(self):
        notes = [self._note] if getattr(self, "_note", "") else []
        if hasattr(self, "free") and self.free.isChecked():
            notes.append("Free feeding keeps a pellet in the well, so no poke is active and "
                         "nothing is earned: the ratio, reward and cue settings don't apply.")
        elif hasattr(self, "active") and self.active.currentText() == "none":
            notes.append("No poke is active, so nothing is earned: the ratio, reward and cue "
                         "settings don't apply, and the phase can't end on rewards.")
        if self.places and self.place.count() and self.place.currentIndex() == 0:
            notes.append(f"'{self.places[-1]}' runs until Stop now, as the last phase. "
                         f"Once this one follows it, give it an end with Edit phase.")
        if hasattr(self, "hint"):
            self.hint.setText("\n".join(notes))

    def settings(self):
        """What the form says, with anything switched off left out."""
        return coherent({
            "name": self.name.text().strip() or "New phase",
            "ends": self.ends.currentData(), "amount": self.amount.text().strip(),
            "fallback": self.fallback.text().strip(),
            "day": self.day.value(), "side": self.side.currentText(),
            "active": self.active.currentText(), "ratio_kind": self.kind.currentText(),
            "ratio": self.ratio.value(), "timeout": self.timeout.value(),
            "quota": self.quota.value(), "quota_period": self.quota_period.text().strip() or "1h",
            "free_feed": self.free.isChecked(), "reward": self.reward.currentData(),
            "tone": self.tone.isChecked(), "tone_hz": self.tone_hz.value(),
            "tone_ms": self.tone_ms.value(),
            "flash": self.flash.isChecked(), "flash_target": self.flash_target.currentText(),
            "flash_color": self.flash_color.currentText(), "flash_ms": self.flash_ms.value(),
            "bnc": self.bnc.isChecked(), "bnc_width": self.bnc_width.value(),
            "bnc_hz": self.bnc_hz.value(), "bnc_count": self.bnc_count.value(),
            **{f"click_{slot}": box.isChecked() for slot, box in self.clicks.items()},
            "only": {name: self._device_change(name) for name in self.per_device},
        })

    def before(self):
        """The phase a new one goes in front of, or None for the end."""
        index = self.place.currentIndex()
        return index - 1 if index > 0 else None

    def yaml(self, indent=2):
        return phase_yaml(self.settings(), indent, self.keys, self.several, self.base)


class DevicesDialog(QDialog):
    """The protocol's FED3s, typed as IDs or ranges. ``connected`` offers the
    FED3s plugged in now."""

    def __init__(self, names, parent=None, connected=None):
        super().__init__(parent)
        self.setWindowTitle("The protocol's FED3s")
        layout = QVBoxLayout(self)
        intro = QLabel("Every FED3 listed here runs every phase. Type IDs or ranges: "
                       "1-4, or FED001, FED007.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.line = QLineEdit(", ".join(names))
        layout.addWidget(self.line)
        self.reading = QLabel()
        self.reading.setObjectName("muted")
        self.reading.setWordWrap(True)
        layout.addWidget(self.reading)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        if connected:
            plugged = buttons.addButton(f"Use the connected ones ({device_span(connected)})",
                                        QDialogButtonBox.ResetRole)
            plugged.clicked.connect(lambda: self.line.setText(", ".join(connected)))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self.ok = buttons.button(QDialogButtonBox.Ok)
        layout.addWidget(buttons)
        self.line.textChanged.connect(self._read)
        self._read()

    def _read(self):
        names, problem = parse_devices(self.line.text())
        self.reading.setText(problem or f"{sch.plural(len(names), 'FED3')}: "
                                        f"{device_span(names)}")
        self.reading.setStyleSheet(f"color: {theme.BAD};" if problem else "")
        self.ok.setEnabled(not problem)

    def names(self):
        return parse_devices(self.line.text())[0]


# ---------------------------------------------------------------------------
# The editor
# ---------------------------------------------------------------------------

class ProtocolEditor(QDialog):
    """Edit a protocol. ``running`` locks it to edits a running run accepts:
    ``edit_check(text)`` says why the run can't take the text, or None.
    ``connected`` names the FED3s plugged in, for the Devices choice."""

    def __init__(self, text="", parent=None, path=None, running=False, t0=None,
                 edit_check=None, connected=None):
        super().__init__(parent)
        self.setWindowTitle("Edit protocol" if not running else "Edit the running protocol")
        self.resize(1200, 760)
        self.path = path
        self.running = running
        self.t0 = t0
        self.edit_check = edit_check
        self.connected = list(connected or [])
        self.compiled = None
        self.problems = []

        layout = QVBoxLayout(self)
        tools = QHBoxLayout()
        # The FED3s, which a running run can't change.
        self.devices_button = theme.quiet(QPushButton())
        self.devices_button.setToolTip("The FED3s this protocol uses. Templates take this "
                                       "list, and Add phase can set each one's rule.")
        self.devices_button.clicked.connect(self._change_devices)
        self.devices_button.setVisible(not running)
        if not running:
            templates = theme.quiet(QPushButton("New from template ▾"))
            menu = QMenu(templates)
            classic = menu.addMenu("Classic FED3 modes")
            for name in CLASSIC:
                classic.addAction(name, lambda n=name: self._use_template(n))
            for name, why in UNSUPPORTED.items():
                action = classic.addAction(f"{name} (not available)")
                action.setEnabled(False)
                action.setToolTip(why)
            classic.setToolTipsVisible(True)
            examples = menu.addMenu("Examples")
            for name in EXAMPLES:
                examples.addAction(name, lambda n=name: self._use_template(n))
            templates.setMenu(menu)
            tools.addWidget(templates)
        tools.addWidget(self.devices_button)
        add = theme.quiet(QPushButton("Add phase…"))
        add.clicked.connect(self._add_phase)
        tools.addWidget(add)
        self.edit_phase = theme.quiet(QPushButton("Edit phase ▾"))
        self.edit_phase.setMenu(QMenu(self.edit_phase))
        self.edit_phase.menu().aboutToShow.connect(
            lambda: self._phase_menu(self.edit_phase.menu(), self._edit_phase))
        tools.addWidget(self.edit_phase)
        self.remove_phase = theme.quiet(QPushButton("Remove phase ▾"))
        self.remove_phase.setMenu(QMenu(self.remove_phase))
        self.remove_phase.menu().aboutToShow.connect(
            lambda: self._phase_menu(self.remove_phase.menu(), self._remove_phase))
        tools.addWidget(self.remove_phase)
        tools.addStretch()
        if running:
            note = QLabel("Phases that are over can't change. The current phase can, and "
                          "the change applies at once.")
            note.setObjectName("muted")
            tools.addWidget(note)
        layout.addLayout(tools)

        split = QSplitter(Qt.Horizontal)
        self.editor = QPlainTextEdit()
        self.editor.setFont(QFont("Consolas", 10))
        self.editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.editor.setPlainText(text)
        split.addWidget(self.editor)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        self.verdict = QLabel()
        side_layout.addWidget(self.verdict)
        # Only there when there's something to fix; click one to go to its line.
        self.problem_list = QListWidget()
        self.problem_list.itemClicked.connect(self._jump)
        self.problem_list.setMaximumHeight(140)
        self.problem_list.hide()
        side_layout.addWidget(self.problem_list)
        timeline = QLabel("Timeline")
        timeline.setObjectName("muted")
        side_layout.addWidget(timeline)
        self.outline = self._readonly_text()
        side_layout.addWidget(self.outline, 1)
        # The schedule text each FED3 receives: for checking the compiler, so
        # closed until someone opens it.
        self.sent = self._readonly_text()
        self.sent.hide()
        self.sent_toggle = QPushButton()
        self.sent_toggle.setObjectName("disclosure")
        self.sent_toggle.setCheckable(True)
        self.sent_toggle.toggled.connect(self._show_sent)
        self._show_sent(False)
        side_layout.addWidget(self.sent_toggle)
        side_layout.addWidget(self.sent, 1)
        split.addWidget(side)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([700, 500])
        layout.addWidget(split, 1)      # the text and panes take the height, not the toolbar

        buttons = QHBoxLayout()
        if not running:
            save_as = theme.quiet(QPushButton("Save as…"))
            save_as.clicked.connect(self._save_as)
            buttons.addWidget(save_as)
        buttons.addStretch()
        cancel = theme.quiet(QPushButton("Cancel"))
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        self.ok = QPushButton("Apply to running devices" if running else "Use this protocol")
        self.ok.clicked.connect(self._accept)
        buttons.addWidget(self.ok)
        layout.addLayout(buttons)

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(300)
        self.timer.timeout.connect(self._check)
        self.editor.textChanged.connect(self.timer.start)
        self._check()

    @staticmethod
    def _readonly_text():
        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setFont(QFont("Consolas", 9))
        text.setLineWrapMode(QPlainTextEdit.NoWrap)
        return text

    def _show_sent(self, opened):
        self.sent.setVisible(opened)
        self.sent_toggle.setText(f"{'▼' if opened else '▶'}  What each FED3 is sent")

    def text(self):
        return self.editor.toPlainText()

    def _set_text(self, text):
        """Replace the text as one undoable change, and re-check at once."""
        cursor = self.editor.textCursor()
        cursor.select(QTextCursor.Document)
        cursor.insertText(text)
        self._check()

    def _use_template(self, name):
        """The template, for the FED3s the protocol already lists."""
        self._set_text(with_devices(TEMPLATES[name], devices_of(self.text()) or self.connected))

    def _change_devices(self):
        dialog = DevicesDialog(devices_of(self.text()), self, connected=self.connected)
        if dialog.exec_() == QDialog.Accepted:
            self._set_text(set_devices(self.text(), dialog.names()))

    def _check(self):
        names = devices_of(self.text())
        self.devices_button.setText(f"Devices: {device_span(names)}…" if names
                                    else "Devices…")
        self.compiled, self.problems = proto.check(self.text())
        errors = [p for p in self.problems if p.level == "error"]
        self.problem_list.clear()
        for problem in self.problems:
            where = f"line {problem.line}: " if problem.line else ""
            item = QListWidgetItem(f"{where}{problem.level}: {problem.message}")
            item.setData(Qt.UserRole, problem.line)
            item.setForeground(Qt.red if problem.level == "error" else Qt.yellow)
            self.problem_list.addItem(item)
        self.problem_list.setVisible(bool(self.problems))
        if self.problems:
            rows = sum(self.problem_list.sizeHintForRow(r) for r in range(self.problem_list.count()))
            self.problem_list.setFixedHeight(min(rows + 8, 140))
        if errors:
            self.verdict.setText(f"✗ {len(errors)} problem{'s' if len(errors) > 1 else ''}")
            self.verdict.setStyleSheet(f"color: {theme.BAD}; font-weight: bold;")
            self.outline.setPlainText("")
            self.sent.setPlainText("")
        else:
            c = self.compiled
            self.verdict.setText(f"✓ checks pass · {sch.plural(len(c.phases), 'phase')} · "
                                 f"{len(c.protocol.devices)} device{'s' if len(c.protocol.devices) > 1 else ''}")
            self.verdict.setStyleSheet(f"color: {theme.GOOD}; font-weight: bold;")
            t0 = self.t0 or datetime.now(sch.standard_timezone()).replace(second=0, microsecond=0)
            self.outline.setPlainText(proto.preview_text(c, t0))
            self.sent.setPlainText("\n".join(
                f"── {name} ({sch.plural(len(s.phases), 'phase')}, crc {s.crc():08x})\n{s.text()}"
                for name, s in c.schedules.items()))
        refused = None
        if not errors and self.edit_check:
            refused = self.edit_check(self.text())
            if refused:
                self.verdict.setText(f"✗ The running devices can't take this: {refused}")
                self.verdict.setStyleSheet(f"color: {theme.BAD}; font-weight: bold;")
                self.verdict.setWordWrap(True)
        self.ok.setEnabled(not errors and not refused)

    def _jump(self, item):
        self._go_to_line(item.data(Qt.UserRole))

    def _go_to_line(self, line):
        if not line:
            return
        block = self.editor.document().findBlockByLineNumber(line - 1)
        self.editor.setTextCursor(QTextCursor(block))
        self.editor.centerCursor()
        self.editor.setFocus()

    # --- phases from the form ------------------------------------------------

    def _phase_menu(self, menu, slot):
        menu.clear()
        blocks = phase_blocks(self.text())
        if blocks is None:
            menu.addAction("Fix the YAML first: the phases can't be found").setEnabled(False)
        elif not blocks:
            menu.addAction("No phases yet").setEnabled(False)
        for n, block in enumerate(blocks or []):
            menu.addAction(f"{n + 1}. {block.name}", lambda n=n: slot(n))

    def _add_phase(self):
        blocks = phase_blocks(self.text())
        if blocks is None:
            QMessageBox.information(self, "Add a phase",
                                    "Fix the YAML first, so the phases can be found.")
            return
        keys, several = syntax_of(self.text())
        base, missing = every_phase_of(self.text())
        if missing:
            QMessageBox.information(self, "Add a phase", "Every phase starts from the "
                                    "protocol's every_phase, and the form can't show all of "
                                    "it:\n\n  • " + "\n  • ".join(missing) +
                                    "\n\nWrite the new phase in the text instead.")
            return
        dialog = PhaseDialog(self, places=[b.name for b in blocks], keys=keys, several=several,
                             devices=devices_of(self.text()), base=base)
        if dialog.exec_() != QDialog.Accepted:
            return
        self._set_text(insert_phase(self.text(), dialog.yaml, dialog.before()))

    def _edit_phase(self, index):
        blocks = phase_blocks(self.text())
        if not blocks or index >= len(blocks):
            return
        block = blocks[index]
        keys, several = syntax_of(self.text())
        devices = devices_of(self.text())
        base, missing = every_phase_of(self.text())
        settings, in_phase = settings_from_phase(block.mapping, keys, devices, base)
        missing += in_phase
        if missing:
            box = QMessageBox(QMessageBox.Information, "Edit a phase",
                              f"The form can't show everything in '{block.name}', so it "
                              f"would lose:\n\n  • " + "\n  • ".join(missing) +
                              "\n\nEdit this phase in the text instead.", parent=self)
            show = box.addButton("Show it in the text", QMessageBox.AcceptRole)
            box.addButton(QMessageBox.Cancel)
            box.exec_()
            if box.clickedButton() is show:
                self._go_to_line(block.start + 1)
            return
        lines = self.text().splitlines()[block.start:block.end]
        note = ("Comments inside this phase will be removed."
                if any(line.lstrip().startswith("#") or " #" in line for line in lines) else "")
        dialog = PhaseDialog(self, settings=settings, note=note, keys=keys, several=several,
                             devices=devices, base=base)
        if dialog.exec_() != QDialog.Accepted:
            return
        self._set_text(replace_phase(self.text(), index, dialog.yaml(block.indent)))
        self._go_to_line(block.start + 1)

    def _remove_phase(self, index):
        blocks = phase_blocks(self.text())
        if not blocks or index >= len(blocks):
            return
        if QMessageBox.question(self, "Remove a phase",
                                f"Remove phase {index + 1}, '{blocks[index].name}'?") \
                != QMessageBox.Yes:
            return
        self._set_text(remove_phase(self.text(), index))

    # --- saving --------------------------------------------------------------

    def _save_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save protocol", self.path or "",
                                              "Protocol (*.yaml *.yml)")
        if path:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(self.text())
            self.path = path

    def _accept(self):
        self._check()
        if self.ok.isEnabled():
            self.accept()


def format_delta(seconds):
    seconds = max(int(seconds), 0)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    if days:
        return f"{days}d {hours:02d}h"
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m"
