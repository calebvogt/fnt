"""The protocol editor: the YAML file, checked as you type, with its timeline.

The file is the protocol, so the editor edits the file. A form covers the
common settings of a phase without typing YAML: **Add phase** writes a new
one, **Edit phase** reads one back into the form and rewrites it, and
**Remove phase** deletes one. Everything else (rotation, daily windows, named
rules) is written in the text, with problems reported against their line as
you go; the form says so rather than touch a phase it can't show in full.
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
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu,
    QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QSplitter, QVBoxLayout,
    QWidget,
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

# The classic FED3 reward cue: a 4 kHz tone and the blue light.
CLASSIC_CUE = {"tone": True, "flash": True, "flash_target": "bar", "flash_ms": 1000}


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


def can_earn(s):
    """Whether a poke can earn a reward under these settings."""
    return not s["free_feed"] and s["active"] != "none"


def coherent(s):
    """The settings with everything that can't apply taken out.

    Free feeding keeps a pellet in the well, so no poke is active and nothing
    is earned; with nothing earned the ratio, timeout, quota, reward and its
    cues mean nothing. A click needs the thing it marks: a counted poke that
    isn't yet a reward, a timeout, a quota.
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
    if s["ratio_kind"] == "fixed" and s["ratio"] == 1:
        s["click_counted"] = False          # every counted poke is a reward
    if not s["timeout"]:
        s["click_timeout"] = False
    if not s["quota"]:
        s["click_depleted"] = False
    return s


def phase_yaml(s, indent=2):
    """One phase list item, from the form's settings."""
    s = coherent(s)
    pad = " " * indent
    lines = [f"{pad}- name: {_plain(s['name'].strip() or 'New phase')}"]
    amount = str(s.get("amount", "")).strip().strip('"\'')
    day = f", day: {s['day']}" if s.get("day") else ""
    side = f", side: {s['side']}" if s.get("side", "any") != "any" else ""
    until = {
        "after": f"{{after: {_plain(amount)}}}",
        "at": f"{{at: {json.dumps(amount)}{day}}}",
        "rewards": f"{{rewards: {amount}}}",
        "pokes": f"{{pokes: {amount}{side}}}",
        "manual": "{manual: true}",
    }.get(s["ends"])
    if s["ends"] == "all_rewards":
        fallback = str(s.get("fallback", "")).strip()
        until = (f"[{{rewards: {amount}, devices: all}}, {{after: {_plain(fallback)}}}]"
                 if fallback else f"{{rewards: {amount}, devices: all}}")
    if until:
        lines.append(f"{pad}  until: {until}")
    rule = [f"active: {s['active']}"]
    if s["ratio_kind"] != "fixed":
        rule.append(f"ratio_kind: {s['ratio_kind']}")
    if s["ratio"] != 1 and s["ratio_kind"] != "progressive":
        rule.append(f"ratio: {s['ratio']}")
    if s["timeout"]:
        rule.append(f"timeout: {sch.format_seconds(s['timeout'])}")
    if s["quota"]:
        rule.append(f"quota: {s['quota']}")
        rule.append(f"quota_period: {_plain(str(s['quota_period']).strip() or '1h')}")
    if s["free_feed"]:
        rule.append("free_feed: 1")
    if s["reward"] == "none":
        rule.append("reward: none")
    actions = []
    if s["tone"]:
        actions.append(f"tone: {{hz: {s['tone_hz']}, ms: {s['tone_ms']}}}")
    if s["flash"]:
        actions.append(f"flash: {{target: {s['flash_target']}, color: {s['flash_color']}, "
                       f"ms: {s['flash_ms']}}}")
    if s["bnc"]:
        actions.append(f"bnc: {{width: {s['bnc_width']}, hz: {s['bnc_hz']}, "
                       f"count: {s['bnc_count']}}}")
    cues = []
    if actions:
        cues.append("reward:")
        cues += [f"  - {action}" for action in actions]
    cues += [f"{slot}: [click]" for slot in CLICK_SLOTS if s[f"click_{slot}"]]
    lines.append(f"{pad}  all:")
    lines += [f"{pad}    {line}" for line in rule]
    if cues:
        lines.append(f"{pad}    cue:")
        lines += [f"{pad}      {line}" for line in cues]
    return "\n".join(lines) + "\n"


def settings_from_phase(mapping):
    """Read a phase back into the form: ``(settings, what the form can't show)``.

    ``mapping`` is the phase as YAML read it. The second value lists what the
    form would lose; the phase can be edited in the form only when it's empty.
    """
    s = dict(RULE_DEFAULTS, ends="never", amount="", fallback="", day=0, side="any")
    missing = []
    if not isinstance(mapping, dict):
        return s, ["it isn't written as a mapping"]
    s["name"] = str(mapping.get("name", ""))
    for key, what in (("devices", "per-device rules"), ("rotate", "a rotation"),
                      ("daily", "daily windows")):
        if key in mapping:
            missing.append(what)
    _read_until(mapping.get("until"), s, missing)
    rule = mapping.get("all", {})
    if isinstance(rule, str):
        missing.append(f"the named rule '{rule}'")
    elif isinstance(rule, dict):
        _read_rule(rule, s, missing)
    else:
        missing.append("its rule")
    return s, missing


def _read_until(until, s, missing):
    if until is None:
        return
    conditions = list(until) if isinstance(until, list) else [until]
    if (len(conditions) == 2 and all(isinstance(c, dict) for c in conditions)
            and set(conditions[0]) == {"rewards", "devices"} and conditions[0]["devices"] == "all"
            and set(conditions[1]) == {"after"}):
        s.update(ends="all_rewards", amount=str(conditions[0]["rewards"]),
                 fallback=str(conditions[1]["after"]))
        return
    if len(conditions) != 1 or not isinstance(conditions[0], dict):
        missing.append("an until with several conditions")
        return
    condition = conditions[0]
    keys = set(condition)
    if keys == {"rewards", "devices"} and condition["devices"] == "all":
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
        if key in ("active", "ratio_kind", "reward") and isinstance(value, str):
            s[key] = value
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
    for slot, cue in cues.items():
        if cue == "none":
            continue
        if slot in CLICK_SLOTS and cue in ("click", ["click"]):
            s[f"click_{slot}"] = True
        elif slot == "reward" and isinstance(cue, list) and _read_reward_cue(cue, s):
            continue
        else:
            missing.append(f"the {slot} cue {_flow(cue)}")


def _read_reward_cue(actions, s):
    """The reward cue's tone, flash and BNC train; False if it has anything else."""
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
    return (f"format: fnt-fed3-protocol/1\n"
            f"name: {_plain(name)}\n"
            f"description: >\n"
            + "".join(f"  {line}\n" for line in explain.splitlines())
            + "devices: [FED]\n"
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
format: fnt-fed3-protocol/1
name: Timed feeding 19:00-21:00
description: >
  Classic FED3 Timed feeding: free feeding between two clock times each
  day, nothing outside them. Change the window's from and to.
devices: [FED]
phases:
  - name: Restricted
    all: {active: none}
    daily:
      - {from: "19:00", to: "21:00", all: {free_feed: 1}}
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
format: fnt-fed3-protocol/1
name: FR1 both pokes
devices: [FED]
phases:
  - name: FR1
    all:
      active: both
      cue:
        reward:
          - tone: {hz: 4000, ms: 200}
          - flash: {target: poke, color: blue, ms: 500}
""",
    "Step up the ratio by rewards": """\
format: fnt-fed3-protocol/1
name: FR1 to FR5, both pokes
devices: [FED]

cues:
  reward:
    - tone: {hz: 4000, ms: 200}
    - flash: {target: poke, color: blue, ms: 500}

rules:
  base: {active: both, cue: {reward: reward}}

phases:
  - name: FR1
    until: {rewards: 50}
    all: {use: base, ratio: 1}
  - name: FR3
    until: {rewards: 100}
    all: {use: base, ratio: 3}
  - name: FR5
    all: {use: base, ratio: 5}
""",
    "Four-patch competition": """\
format: fnt-fed3-protocol/1
name: Four-patch competition
description: >
  Two rich and two poor patches that each refill hourly. Which patches are
  rich is reshuffled every day at lights-off.
devices: [P1, P2, P3, P4]

cues:
  reward:
    - tone: {hz: 4000, ms: 200}
    - flash: {target: bar, color: green, ms: 1000}

rules:
  patch:
    active: both
    cue: {reward: reward, depleted: [click]}
  rich: {use: patch, ratio: 1, quota: 30, quota_period: 1h}
  poor: {use: patch, ratio: 5, quota: 10, quota_period: 1h}

phases:
  - name: Habituation
    until: {at: "19:00", day: 3}
    all: {active: none, free_feed: 1}

  - name: Training
    until: [{rewards: 100, devices: all}, {after: 4d}]
    all: patch

  - name: Competition
    until: {after: 14d}
    rotate:
      every: 1d
      at: "19:00"
      mode: shuffle
      assign: {P1: rich, P2: rich, P3: poor, P4: poor}

  - name: Recovery
    all: {active: none, free_feed: 1}

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
    passes ``settings`` read from the phase instead.
    """

    def __init__(self, parent=None, settings=None, places=None, note=""):
        super().__init__(parent)
        editing = settings is not None
        self.setWindowTitle("Edit a phase" if editing else "Add a phase")
        self.places = places or []
        s = dict(NEW_PHASE, **(settings or {}))
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
        self.active.addItems(["both", "left", "right", "none"])
        form.addRow("Active poke", self.active)
        self.kind = QComboBox()
        self.kind.addItems(["fixed", "random", "progressive"])
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
        self._load(s)

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
        every_poke = self.kind.currentText() == "fixed" and self.ratio.value() == 1
        self.clicks["counted"].setEnabled(earn and not every_poke)
        self.clicks["timeout"].setEnabled(earn and self.timeout.value() > 0)
        self.clicks["depleted"].setEnabled(earn and self.quota.value() > 0)
        self._update_hint()

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
        })

    def before(self):
        """The phase a new one goes in front of, or None for the end."""
        index = self.place.currentIndex()
        return index - 1 if index > 0 else None

    def yaml(self, indent=2):
        return phase_yaml(self.settings(), indent)


# ---------------------------------------------------------------------------
# The editor
# ---------------------------------------------------------------------------

class ProtocolEditor(QDialog):
    """Edit a protocol. ``running`` locks it to edits a running run accepts."""

    def __init__(self, text="", parent=None, path=None, running=False, t0=None):
        super().__init__(parent)
        self.setWindowTitle("Edit protocol" if not running else "Edit the running protocol")
        self.resize(1200, 760)
        self.path = path
        self.running = running
        self.t0 = t0
        self.compiled = None
        self.problems = []

        layout = QVBoxLayout(self)
        tools = QHBoxLayout()
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
        panes = QSplitter(Qt.Vertical)
        self.outline = self._pane(panes, "Timeline")
        self.sent = self._pane(panes, "What each FED3 is sent")
        panes.setSizes([360, 240])
        side_layout.addWidget(panes, 1)
        split.addWidget(side)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([700, 500])
        layout.addWidget(split)

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
    def _pane(splitter, title):
        holder = QWidget()
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 4, 0, 0)
        column.setSpacing(2)
        label = QLabel(title)
        label.setObjectName("muted")
        column.addWidget(label)
        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setFont(QFont("Consolas", 9))
        text.setLineWrapMode(QPlainTextEdit.NoWrap)
        column.addWidget(text, 1)
        splitter.addWidget(holder)
        return text

    def text(self):
        return self.editor.toPlainText()

    def _set_text(self, text):
        """Replace the text as one undoable change, and re-check at once."""
        cursor = self.editor.textCursor()
        cursor.select(QTextCursor.Document)
        cursor.insertText(text)
        self._check()

    def _use_template(self, name):
        self._set_text(TEMPLATES[name])

    def _check(self):
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
        self.ok.setEnabled(not errors)

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
        dialog = PhaseDialog(self, places=[b.name for b in blocks])
        if dialog.exec_() != QDialog.Accepted:
            return
        self._set_text(insert_phase(self.text(), dialog.yaml, dialog.before()))

    def _edit_phase(self, index):
        blocks = phase_blocks(self.text())
        if not blocks or index >= len(blocks):
            return
        block = blocks[index]
        settings, missing = settings_from_phase(block.mapping)
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
        dialog = PhaseDialog(self, settings=settings, note=note)
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
