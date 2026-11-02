"""Ethogram and scoring data model for the Behavior Scoring Studio.

Free of Qt so the rules that matter -- state pairing, exclusions, modifier
validation, export shape -- can be unit tested directly, with the UI layered on
top.

MODELLED ON BORIS
-----------------
BORIS is the reference tool for this kind of scoring, and the parts worth
copying are its ethogram:

* a behavior owns several independent MODIFIER SETS, each with its own name,
  selection type and values, and each value may carry its own key. So `sniff`
  can ask "which partner?" and "which body region?" as separate questions
  rather than sharing one flat pool of modifier strings.
* SUBJECTS are first class, each with a key. Events attribute to whichever
  subject is in focus, and a state is tracked per (subject, behavior) pair so
  two animals can hold the same state at once.
* EXCLUSIONS let a behavior automatically close others. Mutually exclusive
  states otherwise depend on the scorer remembering to stop the old one, which
  is the main source of unpaired states.
* CATEGORIES group behaviors for aggregation in analysis.

Deliberately NOT copied: coding maps, live observations, geometric
measurements, and BORIS's own project format. Exports here are plain tabular
files that any analysis environment can read.
"""

import datetime
import json
from pathlib import Path

#: Selection behaviour of a modifier set.
SINGLE = "single"        # exactly one value
MULTIPLE = "multiple"    # any number of values
NUMERIC = "numeric"      # a typed number
TEXT = "text"            # free text
MODIFIER_TYPES = (SINGLE, MULTIPLE, NUMERIC, TEXT)

POINT = "point"
STATE = "state"

#: How several selected values are joined inside one modifier set.
VALUE_SEP = ","
#: How separate sets are joined in the flattened `modifier` export column.
SET_SEP = "|"


class ModifierValue:
    """One allowed value of a modifier set, with an optional shortcut key."""

    __slots__ = ("value", "key")

    def __init__(self, value, key=""):
        self.value = value
        self.key = key or ""

    def to_dict(self):
        return {"value": self.value, "key": self.key}

    @classmethod
    def from_dict(cls, d):
        if isinstance(d, str):          # tolerate a bare list of strings
            return cls(d)
        return cls(d.get("value", ""), d.get("key", ""))

    def __repr__(self):
        return f"ModifierValue({self.value!r}, key={self.key!r})"


class ModifierSet:
    """One question asked when a behavior is scored.

    A behavior may own several of these; they are independent, so "Partner"
    and "Body region" are separate sets rather than one merged list.
    """

    def __init__(self, name="", type=SINGLE, values=None):
        self.name = name
        self.type = type if type in MODIFIER_TYPES else SINGLE
        self.values = [ModifierValue(v) if isinstance(v, str) else v
                       for v in (values or [])]

    @property
    def value_names(self):
        return [v.value for v in self.values]

    def key_for(self, value):
        for v in self.values:
            if v.value == value:
                return v.key
        return ""

    def value_for_key(self, key):
        """The value bound to `key`, for keyboard-driven modifier entry."""
        if not key:
            return None
        for v in self.values:
            if v.key and v.key.lower() == key.lower():
                return v.value
        return None

    @property
    def needs_values(self):
        """True when the set is a pick-list rather than typed input."""
        return self.type in (SINGLE, MULTIPLE)

    def normalise(self, chosen):
        """Coerce a user's selection into the stored string for this set."""
        if chosen is None:
            return ""
        if isinstance(chosen, (list, tuple, set)):
            if self.type != MULTIPLE:
                chosen = list(chosen)[:1]
            # keep the ethogram's own ordering, not selection order
            order = {v: i for i, v in enumerate(self.value_names)}
            return VALUE_SEP.join(
                sorted((str(c) for c in chosen), key=lambda c: order.get(c, 1e9)))
        return str(chosen)

    def to_dict(self):
        return {"name": self.name, "type": self.type,
                "values": [v.to_dict() for v in self.values]}

    @classmethod
    def from_dict(cls, d):
        return cls(d.get("name", ""), d.get("type", SINGLE),
                   [ModifierValue.from_dict(v) for v in d.get("values", [])])

    def __repr__(self):
        return f"ModifierSet({self.name!r}, {self.type}, {len(self.values)} values)"


class Subject:
    """An animal (or focal individual) that events are attributed to."""

    def __init__(self, name="", key="", description=""):
        self.name = name
        self.key = key or ""
        self.description = description

    def to_dict(self):
        return {"name": self.name, "key": self.key,
                "description": self.description}

    @classmethod
    def from_dict(cls, d):
        return cls(d.get("name", ""), d.get("key", ""), d.get("description", ""))

    def __repr__(self):
        return f"Subject({self.name!r}, key={self.key!r})"


class BehaviorDefinition:
    """One behavior in the ethogram."""

    DEFAULT_COLORS = [
        "#ff6b35", "#4ecdc4", "#ffe66d", "#a8e6cf", "#ff8b94",
        "#7ec8e3", "#c9b1ff", "#f7dc6f", "#82e0aa", "#f1948a",
        "#85c1e9", "#d7bde2", "#f0b27a", "#76d7c4", "#f9e79f",
    ]
    _color_idx = 0

    def __init__(self, name="", key="", event_type=POINT, color="",
                 modifier_sets=None, category="", description="",
                 exclusions=None):
        self.name = name
        self.key = key
        self.event_type = event_type
        if not color:
            color = BehaviorDefinition.DEFAULT_COLORS[
                BehaviorDefinition._color_idx % len(BehaviorDefinition.DEFAULT_COLORS)]
            BehaviorDefinition._color_idx += 1
        self.color = color
        self.modifier_sets = list(modifier_sets or [])
        self.category = category
        self.description = description
        # Names of behaviors this one stops when it starts. Only meaningful
        # between state behaviors.
        self.exclusions = list(exclusions or [])

    @property
    def has_modifiers(self):
        return bool(self.modifier_sets)

    def set_named(self, name):
        for s in self.modifier_sets:
            if s.name == name:
                return s
        return None

    def to_dict(self):
        return {
            "name": self.name,
            "key": self.key,
            "event_type": self.event_type,
            "color": self.color,
            "category": self.category,
            "description": self.description,
            "exclusions": list(self.exclusions),
            "modifier_sets": [s.to_dict() for s in self.modifier_sets],
        }

    @classmethod
    def from_dict(cls, d):
        sets = [ModifierSet.from_dict(s) for s in d.get("modifier_sets", [])]
        # Older ethograms stored a flat list of modifier strings shared by the
        # whole behavior. Promote it to a single single-select set so existing
        # configs keep working.
        if not sets and d.get("modifiers"):
            sets = [ModifierSet("Modifier", SINGLE,
                                [ModifierValue(m) for m in d["modifiers"]])]
        return cls(
            name=d.get("name", ""),
            key=d.get("key", ""),
            event_type=d.get("event_type", POINT),
            color=d.get("color", ""),
            modifier_sets=sets,
            category=d.get("category", ""),
            description=d.get("description", ""),
            exclusions=d.get("exclusions", []),
        )

    def __repr__(self):
        return f"BehaviorDefinition({self.name!r}, {self.event_type}, key={self.key!r})"


class ScoringEvent:
    """One scored event."""

    def __init__(self, frame, time_seconds, subject, behavior, modifiers,
                 event_type, status, comment=""):
        self.frame = frame
        self.time_seconds = time_seconds
        self.subject = subject
        self.behavior = behavior
        # {modifier_set_name: value}; empty when the behavior has no sets.
        self.modifiers = dict(modifiers or {})
        self.event_type = event_type
        self.status = status              # POINT | START | STOP
        self.comment = comment

    @property
    def modifier_text(self):
        """All sets flattened, for the single-column export."""
        return SET_SEP.join(str(self.modifiers[k])
                            for k in sorted(self.modifiers)
                            if str(self.modifiers[k]))

    @property
    def pair_key(self):
        """What a state is tracked by.

        Includes the subject: two animals can hold the same state at once, and
        keying on behavior alone silently loses the first one's start.
        """
        return (self.subject, self.behavior)

    def __repr__(self):
        return (f"ScoringEvent({self.status} {self.behavior!r} "
                f"subj={self.subject!r} f={self.frame})")


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h:02d}:{m:02d}:{seconds % 60:06.3f}"


class ScoringSession:
    """The ethogram, subjects and events for one video."""

    def __init__(self, video_path):
        self.video_path = video_path
        self.ethogram = []          # List[BehaviorDefinition]
        self.subjects = []          # List[Subject]
        self.events = []            # List[ScoringEvent]
        # (subject, behavior) -> the START event still awaiting a STOP
        self.active_states = {}
        self.current_subject = ""
        self.independent_variables = {}

        self.total_frames = 0
        self.fps = 30.0
        self.width = 0
        self.height = 0
        self.duration_seconds = 0.0

    # -- lookup ------------------------------------------------------------

    def behavior_named(self, name):
        for b in self.ethogram:
            if b.name == name:
                return b
        return None

    def behavior_for_key(self, key):
        for b in self.ethogram:
            if b.key and b.key.lower() == str(key).lower():
                return b
        return None

    def subject_for_key(self, key):
        for s in self.subjects:
            if s.key and s.key.lower() == str(key).lower():
                return s
        return None

    @property
    def categories(self):
        return sorted({b.category for b in self.ethogram if b.category})

    @property
    def modifier_set_names(self):
        """Every set name in the ethogram, for building export columns."""
        names = []
        for b in self.ethogram:
            for s in b.modifier_sets:
                if s.name not in names:
                    names.append(s.name)
        return names

    # -- recording ---------------------------------------------------------

    def is_state_active(self, behavior_name, subject=None):
        subject = self.current_subject if subject is None else subject
        return (subject, behavior_name) in self.active_states

    def active_behaviors(self, subject=None):
        if subject is None:
            return [b for (_, b) in self.active_states]
        return [b for (s, b) in self.active_states if s == subject]

    def add_point_event(self, frame, time_s, behavior, modifiers=None,
                        subject=None, comment=""):
        ev = ScoringEvent(frame, time_s,
                          self.current_subject if subject is None else subject,
                          behavior, modifiers, POINT, "POINT", comment)
        self.events.append(ev)
        return ev

    def start_state_event(self, frame, time_s, behavior, modifiers=None,
                          subject=None, comment=""):
        """Open a state, closing anything it excludes for the same subject.

        Exclusions are resolved here rather than in the UI so the rule holds
        however the event was created -- keyboard, button, or a replayed file.
        """
        subject = self.current_subject if subject is None else subject
        closed = self.apply_exclusions(frame, time_s, behavior, subject)
        ev = ScoringEvent(frame, time_s, subject, behavior, modifiers,
                          STATE, "START", comment)
        self.events.append(ev)
        self.active_states[(subject, behavior)] = ev
        return ev, closed

    def stop_state_event(self, frame, time_s, behavior, subject=None):
        subject = self.current_subject if subject is None else subject
        start = self.active_states.pop((subject, behavior), None)
        ev = ScoringEvent(frame, time_s, subject, behavior,
                          start.modifiers if start else {},
                          STATE, "STOP")
        self.events.append(ev)
        return ev

    def apply_exclusions(self, frame, time_s, behavior_name, subject):
        """Stop states that `behavior_name` excludes. Returns what was closed.

        Exclusion is treated as mutual: declaring A excludes B means starting
        either one ends the other, so the ethogram only has to say it once.
        """
        behavior = self.behavior_named(behavior_name)
        if behavior is None:
            return []
        closed = []
        for (subj, active), _ in list(self.active_states.items()):
            if subj != subject or active == behavior_name:
                continue
            other = self.behavior_named(active)
            mutual = (active in behavior.exclusions
                      or (other is not None and behavior_name in other.exclusions))
            if mutual:
                self.stop_state_event(frame, time_s, active, subject)
                closed.append(active)
        return closed

    def toggle_state(self, frame, time_s, behavior, modifiers=None,
                     subject=None, comment=""):
        """Start the state, or stop it if this subject already holds it."""
        subject = self.current_subject if subject is None else subject
        if self.is_state_active(behavior, subject):
            return self.stop_state_event(frame, time_s, behavior, subject), []
        return self.start_state_event(frame, time_s, behavior, modifiers,
                                      subject, comment)

    def undo_last(self):
        if not self.events:
            return None
        ev = self.events.pop()
        if ev.status == "START":
            self.active_states.pop(ev.pair_key, None)
        elif ev.status == "STOP":
            for prev in reversed(self.events):
                if prev.pair_key == ev.pair_key and prev.status == "START":
                    self.active_states[ev.pair_key] = prev
                    break
        return ev

    def remove_event(self, event):
        """Delete one event and rebuild which states are open."""
        if event in self.events:
            self.events.remove(event)
            self.rebuild_active_states()

    def rebuild_active_states(self):
        """Recompute open states from the event list, in time order."""
        open_states = {}
        for ev in sorted(self.events, key=lambda e: (e.time_seconds, e.frame)):
            if ev.status == "START":
                open_states[ev.pair_key] = ev
            elif ev.status == "STOP":
                open_states.pop(ev.pair_key, None)
        self.active_states = open_states

    # -- validation --------------------------------------------------------

    def duplicate_keys(self):
        """Keys bound to more than one thing.

        Behaviors and subjects share one keyboard, so a clash between them is
        just as broken as a clash within either list.
        """
        seen, clashes = {}, {}
        for b in self.ethogram:
            if b.key:
                seen.setdefault(b.key.lower(), []).append(f"behavior '{b.name}'")
        for s in self.subjects:
            if s.key:
                seen.setdefault(s.key.lower(), []).append(f"subject '{s.name}'")
        for key, owners in seen.items():
            if len(owners) > 1:
                clashes[key] = owners
        return clashes

    def unpaired_states(self):
        """STARTs never closed -- their duration is unknown, so analysis of
        them would silently be wrong."""
        self.rebuild_active_states()
        return list(self.active_states.values())

    def undefined_exclusions(self):
        """Exclusions naming behaviors that no longer exist."""
        names = {b.name for b in self.ethogram}
        missing = {}
        for b in self.ethogram:
            gone = [x for x in b.exclusions if x not in names]
            if gone:
                missing[b.name] = gone
        return missing

    # -- analysis ----------------------------------------------------------

    def time_budget(self):
        """Per (subject, behavior): occurrences and duration statistics.

        States contribute their START->STOP durations; point events contribute
        a count only. An unclosed state is counted as an occurrence but adds no
        duration, so a forgotten STOP cannot inflate a total.
        """
        rows = {}

        def row(subject, behavior):
            key = (subject, behavior)
            if key not in rows:
                beh = self.behavior_named(behavior)
                rows[key] = {"subject": subject, "behavior": behavior,
                             "category": beh.category if beh else "",
                             "type": beh.event_type if beh else POINT,
                             "n": 0, "total_seconds": 0.0, "durations": []}
            return rows[key]

        open_starts = {}
        for ev in sorted(self.events, key=lambda e: (e.time_seconds, e.frame)):
            if ev.status == "POINT":
                row(ev.subject, ev.behavior)["n"] += 1
            elif ev.status == "START":
                open_starts[ev.pair_key] = ev
                row(ev.subject, ev.behavior)["n"] += 1
            elif ev.status == "STOP":
                start = open_starts.pop(ev.pair_key, None)
                if start is not None:
                    d = ev.time_seconds - start.time_seconds
                    r = row(ev.subject, ev.behavior)
                    r["total_seconds"] += d
                    r["durations"].append(d)

        out = []
        for r in rows.values():
            durations = r.pop("durations")
            r["mean_seconds"] = (sum(durations) / len(durations)) if durations else 0.0
            if len(durations) > 1:
                mean = r["mean_seconds"]
                var = sum((d - mean) ** 2 for d in durations) / (len(durations) - 1)
                r["sd_seconds"] = var ** 0.5
            else:
                r["sd_seconds"] = 0.0
            out.append(r)
        return sorted(out, key=lambda r: (r["subject"], r["behavior"]))

    # -- persistence -------------------------------------------------------

    def output_folder(self):
        stem = Path(self.video_path).stem
        return Path(self.video_path).parent / f"{stem}_fntScoring"

    def event_rows(self):
        """Export rows: one column per modifier set, plus a flattened one.

        Named per-set columns are what an analysis script actually wants;
        the joined `modifier` column is kept so files written by earlier
        versions still line up.
        """
        set_names = self.modifier_set_names
        rows = []
        for ev in self.events:
            row = {
                "frame": ev.frame,
                "time": format_time(ev.time_seconds),
                "time_seconds": round(ev.time_seconds, 4),
                "subject": ev.subject,
                "behavior": ev.behavior,
                "category": (self.behavior_named(ev.behavior).category
                             if self.behavior_named(ev.behavior) else ""),
                "modifier": ev.modifier_text,
                "type": ev.event_type,
                "status": ev.status,
                "comment": ev.comment,
            }
            for name in set_names:
                row[f"modifier_{name}"] = ev.modifiers.get(name, "")
            rows.append(row)
        return rows

    def config_dict(self):
        return {
            "version": "2.0",
            "video_path": str(self.video_path),
            "current_subject": self.current_subject,
            "subjects": [s.to_dict() for s in self.subjects],
            "behaviors": [b.to_dict() for b in self.ethogram],
            "independent_variables": dict(self.independent_variables),
        }

    def load_config(self, config_path):
        with open(config_path, "r", encoding="utf-8") as fh:
            config = json.load(fh)
        self.ethogram = [BehaviorDefinition.from_dict(b)
                         for b in config.get("behaviors", [])]
        self.subjects = [Subject.from_dict(s) for s in config.get("subjects", [])]
        self.independent_variables = dict(config.get("independent_variables", {}))
        # v1 stored one free-text subject rather than a subject list.
        legacy = config.get("subject", "")
        if legacy and not self.subjects:
            self.subjects = [Subject(legacy)]
        self.current_subject = config.get("current_subject", legacy or "")
        if not self.current_subject and self.subjects:
            self.current_subject = self.subjects[0].name

    def load_events(self, rows):
        """Rebuild events from exported rows (a CSV read back in)."""
        set_names = self.modifier_set_names
        self.events = []
        for row in rows:
            mods = {}
            for name in set_names:
                val = row.get(f"modifier_{name}", "")
                if val and str(val) != "nan":
                    mods[name] = str(val)
            if not mods and row.get("modifier"):
                # v1 file: one unnamed modifier column
                mods = {"Modifier": str(row["modifier"])}
            self.events.append(ScoringEvent(
                frame=int(row.get("frame", 0) or 0),
                time_seconds=float(row.get("time_seconds", 0.0) or 0.0),
                subject=str(row.get("subject", "") or ""),
                behavior=str(row.get("behavior", "") or ""),
                modifiers=mods,
                event_type=str(row.get("type", POINT) or POINT),
                status=str(row.get("status", "POINT") or "POINT"),
                comment=str(row.get("comment", "") or ""),
            ))
        self.rebuild_active_states()
