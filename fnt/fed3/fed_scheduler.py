"""Protocol event scheduler: the model behind the FED3 scheduling table.

Separated from the widget so scheduled events can be serialized into the session
state and restored after a crash — a 3 a.m. mode switch that exists only in a Qt
table is lost the moment FNT dies.

Three correctness problems in the previous inline implementation are addressed:

* **Deleting removed the wrong event.** Delete buttons captured a row index, but
  the list was re-sorted by fire time on every refresh, so the captured index
  pointed at a different event by the time it was clicked. Events now carry a
  stable id and are addressed by it.
* **Alarms in the past fired immediately.** "Today at 09:00" entered at 14:00
  resolved to a time already gone and executed on the next tick.
  :func:`resolve_alarm` rolls such a time forward instead.
* **A missed window fired on restart.** After a crash, every event whose time had
  passed while FNT was down would execute at once on resume. Events overdue by
  more than :data:`MISSED_GRACE_SECONDS` are marked *missed* and not run.
"""

import uuid
from datetime import datetime, time as time_of_day, timedelta

from . import fed_protocol

# How late an event may fire and still be considered on time. Covers a brief
# restart; beyond this the experimental moment has passed and running the action
# is more likely to corrupt a protocol than to rescue it.
MISSED_GRACE_SECONDS = 120

STATUS_PENDING = "Pending"
STATUS_DONE = "Executed"
STATUS_FAILED = "Failed"
STATUS_MISSED = "Missed"
STATUS_DISABLED = "Disabled"

ACTION_SET_MODE = "Set Mode"
ACTION_DISPENSE = "Dispense Pellet"
ACTION_LIGHTS = "Toggle Lights"
ACTION_NEW_TRIAL = "Start New Trial"

ACTIONS = [ACTION_SET_MODE, ACTION_DISPENSE, ACTION_LIGHTS, ACTION_NEW_TRIAL]

REPEAT_NONE = "Once"
REPEAT_DAILY = "Daily"
REPEAT_HOURLY = "Hourly"
REPEATS = [REPEAT_NONE, REPEAT_HOURLY, REPEAT_DAILY]

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"]
DAY_CHOICES = ["Today", "Tomorrow"] + WEEKDAYS

ALL_DEVICES = "All Devices"


class ScheduledEvent:
    """One scheduled protocol action."""

    def __init__(self, target, action, params=None, target_time=None,
                 kind="alarm", repeat=REPEAT_NONE, offset_seconds=0,
                 event_id=None, enabled=True, status=STATUS_PENDING,
                 last_result=""):
        self.id = event_id or uuid.uuid4().hex[:12]
        self.target = target
        self.action = action
        self.params = params or {}
        self.target_time = target_time
        self.kind = kind                    # "alarm" (wall clock) or "timer" (offset)
        self.repeat = repeat
        self.offset_seconds = offset_seconds
        self.enabled = enabled
        self.status = status
        self.last_result = last_result

    # --- display ---------------------------------------------------------

    def describe_action(self):
        if self.action == ACTION_SET_MODE:
            mode = self.params.get("mode", "?")
            # Only name the parameters the mode actually uses, so an FR entry
            # does not claim a timeout it will never apply.
            fields = fed_protocol.mode_fields(mode)
            detail = []
            if "ratio" in fields:
                detail.append(f"ratio {self.params.get('ratio', 1)}")
            if "timeout" in fields:
                detail.append(f"timeout {self.params.get('timeout', 0)}s")
            return mode + (f" ({', '.join(detail)})" if detail else "")
        if self.action == ACTION_LIGHTS:
            return "Lights ON" if self.params.get("lights") else "Lights OFF"
        return self.action

    def describe_trigger(self, now=None):
        """Trigger column text, including a live countdown while pending."""
        if not self.enabled:
            return "Disabled"
        if self.status != STATUS_PENDING or self.target_time is None:
            return self.target_time.strftime("%a %d %b %H:%M:%S") if self.target_time else "—"

        remaining = int((self.target_time - (now or datetime.now())).total_seconds())
        if remaining < 0:
            remaining = 0
        days, rest = divmod(remaining, 86400)
        hours, rest = divmod(rest, 3600)
        minutes, seconds = divmod(rest, 60)
        countdown = (f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}" if days
                     else f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        return f"in {countdown}"

    def describe_when(self):
        if self.target_time is None:
            return "—"
        stamp = self.target_time.strftime("%a %d %b %H:%M:%S")
        return f"{stamp} · {self.repeat.lower()}" if self.repeat != REPEAT_NONE else stamp

    # --- persistence -----------------------------------------------------

    def to_dict(self):
        return {
            "id": self.id,
            "target": self.target,
            "action": self.action,
            "params": self.params,
            "target_time": self.target_time.isoformat() if self.target_time else None,
            "kind": self.kind,
            "repeat": self.repeat,
            "offset_seconds": self.offset_seconds,
            "enabled": self.enabled,
            "status": self.status,
            "last_result": self.last_result,
        }

    @classmethod
    def from_dict(cls, data):
        target_time = data.get("target_time")
        return cls(
            target=data.get("target", ALL_DEVICES),
            action=data.get("action", ACTION_DISPENSE),
            params=data.get("params") or {},
            target_time=datetime.fromisoformat(target_time) if target_time else None,
            kind=data.get("kind", "alarm"),
            repeat=data.get("repeat", REPEAT_NONE),
            offset_seconds=data.get("offset_seconds", 0),
            event_id=data.get("id"),
            enabled=data.get("enabled", True),
            status=data.get("status", STATUS_PENDING),
            last_result=data.get("last_result", ""),
        )


class Scheduler:
    """Holds scheduled events and decides which are due."""

    def __init__(self):
        self.events = []

    # --- collection ------------------------------------------------------

    def add(self, event):
        self.events.append(event)
        self.sort()
        return event

    def remove(self, event_id):
        """Remove by stable id; returns the removed event or None."""
        for index, event in enumerate(self.events):
            if event.id == event_id:
                return self.events.pop(index)
        return None

    def get(self, event_id):
        return next((e for e in self.events if e.id == event_id), None)

    def clear_finished(self):
        """Drop everything that will never fire again."""
        removed = [e for e in self.events
                   if e.status in (STATUS_DONE, STATUS_FAILED, STATUS_MISSED)
                   and e.repeat == REPEAT_NONE]
        self.events = [e for e in self.events if e not in removed]
        return removed

    def sort(self):
        self.events.sort(
            key=lambda e: (e.target_time is None, e.target_time or datetime.max))

    def rename_target(self, old_name, new_name):
        """Keep events attached to a device the user renamed."""
        for event in self.events:
            if event.target == old_name:
                event.target = new_name

    # --- firing ----------------------------------------------------------

    def due(self, now=None):
        """Events to execute right now.

        Anything overdue by more than the grace period is marked *missed* and
        excluded, so a restart does not replay a backlog of protocol changes.
        """
        now = now or datetime.now()
        ready = []
        for event in self.events:
            if not event.enabled or event.status != STATUS_PENDING:
                continue
            if event.target_time is None or event.target_time > now:
                continue
            if (now - event.target_time).total_seconds() > MISSED_GRACE_SECONDS:
                event.status = STATUS_MISSED
                event.last_result = (
                    f"skipped: due {event.target_time:%d %b %H:%M:%S}, "
                    f"FNT was not running")
                continue
            ready.append(event)
        return ready

    def complete(self, event, ok, detail=""):
        """Record an outcome and re-arm the event if it repeats."""
        event.last_result = detail
        if event.repeat == REPEAT_NONE:
            event.status = STATUS_DONE if ok else STATUS_FAILED
            return
        step = timedelta(days=1) if event.repeat == REPEAT_DAILY else timedelta(hours=1)
        now = datetime.now()
        next_time = (event.target_time or now) + step
        while next_time <= now:
            next_time += step
        event.target_time = next_time
        event.status = STATUS_PENDING
        self.sort()

    # --- persistence -----------------------------------------------------

    def to_list(self):
        return [e.to_dict() for e in self.events]

    def load(self, data):
        self.events = [ScheduledEvent.from_dict(d) for d in (data or [])]
        self.sort()


def resolve_alarm(day_choice, when, now=None):
    """Resolve a day choice plus a time of day into the next matching datetime.

    A time already past today rolls forward rather than firing immediately.
    """
    now = now or datetime.now()
    target = datetime.combine(now.date(), when)

    if day_choice == "Today":
        if target <= now:
            target += timedelta(days=1)     # "today at 09:00" entered at 14:00
    elif day_choice == "Tomorrow":
        target += timedelta(days=1)
    elif day_choice in WEEKDAYS:
        ahead = WEEKDAYS.index(day_choice) - now.weekday()
        if ahead < 0 or (ahead == 0 and target <= now):
            ahead += 7
        target += timedelta(days=ahead)
    return target


def resolve_timer(days, when, now=None):
    """Resolve a relative delay into an absolute time. Returns None if zero."""
    seconds = days * 86400 + when.hour * 3600 + when.minute * 60 + when.second
    if seconds <= 0:
        return None, 0
    return (now or datetime.now()) + timedelta(seconds=seconds), seconds


def time_from_qtime(qtime):
    return time_of_day(qtime.hour(), qtime.minute(), qtime.second())
