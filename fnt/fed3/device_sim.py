"""A FED3 running firmware v3, simulated: the firmware's executable spec.

Everything the firmware decides is decided here the same way: what a poke
earns, when a pellet is dispensed, when a phase ends, what a reboot restores
from the card, what each serial command replies. The firmware is written to
match this file, and FNT's tests drive it in place of hardware.

Time is simulated. ``advance`` jumps from one interesting moment to the next
(a pellet landing, a phase coming due, a free-feed refill) instead of ticking,
so weeks of schedule run in milliseconds.

    device = SimDevice(device_id=3, now=datetime(2026, 10, 20, 9, tzinfo=tz))
    device.command("PING")            # -> ["PONG,id=3,fw=3.1,proto=3"]
    device.poke("L")
    device.advance(2)
    device.retrieve()
    device.take()                     # every line the device has sent

See ``docs/fed3/design.md`` for the behaviour this implements.
"""
from __future__ import annotations

import math
import random
import re
import threading
from collections import deque
from datetime import datetime, timedelta, timezone

from . import schedule as sch

FIRMWARE = "3.1"
PROTO = 3
HEADER = ("time,ms,seq,phase,event,side,outcome,left,right,pellets,"
          "phase_rewards,phase_left,phase_right,retrieval_ms,poke_ms,detail")
COLUMNS = HEADER.split(",")

DISPENSE_SECONDS = 1.0          # motor turn until the pellet reaches the well
JAM_RETRY_SECONDS = 600
STATUS_EVERY_SECONDS = 3600
MAX_HOLD_MS = 60000             # a beam broken longer than this is "held"
STAGING_TIMEOUT_SECONDS = 10
TIME_JUMP_LIMIT_SECONDS = 600
CATCH_UP_SECONDS = 60           # a phase change this late is logged as catch-up
MAX_INPUT = 128
RUN_ID = re.compile(r"[A-Za-z0-9]{1,8}")
IDLE_FILE = "IDLE.CSV"
BATTERY, TEMPERATURE, HUMIDITY = "4.10", "22.0", "40"


def iso(moment):
    return moment.isoformat(timespec="seconds")


ISO = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(Z|[+-]\d{2}:\d{2})")


def parse_iso(text):
    """``YYYY-MM-DDTHH:MM:SS`` with ``Z`` or ``±HH:MM``, as the firmware accepts."""
    if not isinstance(text, str) or not ISO.fullmatch(text):
        return None
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def progressive_requirement(rewards):
    """Pokes needed for the next reward after ``rewards`` this phase."""
    return round(5 * math.exp(0.2 * (rewards + 1)) - 5)


def default_schedule():
    return sch.Schedule("default", [sch.Phase("Default")])


class Card:
    """The SD card: named text files that survive a reboot."""

    def __init__(self):
        self.files = {}
        self.failing = False        # tests set this to simulate a dead card

    def read(self, name):
        return None if self.failing else self.files.get(name)

    def write(self, name, text):
        """Write a whole file (temporary name, then rename: all or nothing)."""
        if self.failing:
            return False
        self.files[name] = text
        return True

    def append(self, name, line):
        if self.failing:
            return False
        self.files[name] = self.files.get(name, "") + line + "\n"
        return True

    def remove(self, name):
        self.files.pop(name, None)


def _kv(text):
    return dict(line.split("=", 1) for line in (text or "").splitlines() if "=" in line)


class SimDevice:
    def __init__(self, device_id=1, now=None, card=None, seed=0, clock_set=True):
        self.card = card or Card()
        self.rng = random.Random(seed)
        self.output = []
        self.cues = []                      # (world time, slot, cue text)
        self.world = now or datetime(2026, 10, 20, 9, tzinfo=timezone.utc)
        self.rtc_offset = timedelta(0)      # RTC minus true time
        self.clock_set = clock_set
        self.jams_pending = 0               # tests: next N dispenses jam
        if self.card.read("DEVICE.TXT") is None:
            zone = self.world.utcoffset()
            self._save_device(device_id, zone)
        self.boot("power")

    # ------------------------------------------------------------------
    # The world: what tests do to the device
    # ------------------------------------------------------------------

    def take(self):
        """Every line sent since the last call."""
        out, self.output = self.output, []
        return out

    def rows(self):
        return [line[4:] for line in self.output if line.startswith("ROW,")]

    def advance(self, seconds):
        """Let simulated time pass, handling everything that falls due."""
        target = self.world + timedelta(seconds=seconds)
        while True:
            upcoming = [t for t in self._wakeups() if self.world < t <= target]
            if not upcoming:
                break
            self.world = min(upcoming)
            self._tick()
        self.world = target
        self._tick()

    def poke(self, side, ms=100):
        """An animal breaks one poke's beam for ``ms`` milliseconds."""
        side = side.upper()[0]
        rule = self._rule()
        hold = {"side": side, "start": self.world, "millis": self.millis(),
                "decided": None}
        self.holding = hold
        min_ms = int(rule["min_poke_ms"])
        if min_ms == 0:
            hold["decided"] = self._decide(side)
        elif ms < min_ms:
            self.advance(ms / 1000)
            hold["decided"] = self._decide(side, short=True)
        else:
            self.advance(min_ms / 1000)
            hold["decided"] = self._decide(side)
        held = hold["millis"] + MAX_HOLD_MS
        if ms >= MAX_HOLD_MS:
            # A beam broken this long is logged at the 60 s mark, not on release,
            # so a blocked poke can't hold its row back indefinitely.
            self.advance((held - self.millis()) / 1000)
            self._poke_row(hold, MAX_HOLD_MS, "held")
        remaining = hold["start"] + timedelta(milliseconds=ms) - self.world
        self.advance(max(remaining.total_seconds(), 0))
        self.holding = None
        if ms < MAX_HOLD_MS:
            self._poke_row(hold, ms, "")

    def _poke_row(self, hold, ms, detail):
        decided = hold["decided"]
        self._row("poke", side=hold["side"], outcome=decided["outcome"],
                  phase=decided["phase"], poke_ms=ms, detail=detail, at=hold)

    def retrieve(self):
        """The animal takes the pellet from the well."""
        if self.well != "full":
            return False
        self.well = "empty"
        self.last_retrieve = self.world
        self._row("retrieve", retrieval_ms=self.millis() - self.pellet_millis)
        return True

    def power_off(self, seconds, lose_clock=False):
        """Unplugged for a while: time passes with nothing running, then boot."""
        self.world += timedelta(seconds=seconds)
        self.reboot("power", lose_clock)

    def rule(self):
        """The rule in force now, as canonical key -> value."""
        return self._rule()

    def reboot(self, cause="power", lose_clock=False):
        if lose_clock:
            self.clock_set = False
        self.boot(cause)

    def drift(self, seconds):
        """The RTC gains (or loses) time against the world."""
        self.rtc_offset += timedelta(seconds=seconds)

    # ------------------------------------------------------------------
    # Clocks
    # ------------------------------------------------------------------

    def rtc(self):
        return (self.world + self.rtc_offset).astimezone(self.zone)

    def millis(self):
        return int((self.world - self.boot_world).total_seconds() * 1000)

    def _to_world(self, rtc_moment):
        return rtc_moment - self.rtc_offset

    # ------------------------------------------------------------------
    # Boot and state on the card
    # ------------------------------------------------------------------

    def _save_device(self, device_id, zone):
        minutes = int(zone.total_seconds()) // 60
        sign = "-" if minutes < 0 else "+"
        offset = f"{sign}{abs(minutes) // 60:02d}:{abs(minutes) % 60:02d}"
        self.card.write("DEVICE.TXT", f"id={device_id}\noffset={offset}\n")

    def boot(self, cause):
        self.lights = False          # LIGHTS:on never survives a boot
        self.boot_world = self.world
        self.holding = None
        self.staging = None
        self.staging_failed = None
        self.staging_at = None
        # A pellet in the well stays there through a reboot; one on its way
        # stops with the motor.
        self.well = "full" if getattr(self, "well", None) == "full" else "empty"
        self.dispense_due = None
        self.dispense_reason = None
        self.pellet_millis = 0
        self.last_retrieve = None
        self.jammed = False
        self.jam_retry_at = None
        self.timeout_until = None
        self.counters = {"L": 0, "R": 0, "shared": 0}
        self.next_status = self.world + timedelta(seconds=STATUS_EVERY_SECONDS)

        device = _kv(self.card.read("DEVICE.TXT"))
        self.device_id = int(device.get("id", 1))
        offset = device.get("offset", "+00:00")
        self.zone = datetime.fromisoformat(f"2000-01-01T00:00:00{offset}").tzinfo

        self.schedule, self.sched_crc = default_schedule(), None
        text = self.card.read("SCHED.TXT")
        if text:
            try:
                self.schedule = sch.parse_schedule(text)
                self.sched_crc = self.schedule.crc()
            except sch.ScheduleError:
                pass
        self.next_schedule, self.next_crc = None, None
        text = self.card.read("NEXT.TXT")
        if text:
            try:
                self.next_schedule = sch.parse_schedule(text)
                self.next_crc = self.next_schedule.crc()
            except sch.ScheduleError:
                pass

        run = _kv(self.card.read("RUN.TXT"))
        self.run_id = run.get("run", "-")
        self.t0 = parse_iso(run.get("t0"))
        self.file = run.get("file", IDLE_FILE)
        self.ended = run.get("ended", "1") == "1"
        self.phase = min(int(run.get("phase", 0)), len(self.schedule.phases) - 1)
        self.phase_start = parse_iso(run.get("phase_start")) or self.t0

        self._restore_counts()
        self.output.append(
            f"BOOT,fw={FIRMWARE},reset={cause},clock={'ok' if self.clock_set else 'unset'},"
            f"sched={self._crc_text(self.sched_crc)},run={self.run_id if not self.ended else '-'}")
        self._row("boot", detail=f"reset={cause};ratio counters reset")
        self._tick()

    def _restore_counts(self):
        self.seq = 0
        self.left = self.right = self.pellets = 0
        self.phase_rewards = self.phase_left = self.phase_right = 0
        self.quota_used, self.quota_period = 0, None
        lines = (self.card.read(self.file) or "").splitlines()
        if not lines:
            self.card.append(self.file, HEADER)
            return
        rows = [dict(zip(COLUMNS, line.split(",", len(COLUMNS) - 1)))
                for line in lines[1:]]
        if not rows:
            return
        last = rows[-1]
        self.seq = int(last["seq"])
        self.left, self.right = int(last["left"]), int(last["right"])
        self.pellets = int(last["pellets"])
        if int(last["phase"]) == self.phase:
            self.phase_rewards = int(last["phase_rewards"])
            self.phase_left = int(last["phase_left"])
            self.phase_right = int(last["phase_right"])
        self.quota_used = self.phase_rewards
        period = int(self.schedule.phases[self.phase].rule["quota_period"])
        if period and self.clock_set and self.phase_start:
            # Rewards in the current quota period: scan back to its start,
            # which bounds the scan by the period, not the length of the run.
            index = int((self.rtc() - self.phase_start).total_seconds() // period)
            period_start = self.phase_start + timedelta(seconds=index * period)
            used = 0
            for row in reversed(rows):
                moment = parse_iso(row["time"])
                if moment is None:
                    continue
                if moment < period_start or int(row["phase"]) != self.phase:
                    break
                if row["event"] == "poke" and row["outcome"] == "reward":
                    used += 1
            self.quota_used, self.quota_period = used, index

    def _save_run(self):
        self.card.write("RUN.TXT", (
            f"run={self.run_id}\nt0={iso(self.t0) if self.t0 else ''}\n"
            f"file={self.file}\nended={int(self.ended)}\nphase={self.phase}\n"
            f"phase_start={iso(self.phase_start) if self.phase_start else ''}\n"))

    @staticmethod
    def _crc_text(crc):
        return "default" if crc is None else f"{crc:08x}"

    # ------------------------------------------------------------------
    # Rows
    # ------------------------------------------------------------------

    def _row(self, event, side="", outcome="", phase=None, retrieval_ms="",
             poke_ms="", detail="", at=None):
        self.seq += 1
        moment = (at["start"] if at else self.world) + self.rtc_offset
        millis = at["millis"] if at else self.millis()
        values = [
            iso(moment.astimezone(self.zone)) if self.clock_set else "",
            millis, self.seq, self.phase if phase is None else phase, event, side,
            outcome, self.left, self.right, self.pellets, self.phase_rewards,
            self.phase_left, self.phase_right, retrieval_ms, poke_ms, detail,
        ]
        line = ",".join(str(v) for v in values)
        if self.card.append(self.file, line):
            self.output.append(f"ROW,{line}")
        else:
            self.output.append("ERR,SD,append failed")

    # ------------------------------------------------------------------
    # Rules and pokes
    # ------------------------------------------------------------------

    def _rule(self):
        phase = self.schedule.phases[self.phase]
        return phase.rule_at(self.rtc()) if self.clock_set else dict(phase.rule)

    def _active_run(self):
        return self.run_id != "-" and not self.ended

    def _depleted(self, rule):
        quota = int(rule["quota"])
        if not quota:
            return False
        period = int(rule["quota_period"])
        if period and self.clock_set and self.phase_start:
            index = int((self.rtc() - self.phase_start).total_seconds() // period)
            if index != self.quota_period:
                self.quota_period, self.quota_used = index, 0
        return self.quota_used >= quota

    def _decide(self, side, short=False):
        """What a poke earns, decided the moment it qualifies."""
        rule = self._rule()
        if side == "L":
            self.left += 1
            self.phase_left += 1
        else:
            self.right += 1
            self.phase_right += 1
        active = rule["active"]
        decided = {"phase": self.phase}
        if short:
            outcome = "short"
        elif self.well != "empty":
            outcome = "pellet_in_well"
        elif active == "none" or (active != "both" and side != active[0].upper()):
            outcome = "inactive"
        elif self.timeout_until and self.world < self.timeout_until:
            outcome = "timeout"
        elif self._depleted(rule):
            outcome = "depleted"
        else:
            outcome = self._count(side, rule)
        decided["outcome"] = outcome
        if outcome == "reward":
            self.phase_rewards += 1
            self.quota_used += 1
            timeout = int(rule["timeout"])
            self.timeout_until = (self.world + timedelta(seconds=timeout)
                                  if timeout else None)
            self._cue("reward", rule)
            if rule["reward"] == "pellet":
                self._dispense("reward")
        elif outcome in ("counted", "inactive", "timeout", "depleted"):
            self._cue(outcome, rule)
        self._check_counts()
        return decided

    def _count(self, side, rule):
        if rule["ratio_kind"] == "random":
            return "reward" if self.rng.random() < 1 / int(rule["ratio"]) else "counted"
        key = side if rule["active"] == "both" and rule["count"] == "per_port" else "shared"
        self.counters[key] += 1
        need = (progressive_requirement(self.phase_rewards)
                if rule["ratio_kind"] == "progressive" else int(rule["ratio"]))
        if self.counters[key] >= need:
            self.counters[key] = 0
            return "reward"
        return "counted"

    def _cue(self, slot, rule):
        text = rule[f"cue.{slot}"]
        if text != "none":
            self.cues.append((self.world, slot, text))

    # ------------------------------------------------------------------
    # Dispensing
    # ------------------------------------------------------------------

    def _dispense(self, reason):
        """Start a dispense. Returns started, pellet_in_well or jam."""
        if self.well != "empty":
            return "pellet_in_well"
        if self.jammed:
            return "jam"
        self.well = "dispensing"
        self.dispense_due = self.world + timedelta(seconds=DISPENSE_SECONDS)
        self.dispense_reason = reason
        return "started"

    def _finish_dispense(self):
        self.dispense_due = None
        if self.jams_pending:
            self.jams_pending -= 1
            self.well = "empty"
            # An earned or free pellet is owed and retried; one fed by hand
            # is a bench check, reported and not retried.
            self.jammed = self.dispense_reason != "manual"
            self.jam_retry_at = (self.world + timedelta(seconds=JAM_RETRY_SECONDS)
                                 if self.jammed else None)
            self._row("jam", detail=self.dispense_reason)
            return
        self.jammed = False
        self.jam_retry_at = None
        self.well = "full"
        self.pellets += 1
        self.pellet_millis = self.millis()
        detail = "" if self.dispense_reason == "reward" else self.dispense_reason
        self._row("dispense", detail=detail)

    # ------------------------------------------------------------------
    # Phases
    # ------------------------------------------------------------------

    def _enter_phase(self, index, start, reason=""):
        self.phase = index
        self.phase_start = start
        self.phase_rewards = self.phase_left = self.phase_right = 0
        self.quota_used, self.quota_period = 0, None
        self.counters = {"L": 0, "R": 0, "shared": 0}
        self.timeout_until = None
        self._save_run()
        # The row is stamped when it is written, so timestamps never run
        # backwards; a catch-up row also records when its phase really began.
        detail = self.schedule.phases[index].name
        if reason:
            detail += f";{reason}"
        if reason == "catch-up":
            detail += f";start={iso(start)}"
        self._row("phase", detail=detail)

    def _next_due(self):
        """The earliest due time of the current phase's time conditions."""
        phase = self.schedule.phases[self.phase]
        dues = [sch.due_time(u, self.phase_start, self.t0.astimezone(self.zone))
                for u in phase.untils if u.is_time]
        return min(dues) if dues else None

    def _can_advance(self):
        return (self._active_run() and self.clock_set
                and self.phase + 1 < len(self.schedule.phases))

    def _check_time(self):
        """Move through every phase whose time condition is already due."""
        while self._can_advance():
            due = self._next_due()
            if due is None or due > self.rtc():
                return
            late = (self.rtc() - due).total_seconds() > CATCH_UP_SECONDS
            self._enter_phase(self.phase + 1, due, "catch-up" if late else "")

    def _check_counts(self):
        if not (self._active_run() and self.phase + 1 < len(self.schedule.phases)):
            return
        for until in self.schedule.phases[self.phase].untils:
            if until.kind == "rewards":
                met = self.phase_rewards >= until.count
            elif until.kind == "pokes":
                count = {"left": self.phase_left, "right": self.phase_right,
                         "any": self.phase_left + self.phase_right}[until.side]
                met = count >= until.count
            else:
                continue
            if met:
                self._enter_phase(self.phase + 1, self.rtc())
                return

    # ------------------------------------------------------------------
    # The main loop
    # ------------------------------------------------------------------

    def _wakeups(self):
        times = [self.next_status]
        if self.dispense_due:
            times.append(self.dispense_due)
        if self.jammed and self.jam_retry_at:
            times.append(self.jam_retry_at)
        rule = self._rule()
        if rule["free_feed"] == "1" and self.well == "empty" and self.last_retrieve:
            times.append(self.last_retrieve + timedelta(seconds=int(rule["refill_delay"])))
        if self._can_advance():
            due = self._next_due()
            if due:
                times.append(self._to_world(due))
        if self.clock_set:
            # A window opening or closing can start free feeding.
            now = self.rtc()
            for window in self.schedule.phases[self.phase].windows:
                for minute in (window.start, window.end):
                    edge = now.replace(hour=minute // 60, minute=minute % 60,
                                       second=0, microsecond=0)
                    if edge <= now:
                        edge += timedelta(days=1)
                    times.append(self._to_world(edge))
        return times

    def _tick(self):
        if self.dispense_due and self.world >= self.dispense_due:
            self._finish_dispense()
        self._check_time()
        rule = self._rule()
        if self.jammed:
            if self.world >= self.jam_retry_at:
                self.jammed = False
                self._dispense("retry")
        elif rule["free_feed"] == "1" and self.well == "empty":
            delay = timedelta(seconds=int(rule["refill_delay"]))
            if self.last_retrieve is None or self.world >= self.last_retrieve + delay:
                self._dispense("free")
        if self.world >= self.next_status:
            self.next_status += timedelta(seconds=STATUS_EVERY_SECONDS)
            self._row("status", detail=f"batt={BATTERY};temp={TEMPERATURE};hum={HUMIDITY}")

    # ------------------------------------------------------------------
    # Serial commands
    # ------------------------------------------------------------------

    def command(self, line):
        """Handle one command line; returns (and sends) the reply lines."""
        before = len(self.output)
        self._command(line.rstrip("\r\n"))
        return self.output[before:]

    def _reply(self, text):
        self.output.append(text)

    def _command(self, line):
        if len(line) > MAX_INPUT:
            return self._reply("ERR,TOO_LONG,")
        head, _, arg = line.partition(":")
        if line == "PING":
            return self._reply(f"PONG,id={self.device_id},fw={FIRMWARE},proto={PROTO}")
        if line == "STATUS":
            return self._reply(self._status())
        if head in ("TIME", "TIME!"):
            return self._set_time(arg, force=head == "TIME!")
        if head == "ID":
            if not re.fullmatch(r"\d{1,3}", arg) or not 1 <= int(arg) <= 999:
                return self._reply("ERR,BAD_ID,1-999")
            self.device_id = int(arg)
            self._save_device(self.device_id, self.rtc().utcoffset())
            return self._reply(f"ID,id={self.device_id}")
        if line == "SCHED:BEGIN":
            self.staging, self.staging_failed = sch.ScheduleBuilder(), None
            self.staging_at = self.world
            return self._reply("OK")
        if head == "S":
            return self._stage(arg)
        if head == "SCHED" and arg.startswith("COMMIT,"):
            return self._commit(arg[len("COMMIT,"):])
        if head == "RUN" and arg.startswith("BEGIN,"):
            return self._run_begin(arg[len("BEGIN,"):])
        if line == "RUN:END":
            return self._run_end()
        if head == "GOTO":
            return self._goto(arg)
        if head == "READ":
            return self._read(arg)
        if line == "FEED":
            result = self._dispense("manual")
            return self._reply(f"FEED,result={result}")
        if head == "LIGHTS":
            if arg not in ("on", "off"):
                return self._reply("ERR,BAD_LIGHTS,on|off")
            self.lights = arg == "on"
            return self._reply(f"LIGHTS,on={int(self.lights)}")
        if head == "TEST":
            try:
                cue = sch.parse_cue(arg)
            except ValueError as exc:
                return self._reply(f"ERR,BAD_CUE,{exc}")
            self.cues.append((self.world, "test", cue))
            return self._reply("OK")
        if line == "REBOOT":
            return self.boot("software")
        return self._reply(f"ERR,UNKNOWN,{line[:40]}")

    def _status(self):
        rule = self._rule()
        due = self._next_due() if self._can_advance() else None
        nxt = "-"
        phase = self.schedule.phases[self.phase]
        if due:
            nxt = f"at={iso(due)}"
        else:
            for until in phase.untils:
                if until.kind == "rewards":
                    nxt = f"rewards={self.phase_rewards}/{until.count}"
                    break
                if until.kind == "pokes":
                    count = {"left": self.phase_left, "right": self.phase_right,
                             "any": self.phase_left + self.phase_right}[until.side]
                    nxt = f"pokes={count}/{until.count}"
                    break
                if until.kind == "goto":
                    nxt = "goto"
        fields = {
            "id": self.device_id, "fw": FIRMWARE,
            "time": iso(self.rtc()) if self.clock_set else "",
            "clock": "ok" if self.clock_set else "unset",
            "sd": "err" if self.card.failing else "ok",
            "run": self.run_id, "ended": int(self.ended),
            "sched": self._crc_text(self.sched_crc),
            "next_sched": "-" if self.next_crc is None else f"{self.next_crc:08x}",
            "phase": self.phase, "phase_name": phase.name,
            "phase_start": iso(self.phase_start) if self.phase_start else "",
            "next": nxt, "file": self.file, "seq": self.seq,
            "l": self.left, "r": self.right, "pellets": self.pellets,
            "phase_rewards": self.phase_rewards, "phase_left": self.phase_left,
            "phase_right": self.phase_right,
            "depleted": int(self._depleted(rule)), "well": int(self.well == "full"),
            "jam": int(self.jammed),
            "uptime": int((self.world - self.boot_world).total_seconds()),
            "batt": BATTERY, "temp": TEMPERATURE, "hum": HUMIDITY,
            "lights": int(self.lights),
        }
        return "STATUS," + ",".join(f"{k}={v}" for k, v in fields.items())

    def _set_time(self, arg, force):
        target = parse_iso(arg)
        if target is None:
            return self._reply("ERR,BAD_TIME,")
        before = iso(self.rtc()) if self.clock_set else "unset"
        jump = (target - self.rtc()).total_seconds()
        if (self.clock_set and self._active_run() and not force
                and abs(jump) > TIME_JUMP_LIMIT_SECONDS):
            return self._reply(f"ERR,TIME_JUMP,{int(jump)}")
        self.rtc_offset = target - self.world
        self.zone = target.tzinfo
        self.clock_set = True
        self._save_device(self.device_id, target.utcoffset())
        self._row("clock", detail=f"from={before};to={iso(target)}")
        self._reply(f"TIME,before={before},after={iso(self.rtc())}")
        self._tick()

    def _stage(self, line):
        if self.staging is None:
            return self._reply("ERR,SCHED,0,no SCHED:BEGIN")
        if self.staging_failed:
            return self._reply(f"ERR,SCHED,{len(self.staging.lines) + 1},"
                               f"an earlier line failed")
        self.staging_at = self.world
        try:
            self.staging.add(line)
        except ValueError as exc:
            self.staging_failed = str(exc)
            return self._reply(f"ERR,SCHED,{len(self.staging.lines) + 1},{exc}")
        return self._reply("OK")

    def _commit(self, arg):
        parts = arg.split(",")
        staging = self.staging
        expired = (self.staging_at is None or
                   (self.world - self.staging_at).total_seconds() > STAGING_TIMEOUT_SECONDS)
        self.staging = None
        if staging is None or expired:
            return self._reply("ERR,SCHED_REJECTED,no upload in progress")
        if self.staging_failed:
            return self._reply(f"ERR,SCHED_REJECTED,{self.staging_failed}")
        try:
            schedule = staging.finish()
        except ValueError as exc:
            return self._reply(f"ERR,SCHED_REJECTED,{exc}")
        crc = staging.crc()
        if parts[0].lower() != f"{crc:08x}":
            return self._reply(f"ERR,SCHED_REJECTED,crc is {crc:08x}")
        text = "".join(line + "\n" for line in staging.lines)
        if self._active_run():
            expected = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
            if expected is not None and expected != self.phase:
                return self._reply(f"ERR,SCHED_STALE,phase={self.phase}")
            if self.phase >= len(schedule.phases):
                return self._reply(f"ERR,SCHED_REJECTED,no phase {self.phase}")
            self.card.write("SCHED.TXT", text)
            self.schedule, self.sched_crc = schedule, crc
            self._row("schedule", detail=f"crc={crc:08x}")
            self._reply(f"SCHED,crc={crc:08x},phases={len(schedule.phases)},slot=now")
            self._tick()
            self._check_counts()
            return None
        self.card.write("NEXT.TXT", text)
        self.next_schedule, self.next_crc = schedule, crc
        return self._reply(f"SCHED,crc={crc:08x},phases={len(schedule.phases)},slot=next")

    def _run_begin(self, arg):
        parts = arg.split(",")
        if len(parts) != 3 or not RUN_ID.fullmatch(parts[0]) or parse_iso(parts[1]) is None:
            return self._reply("ERR,BAD_RUN,RUN:BEGIN,<id>,<t0>,<crc>")
        run_id, t0, crc = parts[0], parse_iso(parts[1]), parts[2].lower()
        if run_id == self.run_id and not self.ended:
            # Already running this run: a repeated RUN:BEGIN changes nothing.
            return self._reply(f"RUN,run={run_id},file={self.file},phase={self.phase}")
        have = self.next_crc if self.next_crc is not None else self.sched_crc
        if self._crc_text(have) != crc:
            return self._reply(f"ERR,RUN_SCHED,have={self._crc_text(have)}")
        if self._active_run():
            self._row("run_end", detail=f"replaced by {run_id}")
        if self.next_schedule is not None:
            self.card.write("SCHED.TXT", self.card.read("NEXT.TXT"))
            self.card.remove("NEXT.TXT")
            self.schedule, self.sched_crc = self.next_schedule, self.next_crc
            self.next_schedule, self.next_crc = None, None
        self.run_id, self.t0, self.ended = run_id, t0.astimezone(self.zone), False
        self.lights = False          # a run starts with the lights as its cues leave them
        self.file = f"{run_id}.CSV"
        self.card.write(self.file, HEADER + "\n")
        self.seq = self.left = self.right = self.pellets = 0
        self.phase, self.phase_start = 0, self.t0
        self.phase_rewards = self.phase_left = self.phase_right = 0
        self.quota_used, self.quota_period = 0, None
        self.counters = {"L": 0, "R": 0, "shared": 0}
        self.timeout_until = None
        self._save_run()
        self._row("run_start", detail=f"t0={iso(self.t0)};{self.schedule.phases[0].name}")
        self._reply(f"RUN,run={run_id},file={self.file},phase=0")
        self._tick()

    def _run_end(self):
        if self.run_id == "-":
            return self._reply("ERR,NO_RUN,")
        if not self.ended:
            self.ended = True
            self._save_run()
            self._row("run_end")
        return self._reply(f"RUN,run={self.run_id},ended=1")

    def _goto(self, arg):
        parts = arg.split(",")
        if not self._active_run():
            return self._reply("ERR,NO_RUN,")
        if len(parts) != 2 or not all(p.isdigit() for p in parts):
            return self._reply("ERR,BAD_GOTO,GOTO:<from>,<to>")
        source, target = int(parts[0]), int(parts[1])
        if source != self.phase:
            return self._reply(f"ERR,GOTO_STALE,phase={self.phase}")
        if target >= len(self.schedule.phases):
            return self._reply(f"ERR,BAD_PHASE,{target}")
        self._enter_phase(target, self.rtc(), "goto")
        return self._reply(f"PHASE,n={target},name={self.schedule.phases[target].name}")

    def _read(self, arg):
        run_id, _, start = arg.partition(",")
        name = IDLE_FILE if run_id == "-" else f"{run_id}.CSV"
        text = self.card.read(name)
        if text is None or not start.isdigit():
            return self._reply(f"ERR,NO_FILE,{run_id}")
        last = 0
        for line in text.splitlines()[1:]:
            seq = int(line.split(",", 3)[2])
            if seq >= int(start):
                self._reply(f"ROW,{line}")
            last = seq
        return self._reply(f"READ_END,run={run_id},last={last}")


class SimTransport:
    """A SimDevice behind the interface of link.SerialTransport.

    For the run engine's tests and for FNT's demo mode. Commands are answered
    at once; ``act`` changes the device (a poke, time passing) and makes what
    it sent available to read.
    """

    def __init__(self, device):
        self.device = device
        self.pending = deque()
        self.lock = threading.Lock()
        self.closed = False

    def _pull(self):
        self.pending.extend(self.device.take())

    def write(self, line):
        from .link import LinkError
        if self.closed:
            raise LinkError("port closed")
        with self.lock:
            self.device.command(line)
            self._pull()

    def readline(self, timeout):
        from .link import LinkError
        if self.closed:
            raise LinkError("port closed")
        with self.lock:
            self._pull()
            return self.pending.popleft() if self.pending else None

    def act(self, fn):
        with self.lock:
            result = fn(self.device)
            self._pull()
            return result

    def close(self):
        self.closed = True
