"""The FED3 run engine: devices, runs and their records, with no UI.

The devices run the experiment themselves (``docs/fed3/design.md``). This is
everything FNT does around them:
- it finds them and checks their firmware;
- it keeps their clocks set;
- it uploads schedules and starts and stops runs;
- it keeps a byte-identical copy of each device's data file, filling any gap
  by asking for the rows again;
- it moves phases that depend on several devices;
- it notices problems and keeps them until someone acknowledges them;
- it picks an interrupted run back up.

Everything happens in :meth:`Engine.step`, on one thread. The UI sends commands
with :meth:`Engine.submit` and reads :meth:`Engine.snapshot`. The tests drive
``step`` directly, with simulated devices and a simulated clock.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import queue
import random
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

from . import protocol as proto
from . import schedule as sch
from .device_sim import COLUMNS, HEADER, iso, parse_iso
from .link import DeviceLink, LinkError, SerialTransport, candidate_ports, error, fields

SCAN_EVERY = 5.0
STATUS_EVERY = 30.0
OFFLINE_AFTER = 120.0
CLOCK_SYNC_EVERY = 6 * 3600.0
DRIFT_TOLERANCE = 2.0           # seconds of clock error corrected silently
DRIFT_PROBLEM = 60.0            # seconds of clock error worth flagging
TIME_JUMP_LIMIT = 600.0         # the firmware refuses bigger jumps mid-run
READ_RETRY = 20.0
RETRY_OPEN = 15.0
RUN_FILE = "run.json"
RUN_FORMAT = "fnt-fed3-run/1"

SETUP, RUNNING, STOPPED = "setup", "running", "stopped"


def utc_now():
    return datetime.now(timezone.utc)


@dataclass
class Problem:
    key: str
    device: str
    kind: str
    message: str
    since: datetime
    active: bool = True
    acknowledged: bool = False


@dataclass
class Device:
    """A FED3 on a port."""

    port: str
    description: str = ""
    link: DeviceLink | None = None
    connected: bool = False
    id: int | None = None
    firmware: str = ""
    compatible: bool = False
    status: dict = field(default_factory=dict)
    status_at: float = -1e9
    clock_synced_at: float = -1e9
    last_open_attempt: float = -1e9
    run: str = ""                   # the run its live rows belong to
    rows: deque = field(default_factory=lambda: deque(maxlen=400))
    last_row: dict = field(default_factory=dict)
    last_pellet: datetime | None = None
    drift: float | None = None


class DataCopy:
    """A byte-identical copy of one device's data file for this run.

    Rows are written strictly in ``seq`` order. One that arrives early waits
    for the gap before it to be filled by a ``READ``.

    The copy doesn't end at ``run_end``: after Stop the device keeps running
    its last phase and writing rows to the same file until it starts another
    run, and the copy follows it.
    """

    def __init__(self, path):
        self.path = path
        self.next_seq = 1
        self.pending = {}
        self.read_requested = -1e9
        if os.path.exists(path):
            with open(path, encoding="ascii", newline="") as handle:
                lines = handle.read().splitlines()
            if len(lines) > 1:
                self.next_seq = int(lines[-1].split(",")[2]) + 1
        else:
            with open(path, "w", encoding="ascii", newline="") as handle:
                handle.write(HEADER + "\n")

    @property
    def gap(self):
        return bool(self.pending)

    def add(self, row_text):
        """Take one row (without ``ROW,``). Returns the rows written."""
        seq = int(row_text.split(",", 3)[2])
        if seq < self.next_seq or seq in self.pending:
            return []
        self.pending[seq] = row_text
        written = []
        with open(self.path, "a", encoding="ascii", newline="") as handle:
            while self.next_seq in self.pending:
                text = self.pending.pop(self.next_seq)
                handle.write(text + "\n")
                written.append(text)
                self.next_seq += 1
        return written


def parse_row(text):
    return dict(zip(COLUMNS, text.split(",", len(COLUMNS) - 1)))


def run_id_now(moment):
    """Eight base-36 characters from the time: a valid, sortable run id."""
    value = int(moment.timestamp())
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    out = ""
    while value:
        value, rem = divmod(value, 36)
        out = digits[rem] + out
    return ("R" + out)[-8:]


def slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()[:40] or "run"


class Run:
    """One experiment: its protocol, its devices and its folder."""

    def __init__(self, folder, protocol_text, compiled, binding, run_id, t0, seed):
        self.folder = folder
        self.protocol_text = protocol_text
        self.compiled = compiled
        self.binding = dict(binding)          # logical name -> device id
        self.run_id = run_id
        self.t0 = t0                          # aware, standard time
        self.seed = seed
        self.state = RUNNING
        self.edits = []
        self.fnt_changes = []
        self.copies = {}
        self.started = {}                     # logical -> device confirmed RUN
        self.stopped_at = None                # when Stop was pressed (UTC iso)
        # logical -> the device's own run_end row: where its data after Stop
        # begins. Later than stopped_at for a device that was unplugged then.
        self.ended = {}
        for logical, device_id in self.binding.items():
            self.copies[logical] = DataCopy(os.path.join(folder, f"{logical}_FED{device_id:03d}.csv"))

    @property
    def crcs(self):
        return {name: s.crc() for name, s in self.compiled.schedules.items()}

    def logical_for(self, device_id):
        for logical, bound in self.binding.items():
            if bound == device_id:
                return logical
        return None

    def save(self):
        data = {
            "format": RUN_FORMAT,
            "run_id": self.run_id,
            "name": self.compiled.protocol.name,
            "state": self.state,
            "t0": iso(self.t0),
            "utc_offset": iso(self.t0)[-6:],
            "seed": self.seed,
            "binding": self.binding,
            "protocol_sha256": hashlib.sha256(self.protocol_text.encode()).hexdigest(),
            "protocol_versions": ["protocol.yaml"] + [e["file"] for e in self.edits],
            "edits": self.edits,
            "fnt_phase_changes": self.fnt_changes,
            "schedules": {name: f"{crc:08x}" for name, crc in self.crcs.items()},
        }
        if self.stopped_at:
            data["stopped_at"] = self.stopped_at
        if self.ended:
            data["ended"] = self.ended
        path = os.path.join(self.folder, RUN_FILE)
        with open(path + ".tmp", "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        os.replace(path + ".tmp", path)

    def write_schedules(self):
        os.makedirs(os.path.join(self.folder, "sched"), exist_ok=True)
        for name, schedule in self.compiled.schedules.items():
            with open(os.path.join(self.folder, "sched", f"{name}.txt"), "w",
                      encoding="ascii", newline="") as handle:
                handle.write(schedule.text())


class Engine:
    def __init__(self, runs_root, ports=candidate_ports, open_port=SerialTransport,
                 now=utc_now, monotonic=time.monotonic, zone=None):
        self.runs_root = runs_root
        self.ports = ports
        self.open_port = open_port
        self.now = now
        self.monotonic = monotonic
        self.zone = zone or sch.standard_timezone()
        self.devices = {}                 # port -> Device
        self.problems = {}                # key -> Problem
        self.run = None
        self.state = SETUP
        self.stopped = {}                 # run id -> a stopped Run its devices still write to
        self._searched = set()            # run ids already looked for in runs_root
        self.events = deque(maxlen=300)   # (utc time, device, text) for the UI
        self.event_count = 0              # events ever added, so the UI adds only new ones
        self.preflight = {}               # logical -> "ok" or why not
        self.commands = queue.Queue()
        self.lock = threading.Lock()
        self.last_scan = -1e9
        self._snapshot = {}
        self._log_rows = []

    # ------------------------------------------------------------------
    # Commands from the UI (thread-safe)
    # ------------------------------------------------------------------

    def submit(self, name, **kwargs):
        """Queue a command; it runs on the engine thread in the next step."""
        self.commands.put((name, kwargs))

    def _run_commands(self):
        while True:
            try:
                name, kwargs = self.commands.get_nowait()
            except queue.Empty:
                return
            try:
                getattr(self, f"cmd_{name}")(**kwargs)
            except Exception as exc:          # a bad command must not stop the engine
                self._event("", f"{name} failed: {exc}")

    # ------------------------------------------------------------------
    # The loop
    # ------------------------------------------------------------------

    def step(self):
        self._run_commands()
        tick = self.monotonic()
        if tick - self.last_scan >= SCAN_EVERY:
            self.last_scan = tick
            self._scan()
        for device in list(self.devices.values()):
            if device.link and device.connected:
                try:
                    device.link.poll()
                    self._maintain(device)
                except LinkError as exc:
                    self._disconnect(device, str(exc))
        if self.run and self.state == RUNNING:
            self._run_step()
        for run in self._stopped_runs():
            self._stopped_step(run)
        self._check_problems()
        self._flush_log()
        self._publish()

    def _event(self, device, text):
        self.events.append((self.now(), device, text))
        self.event_count += 1

    def _log(self, device, action, detail="", result="ok", quiet=False):
        """Record an action in log.csv; ``quiet`` keeps routine ones off the screen."""
        if not quiet:
            self._event(device, f"{action}{': ' + detail if detail else ''}"
                                f"{'' if result == 'ok' else ' (' + result + ')'}")
        if self.run:
            self._log_rows.append([iso(self.now()), device, action, detail, result])

    def _flush_log(self):
        if not (self.run and self._log_rows):
            return
        path = os.path.join(self.run.folder, "log.csv")
        new = not os.path.exists(path)
        with open(path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            if new:
                writer.writerow(["time_utc", "device", "action", "detail", "result"])
            writer.writerows(self._log_rows)
        self._log_rows = []

    # ------------------------------------------------------------------
    # Devices
    # ------------------------------------------------------------------

    def _scan(self):
        try:
            available = self.ports()
        except Exception:
            available = {}
        for port, description in available.items():
            device = self.devices.get(port)
            if device is None:
                device = self.devices[port] = Device(port, description)
            if device.connected:
                continue
            if self.monotonic() - device.last_open_attempt < RETRY_OPEN:
                continue
            device.last_open_attempt = self.monotonic()
            self._open(device)
        for port, device in self.devices.items():
            if port not in available and device.connected:
                self._disconnect(device, "unplugged")

    def _open(self, device):
        try:
            transport = self.open_port(device.port)
        except Exception as exc:
            device.firmware = ""
            self._event(device.port, f"can't open: {exc}")
            return
        link = DeviceLink(transport, lambda line, d=device: self._on_line(d, line),
                          clock=self.monotonic)
        device.link = link
        try:
            reply = link.call("PING")
        except LinkError:
            reply = ""
        if reply.startswith("PONG_FED3"):
            device.compatible = False
            device.firmware = "older than 3.0"
            self._problem(device.port, "firmware", device.port,
                          "firmware older than v3: reflash this FED3")
            link.close()
            device.link = None
            return
        info = fields(reply)
        if not reply.startswith("PONG,") or info.get("proto") != "3":
            link.close()
            device.link = None
            return
        device.connected = True
        device.compatible = True
        device.id = int(info["id"])
        device.firmware = info.get("fw", "")
        device.status_at = -1e9
        self._clear(device.port, "firmware")
        self._log(self._name(device), "connected", f"{device.port}, FED{device.id:03d}")
        self._duplicate_ids()

    def _disconnect(self, device, why):
        if device.link:
            device.link.close()
        device.link = None
        if device.connected:
            self._log(self._name(device), "disconnected", why)
        device.connected = False

    def _duplicate_ids(self):
        seen = {}
        for device in self.devices.values():
            if device.connected and device.id is not None:
                seen.setdefault(device.id, []).append(device.port)
        for device_id, ports in seen.items():
            key = f"FED{device_id:03d}"
            if len(ports) > 1:
                self._problem(key, "duplicate_id", key,
                              f"two FED3s are numbered {device_id} ({', '.join(ports)}); "
                              "renumber one in Setup")
            else:
                self._clear(key, "duplicate_id")

    def _name(self, device):
        """A device's name in the UI and log: its logical name in a run."""
        if self.run and device.id is not None:
            logical = self.run.logical_for(device.id)
            if logical:
                return logical
        return f"FED{device.id:03d}" if device.id is not None else device.port

    def device_by_id(self, device_id):
        for device in self.devices.values():
            if device.connected and device.id == device_id:
                return device
        return None

    def _call(self, device, command, timeout=None):
        return device.link.call(command, timeout)

    def _maintain(self, device):
        tick = self.monotonic()
        if tick - device.status_at >= STATUS_EVERY:
            self._read_status(device)
        if tick - device.clock_synced_at >= CLOCK_SYNC_EVERY:
            self._sync_clock(device)

    def _read_status(self, device):
        reply = self._call(device, "STATUS")
        if not reply.startswith("STATUS,"):
            return
        device.status = fields(reply)
        device.status_at = self.monotonic()
        status = device.status
        if status.get("clock") == "ok" and status.get("time"):
            moment = parse_iso(status["time"])
            device.drift = (moment - self.now()).total_seconds() if moment else None
        else:
            device.drift = None
        if status.get("run") not in (None, "-"):
            device.run = status["run"]
            self._find_stopped(device.run)
        name = self._name(device)
        if status.get("clock") == "unset" or (
                device.drift is not None and abs(device.drift) > DRIFT_TOLERANCE):
            self._sync_clock(device)
        if status.get("sd") == "err":
            self._problem(name, "sd", name, "SD card not working")
        else:
            self._clear(name, "sd")
        if status.get("jam") == "1":
            self._problem(name, "jam", name, "jammed: retrying every 10 minutes")
        else:
            self._clear(name, "jam")

    def _sync_clock(self, device, force=False):
        target = self.now().astimezone(self.zone).replace(microsecond=0)
        command = f"{'TIME!' if force else 'TIME'}:{iso(target)}"
        reply = self._call(device, command)
        name = self._name(device)
        device.clock_synced_at = self.monotonic()
        problem = error(reply)
        if problem and problem[0] == "TIME_JUMP":
            self._problem(name, "drift", name,
                          f"clock is {int(float(problem[1]))} s out; confirm to correct it")
            return False
        if reply.startswith("TIME,"):
            info = fields(reply)
            before, after = parse_iso(info.get("before", "")), parse_iso(info.get("after", ""))
            routine = bool(before and after and abs((after - before).total_seconds()) < DRIFT_PROBLEM)
            self._log(name, "clock set", f"{info.get('before')} -> {info.get('after')}",
                      quiet=routine)
            device.drift = 0.0
            self._clear(name, "drift")
            self._clear(name, "clock_unset")
            return True
        return False

    # ------------------------------------------------------------------
    # Lines from devices
    # ------------------------------------------------------------------

    def _on_line(self, device, line, reading_run=None):
        if line.startswith("BOOT,"):
            info = fields(line)
            name = self._name(device)
            device.run = info.get("run", "-")
            device.status_at = -1e9          # read STATUS on the next step
            if info.get("reset") == "watchdog":
                self._problem(name, "watchdog", name,
                              "restarted by its watchdog (a firmware hang)", active=False)
            self._log(name, "boot", f"reset={info.get('reset')}")
            return
        if not line.startswith("ROW,"):
            return
        text = line[4:]
        row = parse_row(text)
        name = self._name(device)
        target = reading_run or device.run
        if reading_run:
            # A replayed row: it belongs in the copy, but it isn't news. Only
            # live rows say what the device is doing now.
            self._copy_row(device, target, text)
            return
        device.rows.append(row)
        device.last_row = row
        if row["event"] == "run_start":
            ours = self.run and row["detail"].startswith(f"t0={iso(self.run.t0)};")
            device.run = self.run.run_id if ours else "?"
        if row["event"] == "dispense":
            device.last_pellet = parse_iso(row["time"]) or self.now()
        elif row["event"] == "jam":
            self._problem(name, "jam", name, "jammed: retrying every 10 minutes")
        elif row["event"] == "phase":
            self._event(name, f"phase {row['phase']}: {row['detail']}")
        self._copy_row(device, device.run, text)

    def _copy_row(self, device, run_id, text):
        run = self._run_by_id(run_id)
        if run and device.id is not None:
            logical = run.logical_for(device.id)
            if logical:
                for written in run.copies[logical].add(text):
                    self._note_end(run, logical, parse_row(written))

    def _note_end(self, run, logical, row):
        """Record in run.json where a device's run ended in its data, so rows
        recorded after Stop can be told apart: every row after it in seq."""
        if row["event"] != "run_end" or logical in run.ended:
            return
        run.ended[logical] = {"time": row["time"] or None, "seq": int(row["seq"]),
                              "detail": row["detail"]}
        run.save()

    # ------------------------------------------------------------------
    # Problems
    # ------------------------------------------------------------------

    def _problem(self, device, kind, subject, message, active=True):
        key = f"{device}:{kind}"
        existing = self.problems.get(key)
        if existing and existing.active and not existing.acknowledged:
            existing.message = message
            return
        if existing and existing.active and existing.acknowledged:
            existing.message = message
            return
        self.problems[key] = Problem(key, device, kind, message, self.now(), active)
        self._log(device, "problem", message, "flagged")

    def _clear(self, device, kind):
        problem = self.problems.get(f"{device}:{kind}")
        if problem and problem.active:
            problem.active = False
            if problem.acknowledged:
                del self.problems[problem.key]

    def cmd_acknowledge(self, key):
        problem = self.problems.get(key)
        if not problem:
            return
        if problem.active:
            problem.acknowledged = True
        else:
            del self.problems[key]
        self._log(problem.device, "acknowledged", problem.message)

    def _check_problems(self):
        if not self.run or self.state != RUNNING:
            return
        tick = self.monotonic()
        checks = self.run.compiled.protocol.checks
        for logical, device_id in self.run.binding.items():
            device = self.device_by_id(device_id)
            if device is None or tick - device.link.last_heard > OFFLINE_AFTER:
                self._problem(logical, "offline", logical, "not connected")
                continue
            self._clear(logical, "offline")
            status = device.status
            if status.get("clock") == "unset":
                self._problem(logical, "clock_unset", logical, "clock lost; setting it")
            if device.drift is not None and abs(device.drift) > DRIFT_PROBLEM:
                self._problem(logical, "drift", logical, f"clock is {device.drift:+.0f} s out")
            expected = f"{self.run.crcs[logical]:08x}"
            if status.get("run") == self.run.run_id and status.get("sched") not in (None, expected):
                self._problem(logical, "schedule", logical,
                              f"holds schedule {status.get('sched')}, expected {expected}")
            elif status.get("run") == self.run.run_id:
                self._clear(logical, "schedule")
            limit = checks.get("no_pellets")
            if limit and self._can_dispense(logical, status):
                since = device.last_pellet or self.run.t0
                quiet = (self.now() - since).total_seconds()
                if quiet > limit:
                    self._problem(logical, "no_pellets", logical,
                                  f"no pellet for {sch.format_seconds(int(quiet) // 60 * 60)}")
                else:
                    self._clear(logical, "no_pellets")

    def _can_dispense(self, logical, status):
        try:
            phase = self.run.compiled.schedules[logical].phases[int(status["phase"])]
        except (KeyError, ValueError, IndexError):
            return False
        rule = phase.rule_at(self.now().astimezone(self.zone))
        return rule["free_feed"] == "1" or (rule["reward"] == "pellet" and rule["active"] != "none")

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    def cmd_start(self, protocol_text, binding, seed=None, name_hint=""):
        """Pre-flight every bound device, then start them all at one t0."""
        if self.state == RUNNING:
            raise RuntimeError("a run is already in progress")
        protocol = proto.load(protocol_text)
        missing = set(protocol.devices) - set(binding)
        if missing:
            raise RuntimeError(f"no FED3 chosen for {', '.join(sorted(missing))}")
        seed = seed if seed is not None else random.randrange(1, 10**9)
        compiled = proto.compile_protocol(protocol, seed=seed)
        self.preflight = {}
        for logical in protocol.devices:
            self.preflight[logical] = self._preflight(logical, binding[logical], compiled)
        if any(result != "ok" for result in self.preflight.values()):
            self._event("", "pre-flight failed; nothing started")
            return
        t0 = self.now().astimezone(self.zone).replace(microsecond=0)
        folder = os.path.join(self.runs_root, f"{t0:%Y-%m-%d_%H%M}_{slug(protocol.name)}")
        suffix = 1
        while os.path.exists(folder):
            suffix += 1
            folder = f"{folder.rstrip('0123456789').rstrip('_')}_{suffix}"
        os.makedirs(folder)
        with open(os.path.join(folder, "protocol.yaml"), "w", encoding="utf-8") as handle:
            handle.write(protocol_text)
        self.run = Run(folder, protocol_text, compiled, binding, run_id_now(t0), t0, seed)
        self.run.write_schedules()
        self.run.save()
        self.state = RUNNING
        self._log("", "run started", f"{protocol.name} as {self.run.run_id}, t0 {iso(t0)}")
        for logical, device_id in binding.items():
            self._begin(logical, self.device_by_id(device_id))

    def _preflight(self, logical, device_id, compiled):
        device = self.device_by_id(device_id)
        if device is None:
            return f"FED{device_id:03d} is not connected"
        try:
            if not self._sync_clock(device):
                return "couldn't set its clock"
            self._read_status(device)
            if device.status.get("sd") != "ok":
                return "SD card not working"
            reply = self._upload(device, compiled.schedules[logical])
        except LinkError as exc:
            return f"stopped answering ({exc})"
        if not reply.startswith("SCHED,") or "slot=next" not in reply:
            return f"schedule upload failed: {reply}"
        self._log(logical, "pre-flight ok", f"FED{device_id:03d}")
        return "ok"

    def _upload(self, device, schedule, phase=None):
        self._call(device, "SCHED:BEGIN")
        for line in schedule.lines():
            reply = self._call(device, f"S:{line}")
            if reply != "OK":
                return reply
        suffix = f",{phase}" if phase is not None else ""
        return self._call(device, f"SCHED:COMMIT,{schedule.crc():08x}{suffix}")

    def _begin(self, logical, device):
        """Put one device into the run. Safe to repeat."""
        if device is None:
            return False
        crc = f"{self.run.crcs[logical]:08x}"
        command = f"RUN:BEGIN,{self.run.run_id},{iso(self.run.t0)},{crc}"
        reply = self._call(device, command)
        problem = error(reply)
        if problem and problem[0] == "RUN_SCHED":
            # Not holding our schedule (a card swapped, a device replaced):
            # upload it, then start.
            self._upload(device, self.run.compiled.schedules[logical])
            reply = self._call(device, command)
        if reply.startswith("RUN,"):
            device.run = self.run.run_id
            self.run.started[logical] = True
            self._read_status(device)
            self._fill_gaps(self.run, logical, device, force=True)
            return True
        self._problem(logical, "not_started", logical, f"couldn't start: {reply}")
        return False

    def _run_step(self):
        for logical, device_id in self.run.binding.items():
            device = self.device_by_id(device_id)
            if device is None:
                continue
            try:
                if device.status.get("run") == self.run.run_id and device.status.get("ended") == "1":
                    # RUN:END reached it but the run wasn't marked stopped (FNT
                    # was closed mid-Stop). A RUN:BEGIN now would start the
                    # run's file on the card over, so leave it be.
                    self._problem(logical, "ended", logical,
                                  "its run was already ended on the device; press Stop run")
                    self._fill_gaps(self.run, logical, device)
                    continue
                if device.status.get("run") != self.run.run_id:
                    if device.status:      # a known other state: join the run
                        if self._begin(logical, device):
                            self._clear(logical, "not_started")
                    continue
                self._clear(logical, "not_started")
                expected = f"{self.run.crcs[logical]:08x}"
                if device.status.get("sched") not in (None, expected):
                    self._reupload(logical, device)
                self._fill_gaps(self.run, logical, device)
            except LinkError as exc:
                self._disconnect(device, str(exc))
        self._fnt_conditions()

    def _fill_gaps(self, run, logical, device, force=False):
        copy = run.copies[logical]
        try:
            seq = int(device.status.get("seq", 0))
        except ValueError:
            seq = 0
        behind = copy.gap or (device.run == run.run_id and seq >= copy.next_seq
                              and device.status.get("run") == run.run_id)
        if not (force or behind):
            return
        if self.monotonic() - copy.read_requested < READ_RETRY and not force:
            return
        copy.read_requested = self.monotonic()
        device.link.on_line = lambda line, d=device: self._on_line(d, line, run.run_id)
        try:
            self._call(device, f"READ:{run.run_id},{copy.next_seq}", timeout=30)
        finally:
            device.link.on_line = lambda line, d=device: self._on_line(d, line)

    def _reupload(self, logical, device):
        phase = int(device.status.get("phase", 0))
        reply = self._upload(device, self.run.compiled.schedules[logical], phase=phase)
        self._log(logical, "schedule re-sent", reply)
        self._read_status(device)

    def _phase_of(self, logical):
        device = self.device_by_id(self.run.binding[logical])
        row = device.last_row if device else {}
        try:
            return (int(row["phase"]), int(row["phase_rewards"]),
                    int(row["phase_left"]), int(row["phase_right"]))
        except (KeyError, ValueError):
            return None

    def _fnt_conditions(self):
        """Phases that end on several devices' counts: FNT moves them."""
        compiled = self.run.compiled
        states = {logical: self._phase_of(logical) for logical in self.run.binding}
        for index, phase in enumerate(compiled.phases):
            counted = [c for c in phase.fnt if c.kind != "manual"]
            if not counted:
                continue
            in_phase = [l for l, s in states.items() if s and s[0] == index]
            if not in_phase:
                continue
            for condition in counted:
                if self._condition_met(condition, states, index):
                    self._advance(index, f"{condition.describe()} reached")
                    return

    @staticmethod
    def _condition_met(condition, states, index):
        def value(state):
            _, rewards, left, right = state
            if condition.kind == "rewards":
                return rewards
            return {"left": left, "right": right, "any": left + right}[condition.side]

        if condition.devices == "any":
            return any(s and s[0] == index and value(s) >= condition.count
                       for s in states.values())
        names = states if condition.devices == "all" else condition.devices
        for name in names:
            state = states.get(name)
            if state is None:
                return False
            if state[0] == index and value(state) < condition.count:
                return False
            if state[0] < index:
                return False
        return True

    def _advance(self, index, why):
        moved = []
        for logical, device_id in self.run.binding.items():
            device = self.device_by_id(device_id)
            state = self._phase_of(logical)
            if device is None or not state or state[0] != index:
                continue
            reply = self._call(device, f"GOTO:{index},{index + 1}")
            if reply.startswith("PHASE,"):
                moved.append(logical)
            self._read_status(device)
        self.run.fnt_changes.append({"time": iso(self.now()), "from": index,
                                     "to": index + 1, "why": why, "devices": moved})
        self.run.save()
        self._log("", "phase advanced", f"{index} -> {index + 1} on {', '.join(moved)}: {why}")

    def cmd_advance(self):
        """Advance phase: the current phase must end on a manual condition."""
        if not self.run:
            return
        phases = {s[0] for s in (self._phase_of(l) for l in self.run.binding) if s}
        for index in sorted(phases):
            phase = self.run.compiled.phases[index]
            if any(c.kind == "manual" for c in phase.fnt):
                self._advance(index, "Advance phase pressed")
                return
        self._event("", "the current phase doesn't end by hand")

    def cmd_stop(self):
        if not self.run or self.state != RUNNING:
            return
        for logical, device_id in self.run.binding.items():
            device = self.device_by_id(device_id)
            if device is None:
                self._log(logical, "not stopped", "not connected", "failed")
                continue
            try:
                self._call(device, "RUN:END")
                self._read_status(device)
                self._fill_gaps(self.run, logical, device, force=True)
            except LinkError as exc:
                self._log(logical, "not stopped", str(exc), "failed")
        self.run.state = STOPPED
        self.state = STOPPED
        self.run.stopped_at = iso(self.now())
        self.run.save()
        self._clear_kind("ended")
        self._log("", "run stopped", "the devices keep their last phase, and their rows "
                                     "are still copied until they start another run")
        self._flush_log()

    def cmd_close(self):
        """Back to Setup after a stopped run. Its rows are still copied."""
        if self.state == STOPPED:
            self.stopped[self.run.run_id] = self.run
            self.run = None
            self.state = SETUP
            self.problems = {k: p for k, p in self.problems.items() if p.kind == "firmware"}

    def cmd_edit(self, protocol_text):
        """Apply an edited protocol to the running devices."""
        if not self.run or self.state != RUNNING:
            raise RuntimeError("no run in progress")
        protocol = proto.load(protocol_text)
        if protocol.devices != self.run.compiled.protocol.devices:
            raise RuntimeError("an edit can't change the list of devices")
        compiled = proto.compile_protocol(protocol, seed=self.run.seed)
        current = {l: (self._phase_of(l) or (0,))[0] for l in self.run.binding}
        for logical, phase in current.items():
            old = self.run.compiled.schedules[logical].phases
            new = compiled.schedules[logical].phases
            if len(new) <= phase or new[phase].name != old[phase].name:
                raise RuntimeError(f"{logical} is in phase {phase} ({old[phase].name}); "
                                   "an edit must keep it in place")
            for index in range(phase):
                if new[index] != old[index]:
                    raise RuntimeError(f"phase {old[index].name} is over and can't change")
        version = len(self.run.edits) + 2
        name = f"protocol.v{version}.yaml"
        with open(os.path.join(self.run.folder, name), "w", encoding="utf-8") as handle:
            handle.write(protocol_text)
        self.run.compiled = compiled
        self.run.protocol_text = protocol_text
        self.run.edits.append({"time": iso(self.now()), "file": name})
        self.run.write_schedules()
        self.run.save()
        for logical, device_id in self.run.binding.items():
            device = self.device_by_id(device_id)
            if device is None:
                continue        # reconciled when it reconnects
            phase = current[logical]
            reply = self._upload(device, compiled.schedules[logical], phase=phase)
            if error(reply) and error(reply)[0] == "SCHED_STALE":
                self._read_status(device)
                reply = self._upload(device, compiled.schedules[logical],
                                     phase=int(device.status.get("phase", phase)))
            self._log(logical, "edit applied", reply)
            self._read_status(device)

    # --- manual actions --------------------------------------------------

    def _device_for(self, who):
        if self.run and who in self.run.binding:
            return self.device_by_id(self.run.binding[who])
        if isinstance(who, int):
            return self.device_by_id(who)
        return self.devices.get(who)

    def cmd_feed(self, device):
        target = self._device_for(device)
        if target:
            self._log(self._name(target), "feed by hand", self._call(target, "FEED"))

    def cmd_lights(self, device, on):
        """A FED3's lights on or off, steady, to find it on the bench. The
        device turns them off itself at RUN:BEGIN and on boot (firmware 3.1)."""
        target = self._device_for(device)
        if not target:
            return
        reply = self._call(target, f"LIGHTS:{'on' if on else 'off'}")
        if reply.startswith("LIGHTS,"):
            target.status["lights"] = fields(reply).get("on", "0")
            self._log(self._name(target), "lights", "on" if on else "off", quiet=True)
        else:
            self._log(self._name(target), "lights", f"need firmware 3.1 ({reply})", "failed")

    def cmd_test_cue(self, device, cue):
        target = self._device_for(device)
        if target:
            self._log(self._name(target), "test cue", f"{cue}: {self._call(target, f'TEST:{cue}')}")

    def cmd_set_id(self, port, number):
        target = self.devices.get(port)
        if target and target.connected:
            reply = self._call(target, f"ID:{number}")
            if reply.startswith("ID,"):
                target.id = int(number)
            self._log(self._name(target), "renumbered", reply)
            self._duplicate_ids()

    def cmd_confirm_time(self, device):
        target = self._device_for(device)
        if target:
            self._sync_clock(target, force=True)

    # --- resuming ----------------------------------------------------------

    def resume_latest(self):
        """Pick up an unfinished run left by a crash or a closed window."""
        if not os.path.isdir(self.runs_root):
            return None
        for name in sorted(os.listdir(self.runs_root), reverse=True):
            path = os.path.join(self.runs_root, name, RUN_FILE)
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
            if data.get("format") != RUN_FORMAT or data.get("state") != RUNNING:
                continue
            folder = os.path.dirname(path)
            self.run = self._load_run(folder, data)
            self.state = RUNNING
            self._log("", "run resumed", f"{data['name']} ({data['run_id']})")
            return folder
        return None

    @staticmethod
    def _load_run(folder, data):
        latest = data["protocol_versions"][-1]
        with open(os.path.join(folder, latest), encoding="utf-8") as handle:
            text = handle.read()
        compiled = proto.compile_protocol(proto.load(text), seed=data["seed"])
        run = Run(folder, text, compiled, data["binding"], data["run_id"],
                  parse_iso(data["t0"]), data["seed"])
        run.state = data.get("state", RUNNING)
        run.edits = data.get("edits", [])
        run.fnt_changes = data.get("fnt_phase_changes", [])
        run.stopped_at = data.get("stopped_at")
        run.ended = data.get("ended", {})
        return run

    # --- stopped runs ------------------------------------------------------
    #
    # After Stop a device keeps running its last phase and writing rows to the
    # same file until it starts another run. The animals are still eating, so
    # the copy in the run folder keeps following the card: in the Stopped
    # view, after New run, and after FNT restarts.

    def _run_by_id(self, run_id):
        if self.run and run_id == self.run.run_id:
            return self.run
        return self.stopped.get(run_id)

    def _stopped_runs(self):
        runs = list(self.stopped.values())
        if self.run and self.state == STOPPED:
            runs.append(self.run)
        return runs

    def _find_stopped(self, run_id):
        """A device reports a run FNT doesn't hold: if it's a stopped run in
        runs_root, keep copying its rows there."""
        if self._run_by_id(run_id) or run_id in self._searched:
            return
        self._searched.add(run_id)
        try:
            names = os.listdir(self.runs_root)
        except OSError:
            return
        for name in names:
            path = os.path.join(self.runs_root, name, RUN_FILE)
            try:
                with open(path, encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, ValueError):
                continue
            if (data.get("format") != RUN_FORMAT or data.get("run_id") != run_id
                    or data.get("state") != STOPPED):
                continue
            try:
                self.stopped[run_id] = self._load_run(os.path.dirname(path), data)
            except (OSError, KeyError, proto.ProtocolError):
                return
            self._log("", "copying a stopped run", f"{data.get('name')} ({run_id})", quiet=True)
            return

    def _stopped_step(self, run):
        for logical, device_id in run.binding.items():
            device = self.device_by_id(device_id)
            if device is None or device.status.get("run") != run.run_id:
                continue
            try:
                if device.status.get("ended") == "0":
                    # Stop didn't reach it (it was unplugged): end it now.
                    reply = self._call(device, "RUN:END")
                    self._log(logical, "run ended", f"it was offline at Stop: {reply}")
                    self._read_status(device)
                self._fill_gaps(run, logical, device)
            except LinkError as exc:
                self._disconnect(device, str(exc))

    def _clear_kind(self, kind):
        for problem in list(self.problems.values()):
            if problem.kind == kind:
                self._clear(problem.device, kind)

    # ------------------------------------------------------------------
    # For the UI
    # ------------------------------------------------------------------

    def _publish(self):
        devices = []
        for device in self.devices.values():
            devices.append({
                "port": device.port, "description": device.description,
                "connected": device.connected, "id": device.id,
                "firmware": device.firmware, "compatible": device.compatible,
                "status": dict(device.status), "drift": device.drift,
                "name": self._name(device), "rows": list(device.rows)[-50:],
                "last_row": dict(device.last_row),
                "lights": device.status.get("lights") == "1",
            })
        run = None
        if self.run:
            run = {
                "folder": self.run.folder, "run_id": self.run.run_id,
                "name": self.run.compiled.protocol.name, "t0": self.run.t0,
                "binding": dict(self.run.binding), "state": self.run.state,
                "phases": [p.name for p in self.run.compiled.phases],
                "manual": [any(c.kind == "manual" for c in p.fnt)
                           for p in self.run.compiled.phases],
                "protocol_text": self.run.protocol_text,
                "copies": {l: c.path for l, c in self.run.copies.items()},
                "rules": {l: [sch.summarize(p.rule) for p in s.phases]
                          for l, s in self.run.compiled.schedules.items()},
                "actives": {l: [p.rule["active"] for p in s.phases]
                            for l, s in self.run.compiled.schedules.items()},
            }
        snapshot = {
            "state": self.state, "now": self.now(), "zone": self.zone,
            "devices": devices, "run": run, "preflight": dict(self.preflight),
            "problems": [vars(p).copy() for p in self.problems.values()],
            "events": list(self.events)[-100:], "event_count": self.event_count,
        }
        with self.lock:
            self._snapshot = snapshot

    def snapshot(self):
        with self.lock:
            return self._snapshot


class EngineThread(threading.Thread):
    """Runs Engine.step a few times a second until stopped."""

    def __init__(self, engine, interval=0.25, before_step=None):
        super().__init__(daemon=True, name="fed3-engine")
        self.engine = engine
        self.interval = interval
        self.before_step = before_step     # demo mode: move the simulated world on
        self.stopping = threading.Event()

    def run(self):
        while not self.stopping.is_set():
            try:
                if self.before_step:
                    self.before_step()
                self.engine.step()
            except Exception as exc:      # keep the engine alive; show what broke
                self.engine._event("", f"engine error: {exc!r}")
            self.stopping.wait(self.interval)

    def stop(self):
        self.stopping.set()
