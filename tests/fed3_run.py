"""The run engine against simulated FED3s: a whole experiment, start to finish.

What matters here is what a lab would discover at the end of a run:
- whether every row the devices wrote reached the run folder, exactly once
  and in order, through unplugs and FNT restarts;
- whether phases that need FNT moved when they should;
- whether every problem stayed on screen until someone saw it.
"""
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fnt.fed3 import device_sim as dev      # noqa: E402
from fnt.fed3 import run as runmod          # noqa: E402
from fnt.fed3.link import fields            # noqa: E402

RESULTS = []
TZ = timezone(timedelta(hours=-7))
START = datetime(2026, 10, 20, 9, 0, tzinfo=TZ)


def check(label, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{f' — {detail}' if detail else ''}")


class OldFirmware:
    """A FED3 still on firmware 2.x."""

    closed = False

    def write(self, line):
        self.reply = "PONG_FED3,ID:9,FW:2.6" if line == "PING" else "ERROR:UNKNOWN"

    def readline(self, timeout):
        reply, self.reply = getattr(self, "reply", None), None
        return reply

    def close(self):
        self.closed = True


class World:
    """Simulated FED3s and one clock shared by them and the engine."""

    def __init__(self, root, ids=(3, 7), skew=None):
        self.root = root
        self.elapsed = 0.0
        self.transports = {}
        self.plugged = {}
        for n, device_id in enumerate(ids):
            device = dev.SimDevice(device_id=device_id, now=START, seed=device_id)
            if skew and device_id in skew:
                device.drift(skew[device_id])
            port = f"COM{n + 3}"
            self.transports[port] = dev.SimTransport(device)
            self.plugged[port] = True
        self.engine = self.new_engine()

    def new_engine(self):
        return runmod.Engine(self.root, ports=self.ports, open_port=self.open,
                             now=self.now, monotonic=lambda: self.elapsed, zone=TZ)

    def now(self):
        return (START + timedelta(seconds=self.elapsed)).astimezone(timezone.utc)

    def ports(self):
        return {p: "Feather M0" for p, on in self.plugged.items() if on}

    def open(self, port):
        transport = self.transports[port]
        if isinstance(transport, dev.SimTransport):
            transport.closed = False
        return transport

    def unplug(self, port):
        self.plugged[port] = False
        self.transports[port].closed = True

    def plug(self, port):
        self.plugged[port] = True

    def device(self, port):
        return self.transports[port].device

    def act(self, port, fn):
        return self.transports[port].act(fn)

    def run_for(self, seconds, step=5.0):
        """Let time pass for devices and engine together."""
        end = self.elapsed + seconds
        while self.elapsed < end:
            dt = min(step, end - self.elapsed)
            for transport in self.transports.values():
                if isinstance(transport, dev.SimTransport):
                    transport.act(lambda d, dt=dt: d.advance(dt))
            self.elapsed += dt
            self.engine.step()

    def card_rows(self, port):
        device = self.device(port)
        return device.card.files[device.file]


def ended(world, device_id):
    """Whether the engine last saw this device's run as ended. Read through the
    engine, so it holds for the simulator and the firmware core alike."""
    status = [d["status"] for d in world.engine.snapshot()["devices"] if d["id"] == device_id]
    return bool(status) and status[0].get("ended") == "1"


def copy_text(world, logical, folder=None):
    """A device's copy in the current run's folder, or in ``folder``."""
    if folder is None:
        path = world.engine.run.copies[logical].path
    else:
        path = next(os.path.join(folder, n) for n in os.listdir(folder)
                    if n == f"{logical}.csv" or n.startswith(f"{logical}_FED"))
    with open(path, encoding="ascii") as f:
        return f.read()


PROTOCOL = """\
format: fnt-fed3-protocol/2
name: Engine test
devices: [FED003, FED007]
cues:
  ding: [{tone: {hz: 4000, ms: 100}}]
phases:
  - name: Train
    until: [{rewards: 3, of: all}, {after: 2d}]
    rule: {active: both, cue: {reward: ding}}
  - name: Hold
    until: {manual: true}
    rule: {active: left, ratio: 2}
  - name: Last
    rule: {active: none, free_feed: 1}
checks:
  no_pellets: 2h
"""


def feed(world, port, pokes=1, side="L"):
    for _ in range(pokes):
        world.act(port, lambda d: d.poke(side))
        world.run_for(2, step=1)
        world.act(port, lambda d: d.retrieve())


def main():
    root = tempfile.mkdtemp(prefix="fed3runs-")
    try:
        run_checks(root)
        guard_checks(os.path.join(root, "guards"))
        fault_checks(os.path.join(root, "faults"))
    finally:
        shutil.rmtree(root, ignore_errors=True)
    passed = sum(RESULTS)
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    return 0 if all(RESULTS) else 1


def run_checks(root):
    print("--- finding devices ---")
    world = World(root, skew={7: 95})
    world.transports["COM9"] = OldFirmware()
    world.plugged["COM9"] = True
    world.run_for(1)
    snap = world.engine.snapshot()
    connected = sorted(d["id"] for d in snap["devices"] if d["connected"])
    check("both v3 devices are found, by number", connected == [3, 7], f"{connected}")
    check("old firmware is flagged, not connected",
          any(p["kind"] == "firmware" for p in snap["problems"]))
    world.run_for(1)
    drift = [d["drift"] for d in world.engine.snapshot()["devices"] if d["id"] == 7][0]
    check("a clock 95 s out is corrected on connect", drift == 0.0, f"{drift}")

    print("\n--- pre-flight and start ---")
    world.engine.submit("start", protocol_text=PROTOCOL, binding={"FED003": 3, "FED007": 11}, seed=1)
    world.run_for(1)
    snap = world.engine.snapshot()
    check("a missing device fails pre-flight, and nothing starts",
          snap["state"] == "setup" and "not connected" in snap["preflight"]["FED007"]
          and snap["preflight"]["FED003"] == "ok")
    world.engine.submit("start", protocol_text=PROTOCOL, binding={"FED003": 3, "FED007": 7}, seed=1)
    world.run_for(1)
    snap = world.engine.snapshot()
    run = snap["run"]
    check("with both present, the run starts", snap["state"] == "running" and run)
    folder = run["folder"]
    check("the run folder has its records",
          all(os.path.exists(os.path.join(folder, f)) for f in
              ("run.json", "protocol.yaml", "log.csv", "sched/FED003.txt", "sched/FED007.txt")))
    statuses = {d["id"]: d["status"] for d in snap["devices"] if d["connected"]}
    check("both devices are in the run, on one t0",
          all(statuses[i].get("run") == run["run_id"] for i in (3, 7))
          and statuses[3]["phase_start"] == statuses[7]["phase_start"])

    print("\n--- the data copy ---")
    feed(world, "COM3", 2)
    feed(world, "COM4", 1)
    world.run_for(60)
    check("each copy matches the device's card byte for byte",
          copy_text(world, "FED003") == world.card_rows("COM3")
          and copy_text(world, "FED007") == world.card_rows("COM4"))

    world.unplug("COM4")
    world.run_for(130)
    check("an unplugged device is flagged after 2 minutes",
          any(p["kind"] == "offline" and p["device"] == "FED007"
              for p in world.engine.snapshot()["problems"]))
    for _ in range(4):
        world.act("COM4", lambda d: d.poke("R"))
        world.act("COM4", lambda d: d.advance(2))
        world.act("COM4", lambda d: d.retrieve())
    world.plug("COM4")
    world.run_for(60)
    check("rows written while it was away are filled in on reconnect",
          copy_text(world, "FED007") == world.card_rows("COM4"))
    problem = [p for p in world.engine.snapshot()["problems"] if p["kind"] == "offline"]
    check("the offline problem stays until acknowledged",
          problem and not problem[0]["active"])
    world.engine.submit("acknowledge", key=problem[0]["key"])
    world.run_for(1)
    check("and goes when it is",
          not any(p["kind"] == "offline" for p in world.engine.snapshot()["problems"]))

    print("\n--- phases FNT moves ---")
    snap = world.engine.snapshot()
    phases = {d["name"]: d["status"].get("phase") for d in snap["devices"] if d["connected"]}
    check("B has reached 3 rewards but A hasn't, so nobody moves", phases == {"FED003": "0", "FED007": "0"},
          f"{phases}")
    feed(world, "COM3", 1)
    world.run_for(40)
    snap = world.engine.snapshot()
    phases = {d["name"]: d["status"].get("phase") for d in snap["devices"] if d["connected"]}
    check("when both have, FNT moves both", phases == {"FED003": "1", "FED007": "1"}, f"{phases}")
    import json
    with open(os.path.join(folder, "run.json")) as f:
        changes = json.load(f)["fnt_phase_changes"]
    check("and records it in run.json", len(changes) == 1 and changes[0]["to"] == 1)
    world.engine.submit("advance")
    world.run_for(40)
    snap = world.engine.snapshot()
    phases = {d["name"]: d["status"].get("phase") for d in snap["devices"] if d["connected"]}
    check("Advance phase moves a phase that ends by hand", phases == {"FED003": "2", "FED007": "2"})

    print("\n--- editing mid-run ---")
    edited = PROTOCOL.replace("rule: {active: none, free_feed: 1}",
                              "rule: {active: none, free_feed: 1, refill_delay: 60s}")
    world.engine.submit("edit", protocol_text=edited)
    world.run_for(40)
    check("an edit reaches the devices",
          world.device("COM3").rule()["refill_delay"] == "60")
    check("and is kept as a new version", os.path.exists(os.path.join(folder, "protocol.v2.yaml")))
    bad = PROTOCOL.replace("name: Hold", "name: Renamed")
    world.engine.submit("edit", protocol_text=bad)
    world.run_for(1)
    check("an edit that rewrites a finished phase is refused",
          world.device("COM3").rule()["refill_delay"] == "60"
          and any("failed" in e[2] for e in world.engine.snapshot()["events"]))

    print("\n--- problems ---")
    world.act("COM3", lambda d: d.reboot("watchdog"))
    world.run_for(40)
    check("a watchdog reset is flagged",
          any(p["kind"] == "watchdog" for p in world.engine.snapshot()["problems"]))
    world.act("COM4", lambda d: setattr(d, "jams_pending", 50))
    world.act("COM4", lambda d: d.retrieve())
    world.run_for(120)          # the edited refill_delay is 60 s, then a 1 s dispense
    check("a jam is flagged", any(p["kind"] == "jam" and p["device"] == "FED007"
                                  for p in world.engine.snapshot()["problems"]))
    world.run_for(3 * 3600, step=60)
    check("no pellets for longer than checks.no_pellets is flagged",
          any(p["kind"] == "no_pellets" and p["device"] == "FED007"
              for p in world.engine.snapshot()["problems"]))

    print("\n--- FNT restarts mid-run ---")
    world.engine = world.new_engine()
    resumed = world.engine.resume_latest()
    check("a new engine finds the unfinished run", resumed == folder)
    world.engine._publish()
    recent = world.engine.snapshot()["run"]["recent"]["FED003"]
    last = copy_text(world, "FED003").splitlines()[-1].split(",")
    check("and its cards start from the run's saved rows, not empty",
          recent and recent[-1]["seq"] == last[2] and recent[-1]["event"] == last[4])
    feed(world, "COM3", 3)
    world.run_for(120)
    check("and carries on copying without gaps or repeats",
          copy_text(world, "FED003") == world.card_rows("COM3")
          and copy_text(world, "FED007") == world.card_rows("COM4"))

    print("\n--- stopping ---")
    world.unplug("COM4")            # B misses the Stop
    world.run_for(1)
    world.engine.submit("stop")
    world.run_for(5)
    snap = world.engine.snapshot()
    check("the run stops", snap["state"] == "stopped")
    check("the copy has the device's run_end row",
          any(line.split(",")[4] == "run_end" for line in copy_text(world, "FED003").splitlines()[1:])
          and copy_text(world, "FED003") == world.card_rows("COM3"))
    with open(os.path.join(folder, "run.json")) as f:
        data = json.load(f)
    check("run.json records the stop", data["state"] == "stopped" and "stopped_at" in data)
    end_row = [line for line in copy_text(world, "FED003").splitlines()[1:]
               if line.split(",")[4] == "run_end"][0].split(",")
    check("and where each device's data after Stop begins, from its own run_end row",
          data.get("ended", {}).get("FED003") == {"time": end_row[0], "seq": int(end_row[2]),
                                             "detail": ""}
          and "FED007" not in data.get("ended", {}))
    world.run_for(600)
    check("the devices keep feeding after Stop",
          world.device("COM3").rule()["free_feed"] == "1")
    feed(world, "COM3", 3)
    world.run_for(60)
    after_end = copy_text(world, "FED003").split(",run_end,", 1)[1].splitlines()[1:]
    check("and what they write after Stop is still copied",
          after_end and copy_text(world, "FED003") == world.card_rows("COM3"), f"{len(after_end)} rows")
    world.plug("COM4")
    world.run_for(60)
    check("a device that was unplugged at Stop is stopped when it's back",
          ended(world, 7) and copy_text(world, "FED007") == world.card_rows("COM4"))
    with open(os.path.join(folder, "run.json")) as f:
        data = json.load(f)
    check("and run.json gets its later end too, keeping when Stop was pressed",
          "FED007" in data["ended"] and data["ended"]["FED007"]["time"] > data["ended"]["FED003"]["time"]
          and data["state"] == "stopped" and "stopped_at" in data)
    world.engine.submit("close")
    world.run_for(1)
    check("closing returns to Setup", world.engine.snapshot()["state"] == "setup")
    feed(world, "COM3", 2)
    world.run_for(60)
    check("rows are still copied to the stopped run after New run",
          copy_text(world, "FED003", folder) == world.card_rows("COM3"))
    world.engine = world.new_engine()
    world.engine.resume_latest()
    feed(world, "COM3", 2)
    world.run_for(60)
    check("and after FNT restarts",
          world.engine.snapshot()["state"] == "setup"
          and copy_text(world, "FED003", folder) == world.card_rows("COM3"))

    print("\n--- renumbering after Stop ---")
    numbers = lambda: {d["port"]: d["id"] for d in world.engine.snapshot()["devices"]  # noqa: E731
                       if d["connected"]}
    world.engine.submit("set_id", port="COM3", number=7)
    world.run_for(1)
    check("choosing a number another FED3 has swaps the two",
          numbers() == {"COM3": 7, "COM4": 3}, f"{numbers()}")
    feed(world, "COM3", 2)
    feed(world, "COM4", 2, side="R")
    world.run_for(60)
    check("each device's rows still reach its own copy of the stopped run",
          copy_text(world, "FED003", folder) == world.card_rows("COM3")
          and copy_text(world, "FED007", folder) == world.card_rows("COM4"))
    with open(os.path.join(folder, "run.json")) as f:
        data = json.load(f)
    changes = [(c["device"], c["from"], c["to"]) for c in data.get("renumbered", [])]
    check("run.json keeps who played what at Start, and records the renumbering",
          data["binding"] == {"FED003": 3, "FED007": 7} and changes == [("FED003", 3, 7), ("FED007", 7, 3)],
          f"{changes}")
    world.engine = world.new_engine()
    world.engine.resume_latest()
    feed(world, "COM3", 1)
    feed(world, "COM4", 1, side="R")
    world.run_for(60)
    check("and after FNT restarts",
          copy_text(world, "FED003", folder) == world.card_rows("COM3")
          and copy_text(world, "FED007", folder) == world.card_rows("COM4"))
    world.engine.submit("set_id", port="COM4", number=7)
    world.run_for(1)
    check("and back", numbers() == {"COM3": 3, "COM4": 7}, f"{numbers()}")

    print("\n--- a Stop that FNT didn't finish ---")
    world.engine.submit("start", protocol_text=PROTOCOL, binding={"FED003": 3, "FED007": 7}, seed=2)
    world.run_for(5)
    run = world.engine.run
    world.engine.submit("set_id", port="COM3", number=9)
    world.run_for(1)
    check("a FED3 in the running run can't be renumbered",
          numbers() == {"COM3": 3, "COM4": 7}
          and any("renumber it after Stop" in e[2] for e in world.engine.snapshot()["events"]))
    feed(world, "COM3", 2)
    world.run_for(40)
    rows_before = world.card_rows("COM3")
    world.act("COM3", lambda d: d.command("RUN:END"))      # as if FNT closed mid-Stop
    world.engine = world.new_engine()
    check("the run is picked up as still running", world.engine.resume_latest() == run.folder)
    world.run_for(90)
    check("a device whose run already ended isn't started over",
          ended(world, 3) and world.card_rows("COM3").startswith(rows_before))
    check("it's flagged instead",
          any(p["kind"] == "ended" and p["device"] == "FED003"
              for p in world.engine.snapshot()["problems"]))
    world.engine.submit("stop")
    world.run_for(5)
    check("and Stop finishes the job",
          world.engine.snapshot()["state"] == "stopped"
          and not any(p["kind"] == "ended" and p["active"]
                      for p in world.engine.snapshot()["problems"]))


ONE_PHASE = """\
format: fnt-fed3-protocol/2
name: Guard test 1
devices: [FED001, FED002]
phases:
  - name: FR1
    rule: {active: both}
checks:
  no_pellets: 2h
"""

RACE = """\
format: fnt-fed3-protocol/2
name: Race
devices: [FED001, FED002]
phases:
  - name: Race
    until: {rewards: 3, of: any}
    rule: {active: both}
  - name: Next
    rule: {active: left}
"""


class Unopenable:
    """A port another program holds."""

    def __init__(self):
        raise OSError("access denied")


class OtherBoard:
    """An Adafruit board that isn't a FED3."""

    pings = 0
    closed = False

    def write(self, line):
        OtherBoard.pings += line == "PING"
        self.reply = "hello"

    def readline(self, timeout):
        reply, self.reply = getattr(self, "reply", None), None
        return reply

    def close(self):
        self.closed = True


def begin_elsewhere(device):
    """Start a run on a device behind FNT's back, with the schedule it holds."""
    status = [line for line in device.command("STATUS") if line.startswith("STATUS,")][0]
    device.command(f"RUN:BEGIN,ELSEWHRE,{START.isoformat()},{fields(status)['sched']}")


def seen(world, device_id, key):
    """A STATUS field as the engine last read it: the same for the simulator
    and the firmware core."""
    found = [d["status"].get(key) for d in world.engine.snapshot()["devices"]
             if d["id"] == device_id and d["connected"]]
    return found[0] if found else None


def problems(world, active=True):
    return sorted((p["device"], p["kind"]) for p in world.engine.snapshot()["problems"]
                  if p["active"] or not active)


def guard_checks(root):
    os.makedirs(root)
    print("\n--- Start won't end a run that's still going ---")
    world = World(root, ids=(1, 2))
    world.run_for(30)
    # As if another PC (or an FNT that lost its run folder) started a run on it.
    world.act("COM3", begin_elsewhere)
    world.run_for(40)
    snap = world.engine.snapshot()
    live = {d["port"]: d["live_run"] for d in snap["devices"]}
    check("a FED3 in someone else's unended run is marked as such", live == {"COM3": True,
                                                                          "COM4": False},
          f"{live}")
    world.engine.submit("start", protocol_text=ONE_PHASE, binding={"FED001": 1, "FED002": 2})
    world.run_for(5)
    snap = world.engine.snapshot()
    check("Start refuses it, saying which run, and starts nothing",
          snap["state"] == "setup" and "still in run" in snap["preflight"].get("FED001", "")
          and seen(world, 1, "run") == "ELSEWHRE", f"{snap['preflight']}")
    world.engine.submit("start", protocol_text=ONE_PHASE, binding={"FED001": 1, "FED002": 2}, replace=True)
    world.run_for(5)
    check("unless told to replace it",
          world.engine.snapshot()["state"] == "running"
          and seen(world, 1, "run") == world.engine.run.run_id)

    print("\n--- no false 'no pellets' after a restart ---")
    for _ in range(4):
        feed(world, "COM3")
        feed(world, "COM4")
        world.run_for(3600)
    check("none before", ("FED001", "no_pellets") not in problems(world))
    world.engine = world.new_engine()
    world.engine.resume_latest()
    world.run_for(120)
    check("and none after FNT restarts, with the last pellet an hour ago",
          not [p for p in problems(world) if p[1] == "no_pellets"], f"{problems(world)}")
    world.run_for(2 * 3600)
    check("but a real gap is still flagged",
          ("FED001", "no_pellets") in problems(world) and ("FED002", "no_pellets") in problems(world))
    world.engine.submit("stop")
    world.run_for(5)
    world.engine.submit("close")
    world.run_for(1)

    print("\n--- a phase change FNT made while a device was away ---")
    world.engine.submit("start", protocol_text=RACE, binding={"FED001": 1, "FED002": 2})
    world.run_for(10)
    world.unplug("COM4")
    world.run_for(10)
    feed(world, "COM3", 3)
    world.run_for(30)
    check("the others move on", seen(world, 1, "phase") == "1")
    world.plug("COM4")
    world.run_for(60)
    check("the one that was unplugged catches up when it's back",
          seen(world, 2, "phase") == "1", f"phase {seen(world, 2, 'phase')}")
    with open(os.path.join(world.engine.run.folder, "run.json")) as f:
        data = json.load(f)
    check("run.json says who missed it, and nobody is still waiting",
          data["fnt_phase_changes"][0]["devices"] == ["FED001"]
          and data["fnt_phase_changes"][0].get("missed") == [])

    print("\n--- an edit the running run can't take ---")
    bad = RACE.replace("  - name: Race", "  - name: Renamed")
    world.engine.submit("edit", protocol_text=bad, token=7)
    world.run_for(1)
    result = world.engine.snapshot()["edit_result"]
    check("is refused, and the refusal is kept for the window to show",
          result["token"] == 7 and "is over" in (result["error"] or ""), f"{result}")
    world.engine.submit("edit", protocol_text=RACE.replace("active: left", "active: right"),
                        token=8)
    world.run_for(1)
    result = world.engine.snapshot()["edit_result"]
    check("a good one is applied, with no error", result == {"token": 8, "error": None})
    world.engine.submit("stop")
    world.run_for(5)

    print("\n--- run folders ---")
    folder = world.engine.run.folder
    world.engine.submit("close")
    world.run_for(1)
    elsewhere = os.path.join(os.path.dirname(root), "elsewhere")
    world.engine.submit("set_runs_root", path=elsewhere)
    world.run_for(1)
    world.engine = world.new_engine()
    world.engine.runs_root = elsewhere
    world.engine.other_roots = [root]
    feed(world, "COM3", 2)
    world.run_for(60)
    check("a stopped run in the old folder is still followed",
          copy_text(world, "FED001", folder) == world.card_rows("COM3"))
    os.makedirs(os.path.join(elsewhere, "2099-01-01_0000_broken"))
    with open(os.path.join(elsewhere, "2099-01-01_0000_broken", "run.json"), "w") as f:
        f.write("{not json")
    os.makedirs(os.path.join(elsewhere, "2099-01-02_0000_old"))
    with open(os.path.join(elsewhere, "2099-01-02_0000_old", "run.json"), "w") as f:
        json.dump({"format": runmod.RUN_FORMAT, "state": "running", "run_id": "R1",
                   "name": "Old", "protocol_versions": ["protocol.yaml"]}, f)
    engine = world.new_engine()
    engine.runs_root = elsewhere
    try:
        resumed, crashed = engine.resume_latest(), None
    except Exception as exc:        # noqa: BLE001
        resumed, crashed = None, exc
    kinds = sorted(p.kind.split(":")[0] for p in engine.problems.values())
    check("a run.json that can't be read, or a run that can't be loaded, doesn't stop "
          "the window opening; both are reported",
          crashed is None and resumed is None and kinds == ["resume", "resume"],
          f"{crashed!r} {kinds}")
    world.engine.runs_root = root
    taken = os.path.join(root, f"{(world.now() + timedelta(seconds=5)).astimezone(TZ):%Y-%m-%d_%H%M}"
                               "_guard-test-1")
    os.makedirs(taken)                  # a run already started this minute
    world.engine.submit("start", protocol_text=ONE_PHASE, binding={"FED001": 1, "FED002": 2},
                        replace=True)
    world.run_for(5)
    check("a second run in the same minute gets _2, keeping the name whole",
          world.engine.run.folder == taken + "_2", world.engine.run.folder)
    world.engine.submit("stop")
    world.run_for(5)

    print("\n--- an ID changed on the FED3 itself ---")
    world.act("COM4", lambda d: d.command("ID:9"))       # as its boot menu does
    world.run_for(40)
    ids = {d["port"]: d["id"] for d in world.engine.snapshot()["devices"] if d["connected"]}
    check("FNT picks up the new ID", ids.get("COM4") == 9, f"{ids}")
    check("and logs it", any("ID changed on the FED3" in e[2] for e in world.engine.events))

    print("\n--- ports that aren't FED3s ---")
    world.transports["COM8"] = OtherBoard()
    world.plugged["COM8"] = True
    world.open_real = world.open
    opened = world.open

    def open_port(port):
        if port == "COM9":
            return Unopenable()
        return opened(port)
    world.plugged["COM9"] = True
    world.transports["COM9"] = None
    world.open = open_port
    world.engine = world.new_engine()
    world.run_for(120)
    cant = [e for e in world.engine.events if "can't open" in e[2]]
    check("a port another program holds is mentioned once, not every retry", len(cant) == 1,
          f"{len(cant)}")
    check("another Adafruit board is asked once, then left alone", OtherBoard.pings == 1,
          f"{OtherBoard.pings} PINGs")


FAULTS = """\
format: fnt-fed3-protocol/2
name: Faults
devices: [FED001, FED003]
phases:
  - name: One
    until: {rewards: 2, of: each}
    rule: {active: left}
  - name: Two
    rule: {active: left, ratio: 2}
"""


def card_file(world, port):
    """(card, name) of the data file a device is writing."""
    device = world.device(port)
    return device.card, device.file


def drop_row(world, port, seq):
    """Take one row off a device's card, as firmware 3.1 left it when the card
    refused a row: the number was used, and the row was never written."""
    card, name = card_file(world, port)
    lines = card.files[name].splitlines(keepends=True)
    text = "".join(line for line in lines if line.split(",")[2:3] != [str(seq)])
    if hasattr(card, "path"):           # the firmware core's card is a folder
        with open(os.path.join(card.path, name), "w", encoding="ascii", newline="") as f:
            f.write(text)
    else:
        card.files[name] = text


def away(world, port, fn):
    """Something a device does while unplugged. What it sends then goes
    nowhere: the simulated port would otherwise hand it over on reconnect."""
    result = world.act(port, fn)
    world.transports[port].pending.clear()
    return result


class Trickle(dev.SimTransport):
    """A FED3 that replays a file the way a real one does: a few rows at a
    time, between everything else. A poll gets what has arrived (``budget``
    lines a step); waiting for a reply gets whatever comes before it."""

    budget = 0

    def readline(self, timeout):
        if not timeout:
            if self.budget <= 0:
                return None
            self.budget -= 1
        return super().readline(timeout)


class LowBattery:
    """A FED3 whose battery has run down."""

    closed = False

    def write(self, line):
        self.reply = {"PING": "PONG,id=21,fw=3.2,proto=3"}.get(
            line, "STATUS,id=21,clock=ok,sd=ok,run=-,ended=1,jam=0,batt=3.41" if line == "STATUS"
            else "TIME,before=unset,after=2026-10-20T09:00:00-07:00")

    def readline(self, timeout):
        reply, self.reply = getattr(self, "reply", None), None
        return reply

    def close(self):
        self.closed = True


def fault_checks(root):
    os.makedirs(root)
    print("\n--- a FED3 that powers up before its card is ready ---")
    world = World(root, ids=(1, 3))
    world.run_for(10)
    world.engine.submit("start", protocol_text=FAULTS, binding={"FED001": 1, "FED003": 3})
    world.run_for(10)
    feed(world, "COM3", 1)
    feed(world, "COM4", 3)
    world.run_for(40)
    world.device("COM4").card.failing = True
    world.act("COM4", lambda d: d.reboot("power"))
    world.run_for(45)                   # long enough for FNT to read its STATUS and set its clock
    ids = {d["port"]: d["id"] for d in world.engine.snapshot()["devices"] if d["connected"]}
    check("while it can't read its card, FNT doesn't take the default ID for a new one",
          ids == {"COM3": 1, "COM4": 3} and ("FED001", "duplicate_id") not in problems(world),
          f"{ids}")
    world.device("COM4").card.failing = False
    world.run_for(60)
    check("with the card back it's in the run again, in its phase, under its own ID",
          seen(world, 3, "run") == world.engine.run.run_id and seen(world, 3, "phase") == "1"
          and seen(world, 3, "id") == "3", f"{seen(world, 3, 'run')} id {seen(world, 3, 'id')}")
    check("its card still says who it is",
          world.device("COM4").card.files["DEVICE.TXT"].startswith("id=3\n"))
    check("the copy matches the card", copy_text(world, "FED003") == world.card_rows("COM4"))
    check("and it's flagged", ("FED003", "sd_boot") in problems(world, active=False),
          f"{problems(world, active=False)}")

    print("\n--- rows a card refused ---")
    world.device("COM4").card.failing = True
    world.act("COM4", lambda d: d.poke("R"))
    world.run_for(2, step=1)
    world.device("COM4").card.failing = False
    world.run_for(40)
    check("a refused row is flagged even though STATUS says the card is fine again",
          ("FED003", "sd") in problems(world, active=False)
          and seen(world, 3, "sd") == "ok")
    check("and reaches the copy once the card takes it",
          copy_text(world, "FED003") == world.card_rows("COM4"))
    # Firmware 3.1 used the row's number and never wrote it: a gap on the card.
    world.unplug("COM4")
    world.run_for(5)
    for _ in range(3):
        away(world, "COM4", lambda d: d.poke("R"))
    lost = world.engine.run.copies["FED003"].next_seq + 1
    drop_row(world, "COM4", lost)
    world.plug("COM4")
    world.run_for(60)
    feed(world, "COM4", 2)
    world.run_for(60)
    copy = world.engine.run.copies["FED003"]
    check("a number missing from the card doesn't hold the copy up for good",
          not copy.gap and copy_text(world, "FED003") == world.card_rows("COM4"),
          f"waiting at {copy.next_seq}, {len(copy.pending)} rows held")
    lost_rows = [p["message"] for p in world.engine.snapshot()["problems"]
                 if p["kind"] == "lost_rows"]
    check("and the lost row is flagged by number", lost_rows and f"row {lost} " in lost_rows[0],
          f"{lost_rows}")

    print("\n--- a restart nobody saw ---")
    world.unplug("COM3")
    world.run_for(5)
    away(world, "COM3", lambda d: d.reboot("watchdog"))     # its BOOT line goes to nobody
    world.plug("COM3")
    world.run_for(60)
    check("a watchdog restart is flagged from its row in the data",
          ("FED001", "watchdog") in problems(world, active=False))

    print("\n--- FNT's copy after the PC lost power mid-write ---")
    feed(world, "COM3", 2)
    world.run_for(30)
    folder = world.engine.run.folder
    with open(world.engine.run.copies["FED001"].path, "a", encoding="ascii") as f:
        f.write("2026-10-20T09:0")
    world.engine = world.new_engine()
    check("a torn last line doesn't stop the run being picked up",
          world.engine.resume_latest() == folder and world.engine.state == "running")
    feed(world, "COM3", 1)
    world.run_for(60)
    check("and the copy matches the card again",
          copy_text(world, "FED001") == world.card_rows("COM3"))

    print("\n--- catching up on a long file ---")
    world.unplug("COM3")
    world.run_for(5)
    for _ in range(150):
        away(world, "COM3", lambda d: d.poke("R"))
    slow = world.transports["COM3"] = Trickle(world.device("COM3"))
    world.plug("COM3")
    for _ in range(40):                 # until it's found and the READ is sent
        world.run_for(1, step=1)
        found = world.engine.device_by_id(1)
        if found and found.link.replaying is not None:
            break
    partial = []
    for _ in range(12):
        slow.budget = 5
        world.run_for(1, step=1)
        partial.append(world.engine.run.copies["FED001"].next_seq)
    check("the rows come in over many steps, none of them waiting on the replay",
          len(set(partial)) > 5 and copy_text(world, "FED001") != world.card_rows("COM3"),
          f"{partial[:4]}...")
    # Every replayed row is one of those pokes; live, there was only the clock being set.
    check("replayed rows go to the copy without passing for what the device just did",
          found.last_row.get("event") != "poke", f"{found.last_row.get('event')}")
    behind = world.engine.snapshot()["run"]["behind"]["FED001"]
    check("the snapshot says how far behind the copy is", behind and behind > 0, f"{behind}")
    world.act("COM3", lambda d: d.poke("L"))
    slow.budget = 100000
    world.run_for(1, step=1)
    news = next(d["last_row"] for d in world.engine.snapshot()["devices"] if d["id"] == 1)
    check("a row written during the replay is news",
          news.get("side") == "L", f"{news.get('seq')} {news.get('side')}")
    world.run_for(120)
    check("and the copy ends up matching the card",
          copy_text(world, "FED001") == world.card_rows("COM3")
          and world.engine.snapshot()["run"]["behind"]["FED001"] == 0)

    print("\n--- the engine, when something breaks ---")
    world.transports["COM7"] = LowBattery()
    world.plugged["COM7"] = True
    world.run_for(40)
    check("a low battery is flagged", ("FED021", "battery") in problems(world),
          f"{problems(world)}")
    # A new snapshot, not a later "published": that's real time, and Windows'
    # clock can show the same value for all five steps.
    before = world.engine.snapshot()
    real = world.engine._check_problems
    world.engine._check_problems = lambda: 1 / 0
    world.run_for(5)
    snap = world.engine.snapshot()
    check("an error in a step is shown as a problem, and the snapshot still goes out",
          ("", "engine") in problems(world) and snap is not before)
    world.engine._check_problems = real
    world.run_for(5)
    check("and clears when the steps work again", ("", "engine") not in problems(world))
    world.engine.submit("stop")
    world.run_for(5)


if __name__ == "__main__":
    sys.exit(main())
