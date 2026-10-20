"""The run engine against simulated FED3s: a whole experiment, start to finish.

What matters here is what a lab would discover at the end of a run:
- whether every row the devices wrote reached the run folder, exactly once
  and in order, through unplugs and FNT restarts;
- whether phases that need FNT moved when they should;
- whether every problem stayed on screen until someone saw it.
"""
import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fnt.fed3 import device_sim as dev      # noqa: E402
from fnt.fed3 import run as runmod          # noqa: E402

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
                    if n.startswith(f"{logical}_FED"))
    with open(path, encoding="ascii") as f:
        return f.read()


PROTOCOL = """\
format: fnt-fed3-protocol/1
name: Engine test
devices: [A, B]
cues:
  ding: [{tone: {hz: 4000, ms: 100}}]
phases:
  - name: Train
    until: [{rewards: 3, devices: all}, {after: 2d}]
    all: {active: both, cue: {reward: ding}}
  - name: Hold
    until: {manual: true}
    all: {active: left, ratio: 2}
  - name: Last
    all: {active: none, free_feed: 1}
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
    world.engine.submit("start", protocol_text=PROTOCOL, binding={"A": 3, "B": 11}, seed=1)
    world.run_for(1)
    snap = world.engine.snapshot()
    check("a missing device fails pre-flight, and nothing starts",
          snap["state"] == "setup" and "not connected" in snap["preflight"]["B"]
          and snap["preflight"]["A"] == "ok")
    world.engine.submit("start", protocol_text=PROTOCOL, binding={"A": 3, "B": 7}, seed=1)
    world.run_for(1)
    snap = world.engine.snapshot()
    run = snap["run"]
    check("with both present, the run starts", snap["state"] == "running" and run)
    folder = run["folder"]
    check("the run folder has its records",
          all(os.path.exists(os.path.join(folder, f)) for f in
              ("run.json", "protocol.yaml", "log.csv", "sched/A.txt", "sched/B.txt")))
    statuses = {d["id"]: d["status"] for d in snap["devices"] if d["connected"]}
    check("both devices are in the run, on one t0",
          all(statuses[i].get("run") == run["run_id"] for i in (3, 7))
          and statuses[3]["phase_start"] == statuses[7]["phase_start"])

    print("\n--- the data copy ---")
    feed(world, "COM3", 2)
    feed(world, "COM4", 1)
    world.run_for(60)
    check("each copy matches the device's card byte for byte",
          copy_text(world, "A") == world.card_rows("COM3")
          and copy_text(world, "B") == world.card_rows("COM4"))

    world.unplug("COM4")
    world.run_for(130)
    check("an unplugged device is flagged after 2 minutes",
          any(p["kind"] == "offline" and p["device"] == "B"
              for p in world.engine.snapshot()["problems"]))
    for _ in range(4):
        world.act("COM4", lambda d: d.poke("R"))
        world.act("COM4", lambda d: d.advance(2))
        world.act("COM4", lambda d: d.retrieve())
    world.plug("COM4")
    world.run_for(60)
    check("rows written while it was away are filled in on reconnect",
          copy_text(world, "B") == world.card_rows("COM4"))
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
    check("B has reached 3 rewards but A hasn't, so nobody moves", phases == {"A": "0", "B": "0"},
          f"{phases}")
    feed(world, "COM3", 1)
    world.run_for(40)
    snap = world.engine.snapshot()
    phases = {d["name"]: d["status"].get("phase") for d in snap["devices"] if d["connected"]}
    check("when both have, FNT moves both", phases == {"A": "1", "B": "1"}, f"{phases}")
    import json
    with open(os.path.join(folder, "run.json")) as f:
        changes = json.load(f)["fnt_phase_changes"]
    check("and records it in run.json", len(changes) == 1 and changes[0]["to"] == 1)
    world.engine.submit("advance")
    world.run_for(40)
    snap = world.engine.snapshot()
    phases = {d["name"]: d["status"].get("phase") for d in snap["devices"] if d["connected"]}
    check("Advance phase moves a phase that ends by hand", phases == {"A": "2", "B": "2"})

    print("\n--- editing mid-run ---")
    edited = PROTOCOL.replace("all: {active: none, free_feed: 1}",
                              "all: {active: none, free_feed: 1, refill_delay: 60s}")
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
    check("a jam is flagged", any(p["kind"] == "jam" and p["device"] == "B"
                                  for p in world.engine.snapshot()["problems"]))
    world.run_for(3 * 3600, step=60)
    check("no pellets for longer than checks.no_pellets is flagged",
          any(p["kind"] == "no_pellets" and p["device"] == "B"
              for p in world.engine.snapshot()["problems"]))

    print("\n--- FNT restarts mid-run ---")
    world.engine = world.new_engine()
    resumed = world.engine.resume_latest()
    check("a new engine finds the unfinished run", resumed == folder)
    feed(world, "COM3", 3)
    world.run_for(120)
    check("and carries on copying without gaps or repeats",
          copy_text(world, "A") == world.card_rows("COM3")
          and copy_text(world, "B") == world.card_rows("COM4"))

    print("\n--- stopping ---")
    world.unplug("COM4")            # B misses the Stop
    world.run_for(1)
    world.engine.submit("stop")
    world.run_for(5)
    snap = world.engine.snapshot()
    check("the run stops", snap["state"] == "stopped")
    check("the copy has the device's run_end row",
          any(line.split(",")[4] == "run_end" for line in copy_text(world, "A").splitlines()[1:])
          and copy_text(world, "A") == world.card_rows("COM3"))
    with open(os.path.join(folder, "run.json")) as f:
        data = json.load(f)
    check("run.json records the stop", data["state"] == "stopped" and "stopped_at" in data)
    end_row = [line for line in copy_text(world, "A").splitlines()[1:]
               if line.split(",")[4] == "run_end"][0].split(",")
    check("and where each device's data after Stop begins, from its own run_end row",
          data.get("ended", {}).get("A") == {"time": end_row[0], "seq": int(end_row[2]),
                                             "detail": ""}
          and "B" not in data.get("ended", {}))
    world.run_for(600)
    check("the devices keep feeding after Stop",
          world.device("COM3").rule()["free_feed"] == "1")
    feed(world, "COM3", 3)
    world.run_for(60)
    after_end = copy_text(world, "A").split(",run_end,", 1)[1].splitlines()[1:]
    check("and what they write after Stop is still copied",
          after_end and copy_text(world, "A") == world.card_rows("COM3"), f"{len(after_end)} rows")
    world.plug("COM4")
    world.run_for(60)
    check("a device that was unplugged at Stop is stopped when it's back",
          ended(world, 7) and copy_text(world, "B") == world.card_rows("COM4"))
    with open(os.path.join(folder, "run.json")) as f:
        data = json.load(f)
    check("and run.json gets its later end too, keeping when Stop was pressed",
          "B" in data["ended"] and data["ended"]["B"]["time"] > data["ended"]["A"]["time"]
          and data["state"] == "stopped" and "stopped_at" in data)
    world.engine.submit("close")
    world.run_for(1)
    check("closing returns to Setup", world.engine.snapshot()["state"] == "setup")
    feed(world, "COM3", 2)
    world.run_for(60)
    check("rows are still copied to the stopped run after New run",
          copy_text(world, "A", folder) == world.card_rows("COM3"))
    world.engine = world.new_engine()
    world.engine.resume_latest()
    feed(world, "COM3", 2)
    world.run_for(60)
    check("and after FNT restarts",
          world.engine.snapshot()["state"] == "setup"
          and copy_text(world, "A", folder) == world.card_rows("COM3"))

    print("\n--- a Stop that FNT didn't finish ---")
    world.engine.submit("start", protocol_text=PROTOCOL, binding={"A": 3, "B": 7}, seed=2)
    world.run_for(5)
    run = world.engine.run
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
          any(p["kind"] == "ended" and p["device"] == "A"
              for p in world.engine.snapshot()["problems"]))
    world.engine.submit("stop")
    world.run_for(5)
    check("and Stop finishes the job",
          world.engine.snapshot()["state"] == "stopped"
          and not any(p["kind"] == "ended" and p["active"]
                      for p in world.engine.snapshot()["problems"]))


if __name__ == "__main__":
    sys.exit(main())
