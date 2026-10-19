"""The simulated v3 device: the behaviour the firmware has to reproduce.

Each check is a promise the device makes to an animal or to the data: a reward
on the right poke at the right count, a phase that changes at 19:00 and not at
19:00 plus however long the PC was off, counts that survive a power cut, a
pellet that keeps coming when the PC is gone. The firmware is checked against
the same list on the bench.
"""
import os
import random
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fnt.fed3 import device_sim as dev       # noqa: E402
from fnt.fed3 import protocol as proto       # noqa: E402
from fnt.fed3 import schedule as sch         # noqa: E402

RESULTS = []
TZ = timezone(timedelta(hours=-7))
# The device under test. The firmware's host build swaps in its own factory
# (fnt-fed3 test/run_sim_tests.py) to run these same checks on the C++ core.
MAKE = dev.SimDevice


def check(label, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{f' — {detail}' if detail else ''}")


def at(day, hour, minute=0, second=0):
    return datetime(2026, 10, day, hour, minute, second, tzinfo=TZ)


def compile_one(phases, devices="[A]", extra="", seed=None):
    text = (f"format: {proto.FORMAT}\nname: Test\ndevices: {devices}\n{extra}"
            f"phases:\n{phases}")
    return proto.compile_protocol(proto.load(text), seed=seed)


def upload(device, schedule, phase=None):
    device.command("SCHED:BEGIN")
    replies = [device.command(f"S:{line}")[-1] for line in schedule.lines()]
    suffix = f",{phase}" if phase is not None else ""
    return replies, device.command(f"SCHED:COMMIT,{schedule.crc():08x}{suffix}")[-1]


def start(device, schedule, run_id="RUN1", t0=None):
    upload(device, schedule)
    t0 = t0 or dev.parse_iso(status(device)["time"])
    return device.command(f"RUN:BEGIN,{run_id},{dev.iso(t0)},{schedule.crc():08x}")[-1]


def rows(device, event=None):
    """Data rows on the card, as dicts."""
    text = device.card.files[status(device)["file"]]
    out = [dict(zip(dev.COLUMNS, line.split(",", len(dev.COLUMNS) - 1)))
           for line in text.splitlines()[1:]]
    return [r for r in out if event is None or r["event"] == event]


def outcomes(device):
    return [r["outcome"] for r in rows(device, "poke")]


def fresh(phases, devices="[A]", extra="", now=None, seed=None):
    compiled = compile_one(phases, devices, extra, seed)
    device = MAKE(device_id=3, now=now or at(20, 9))
    reply = start(device, compiled.schedules[compiled.protocol.devices[0]])
    return device, compiled, reply


def sent_rows(device):
    """Rows the device sent for the current run's file, from its run_start on."""
    sent = [line[4:] for line in device.output if line.startswith("ROW,")]
    first = next(i for i, row in enumerate(sent) if ",run_start," in row)
    return sent[first:]


def status(device):
    line = device.command("STATUS")[-1]
    return dict(f.split("=", 1) for f in line.split(",")[1:])


def main(make=None):
    global MAKE
    if make is not None:
        MAKE = make
    print("--- boot ---")
    device = MAKE(device_id=3, now=at(20, 9))
    out = device.take()
    check("a boot announces itself", out[0].startswith("BOOT,fw=3.0,reset=power,clock=ok"))
    check("and is logged", out[1].startswith("ROW,") and ",boot," in out[1])
    check("a blank card logs to IDLE.CSV with a header",
          device.card.files["IDLE.CSV"].startswith(dev.HEADER + "\n"))
    check("PING answers with id and protocol",
          device.command("PING") == ["PONG,id=3,fw=3.0,proto=3"])
    check("a blank card runs FR1 on the left",
          sch.summarize(device.rule()) == "FR1 · left")
    check("ID renumbers and is kept", device.command("ID:12") == ["ID,id=12"]
          and MAKE(card=device.card, now=at(20, 9)).command("PING")[-1].startswith("PONG,id=12,"))
    check("unknown commands are refused", device.command("MODE:FR1")[0].startswith("ERR,UNKNOWN"))
    check("over-long commands are refused", device.command("X" * 200)[0].startswith("ERR,TOO_LONG"))

    print("\n--- uploading and starting ---")
    compiled = compile_one("""\
  - name: FR3
    until: {rewards: 2}
    all: {ratio: 3, cue: {reward: [click], counted: [{tone: {hz: 2000, ms: 10}}]}}
  - name: Both
    all: {active: both}
""")
    schedule = compiled.schedules["A"]
    device = MAKE(device_id=3, now=at(20, 9))
    replies, commit = upload(device, schedule)
    check("every line is accepted", all(r == "OK" for r in replies))
    check("outside a run the schedule waits", commit.endswith("slot=next"))
    check("and is not yet in force", sch.summarize(device.rule()) == "FR1 · left")
    check("a start with the wrong crc is refused",
          device.command(f"RUN:BEGIN,RUN1,{dev.iso(at(20, 9))},deadbeef")[0]
          .startswith("ERR,RUN_SCHED"))
    reply = device.command(f"RUN:BEGIN,RUN1,{dev.iso(at(20, 9))},{schedule.crc():08x}")[-1]
    check("the right crc starts the run", reply == "RUN,run=RUN1,file=RUN1.CSV,phase=0")
    check("and the uploaded schedule is now in force",
          sch.summarize(device.rule()) == "FR3 · left")
    first = rows(device)
    check("the run's file starts with run_start", first[0]["event"] == "run_start"
          and first[0]["seq"] == "1")
    again = device.command(f"RUN:BEGIN,RUN1,{dev.iso(at(20, 9))},{schedule.crc():08x}")[-1]
    check("repeating RUN:BEGIN changes nothing", again == reply and len(rows(device)) == 1)

    bad = sch.Schedule("x", [sch.Phase("A")])
    device2 = MAKE(now=at(20, 9))
    device2.command("SCHED:BEGIN")
    device2.command("S:NAME x")
    device2.command("S:PHASE A")
    err = device2.command("S:SET ratio=0")[0]
    check("a bad line is refused with its number", err.startswith("ERR,SCHED,3,"))
    check("and poisons the commit",
          device2.command(f"SCHED:COMMIT,{bad.crc():08x}")[0].startswith("ERR,SCHED_REJECTED"))
    device2.command("SCHED:BEGIN")
    for line in bad.lines():
        device2.command(f"S:{line}")
    check("a wrong crc is refused",
          device2.command("SCHED:COMMIT,00000000")[0].startswith("ERR,SCHED_REJECTED"))
    device2.command("SCHED:BEGIN")
    for line in bad.lines():
        device2.command(f"S:{line}")
    device2.advance(30)
    check("an upload left idle expires",
          device2.command(f"SCHED:COMMIT,{bad.crc():08x}")[0].startswith("ERR,SCHED_REJECTED"))

    print("\n--- what a poke earns ---")
    device.poke("L")
    device.poke("L")
    device.poke("R")
    check("counted pokes play the counted cue",
          [slot for _, slot, _ in device.cues] == ["counted", "counted"])
    device.poke("L")
    check("the third left poke is rewarded",
          outcomes(device) == ["counted", "counted", "inactive", "reward"])
    check("and the reward cue plays", device.cues[-1][1:] == ("reward", "click"))
    device.poke("L")
    check("a poke while the pellet is on its way doesn't count",
          outcomes(device)[-1] == "pellet_in_well")
    device.advance(2)
    check("the pellet lands as a dispense row", rows(device)[-1]["event"] == "dispense")
    device.advance(8)
    device.retrieve()
    retrieve = rows(device, "retrieve")[-1]
    landed = rows(device, "dispense")[-1]
    check("retrieval time is measured from the dispense",
          int(retrieve["retrieval_ms"]) == int(retrieve["ms"]) - int(landed["ms"]),
          retrieve["retrieval_ms"])
    for _ in range(3):
        device.poke("L")
    check("the second reward ends the phase", rows(device, "phase")[-1]["detail"] == "Both")
    check("and the next phase's rule takes over",
          sch.summarize(device.rule()) == "FR1 · both")
    poke_rows = rows(device, "poke")
    check("the rewarding poke is logged in the phase it was made in",
          poke_rows[-1]["phase"] == "0" and poke_rows[-1]["outcome"] == "reward")
    check("phase counts restart", status(device)["phase_rewards"] == "0")
    check("run totals don't", status(device)["l"] == "7")
    check("each row went out as it was written", sent_rows(device)
          == device.card.files["RUN1.CSV"].splitlines()[1:])

    print("\n--- ratios ---")
    device, _, _ = fresh("""\
  - name: Shared
    all: {active: both, ratio: 2}
""")
    for side in "LRLR":
        device.poke(side)
        device.advance(2)
        device.retrieve()
    check("shared: pokes on either side add to one count",
          outcomes(device) == ["counted", "reward", "counted", "reward"])
    device, _, _ = fresh("""\
  - name: PerPort
    all: {active: both, ratio: 2, count: per_port}
""")
    for side in "LRLR":
        device.poke(side)
        device.advance(2)
        device.retrieve()
    check("per_port: each side counts on its own",
          outcomes(device) == ["counted", "counted", "reward", "reward"])
    device, _, _ = fresh("""\
  - name: PR
    all: {ratio_kind: progressive}
""")
    needed = []
    pokes = 0
    while len(rows(device, "dispense")) < 4:
        device.poke("L")
        pokes += 1
        if outcomes(device)[-1] == "reward":
            needed.append(pokes)
            pokes = 0
            device.advance(2)
            device.retrieve()
    check("progressive ratio follows 1, 2, 4, 6", needed == [1, 2, 4, 6], f"{needed}")
    device, _, _ = fresh("""\
  - name: RR
    all: {ratio_kind: random, ratio: 4}
""")
    for _ in range(400):
        device.poke("L")
        device.advance(2)
        device.retrieve()
    rate = outcomes(device).count("reward") / 400
    check("random ratio rewards about 1 in 4", 0.18 < rate < 0.32, f"{rate:.2f}")

    print("\n--- timeout, quota, short pokes ---")
    device, _, _ = fresh("""\
  - name: TO
    all: {timeout: 30s}
""")
    device.poke("L")
    device.advance(2)
    device.retrieve()
    device.poke("L")
    device.advance(30)
    device.poke("L")
    check("pokes in the timeout don't count, and it ends",
          outcomes(device) == ["reward", "timeout", "reward"])
    device, _, _ = fresh("""\
  - name: Q
    all: {quota: 2, quota_period: 1h, cue: {depleted: [click]}}
""")
    for _ in range(3):
        device.poke("L")
        device.advance(2)
        device.retrieve()
    check("the quota depletes the patch", outcomes(device) == ["reward", "reward", "depleted"])
    check("depleted pokes play their cue", device.cues[-1][1] == "depleted")
    check("STATUS reports it", status(device)["depleted"] == "1")
    device.advance(3600)
    device.poke("L")
    check("it refills next period", outcomes(device)[-1] == "reward")
    device, _, _ = fresh("""\
  - name: Long
    all: {min_poke_ms: 200}
""")
    device.poke("L", ms=150)
    device.advance(2)
    device.poke("L", ms=300)
    check("a poke shorter than min_poke_ms doesn't count",
          outcomes(device) == ["short", "reward"])
    check("and its duration is logged", rows(device, "poke")[0]["poke_ms"] == "150")

    device, _, _ = fresh("""\
  - name: Held
    all: {ratio: 1}
""")
    device.poke("L", ms=75000)
    held = rows(device, "poke")
    check("a beam held past 60 s is logged once, as held, at 60 s",
          len(held) == 1 and held[0]["poke_ms"] == "60000" and held[0]["detail"] == "held")
    check("and it still earned its reward", held[0]["outcome"] == "reward")
    device.poke("L")
    check("the next poke is decided normally", outcomes(device)[-1] == "pellet_in_well")

    print("\n--- dispensing on its own ---")
    device, _, _ = fresh("""\
  - name: Free
    all: {active: none, free_feed: 1, refill_delay: 60s}
""")
    device.advance(2)
    check("free feeding fills the well", len(rows(device, "dispense")) == 1)
    device.retrieve()
    device.advance(30)
    check("and waits refill_delay after a retrieval", len(rows(device, "dispense")) == 1)
    device.advance(32)
    check("then refills", len(rows(device, "dispense")) == 2)
    check("free pellets are marked", rows(device, "dispense")[-1]["detail"] == "free")
    device.poke("L")
    check("pokes are still logged", outcomes(device) == ["pellet_in_well"])

    device, _, _ = fresh("""\
  - name: Fed
    all: {active: none}
    daily:
      - {from: "19:00", to: "21:00", all: {free_feed: 1}}
""")
    device.advance(10 * 3600 + 30)
    check("a timed-feeding window opens on time",
          [r["time"][11:16] for r in rows(device, "dispense")] == ["19:00"])
    for _ in range(3):
        device.advance(300)
        device.retrieve()
    device.advance(3 * 3600)
    times = [r["time"][11:16] for r in rows(device, "dispense")]
    check("and stops at 21:00", times[-1] < "21:00", f"{times}")

    device, _, _ = fresh("""\
  - name: J
    all: {ratio: 1}
""")
    device.jams_pending = 1
    device.poke("L")
    device.advance(2)
    check("a jam is logged instead of a dispense",
          [r["event"] for r in rows(device)][-2:] == ["poke", "jam"])
    check("STATUS reports it", status(device)["jam"] == "1")
    device.advance(dev.JAM_RETRY_SECONDS + 2)
    check("and the dispense is retried later",
          rows(device)[-1]["event"] == "dispense" and rows(device)[-1]["detail"] == "retry")
    check("manual FEED with a pellet in the well says so",
          device.command("FEED") == ["FEED,result=pellet_in_well"])

    print("\n--- phases on the clock ---")
    rotating = compile_one("""\
  - name: Hab
    until: {at: "19:00", day: 2}
    all: {active: none, free_feed: 1}
  - name: Rot
    until: {after: 3d}
    rotate: {every: 1d, at: "19:00", mode: cycle, assign: {A: rich, B: poor}}
  - name: End
    all: {active: none, free_feed: 1}
""", devices="[A, B]", extra="rules:\n  rich: {ratio: 1}\n  poor: {ratio: 5}\n")
    device = MAKE(now=at(20, 9))
    start(device, rotating.schedules["A"])
    device.advance(6 * 86400)
    changes = [(r["time"], r["detail"]) for r in rows(device, "phase")]
    expected = [(dev.iso(at(21, 19)), "Rot 1"), (dev.iso(at(22, 19)), "Rot 2"),
                (dev.iso(at(23, 19)), "Rot 3"), (dev.iso(at(24, 19)), "End")]
    check("each phase changes at 19:00, day after day", changes == expected, f"{changes}")

    device = MAKE(now=at(20, 9))
    start(device, rotating.schedules["A"])
    device.advance(3600)
    device.power_off(3 * 86400)
    changes = [r["detail"] for r in rows(device, "phase")]
    expected = [f"Rot 1;catch-up;start={dev.iso(at(21, 19))}",
                f"Rot 2;catch-up;start={dev.iso(at(22, 19))}"]
    check("after three days off, it replays the boundaries it missed",
          changes == expected, f"{changes}")
    check("stamped when written, so time never runs backwards",
          all(r["time"] == dev.iso(at(23, 10)) for r in rows(device, "phase")))
    check("and lands where it would have been", status(device)["phase_name"] == "Rot 2"
          and status(device)["phase_start"] == dev.iso(at(22, 19)))
    device.advance(86400)
    check("then carries on at the next 19:00",
          rows(device, "phase")[-1]["time"] == dev.iso(at(23, 19)))

    print("\n--- what a reboot keeps ---")
    device, _, _ = fresh("""\
  - name: Q
    until: {rewards: 10}
    all: {quota: 3, quota_period: 1h, ratio: 2}
  - name: Next
""")
    for _ in range(4):
        device.poke("L")
        device.advance(2)
        device.retrieve()
    device.poke("L")
    before = status(device)
    device.reboot("watchdog")
    after = status(device)
    check("a reboot reports why", any(r["detail"].startswith("reset=watchdog")
                                      for r in rows(device, "boot")))
    check("totals and phase counts survive",
          all(before[k] == after[k] for k in ("l", "pellets", "phase_rewards", "phase")))
    check("seq carries on", int(after["seq"]) == int(before["seq"]) + 1)
    device.poke("L")
    device.poke("L")
    check("the ratio counter restarted, as documented",
          outcomes(device)[-2:] == ["counted", "reward"])
    device.advance(2)
    device.retrieve()
    device.poke("L")
    device.poke("L")
    check("quota used this period survived the reboot",
          outcomes(device)[-1] == "depleted", f"{outcomes(device)}")

    device, _, _ = fresh("""\
  - name: P
    all: {ratio: 1}
""")
    device.poke("L")
    device.advance(2)
    device.reboot()
    check("a pellet in the well is still there after a reboot",
          status(device)["well"] == "1")
    device.advance(5)
    device.retrieve()
    check("and its retrieval is logged, timed from the boot",
          rows(device, "retrieve")[-1]["retrieval_ms"] == "5000")

    device, _, _ = fresh("""\
  - name: P
    all: {ratio: 1}
""")
    device.card.failing = True
    device.poke("L")
    check("a dead card is reported", any(line.startswith("ERR,SD") for line in device.output))
    check("and STATUS says so", status(device)["sd"] == "err")
    device.advance(2)
    check("the device keeps feeding", status(device)["well"] == "1")
    device.card.failing = False

    print("\n--- the clock ---")
    device, compiled, _ = fresh("""\
  - name: One
    until: {after: 2h}
  - name: Two
""")
    check("a big jump during a run is refused",
          device.command(f"TIME:{dev.iso(at(25, 9))}")[0].startswith("ERR,TIME_JUMP"))
    reply = device.command(f"TIME:{dev.iso(at(20, 9, 0, 40))}")[-1]
    check("a small correction is applied and reported",
          reply.startswith("TIME,before=") and "after=2026-10-20T09:00:40" in reply)
    check("and logged", rows(device, "clock")[-1]["detail"].startswith("from="))
    device.reboot(lose_clock=True)
    device.advance(3 * 3600)
    check("with the clock lost, time conditions wait",
          status(device)["phase_name"] == "One" and status(device)["clock"] == "unset")
    check("and rows carry no time", rows(device)[-1]["time"] == "")
    device.command(f"TIME:{dev.iso(at(20, 12, 1))}")
    check("setting it catches up", status(device)["phase_name"] == "Two"
          and rows(device, "phase")[-1]["detail"]
          == f"Two;catch-up;start={dev.iso(at(20, 11))}")
    device.command(f"TIME!:{dev.iso(at(22, 12))}")
    check("TIME! overrides the jump limit", status(device)["time"] == dev.iso(at(22, 12)))

    print("\n--- FNT-driven changes ---")
    device, _, _ = fresh("""\
  - name: Train
    until: [{rewards: 100, devices: all}, {after: 4d}]
  - name: Next
""")
    check("a stale GOTO is refused", device.command("GOTO:1,2")[0].startswith("ERR,GOTO_STALE"))
    check("GOTO moves from the named phase", device.command("GOTO:0,1")[-1] == "PHASE,n=1,name=Next")
    check("repeating it does nothing", device.command("GOTO:0,1")[0].startswith("ERR,GOTO_STALE"))
    check("it's logged", rows(device, "phase")[-1]["detail"] == "Next;goto")

    print("\n--- editing a running schedule ---")
    device, compiled, _ = fresh("""\
  - name: A1
    until: {after: 1d}
    all: {ratio: 5}
  - name: A2
""")
    edited = compile_one("""\
  - name: A1
    until: {after: 2h}
    all: {ratio: 2}
  - name: A2
""")
    device.advance(3600)
    device.poke("L")
    _, reply = upload(device, edited.schedules["A"], phase=0)
    check("an edit applies at once", reply.endswith("slot=now")
          and sch.summarize(device.rule()) == "FR2 · left")
    check("counts in the phase carry over", status(device)["phase_left"] == "1")
    device.advance(3600 + 1)
    check("its new end time counts from the original phase start",
          rows(device, "phase")[-1]["time"] == dev.iso(at(20, 11)))
    _, reply = upload(device, edited.schedules["A"], phase=0)
    check("an edit for a phase the device has left is refused",
          reply.startswith("ERR,SCHED_STALE,phase=1"))

    print("\n--- stopping ---")
    device, _, _ = fresh("""\
  - name: Fed
    until: {after: 1h}
    all: {active: none}
    daily:
      - {from: "10:00", to: "11:00", all: {free_feed: 1}}
  - name: After
""")
    device.command("RUN:END")
    device.advance(2 * 3600)
    check("after RUN:END the phase doesn't change", status(device)["phase_name"] == "Fed")
    check("but its windows still feed", len(rows(device, "dispense")) >= 1)
    check("RUN:END twice is harmless", device.command("RUN:END") == ["RUN,run=RUN1,ended=1"])

    print("\n--- re-reading rows ---")
    out = device.command("READ:RUN1,3")
    file_rows = device.card.files["RUN1.CSV"].splitlines()[1:]
    check("READ replays from a seq onward",
          [line[4:] for line in out[:-1]] == file_rows[2:])
    check("and says where it ended", out[-1] == f"READ_END,run=RUN1,last={len(file_rows)}")
    check("an unknown run is reported", device.command("READ:NOPE,1")[0].startswith("ERR,NO_FILE"))

    print("\n--- an unattended week, at random ---")
    weekly = compile_one("""\
  - name: Hab
    until: {at: "19:00", day: 2}
    all: {active: none, free_feed: 1}
  - name: Train
    until: [{rewards: 40}, {after: 2d}]
    all: {active: both, cue: {reward: [click]}}
  - name: Rot
    until: {after: 3d}
    rotate: {every: 1d, at: "19:00", mode: cycle, assign: {A: rich, B: poor}}
  - name: End
    all: {active: none, free_feed: 1}
""", devices="[A, B]", extra="rules:\n  rich: {active: both, quota: 20, quota_period: 1h}\n"
                             "  poor: {active: both, ratio: 5}\n")
    rng = random.Random(5)
    device = MAKE(now=at(20, 9), seed=5)
    start(device, weekly.schedules["A"])
    elapsed = 0.0
    while elapsed < 8 * 86400:
        roll = rng.random()
        if roll < 0.6:
            ms = rng.randint(20, 900)
            device.poke(rng.choice("LR"), ms=ms)
            elapsed += ms / 1000
        elif roll < 0.85:
            device.retrieve()
        elif roll < 0.9995:
            seconds = rng.randint(1, 900)
            device.advance(seconds)
            elapsed += seconds
        else:
            seconds = rng.randint(60, 4 * 3600)
            device.power_off(seconds)
            elapsed += seconds
    data = rows(device)
    seqs = [int(r["seq"]) for r in data]
    check("seq never repeats or skips", seqs == list(range(1, len(seqs) + 1)))
    dispensed = len(rows(device, "dispense"))
    check("the pellet total matches the dispense rows",
          int(data[-1]["pellets"]) == dispensed)
    check("left and right totals match the poke rows",
          int(data[-1]["left"]) == sum(r["side"] == "L" for r in rows(device, "poke"))
          and int(data[-1]["right"]) == sum(r["side"] == "R" for r in rows(device, "poke")))
    phases = [r["detail"].split(";")[0] for r in rows(device, "phase")]
    check("it reached the last phase through every phase in order",
          phases == ["Train", "Rot 1", "Rot 2", "Rot 3", "End"], f"{phases}")
    rot_starts = [r["time"][11:16] for r in rows(device, "phase")
                  if r["detail"].startswith(("Rot 2", "Rot 3", "End"))]
    check("rotation boundaries stayed on 19:00 through power cuts",
          rot_starts == ["19:00"] * 3, f"{rot_starts}")
    check("every row reached the host",
          sent_rows(device) == device.card.files["RUN1.CSV"].splitlines()[1:])

    passed = sum(RESULTS)
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
