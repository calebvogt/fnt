"""Protocol files and device schedules: what a run will do, decided before it starts.

The compiler is the only thing between a protocol someone wrote and weeks of
behaviour on a device that no longer asks the PC. Every check here is a way that
gap could silently change an experiment: a phase inheriting a ratio it was never
given, a rotation that differs after a crash, a clock time YAML read as a
number, a schedule line too long for the firmware's buffer.

The examples in docs/fed3/design.md are compiled here too, so the spec and the
code can't drift apart.
"""
import io
import os
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fnt.fed3 import protocol as proto      # noqa: E402
from fnt.fed3 import schedule as sch        # noqa: E402
from fnt.fed3 import cli                    # noqa: E402

RESULTS = []
TZ = timezone(timedelta(hours=-7))          # a standard-time zone, no DST


def check(label, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{f' — {detail}' if detail else ''}")


def raises(fn, *args):
    try:
        fn(*args)
    except (ValueError, proto.ProtocolError) as exc:
        return str(exc) or True
    return False


def problems_of(text):
    """Problems for a protocol, as (line, level, message)."""
    _, problems = proto.check(text)
    return [(p.line, p.level, p.message) for p in problems]


def has_error(text, fragment, line=None):
    found = [(l, m) for l, level, m in problems_of(text)
             if level == "error" and fragment in m]
    if not found:
        return False
    return line is None or any(l == line for l, _ in found)


def protocol_text(phases, extra=""):
    return (f"format: {proto.FORMAT}\nname: Test\ndevices: [A, B]\n{extra}"
            f"phases:\n{phases}")


def at(day, hour, minute=0, second=0):
    return datetime(2026, 10, day, hour, minute, second, tzinfo=TZ)


def doc_blocks():
    with open(os.path.join(ROOT, "docs", "fed3", "design.md"), encoding="utf-8") as f:
        doc = f.read()
    blocks, lang, body = [], None, None
    for line in doc.splitlines(keepends=True):
        if line.startswith("```"):
            if body is None:
                lang, body = line[3:].strip(), []
            else:
                blocks.append((lang, "".join(body)))
                body = None
        elif body is not None:
            body.append(line)
    protocols = [b for lang, b in blocks
                 if lang == "yaml" and b.startswith("format:") and "…" not in b]
    schedules = [b for lang, b in blocks if not lang and b.startswith("NAME ")]
    return protocols, schedules


def main():
    print("--- loading the compiler does not load Qt ---")
    check("no PyQt5 after importing protocol, schedule and cli",
          "PyQt5" not in sys.modules,
          "fnt-fed and the tests must run on a machine with no display")

    print("\n--- rule values are validated and canonical ---")
    check("a ratio is normalized", sch.parse_value("ratio", "007") == "7")
    check("a ratio of 0 is refused", raises(sch.parse_value, "ratio", "0"))
    check("an unknown enum value is refused",
          raises(sch.parse_value, "active", "middle"))
    check("an unknown key is refused", raises(sch.parse_value, "light.bar", "x"))

    print("\n--- cues ---")
    check("a cue is canonical",
          sch.parse_cue("tone:04000:200+flash:poke:0:0:30:0:500")
          == "tone:4000:200+flash:poke:0:0:30:0:500")
    check("at most three actions",
          raises(sch.parse_cue, "click+flash:bar:1:1:1:1:9+bnc:1:1:1+flash:poke_l:1:1:1:1:9"))
    check("two sounds at once are refused",
          raises(sch.parse_cue, "click+tone:4000:200"),
          "the buzzer plays one sound; the first would be cut off")
    check("two flashes on the same LEDs are refused",
          raises(sch.parse_cue, "flash:pokes:1:1:1:1:9+flash:poke_l:1:1:1:1:9"))
    check("bar and a poke can flash together",
          sch.parse_cue("flash:bar:1:1:1:1:9+flash:poke:1:1:1:1:9"))
    check("a BNC pulse wider than its period is refused",
          raises(sch.parse_cue, "bnc:50:20:10"))
    check("an unknown action is refused", raises(sch.parse_cue, "buzz:1"))

    print("\n--- conditions: when a phase ends ---")
    for until in (sch.Until("after", seconds=3600), sch.Until("at", clock=1140),
                  sch.Until("at", clock=420, day=3), sch.Until("rewards", count=50),
                  sch.Until("pokes", count=9, side="left"), sch.Until("goto")):
        check(f"'{until.text()}' round-trips", sch.parse_until(until.text()) == until)
    check("a malformed condition is refused", raises(sch.parse_until, "after=10 at=19:00"))

    t0 = at(20, 9)
    seven_pm = sch.Until("at", clock=19 * 60)
    check("at 19:00 from 09:00 is the same day",
          sch.due_time(seven_pm, at(20, 9), t0) == at(20, 19))
    check("at 19:00 from exactly 19:00 is the next day",
          sch.due_time(seven_pm, at(20, 19), t0) == at(21, 19),
          "otherwise a daily rotation ends every phase the moment it starts")
    check("at 19:00 from 19:00:01 is the next day",
          sch.due_time(seven_pm, at(20, 19, 0, 1), t0) == at(21, 19))
    check("at 19:00 on day 3 is counted from t0's date",
          sch.due_time(sch.Until("at", clock=1140, day=3), at(20, 15), t0) == at(22, 19))
    check("a day already past ends the phase as it starts",
          sch.due_time(sch.Until("at", clock=1140, day=1), at(21, 8), t0) == at(21, 8))
    check("after counts from the phase start",
          sch.due_time(sch.Until("after", seconds=90), at(20, 9), t0)
          == at(20, 9, 1, 30))
    check("count conditions have no due time",
          sch.due_time(sch.Until("rewards", count=5), at(20, 9), t0) is None)

    # Catch-up: each phase starts at its predecessor's due time, so a device
    # that was off for days replays exactly the boundaries it missed.
    start, boundaries = at(20, 16, 42), []
    for _ in range(5):
        start = sch.due_time(seven_pm, start, t0)
        boundaries.append(start)
    check("chained daily boundaries land on consecutive 19:00s",
          boundaries == [at(20 + i, 19) for i in range(5)])

    print("\n--- daily windows ---")
    night = sch.Window(19 * 60, 7 * 60)
    check("a window across midnight is open at 23:00", night.active_at(at(20, 23)))
    check("and at 06:59", night.active_at(at(21, 6, 59)))
    check("and closed at 07:00", not night.active_at(at(21, 7)))
    check("and closed at noon", not night.active_at(at(21, 12)))
    check("overlap across midnight is found",
          sch.windows_overlap(night, sch.Window(6 * 60, 8 * 60)))
    check("touching windows don't overlap",
          not sch.windows_overlap(night, sch.Window(7 * 60, 19 * 60)))

    print("\n--- the examples in the design doc ---")
    protocols, schedules = doc_blocks()
    check("the doc has three full examples", len(protocols) == 3, f"{len(protocols)}")
    compiled_examples = []
    for text in protocols:
        compiled, problems = proto.check(text)
        errors = [p.message for p in problems if p.level == "error"]
        name = text.split("name: ", 1)[1].split("\n", 1)[0]
        check(f"'{name}' compiles", compiled is not None and not errors,
              "; ".join(errors))
        compiled_examples.append(compiled)
    by_name = {c.protocol.name: c for c in compiled_examples if c}
    stepped = by_name.get("FR1 to FR5, both pokes")
    compiled_doc = [b for b in schedules if b.startswith("NAME FR1 to FR5")]
    check("example 1 compiles to exactly the text the doc shows",
          stepped and compiled_doc and stepped.schedules["FED"].text() == compiled_doc[0])
    four = by_name.get("Four-patch competition")
    sample = [b for b in schedules if b.startswith("NAME Four-patch")]
    if four and sample:
        shown = [l for l in sample[0].splitlines() if l != "…"]
        actual = four.schedules["P1"].lines()[:len(shown)]
        check("the Part 2 sample matches P1's compiled schedule", shown == actual,
              "" if shown == actual else f"\n      doc: {shown}\n      got: {actual}")
    else:
        check("the Part 2 sample matches P1's compiled schedule", False, "missing")

    print("\n--- schedule text: what the device would parse ---")
    for compiled in filter(None, compiled_examples):
        for device, schedule in compiled.schedules.items():
            text = schedule.text()
            parsed = sch.parse_schedule(text)
            check(f"{compiled.protocol.name} / {device}: text -> parse -> text "
                  "is unchanged", parsed.text() == text)
            check(f"{compiled.protocol.name} / {device}: every line fits",
                  all(len(l) <= sch.MAX_LINE for l in schedule.lines()))
    check("the crc is CRC-32 of the text",
          stepped.schedules["FED"].crc()
          == __import__("zlib").crc32(stepped.schedules["FED"].text().encode()))

    def bad(text):
        try:
            sch.parse_schedule(text)
        except sch.ScheduleError as exc:
            return exc
        return None

    exc = bad("NAME x\nSET ratio=2\n")
    check("SET before PHASE is refused, with its line", exc and exc.line_no == 2)
    check("an unknown key is refused", bad("NAME x\nPHASE a\nSET light.bar=1\n"))
    check("a key set twice in a phase is refused",
          bad("NAME x\nPHASE a\nSET ratio=2\nSET ratio=3\n"))
    check("overlapping windows are refused",
          bad("NAME x\nPHASE a\nWINDOW 19:00-07:00 free_feed=1\n"
              "WINDOW 06:00-08:00 ratio=2\n"))
    check("windows with the same times merge into one",
          len(sch.parse_schedule("NAME x\nPHASE a\nWINDOW 19:00-07:00 free_feed=1\n"
                                 "WINDOW 19:00-07:00 ratio=2\n").phases[0].windows) == 1)
    check("too many end conditions are refused",
          bad("NAME x\nPHASE a\n" + "UNTIL after=60\n" * 7))
    check("a schedule without phases is refused", bad("NAME x\n"))

    # Every key set, with the longest legal cues: still every line fits.
    long_cue = "tone:20000:10000+flash:pokes:255:255:255:255:60000+bnc:1:500:1000"
    busy = sch.Phase("Busy", rule={**sch.DEFAULTS, **{
        "active": "both", "count": "per_port", "ratio": "999",
        "ratio_kind": "random", "timeout": "3600", "min_poke_ms": "5000",
        "reward": "none", "quota": "9999", "quota_period": "604800",
        "free_feed": "1", "refill_delay": "3600",
        **{k: long_cue for k in sch.KEY_ORDER if k.startswith("cue.")}}})
    busy_schedule = sch.Schedule("x", [busy])
    check("a phase with every key at its longest still fits the device",
          not busy_schedule.problems()
          and sch.parse_schedule(busy_schedule.text()).text() == busy_schedule.text())

    print("\n--- each phase means the same thing wherever it is ---")
    text = protocol_text("""\
  - name: Five
    until: {after: 1h}
    all: {ratio: 5, timeout: 30s}
  - name: Plain
    all: {active: both}
""")
    compiled, _ = proto.check(text)
    plain = compiled.schedules["A"].phases[1].rule
    check("a phase does not inherit the previous phase's ratio",
          plain["ratio"] == "1" and plain["timeout"] == "0")
    check("durations become seconds",
          compiled.schedules["A"].phases[0].rule["timeout"] == "30")

    print("\n--- rules: names, use, and merge order ---")
    text = protocol_text("""\
  - name: One
    until: {after: 1d}
    all: base
    devices: {B: {use: base, ratio: 3, cue: {counted: [click]}}}
  - name: Two
    all: {active: none, free_feed: true}
""", extra="""\
cues:
  ding: [{tone: {hz: 3000, ms: 100}}]
rules:
  base: {active: both, cue: {reward: ding}}
""")
    compiled, problems = proto.check(text)
    a, b = (compiled.schedules[d].phases[0].rule for d in "AB")
    check("a named rule applies", a["active"] == "both" and a["cue.reward"] == "tone:3000:100")
    check("a device override keeps the rule it uses",
          b["ratio"] == "3" and b["cue.reward"] == "tone:3000:100"
          and b["cue.counted"] == "click")
    check("other devices are untouched by it", a["ratio"] == "1" and a["cue.counted"] == "none")
    check("true becomes 1", compiled.schedules["A"].phases[1].rule["free_feed"] == "1")

    print("\n--- rotation ---")
    rotating = protocol_text("""\
  - name: Rot
    until: {after: 4d}
    rotate:
      every: 1d
      at: "19:00"
      mode: MODE
      assign: {A: rich, B: poor}
  - name: End
    all: {active: none, free_feed: 1}
""", extra="""\
rules:
  rich: {ratio: 1}
  poor: {ratio: 5}
""")
    cycle = proto.compile_protocol(proto.load(rotating.replace("MODE", "cycle")))
    names = [p.name for p in cycle.phases]
    check("a rotation expands into one phase per period",
          names == ["Rot 1", "Rot 2", "Rot 3", "Rot 4", "End"], f"{names}")
    ratios = [[cycle.schedules[d].phases[i].rule["ratio"] for i in range(4)] for d in "AB"]
    check("cycle alternates the assignment", ratios == [["1", "5", "1", "5"],
                                                        ["5", "1", "5", "1"]], f"{ratios}")
    check("each period ends at the next 19:00",
          all(p.untils == [sch.Until("at", clock=1140)] for p in cycle.phases[:4]))
    check("every device gets the same phases",
          [p.name for p in cycle.schedules["A"].phases]
          == [p.name for p in cycle.schedules["B"].phases])

    shuffled = proto.load(rotating.replace("MODE", "shuffle"))
    check("a shuffle won't compile without a seed",
          raises(proto.compile_protocol, shuffled))
    first = proto.compile_protocol(shuffled, seed=42)
    again = proto.compile_protocol(shuffled, seed=42)
    check("the same seed gives the same schedules",
          all(first.schedules[d].text() == again.schedules[d].text() for d in "AB"),
          "a resumed run must rebuild exactly what the devices hold")
    labels = [sorted(p.labels.values()) for p in first.phases[:4]]
    check("each period deals out the same set of rules",
          all(l == ["poor", "rich"] for l in labels))
    check("the first period is the assignment as written",
          first.phases[0].labels == {"A": "rich", "B": "poor"})
    seeds_differ = any(
        proto.compile_protocol(shuffled, seed=s).schedules["A"].text()
        != first.schedules["A"].text() for s in range(1, 20))
    check("other seeds give other sequences", seeds_differ)

    check("a rotation must last whole periods",
          has_error(rotating.replace("4d", "36h").replace("MODE", "cycle"),
                    "whole number"))
    check("rotate.at needs every: 1d",
          has_error(rotating.replace("every: 1d", "every: 12h").replace("MODE", "cycle"),
                    "every: 1d"))
    check("an unknown rule in assign is found",
          has_error(rotating.replace("B: poor", "B: pour").replace("MODE", "cycle"),
                    "did you mean 'poor'"))

    print("\n--- conditions that need FNT ---")
    text = protocol_text("""\
  - name: Train
    until: [{rewards: 100, devices: all}, {manual: true}, {after: 4d}]
    all: {active: both}
  - name: Next
    all: {active: both}
""")
    compiled, problems = proto.check(text)
    check("FNT conditions become one goto, in place",
          compiled.phases[0].untils == [sch.Until("goto"),
                                        sch.Until("after", seconds=4 * 86400)])
    check("and are kept for FNT to evaluate",
          [c.kind for c in compiled.phases[0].fnt] == ["rewards", "manual"])
    check("with a local fallback there is no warning",
          not any(level == "warning" for _, level, _ in
                  [(p.line, p.level, p.message) for p in problems]))
    only_fnt = text.replace(", {after: 4d}", "")
    check("without one, the phase is flagged",
          any(level == "warning" and "fallback" in m
              for _, level, m in problems_of(only_fnt)))

    print("\n--- daily windows in a protocol ---")
    text = protocol_text("""\
  - name: Fed
    all: {active: none}
    daily:
      - {from: "19:00", to: "21:00", all: {free_feed: 1}}
      - {from: "22:00", to: "23:00", devices: {B: {ratio: 3}}}
""")
    compiled, _ = proto.check(text)
    a_windows = compiled.schedules["A"].phases[0].windows
    b_windows = compiled.schedules["B"].phases[0].windows
    check("a window for all devices reaches every device",
          a_windows[0].settings == {"free_feed": "1"})
    check("a window for one device reaches only that device",
          len(a_windows) == 1 and len(b_windows) == 2
          and b_windows[1].settings == {"ratio": "3"})
    check("the rule in force follows the window",
          compiled.schedules["A"].phases[0].rule_at(at(20, 20))["free_feed"] == "1"
          and compiled.schedules["A"].phases[0].rule_at(at(20, 21))["free_feed"] == "0")

    print("\n--- mistakes are reported where they are ---")
    good = protocol_text("""\
  - name: Only
    all: {active: both}
""")
    check("the minimal protocol is clean", problems_of(good) == [])
    check("an unquoted clock time is explained",
          has_error(protocol_text("""\
  - name: Only
    daily:
      - {from: 19:00, to: "21:00", all: {free_feed: 1}}
"""), "quotes", line=7))
    check("a misspelled key gets a suggestion, on its line",
          has_error(protocol_text("""\
  - name: Only
    all: {ratoi: 2}
"""), "did you mean 'ratio'", line=6))
    check("asking for steady lights says how lights work now",
          has_error(protocol_text("""\
  - name: Only
    all: {light: {bar: white}}
"""), "no steady lights"))
    check("pellets as a condition points to rewards",
          has_error(protocol_text("""\
  - name: One
    until: {pellets: 5}
  - name: Two
"""), "use rewards"))
    check("the last phase can't have an end",
          has_error(protocol_text("""\
  - name: Only
    until: {after: 1h}
"""), "last phase"))
    check("a phase before the last must have one",
          has_error(protocol_text("""\
  - name: One
  - name: Two
"""), "never ends"))
    check("phase names are unique",
          has_error(protocol_text("""\
  - name: Same
    until: {after: 1h}
  - name: Same
"""), "two phases"))
    check("after needs a unit",
          has_error(protocol_text("""\
  - name: One
    until: {after: 48}
  - name: Two
"""), "duration"))
    check("a zero-length after is refused, not dropped",
          has_error(protocol_text("""\
  - name: One
    until: {after: 0s}
  - name: Two
"""), "between 1s"))
    check("a zero rotation period is refused",
          has_error(protocol_text("""\
  - name: One
    until: {after: 1d}
    rotate: {every: 0s, assign: {A: x, B: x}}
  - name: Two
""", extra="rules:\n  x: {ratio: 2}\n"), "at least 1m"))
    check("an unknown device is found",
          has_error(protocol_text("""\
  - name: Only
    devices: {C: {ratio: 2}}
"""), "not in devices"))
    check("rules that use each other in a loop are found",
          has_error(protocol_text("""\
  - name: Only
    all: x
""", extra="rules:\n  x: {use: y}\n  y: {use: x}\n"), "loop"))
    check("a bad cue is reported with the reason",
          has_error(protocol_text("""\
  - name: Only
    all: {cue: {reward: [click, {tone: {hz: 4000, ms: 50}}]}}
"""), "one at a time"))
    check("overlapping daily windows are found",
          has_error(protocol_text("""\
  - name: Only
    daily:
      - {from: "19:00", to: "07:00", all: {free_feed: 1}}
      - {from: "06:00", to: "08:00", all: {ratio: 2}}
"""), "overlap"))
    check("a duplicated key is found",
          has_error(protocol_text("""\
  - name: Only
    all: {ratio: 2}
    all: {ratio: 3}
"""), "twice", line=7))
    check("broken YAML gives its line",
          has_error(protocol_text("  - name: [unclosed\n"), "not valid YAML"))
    check("a protocol too long for the device is refused",
          has_error(protocol_text("""\
  - name: Hourly
    until: {after: 11d}
    rotate: {every: 1h, assign: {A: x, B: x}}
  - name: End
""", extra="rules:\n  x: {ratio: 2}\n"), "phases"))
    check("non-ASCII names are refused before they reach a device",
          has_error(protocol_text("""\
  - name: Fütterung
"""), "ASCII"))

    print("\n--- preview ---")
    t0 = at(20, 9)
    rows = proto.preview(four, t0)
    check("habituation ends at 19:00 on day 3", rows[0].end == at(22, 19))
    check("training starts then", rows[1].start == at(22, 19))
    check("and its end is unknown, but bounded by the 4-day fallback",
          rows[1].end is None and rows[1].latest_end == at(26, 19))
    check("so competition starts by then at the latest",
          rows[2].start is None and rows[2].latest_start == at(26, 19))
    check("training is marked as needing FNT", rows[1].needs_fnt)
    rows = proto.preview(stepped, t0)
    check("a count-only phase leaves later starts unknown",
          rows[1].start is None and rows[1].latest_start is None)
    text = proto.preview_text(four, t0, days=1)
    check("the text preview stops at the horizon", "stopping at the 1-day mark" in text)

    print("\n--- the fnt-fed command ---")
    with tempfile.TemporaryDirectory() as tmp:
        good_path = os.path.join(tmp, "good.yaml")
        bad_path = os.path.join(tmp, "bad.yaml")
        with open(good_path, "w", encoding="utf-8") as f:
            f.write(protocols[1])
        with open(bad_path, "w", encoding="utf-8") as f:
            f.write(protocol_text("  - name: Only\n    all: {ratoi: 2}\n"))
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code_good = cli.main(["check", good_path])
            code_bad = cli.main(["check", bad_path])
            cli.main(["compile", good_path, "--device", "P1", "--seed", "3"])
            cli.main(["preview", good_path, "--seed", "3", "--start",
                      "2026-10-20T09:00", "--days", "5"])
        check("check passes a good file", code_good == 0)
        check("and fails a bad one, naming file and line",
              code_bad == 1 and f"{bad_path}:6:" in err.getvalue())
        check("compile prints the schedule", "PHASE Competition 1" in out.getvalue())
        check("preview prints the timeline", "Habituation" in out.getvalue())

    print("\n--- summaries ---")
    def summary(**keys):
        return sch.summarize({**sch.DEFAULTS, **keys})
    check("FR1 left", summary() == "FR1 · left")
    check("PR", summary(ratio_kind="progressive", active="right") == "PR · right")
    check("free feed", summary(active="none", free_feed="1") == "free feed")
    check("quota and timeout",
          summary(ratio="5", active="both", timeout="30", quota="30",
                  quota_period="3600") == "FR5 · both · TO30s · Q30/1h")
    check("extinction", summary(reward="none") == "FR1 no reward · left")

    passed = sum(RESULTS)
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
