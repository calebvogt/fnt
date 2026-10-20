"""``fnt-fed``: check, compile and preview FED3 protocol files without hardware.

    fnt-fed check protocol.yaml
    fnt-fed compile protocol.yaml [--device P1] [--seed N]
    fnt-fed preview protocol.yaml [--start 2026-10-20T19:00] [--days 21] [--seed N]

Times are in the PC's standard time (daylight saving ignored), as on the devices.
"""
import argparse
import random
import sys
from datetime import datetime

from . import protocol as proto
from . import schedule as sch


def _read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _report(path, problems):
    for problem in problems:
        print(problem.format(path), file=sys.stderr)
    return any(p.level == "error" for p in problems)


def _load(path, seed):
    """Load and compile, printing problems. Returns (compiled, seed) or None."""
    try:
        protocol = proto.load(_read(path))
    except proto.ProtocolError as exc:
        _report(path, exc.problems)
        return None
    _report(path, protocol.warnings)
    if protocol.needs_seed and seed is None:
        seed = random.randrange(1, 10**9)
        print(f"# shuffled rotation: using --seed {seed}", file=sys.stderr)
    try:
        return proto.compile_protocol(protocol, seed), seed
    except proto.ProtocolError as exc:
        _report(path, exc.problems)
        return None


def cmd_check(args):
    compiled, problems = proto.check(_read(args.file))
    if _report(args.file, problems):
        return 1
    fnt_phases = [p.name for p in compiled.phases if p.fnt]
    print(f"{args.file}: ok · {sch.plural(len(compiled.phases), 'phase')} · "
          f"{sch.plural(len(compiled.protocol.devices), 'device')}")
    if fnt_phases:
        print(f"phases that need FNT running to end: {', '.join(fnt_phases)}")
    return 0


def cmd_compile(args):
    result = _load(args.file, args.seed)
    if result is None:
        return 1
    compiled, _ = result
    devices = [args.device] if args.device else compiled.protocol.devices
    for device in devices:
        if device not in compiled.schedules:
            print(f"no device '{device}' in this protocol", file=sys.stderr)
            return 1
        schedule = compiled.schedules[device]
        if len(devices) > 1:
            print(f"# {device}  crc={schedule.crc():08x}")
        sys.stdout.write(schedule.text())
        if len(devices) > 1:
            print()
    return 0


def cmd_preview(args):
    result = _load(args.file, args.seed)
    if result is None:
        return 1
    compiled, _ = result
    zone = sch.standard_timezone()
    if args.start:
        try:
            t0 = datetime.fromisoformat(args.start)
        except ValueError:
            print(f"--start: '{args.start}' is not an ISO date-time", file=sys.stderr)
            return 1
        t0 = t0.replace(tzinfo=zone) if t0.tzinfo is None else t0.astimezone(zone)
    else:
        t0 = datetime.now(zone).replace(second=0, microsecond=0)
    print(proto.preview_text(compiled, t0, args.days))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="fnt-fed", description="Check, compile and preview FED3 protocol files.")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="find errors in a protocol file")
    check.add_argument("file")
    check.set_defaults(func=cmd_check)

    comp = sub.add_parser("compile", help="print each device's schedule")
    comp.add_argument("file")
    comp.add_argument("--device", help="only this device")
    comp.add_argument("--seed", type=int, help="seed for shuffled rotations")
    comp.set_defaults(func=cmd_compile)

    prev = sub.add_parser("preview", help="print the timeline")
    prev.add_argument("file")
    prev.add_argument("--start", help="t0 as an ISO date-time (default: now)")
    prev.add_argument("--days", type=int, help="stop after this many days")
    prev.add_argument("--seed", type=int, help="seed for shuffled rotations")
    prev.set_defaults(func=cmd_preview)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except OSError as exc:
        print(f"fnt-fed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
