#!/usr/bin/env python3
"""End-to-end check of the FED3 stack against an emulated device.

Runs the real :mod:`fnt.fed3` serial, transfer and mirror code against a FED3
firmware emulator on a pseudo-terminal, so the data path can be verified without
a device on the bench.

Two firmware profiles are exercised, because they behave differently in ways
that have caused silent data loss:

``legacy``
    Pre-2.0 firmware. Answers ``PING`` without a ``FW:`` field, has no ``STATUS``
    or ``FSIZE``, and takes everything after ``GET_FILE:`` as a filename — so a
    ranged request asks for a file named ``"<name>,0"`` and fails. FNT must
    *refuse* such a device outright; it is not driven in a reduced mode.

``modern``
    Firmware 2.0. Ranged transfers.

Two interleaving cases are checked on top of that, because firmware 2.0 streams
files from ``loop()`` and so *can* emit a line between two chunks of payload:

*Line-aligned* injection is recognised as device chatter, routed to the event
handler, and excluded from the file, which still verifies against the device CRC.

*Mid-line* injection cannot be recognised — the fragment fuses onto a partial CSV
row — and the guarantee is weaker but still sound: the CRC rejects the chunk, no
corrupt bytes are written, and the next pull succeeds. This is why firmware 2.0
queues events for the duration of a transfer instead of relying on the host to
sort it out.

Run it with the FED3 disconnected; it touches no real serial port::

    python tests/fed3_loopback.py
"""

import os
import sys

if not hasattr(os, "openpty"):      # pty is POSIX-only
    # A skip, not a failure: the emulator needs a pty, and Windows has none.
    print("SKIPPED — fed3_loopback needs a POSIX pty; run it on Linux or macOS.")
    raise SystemExit(0)

import pty
import threading
import zlib

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QCoreApplication, QEventLoop, QTimer  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fnt.fed3 import fed_protocol as proto           # noqa: E402
from fnt.fed3.fed_export import Fed3Transfer          # noqa: E402
from fnt.fed3.fed_mirror import DeviceMirror          # noqa: E402
from fnt.fed3.fed_serial import Fed3Link              # noqa: E402

FW_VERSION = "2.0"

CARD = {
    "FED004_082426_00.CSV": (
        b"MM:DD:YYYY hh:mm:ss,Library_Version,Session_type,Device_Number,"
        b"Event,Left_Poke_Count,Right_Poke_Count,Pellet_Count\r\n"
        + b"".join(b"08:24:2026 10:%02d:00,1.16.3,FR1,4,Left,%d,0,%d\r\n" % (i, i, i // 3)
                   for i in range(40))
    ),
    "FED004_082426_01.CSV": b"MM:DD:YYYY hh:mm:ss,Event\r\n08:24:2026 11:00:00,Pellet\r\n",
}

# The log the emulated device is writing to now, and one it has finished with.
# LIST_FILES returns them in creation order, so the live file is deliberately
# *last* in the listing — the ordering that used to leave live data behind the
# whole back catalogue.
ARCHIVED_FILE = "FED004_082426_00.CSV"
LIVE_FILE = "FED004_082426_01.CSV"


class FakeFed3(threading.Thread):
    """A FED3 firmware emulator speaking the wire protocol over a PTY."""

    daemon = True

    def __init__(self, profile="modern", inject=None):
        super().__init__()
        self.profile = profile
        # None, "line" or "midline": where to emit a live event line during a
        # payload, reproducing a nosepoke that lands mid-download.
        self.inject = inject
        self.master, self._slave = pty.openpty()
        self.port = os.ttyname(self._slave)
        # The slave fd is deliberately kept open. Closing it after taking the
        # name leaves the master seeing a hangup the moment pyserial closes its
        # own handle, which makes the emulator look like a dead device.
        self._running = True
        self._buf = b""

    def stop(self):
        self._running = False
        self.join(timeout=2)
        for fd in (self.master, self._slave):
            try:
                os.close(fd)
            except OSError:
                pass

    # --- wire helpers ----------------------------------------------------

    def _write(self, data):
        if isinstance(data, str):
            data = data.encode()
        os.write(self.master, data)

    def _line(self, text):
        self._write(text + "\r\n")

    # --- main loop -------------------------------------------------------

    def run(self):
        import select
        while self._running:
            r, _, _ = select.select([self.master], [], [], 0.1)
            if not r:
                continue
            try:
                chunk = os.read(self.master, 4096)
            except OSError:
                return
            self._buf += chunk
            while b"\n" in self._buf:
                raw, _, self._buf = self._buf.partition(b"\n")
                command = raw.decode("utf-8", "ignore").strip()
                if command:
                    self._handle(command)

    def _handle(self, command):
        if command == "PING":
            if self.profile == "legacy":
                self._line("PONG_FED3,ID:4")
            else:
                self._line(f"PONG_FED3,ID:4,FW:{FW_VERSION}")
        elif command == "STATUS":
            if self.profile == "legacy":
                return              # legacy firmware answers with silence
            self._line(f"STATUS,ID:4,FW:{FW_VERSION},TIME:2026-08-24T10:00:00,"
                       "MODE:1,SESSION:FR1,FR:1,L:40,R:0,P:13,FILE:x.CSV")
        elif command == "LIST_FILES":
            for name, body in CARD.items():
                self._line(f"FILE:{name},{len(body)}")
            self._line("END_LIST")
        elif command.startswith("GET_FILE:"):
            self._get_file(command[len("GET_FILE:"):])
        elif command.startswith("FSIZE:"):
            if self.profile == "legacy":
                return
            name = command[len("FSIZE:"):].strip()
            body = CARD.get(name)
            self._line(f"FSIZE:{name},{len(body)}" if body
                       else f"ERROR:FILE_NOT_FOUND:{name}")
        else:
            self._line(f"ERROR:UNKNOWN_COMMAND:{command}")

    def _injection_point(self, payload):
        """Byte offset to splice an event line at, or None."""
        if self.inject is None or len(payload) < 400:
            return None
        if self.inject == "line":
            # Immediately after a CRLF, so the host sees a whole "EVT,..." line.
            return payload.index(b"\r\n", 300) + 2
        return 300          # mid-row: the fragment fuses onto a partial CSV line

    def _get_file(self, args):
        offset = 0
        if self.profile == "legacy":
            # The bug this profile exists to reproduce: no comma parsing, so a
            # ranged request names a file that does not exist.
            name = args.strip()
        else:
            name, _, rest = args.partition(",")
            name = name.strip()
            if rest.strip():
                offset = int(rest)

        body = CARD.get(name)
        if body is None:
            self._line(f"ERROR:FILE_NOT_FOUND:{name}")
            return

        offset = min(offset, len(body))
        payload = body[offset:]
        if self.profile == "legacy":
            self._line(f"FILE_DATA_START:{name}")
        else:
            self._line(f"FILE_DATA_START:{name},{offset},{len(body)}")

        cut = self._injection_point(payload)
        if cut is None:
            self._write(payload)
        else:
            self._write(payload[:cut])
            # A nosepoke between two chunks of file content.
            self._line("EVT,2026-08-24T10:40:01,LEFT,41,0,13,987654")
            self._write(payload[cut:])

        self._write(b"\x04\r\n")
        self._line(f"CRC32:{zlib.crc32(payload) & 0xFFFFFFFF:08X}")


# ---------------------------------------------------------------- harness

def wait_until(predicate, timeout_ms, app):
    loop = QEventLoop()
    deadline = QTimer(); deadline.setSingleShot(True); deadline.timeout.connect(loop.quit)
    poll = QTimer(); poll.timeout.connect(lambda: loop.quit() if predicate() else None)
    deadline.start(timeout_ms); poll.start(25)
    loop.exec_(); poll.stop(); deadline.stop()
    return predicate()


RESULTS = []


def check(label, ok, detail=""):
    RESULTS.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{f' — {detail}' if detail else ''}")


def run_profile(app, profile, tmpdir, inject=None):
    labels = {None: "", "line": " (event spliced at a line boundary)",
              "midline": " (event spliced mid-row)"}
    print(f"\n--- {profile} firmware{labels[inject]} ---")
    # CARD is the emulated SD card and profiles mutate it — growing a log,
    # truncating one — so each profile starts from the same card rather than
    # inheriting whatever the previous one left behind.
    pristine = dict(CARD)
    try:
        _run_profile(app, profile, tmpdir, inject)
    finally:
        CARD.clear()
        CARD.update(pristine)


def _run_profile(app, profile, tmpdir, inject=None):
    device = FakeFed3(profile, inject=inject)
    device.start()

    link = Fed3Link(device.port, owner=profile, silence_timeout=0)
    transfer = Fed3Transfer(link.send)
    seen_events = []
    firmware = {}

    def on_line(line):
        if transfer.handle_line(line):
            return
        text = line.strip()
        if text.startswith("PONG_FED3"):
            _id, fw = proto.parse_pong(text)
            firmware["fw"] = fw
        elif proto.parse_event(text) is not None:
            seen_events.append(text)

    link.line_received.connect(on_line)
    connected = {"v": False}
    link.connected.connect(lambda: connected.__setitem__("v", True))
    link.start()
    check("link opens", wait_until(lambda: connected["v"], 8000, app))

    link.send(proto.CMD_PING)
    got_pong = wait_until(lambda: "fw" in firmware, 6000, app)
    check("PING answered", got_pong)

    supported = proto.is_supported(firmware.get("fw"))
    if profile == "legacy":
        # The whole point: an old board is rejected, and rejected on the strength
        # of the handshake alone — before anything is asked of its SD card.
        check("old firmware rejected", not supported,
              f"is_supported({firmware.get('fw')!r}) should be False")
        check("no FW field reported", firmware.get("fw") is None,
              f"got {firmware.get('fw')!r}")
        link.stop()
        device.stop()
        return
    check("firmware accepted", supported, f"FW {firmware.get('fw')!r}")

    # Two destinations: the session owns the log the device is writing to now,
    # the archive holds the rest of the card and persists across sessions.
    session_dir = os.path.join(tmpdir, profile, "session")
    archive_dir = os.path.join(tmpdir, profile, "archive")
    os.makedirs(session_dir, exist_ok=True)
    os.makedirs(archive_dir, exist_ok=True)
    mirror = DeviceMirror(transfer, session_dir, archive_dir)
    mirror.adopt_current_file(LIVE_FILE)
    failures = []
    pull_order = []
    mirror.failed.connect(failures.append)
    mirror.updated.connect(lambda name, *_: pull_order.append(name))
    mirror.sync_now(force=True)
    check("mirror pull completes", wait_until(lambda: not mirror.busy, 30000, app))

    def mirrored_bytes(name):
        # Each file is looked for where it belongs, so a file written into the
        # wrong store reads as missing rather than quietly passing.
        where = session_dir if name in mirror.session_files else archive_dir
        try:
            with open(os.path.join(where, name), "rb") as f:
                return f.read()
        except OSError:
            return b""

    if inject == "midline":
        # An unrecognisable splice must be *rejected*, never written. The chunk
        # is discarded on the CRC mismatch and the offset is left untouched, so
        # a retry re-fetches the same range.
        check("corrupt chunk refused", bool(failures),
              "CRC did not reject the spliced payload" if not failures else "")
        for name, body in CARD.items():
            got = mirrored_bytes(name)
            check(f"{name} never written corrupt", got in (b"", body),
                  f"{len(got)} bytes on disk")

        device.inject = None            # the poke is over; retry must now succeed
        mirror.enabled = True
        failures.clear()
        mirror.sync_now(force=True)
        check("retry after interference completes",
              wait_until(lambda: not mirror.busy, 30000, app))
        check("retry has no failures", not failures, "; ".join(failures))
        for name, body in CARD.items():
            check(f"{name} recovered byte-exact", mirrored_bytes(name) == body)
    else:
        check("no mirror failures", not failures, "; ".join(failures))
        for name, body in CARD.items():
            got = mirrored_bytes(name)
            check(f"{name} mirrored byte-exact", got == body,
                  f"{len(got)}/{len(body)} bytes")

        # The running experiment must not queue behind the card's back
        # catalogue: whatever else is on the card, the live log goes first.
        check("live file pulled first", pull_order[:1] == [LIVE_FILE],
              f"order was {pull_order}")
        check("live file in the session folder",
              os.path.exists(os.path.join(session_dir, LIVE_FILE)))
        check("history in the archive, not the session",
              not os.path.exists(os.path.join(session_dir, ARCHIVED_FILE))
              and os.path.exists(os.path.join(archive_dir, ARCHIVED_FILE)))

    if inject == "line":
        check("spliced event reached the host, not the file", bool(seen_events),
              f"{len(seen_events)} event(s)")

    # Append-only growth: the device's log grows, the mirror must pick up only
    # the new bytes (modern) or re-fetch cleanly (legacy), and stay byte-exact.
    name = LIVE_FILE
    CARD[name] += b"08:24:2026 11:05:00,Pellet\r\n"
    mirror.enabled = True
    mirror.sync_now(force=True)
    wait_until(lambda: not mirror.busy, 30000, app)
    check("growing log tracked byte-exact", mirrored_bytes(name) == CARD[name])

    if inject is None:
        # NEW_TRIAL can hand back a name the host already mirrored: FED3 deletes
        # and reuses the filename of any log with fewer than three lines, so the
        # file the host has bytes for is replaced rather than appended to. The
        # mirror must notice the shrink and re-copy from zero, or it would keep
        # the old content and graft the new log's tail onto it.
        CARD[name] = b"MM:DD:YYYY hh:mm:ss,Event\r\n"
        mirror.enabled = True
        mirror.sync_now(force=True)
        wait_until(lambda: not mirror.busy, 30000, app)
        check("recycled log re-copied, not appended to",
              mirrored_bytes(name) == CARD[name],
              f"{mirrored_bytes(name)!r}")

        # A file that shrank but holds real data is an anomaly, not a recycle:
        # it must be reported and left alone rather than overwritten.
        failures.clear()
        CARD[ARCHIVED_FILE] = b"truncated\r\n"
        mirror.enabled = True
        mirror.sync_now(force=True)
        wait_until(lambda: not mirror.busy, 30000, app)
        check("unexpected shrink reported, file untouched",
              any("shrank" in f for f in failures)
              and mirrored_bytes(ARCHIVED_FILE).startswith(b"MM:DD:YYYY"),
              "; ".join(failures) or "no failure reported")

    mirror.stop()
    link.stop()
    device.stop()


def main():
    import tempfile
    app = QCoreApplication(sys.argv)
    with tempfile.TemporaryDirectory(prefix="fnt_fed3_loopback_") as tmp:
        run_profile(app, "legacy", tmp)
        run_profile(app, "modern", tmp)
        run_profile(app, "modern", os.path.join(tmp, "aligned"), inject="line")
        run_profile(app, "modern", os.path.join(tmp, "midrow"), inject="midline")
    total, passed = len(RESULTS), sum(RESULTS)
    print(f"\n{passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
