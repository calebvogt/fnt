"""One FED3 (firmware v3) on one serial port.

The device handles one command at a time and answers it before reading the
next, so a reply is simply the first line after a command that isn't a ``ROW``
or ``BOOT``. ``ROW`` and ``BOOT`` lines (and ``ERR,SD,append failed``, for a row
the card refused) can arrive at any moment, including between a command and
its reply; they go to ``on_line``.

A ``READ`` is the exception: the device replays the file a few rows at a time
between everything else it does, and ends with ``READ_END``. :meth:`read`
sends one and returns at once; the rows and the ``READ_END`` go to ``on_line``
as they arrive, so catching up on a long file never holds the engine up.

A reply can also be held up on its way: a line that fills its last 64-byte USB
packet sits in Windows' serial driver until the device sends something else,
and firmware before the fix ends such lines without a short packet. A command
that has heard nothing for ``NUDGE_AFTER`` gets one ``PING`` behind it; the
held reply comes out ahead of the ``PONG``, which is dropped.

There is no thread here. The run engine calls :meth:`poll` for unsolicited
lines and :meth:`call` for commands, all from its own loop.
"""
from __future__ import annotations

import time

UNSOLICITED = ("ROW,", "BOOT,", "ERR,SD,append")
REPLAY_END = ("READ_END,", "ERR,NO_FILE")
CANDIDATE_VIDS = {0x239A}           # Adafruit: the FED3's Feather M0
NUDGE_AFTER = 1.0                   # seconds of silence before a held reply is pushed out


class LinkError(Exception):
    """The device didn't answer, or the port went away."""


class SerialTransport:
    """A real port, through pyserial."""

    def __init__(self, port, baud=115200):
        import serial
        # Opening asserts DTR, which is what tells the firmware someone is
        # listening; it doesn't send anything while DTR is low.
        self.serial = serial.Serial(port, baud, timeout=0)
        self.buffer = b""

    def write(self, line):
        try:
            self.serial.write(line.encode("ascii") + b"\n")
        except Exception as exc:
            raise LinkError(str(exc)) from exc

    def readline(self, timeout):
        """One line without its newline, or None if none arrives in time."""
        deadline = time.monotonic() + timeout
        while True:
            newline = self.buffer.find(b"\n")
            if newline >= 0:
                line, self.buffer = self.buffer[:newline], self.buffer[newline + 1:]
                return line.decode("ascii", "replace").rstrip("\r")
            try:
                waiting = self.serial.in_waiting
                chunk = self.serial.read(waiting or 1) if waiting or timeout else b""
            except Exception as exc:
                raise LinkError(str(exc)) from exc
            if chunk:
                self.buffer += chunk
                continue
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.005)

    def close(self):
        try:
            self.serial.close()
        except Exception:
            pass


def candidate_ports():
    """Serial ports that look like a FED3: ``{device: description}``."""
    from serial.tools import list_ports
    return {p.device: p.description or p.device for p in list_ports.comports()
            if p.vid in CANDIDATE_VIDS}


class DeviceLink:
    def __init__(self, transport, on_line, timeout=3.0, clock=time.monotonic):
        self.transport = transport
        self.on_line = on_line
        self.timeout = timeout
        self.clock = clock
        self.last_heard = clock()
        self.closed = False
        # When the last row of a READ sent with read() arrived (or when it was
        # sent), until its READ_END; None when no replay is under way.
        self.replaying = None
        self.nudges = 0             # PINGs sent to push a reply out, whose PONG is still to come
        self.held = 0               # replies that came only after a nudge

    def _nudged(self, line):
        """Whether this is the PONG of a nudge, dropped here."""
        if self.nudges and line.startswith("PONG,"):
            self.nudges -= 1
            self.last_heard = self.clock()
            return True
        return False

    def _dispatch(self, line):
        self.last_heard = self.clock()
        if self.replaying is not None:
            if line.startswith(REPLAY_END):
                self.replaying = None
            elif line.startswith("ROW,"):
                self.replaying = self.clock()
        self.on_line(line)

    def _unsolicited(self, line):
        return line.startswith(UNSOLICITED) or (self.replaying is not None
                                                and line.startswith(REPLAY_END))

    def read(self, command):
        """Send a ``READ`` and return without waiting for the replay. Its rows,
        then its ``READ_END`` (or ``ERR,NO_FILE``), go to on_line as they come."""
        self.poll()
        self.transport.write(command)
        self.replaying = self.clock()
        self.poll()

    def poll(self, limit=200):
        """Hand every line already waiting to on_line, without blocking."""
        for _ in range(limit):
            line = self._readline(0)
            if line is None:
                return
            if not self._nudged(line):
                self._dispatch(line)

    def call(self, command, timeout=None):
        """Send one command and return its reply line.

        For READ, returns the READ_END (or ERR) line; the rows it replays go
        to on_line like any other row.
        """
        self.poll()
        self.transport.write(command)
        deadline = time.monotonic() + (timeout or self.timeout)
        # PING's own reply is too short to be held, and its PONG is the answer.
        can_nudge = command != "PING"
        nudged = False
        reading = command.startswith("READ:")
        while True:
            wait = deadline
            if can_nudge and not nudged:
                wait = min(deadline, time.monotonic() + NUDGE_AFTER)
            line = self._readline(max(0.0, wait - time.monotonic()))
            if line is None and wait < deadline:
                self.transport.write("PING")
                self.nudges += 1
                nudged = True
                continue
            if line is None:
                raise LinkError(f"no reply to {command.split(':')[0]}")
            if self._nudged(line):
                continue
            if self._unsolicited(line):
                self._dispatch(line)
                continue
            self.last_heard = self.clock()
            if reading and not line.startswith(("READ_END,", "ERR,")):
                self.on_line(line)
                continue
            if nudged:
                self.held += 1
            return line

    def _readline(self, timeout):
        if self.closed:
            raise LinkError("link closed")
        return self.transport.readline(timeout)

    def close(self):
        self.closed = True
        self.transport.close()


def fields(line):
    """``TYPE,k=v,k=v`` -> ``{k: v}``. The first field (the type) is dropped."""
    out = {}
    for part in line.split(",")[1:]:
        key, sep, value = part.partition("=")
        if sep:
            out[key] = value
    return out


def error(line):
    """``ERR,<code>,<detail>`` -> (code, detail), or None if not an error."""
    if not line.startswith("ERR,"):
        return None
    _, code, detail = (line.split(",", 2) + ["", ""])[:3]
    return code, detail
