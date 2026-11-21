"""One FED3 (firmware v3) on one serial port.

The device handles one command at a time and answers it before reading the
next, so a reply is simply the first line after a command that isn't a ``ROW``
or ``BOOT`` (a ``READ`` ends at ``READ_END``). ``ROW`` and ``BOOT`` lines can
arrive at any moment, including between a command and its reply; they go to
``on_line``.

There is no thread here. The run engine calls :meth:`poll` for unsolicited
lines and :meth:`call` for commands, all from its own loop.
"""
from __future__ import annotations

import time

UNSOLICITED = ("ROW,", "BOOT,")
CANDIDATE_VIDS = {0x239A}           # Adafruit: the FED3's Feather M0


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

    def _dispatch(self, line):
        self.last_heard = self.clock()
        self.on_line(line)

    def poll(self, limit=200):
        """Hand every line already waiting to on_line, without blocking."""
        for _ in range(limit):
            line = self._readline(0)
            if line is None:
                return
            self._dispatch(line)

    def call(self, command, timeout=None):
        """Send one command and return its reply line.

        For READ, returns the READ_END (or ERR) line; the rows it replays go
        to on_line like any other row.
        """
        self.poll()
        self.transport.write(command)
        deadline = time.monotonic() + (timeout or self.timeout)
        reading = command.startswith("READ:")
        while True:
            line = self._readline(max(0.0, deadline - time.monotonic()))
            if line is None:
                raise LinkError(f"no reply to {command.split(':')[0]}")
            if line.startswith(UNSOLICITED):
                self._dispatch(line)
                continue
            self.last_heard = self.clock()
            if reading and not line.startswith(("READ_END,", "ERR,")):
                self.on_line(line)
                continue
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
