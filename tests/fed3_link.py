"""The serial link over a real (pseudo) terminal, with a simulated FED3 behind it.

The engine tests use an in-process transport; this one goes through pyserial
and the operating system's tty layer, where lines arrive split across reads,
rows interleave with replies, and a port can vanish mid-command.
POSIX only (it needs a pty); skipped on Windows.
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS = []


def check(label, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{f' — {detail}' if detail else ''}")


class Responder(threading.Thread):
    """The device end of the pty: commands in, replies and rows out."""

    def __init__(self, fd, device, chunk=None):
        super().__init__(daemon=True)
        self.fd = fd
        self.device = device
        self.chunk = chunk          # write replies a few bytes at a time
        self.silent = False
        self.lock = threading.Lock()
        self.stopping = False

    def send(self, lines):
        data = "".join(line + "\n" for line in lines).encode()
        if self.chunk:
            for i in range(0, len(data), self.chunk):
                os.write(self.fd, data[i:i + self.chunk])
                time.sleep(0.001)
        else:
            os.write(self.fd, data)

    def act(self, fn):
        with self.lock:
            fn(self.device)
            self.send(self.device.take())

    def run(self):
        buffer = b""
        while not self.stopping:
            try:
                data = os.read(self.fd, 256)
            except OSError:
                return
            buffer += data
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if self.silent:
                    continue
                with self.lock:
                    self.device.command(line.decode())
                    self.send(self.device.take())


def main():
    if os.name != "posix":
        print("  SKIP  needs a POSIX pty")
        return 0
    import pty
    import termios
    import tty

    from fnt.fed3 import device_sim as dev
    from fnt.fed3.link import DeviceLink, LinkError, SerialTransport, fields

    master, slave = pty.openpty()
    tty.setraw(slave)
    attrs = termios.tcgetattr(slave)
    attrs[3] &= ~termios.ECHO
    termios.tcsetattr(slave, termios.TCSANOW, attrs)
    device = dev.SimDevice(device_id=5)
    device.take()
    responder = Responder(master, device, chunk=7)
    responder.start()
    seen = []
    transport = SerialTransport(os.ttyname(slave))
    link = DeviceLink(transport, seen.append, timeout=2.0)

    print("--- commands and replies ---")
    check("PING answers, though it arrives 7 bytes at a time",
          link.call("PING") == "PONG,id=5,fw=3.1,proto=3")
    status = fields(link.call("STATUS"))
    check("STATUS parses into fields", status.get("id") == "5" and status.get("clock") == "ok")

    print("\n--- rows between commands ---")
    responder.act(lambda d: d.poke("L"))
    time.sleep(0.2)
    link.poll()
    check("a poke's row arrives without being asked for",
          any(line.startswith("ROW,") and ",poke," in line for line in seen))
    seen.clear()
    reply = link.call("FEED")
    check("rows written by a command go to on_line, its reply is returned",
          reply.startswith("FEED,result=") and all(not s.startswith("FEED") for s in seen))
    seen.clear()
    end = link.call("READ:-,1")
    rows = [s for s in seen if s.startswith("ROW,")]
    check("READ returns READ_END, with its rows on on_line",
          end.startswith("READ_END,run=-") and len(rows) >= 2)

    print("\n--- failures ---")
    responder.silent = True
    started = time.monotonic()
    try:
        link.call("STATUS", timeout=0.5)
        timed_out = False
    except LinkError:
        timed_out = True
    check("a device that doesn't answer raises LinkError, on time",
          timed_out and time.monotonic() - started < 1.5)
    responder.silent = False
    link.poll()
    check("and the link recovers when it answers again",
          link.call("PING").startswith("PONG,"))
    responder.stopping = True
    os.close(master)
    try:
        link.call("PING", timeout=0.5)
        vanished = False
    except LinkError:
        vanished = True
    check("a port that disappears raises LinkError", vanished)
    link.close()
    os.close(slave)

    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed")
    return 0 if all(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
