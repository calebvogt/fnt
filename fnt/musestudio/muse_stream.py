"""Backend for MuseStudio: device discovery, the OpenMuse streamer subprocess,
an LSL reader thread, and a CSV recorder.

Architecture (producer/consumer):
  - Producer: ``OpenMuse stream --address <addr>`` runs as a subprocess and
    publishes Lab Streaming Layer (LSL) streams (Muse_EEG, fNIRS, PPG, ...).
  - Consumer: ``LSLReaderThread`` resolves those LSL streams, opens inlets,
    and pulls chunks in a loop, emitting them to the GUI and (optionally)
    writing them to disk via ``MuseRecorder``.

OpenMuse has no direct Python callback API, hence the LSL hop. OpenMuse and
its decoding (especially fNIRS) are reverse-engineered and experimental.
"""

import collections
import csv
import os
import re
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal


# NOTE: an earlier revision pointed LSLAPICFG at a generated lsl_api.cfg to
# silence liblsl's start-up logging. That was removed: the log tee in
# logbuffer.py already keeps the terminal quiet, so the config file bought
# nothing while adding a way for stream resolution to break. Don't reintroduce
# it — suppress liblsl output at the fd level instead.

# Note: mne_lsl (liblsl) is imported lazily inside LSLReaderThread.run() so that
# device discovery, the streamer subprocess, and the CSV recorder remain usable
# (and testable) without the LSL stack installed.


# Stream name prefixes published by OpenMuse that we care about. OpenMuse names
# its streams like "Muse_EEG", "Muse_ACCGYRO", and fNIRS/PPG variants; matching
# on the "Muse" prefix keeps us robust to exact suffix naming.
MUSE_STREAM_PREFIX = "Muse"


def find_devices(timeout=15):
    """Run ``OpenMuse find`` and return a list of discovered device addresses.

    Returns a tuple ``(devices, raw_output)`` where ``devices`` is a list of
    dicts ``{"name": str, "address": str}`` parsed best-effort from the CLI
    output, and ``raw_output`` is the full combined stdout/stderr for display.
    """
    try:
        proc = subprocess.run(
            [_openmuse_exe(), "find"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "OpenMuse CLI not found. Reinstall project dependencies:\n"
            "    pip install -e ."
        )
    except subprocess.TimeoutExpired as exc:
        raw = (exc.stdout or "") + (exc.stderr or "")
        return _parse_find_output(raw), raw

    raw = (proc.stdout or "") + (proc.stderr or "")
    return _parse_find_output(raw), raw


def _parse_find_output(text):
    """Best-effort parse of ``OpenMuse find`` output into device dicts.

    Matches both BLE MAC addresses (Windows/Linux) and CoreBluetooth UUIDs
    (macOS), pairing each with any "Muse..." name on the same line.
    """
    devices = []
    seen = set()
    # MAC like AA:BB:CC:DD:EE:FF or a macOS CoreBluetooth UUID.
    addr_re = re.compile(
        r"([0-9A-Fa-f]{2}(?::[0-9A-Fa-f]{2}){5}"
        r"|[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
        r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12})"
    )
    # Muse device names look like "Muse-1A2B" or "MuseS-1A2B" (no spaces).
    name_re = re.compile(r"(Muse[\w\-]*)", re.IGNORECASE)
    for line in text.splitlines():
        m = addr_re.search(line)
        nm = name_re.search(line)
        if not m or not nm:  # require both a Muse name and an address
            continue
        address = m.group(1)
        if address in seen:
            continue
        seen.add(address)
        devices.append({"name": nm.group(1).strip(), "address": address})
    return devices


def _openmuse_exe():
    """Resolve the OpenMuse console script.

    Looks beside the running interpreter FIRST, then on PATH. That order matters:
    the console script is installed into the same environment as this package,
    but PATH only contains that environment's ``bin`` when the env was activated
    in the launching shell. Launch the app any other way -- a bare interpreter
    path, an IDE, a .app bundle, a desktop launcher -- and a PATH-only lookup
    fails with "OpenMuse CLI not found. Reinstall project dependencies", which
    sends the user off reinstalling a package that is already correctly
    installed. Observed doing exactly that on 2026-09-02.

    Returns an absolute path when one is found, otherwise the bare name so the
    existing FileNotFoundError path still reports something sensible.
    """
    exe_dir = os.path.dirname(sys.executable)
    for name in ("OpenMuse", "openmuse"):
        candidate = os.path.join(exe_dir, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    found = shutil.which("OpenMuse") or shutil.which("openmuse")
    return found or "OpenMuse"


class MuseStreamProcess:
    """Manages the ``OpenMuse stream --address <addr>`` subprocess (producer)."""

    def __init__(self, address):
        self.address = address
        self._proc = None
        self._lines = collections.deque(maxlen=400)
        self._pump = None

    def start(self):
        if self._proc is not None and self._proc.poll() is None:
            return  # already running
        self._lines.clear()
        self._proc = subprocess.Popen(
            [_openmuse_exe(), "stream", "--address", self.address],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        # Drain stdout continuously. This is NOT bookkeeping: the pipe buffer is
        # about 64 KB, OpenMuse logs steadily, and a full pipe BLOCKS the writer.
        # Left unread the streamer stalls and then dies, which presents as
        # "connected, no data, no battery" with no explanation anywhere -- the
        # exact symptom hit on 2026-09-02, where the app held inlets to streams
        # whose producer was gone. Draining also makes the streamer's own error
        # messages available to show the operator instead of discarding them.
        self._pump = threading.Thread(target=self._drain, daemon=True)
        self._pump.start()

    def _drain(self):
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                self._lines.append(line.rstrip())
        except Exception:  # noqa: BLE001
            pass

    def is_alive(self):
        return self._proc is not None and self._proc.poll() is None

    def exit_code(self):
        return self._proc.poll() if self._proc is not None else None

    def read_output_nonblocking(self):
        """Everything the streamer has printed since it started."""
        return "\n".join(self._lines)

    def tail(self, n=12):
        """The last few streamer lines — usually where the real error is."""
        return "\n".join(list(self._lines)[-n:])

    def stop(self, timeout=5):
        """Terminate the streamer and return its captured output, if any."""
        if self._proc is None:
            return ""
        out = ""
        try:
            self._proc.terminate()
            try:
                out = self._proc.communicate(timeout=timeout)[0] or ""
            except subprocess.TimeoutExpired:
                self._proc.kill()
                out = self._proc.communicate(timeout=timeout)[0] or ""
        except Exception:
            pass
        finally:
            self._proc = None
        return out


class MuseRecorder:
    """Writes one CSV per stream into the given directory.

    Thread-safe: ``write`` is called from the reader thread while ``stop`` is
    called from the GUI thread.
    """

    def __init__(self, out_dir, precision=6):
        """``precision`` = decimal places per value.

        The default 6 is lossless for anything the headband produces. Long
        overnight runs pass 2: EEG resolution is then 0.01 µV — two orders of
        magnitude below the device's own ~2 µV noise floor, so nothing real is
        lost — and it removes roughly a third of a 1.5 GB night.
        """
        self.session_dir = out_dir
        self.precision = int(precision)
        self._fmt = f"{{:.{int(precision)}f}}"
        os.makedirs(out_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._files = {}   # stream_name -> (file_handle, csv.writer)
        self._counts = {}  # stream_name -> int
        self._closed = False

    def write(self, stream_name, timestamps, data, channel_names):
        """Append a chunk. ``data`` is (n_samples, n_channels); ``timestamps``
        is (n_samples,)."""
        if data is None or len(data) == 0:
            return
        with self._lock:
            if self._closed:
                return
            writer = self._files.get(stream_name)
            if writer is None:
                writer = self._open_stream_file(stream_name, channel_names)
            fh, w = writer
            for i in range(len(timestamps)):
                # Timestamps keep full precision regardless — they carry the
                # alignment between streams and must not be rounded.
                w.writerow([f"{timestamps[i]:.6f}",
                            *(self._fmt.format(v) for v in data[i])])
            self._counts[stream_name] = self._counts.get(stream_name, 0) + len(timestamps)
            fh.flush()

    def _open_stream_file(self, stream_name, channel_names):
        safe = re.sub(r"[^\w\-]", "_", stream_name)
        path = os.path.join(self.session_dir, f"{safe}.csv")
        fh = open(path, "w", newline="")
        w = csv.writer(fh)
        w.writerow(["lsl_timestamp", *channel_names])
        self._files[stream_name] = (fh, w)
        return self._files[stream_name]

    def counts(self):
        with self._lock:
            return dict(self._counts)

    def stop(self):
        with self._lock:
            self._closed = True
            for fh, _ in self._files.values():
                try:
                    fh.close()
                except Exception:
                    pass
            self._files.clear()
        return self.session_dir


class LSLReaderThread(QThread):
    """Resolves Muse LSL streams and pulls chunks in a loop (consumer)."""

    # stream_name, timestamps (n,), data (n, n_channels)
    samples_ready = pyqtSignal(str, object, object)
    connected = pyqtSignal(list)       # list of stream names found
    disconnected = pyqtSignal()
    error = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, resolve_timeout=45.0, address=None, parent=None,
                 streamer_dead=None):
        super().__init__(parent)
        self._running = False
        self._resolve_timeout = resolve_timeout
        # Optional callable: True when the producer process has exited, so a
        # dead streamer is reported as such instead of as a 45 s timeout.
        self._streamer_dead = streamer_dead
        self._address = address       # restrict to this device's streams if given
        self._recorder = None
        self._rec_lock = threading.Lock()
        self._channel_names = {}  # stream_name -> [names]
        self._sfreq = {}          # stream_name -> sample rate (Hz)

    # --- recording control (called from GUI thread) ---
    def start_recording(self, recorder):
        with self._rec_lock:
            self._recorder = recorder

    def stop_recording(self):
        with self._rec_lock:
            rec = self._recorder
            self._recorder = None
        if rec is not None:
            return rec.stop()
        return None

    def channel_names(self):
        """Return a copy of {stream_name: [channel names]} (valid after connect)."""
        return dict(self._channel_names)

    def sample_rate(self, stream_name):
        """Sample rate (Hz) of a resolved stream, or None."""
        return self._sfreq.get(stream_name)

    def stop(self):
        self._running = False

    def run(self):
        self._running = True
        inlets = []
        try:
            from mne_lsl.lsl import StreamInlet, resolve_streams

            # Retry until the streamer has had time to come up.
            #
            # A single 10 s attempt was not enough and this was the real cause
            # of "No Muse LSL streams found": OpenMuse's own start-up runs
            # connect -> subscribe to notifications -> "Waiting for device
            # info..." -> "Streaming data...", which takes well over ten
            # seconds on this headband. The reader gave up while the streamer
            # was still starting, reported a missing stream, and tore itself
            # down — and that teardown is what then aborted the process.
            #
            # Polling also lets us notice the streamer DYING, which is a
            # different failure needing a different message.
            deadline = time.monotonic() + self._resolve_timeout
            muse_infos = []
            attempt = 0
            while time.monotonic() < deadline and self._running:
                attempt += 1
                left = max(1, int(deadline - time.monotonic()))
                self.status.emit(
                    f"Waiting for the headband to start streaming… ({left}s left)")
                infos = resolve_streams(timeout=2.0)
                muse_infos = [si for si in infos
                              if si.name.startswith(MUSE_STREAM_PREFIX)]
                # Wait for the FULL expected set, not merely the first stream to
                # appear. Accepting whatever the first successful poll returned
                # is a race, and it silently cost real data: session 204249
                # recorded only EEG and ACCGYRO — no optics, no battery — and
                # 205111 lost battery, with nothing anywhere saying so. For
                # fNIRS work a missing optics stream makes the session useless
                # and it is only discoverable by reading the config afterwards.
                kinds = {k for k in EXPECTED_STREAM_KINDS
                         if any(k in si.name.upper() for si in muse_infos)}
                if len(kinds) >= len(EXPECTED_STREAM_KINDS):
                    break
                if muse_infos and time.monotonic() > deadline - 8.0:
                    # Out of time but something is there: take it and say what
                    # is missing rather than failing outright.
                    missing = sorted(set(EXPECTED_STREAM_KINDS) - kinds)
                    self.status.emit(
                        "Started WITHOUT " + ", ".join(missing)
                        + " — those sensors will be absent from this recording.")
                    break
                if self._streamer_dead is not None and self._streamer_dead():
                    self.error.emit(
                        "The OpenMuse streamer exited before it published any "
                        "data. The headband may have refused the connection — "
                        "switch it off and on, then try again.")
                    return
            # If we know the device address, prefer its streams (OpenMuse embeds
            # the id in the stream name) so a second headband/app doesn't leak in.
            if self._address:
                matched = [si for si in muse_infos if self._address in si.name]
                if matched:
                    muse_infos = matched
            if not muse_infos:
                self.error.emit(
                    "No Muse LSL streams found. Is the OpenMuse streamer "
                    "running and the headband connected?"
                )
                return

            names = []
            for si in muse_infos:
                inlet = StreamInlet(si, max_buffered=4)
                inlet.open_stream(timeout=5.0)
                sinfo = inlet.get_sinfo()
                self._channel_names[si.name] = self._channel_names_for(sinfo)
                self._sfreq[si.name] = getattr(sinfo, "sfreq", None)
                inlets.append((si.name, inlet))
                names.append(si.name)

            self.connected.emit(names)
            self.status.emit(f"Streaming: {', '.join(names)}")

            while self._running:
                got_any = False
                for name, inlet in inlets:
                    data, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=512)
                    if timestamps is None or len(timestamps) == 0:
                        continue
                    got_any = True
                    data = np.asarray(data, dtype=float)
                    timestamps = np.asarray(timestamps, dtype=float)
                    self.samples_ready.emit(name, timestamps, data)
                    with self._rec_lock:
                        rec = self._recorder
                    if rec is not None:
                        rec.write(name, timestamps, data, self._channel_names[name])
                if not got_any:
                    time.sleep(0.005)  # avoid busy-spin when no data is pending
        except Exception as exc:  # noqa: BLE001 - surface any backend failure to UI
            self.error.emit(f"{type(exc).__name__}: {exc}")
        finally:
            for _, inlet in inlets:
                try:
                    inlet.close_stream()
                except Exception:
                    pass
            self.disconnected.emit()

    @staticmethod
    def _channel_names_for(sinfo):
        try:
            ch = sinfo.get_channel_names()
            if ch and all(ch):
                return list(ch)
        except Exception:
            pass
        n = getattr(sinfo, "n_channels", 0) or 0
        return [f"ch{i}" for i in range(n)]
