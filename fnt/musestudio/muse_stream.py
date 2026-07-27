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

import csv
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime

import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal

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
            "OpenMuse CLI not found. Install the muse extra:\n"
            '    pip install -e ".[muse]"'
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
    """Resolve the OpenMuse entry point.

    Prefer the installed console script, but fall back to ``python -m`` form is
    handled by callers if needed. Returns a string suitable as argv[0].
    """
    return "OpenMuse"


class MuseStreamProcess:
    """Manages the ``OpenMuse stream --address <addr>`` subprocess (producer)."""

    def __init__(self, address):
        self.address = address
        self._proc = None

    def start(self):
        if self._proc is not None and self._proc.poll() is None:
            return  # already running
        self._proc = subprocess.Popen(
            [_openmuse_exe(), "stream", "--address", self.address],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

    def is_alive(self):
        return self._proc is not None and self._proc.poll() is None

    def read_output_nonblocking(self):
        """Drain any buffered subprocess output (for surfacing errors)."""
        # We don't actively pump stdout here to keep things simple; on stop we
        # collect whatever is available. Kept for future log-streaming.
        return ""

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
    """Writes per-stream samples to CSV files in a timestamped session folder.

    Thread-safe: ``write`` is called from the reader thread while ``start``/
    ``stop`` are called from the GUI thread.
    """

    def __init__(self, base_dir):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(base_dir, f"MuseStudio_{ts}")
        os.makedirs(self.session_dir, exist_ok=True)
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
                w.writerow([f"{timestamps[i]:.6f}", *(f"{v:.6f}" for v in data[i])])
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

    def __init__(self, resolve_timeout=10.0, parent=None):
        super().__init__(parent)
        self._running = False
        self._resolve_timeout = resolve_timeout
        self._recorder = None
        self._rec_lock = threading.Lock()
        self._channel_names = {}  # stream_name -> [names]

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

    def stop(self):
        self._running = False

    def run(self):
        self._running = True
        try:
            from mne_lsl.lsl import StreamInlet, resolve_streams

            self.status.emit("Resolving LSL streams from OpenMuse...")
            infos = resolve_streams(timeout=self._resolve_timeout)
            muse_infos = [si for si in infos if si.name.startswith(MUSE_STREAM_PREFIX)]
            if not muse_infos:
                self.error.emit(
                    "No Muse LSL streams found. Is the OpenMuse streamer "
                    "running and the headband connected?"
                )
                return

            inlets = []
            names = []
            for si in muse_infos:
                inlet = StreamInlet(si, max_buffered=4)
                inlet.open_stream(timeout=5.0)
                sinfo = inlet.get_sinfo()
                ch_names = self._channel_names_for(sinfo)
                self._channel_names[si.name] = ch_names
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

            for _, inlet in inlets:
                try:
                    inlet.close_stream()
                except Exception:
                    pass
        except Exception as exc:  # noqa: BLE001 - surface any backend failure to UI
            self.error.emit(f"{type(exc).__name__}: {exc}")
        finally:
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
