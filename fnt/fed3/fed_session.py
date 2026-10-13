"""Recording session layout, interaction logging and crash recovery.

Every recording gets a timestamped folder::

    YYYY-MM-DD_HHMMSS_FNT_FED3_session/
        session_config.json      # settings snapshot written at start
        session_state.json       # live state, rewritten atomically for resume
        session_log.txt          # human-readable narrative
        interactions.csv         # machine-readable record of every action
        Data/
            FED/<device>/events.csv          # behavioural events, host + device clock
            FED/<device>/mirror/<FILE>.CSV   # byte-exact copy of the SD card file
            FED/<device>/mirror_state.json   # per-file byte offsets
            Sync/clock_sync.csv              # host vs device RTC at each sync

Time base
---------
Every host-side timestamp in a session — FED events, interactions,
clock syncs — is :func:`host_now`, a single wall-clock epoch float. That is what
makes events directly comparable without post-hoc alignment.
The device RTC is recorded alongside, never substituted for it, so clock drift
stays measurable rather than baked in.

Crash recovery
--------------
``session_state.json`` is rewritten (write-temp-then-replace, so it is never
observed half-written) whenever something durable changes. A session whose state
still says ``"running"`` was interrupted; :func:`find_resumable_sessions` finds
those so the GUI can offer to reopen one and append to its logs rather than
starting a fresh folder and orphaning the data.
"""

import csv
import json
import os
import time
from datetime import datetime

FOLDER_SUFFIX = "FNT_FED3_session"

STATUS_RUNNING = "running"
STATUS_CLOSED = "closed"


def host_now():
    """The session time base: seconds since the Unix epoch, host wall clock."""
    return time.time()


def host_iso(ts=None):
    """Format a :func:`host_now` value as ISO-8601 with milliseconds."""
    return datetime.fromtimestamp(ts if ts is not None else host_now()).isoformat(
        sep=" ", timespec="milliseconds")


def default_session_root():
    """Where sessions live, falling back to ~ if Documents is not writable."""
    preferred = os.path.expanduser("~/Documents/FED3_Sessions")
    try:
        os.makedirs(preferred, exist_ok=True)
        return preferred
    except OSError:
        fallback = os.path.expanduser("~/FED3_Sessions")
        os.makedirs(fallback, exist_ok=True)
        return fallback


def _atomic_write_json(path, payload):
    """Write JSON so a crash mid-write cannot corrupt the existing file."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def device_archive_dir(sessions_root, device_name):
    """Where a device's historical SD files are kept, across all sessions.

    Deliberately outside any one session folder. The card's back catalogue is a
    property of the device, not of the run that happened to be recording when it
    was copied; keeping it here means it is downloaded once ever rather than
    re-downloaded from byte zero into every new session folder.
    """
    path = os.path.join(sessions_root, "device_archive", _safe_name(device_name))
    os.makedirs(path, exist_ok=True)
    return path


class DeviceNames:
    """User-assigned device labels, remembered between launches.

    Keyed by the on-board FED3 ID rather than by slot or port, because that is
    the only identifier that survives a replug: ports get renumbered by the
    kernel and slots are a UI concept. A cage labelled "Cage A — FR1" keeps that
    label when it is unplugged, moved to another USB socket, and plugged back in.
    """

    def __init__(self, path=None):
        self.path = path or os.path.join(default_session_root(), "device_names.json")
        self._labels = {}
        try:
            with open(self.path, encoding="utf-8") as f:
                loaded = json.load(f)
        except (OSError, ValueError):
            return
        if isinstance(loaded, dict):
            self._labels = {str(k): str(v) for k, v in loaded.items() if v}

    def get(self, device_id):
        """The stored label for a device ID, or "" if it has never been named."""
        if device_id in (None, ""):
            return ""
        return self._labels.get(str(device_id), "")

    def set(self, device_id, label):
        """Store (or, for an empty label, forget) the name for a device ID."""
        if device_id in (None, ""):
            return
        key = str(device_id)
        label = (label or "").strip()
        if label:
            if self._labels.get(key) == label:
                return
            self._labels[key] = label
        elif key in self._labels:
            del self._labels[key]
        else:
            return
        try:
            _atomic_write_json(self.path, self._labels)
        except OSError:
            # A name that cannot be written back is not worth interrupting a
            # recording over; it just will not survive this launch.
            pass


class RecordingSession:
    """The on-disk tree for one recording, plus its logs."""

    def __init__(self, base_dir, root=None):
        if root is None:
            stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            self.name = f"{stamp}_{FOLDER_SUFFIX}"
            self.root = os.path.join(base_dir, self.name)
            self.resumed = False
        else:
            self.root = root
            self.name = os.path.basename(root)
            self.resumed = True

        self.data_dir = os.path.join(self.root, "Data")
        self.fed_dir = os.path.join(self.data_dir, "FED")
        self.sync_dir = os.path.join(self.data_dir, "Sync")
        for d in (self.root, self.data_dir, self.fed_dir, self.sync_dir):
            os.makedirs(d, exist_ok=True)

        self.config_path = os.path.join(self.root, "session_config.json")
        self.state_path = os.path.join(self.root, "session_state.json")
        self.log_path = os.path.join(self.root, "session_log.txt")
        self.interactions_path = os.path.join(self.root, "interactions.csv")
        self.clock_sync_path = os.path.join(self.sync_dir, "clock_sync.csv")

        self.started_at = host_now() if not self.resumed else None

    # --- per-device paths -------------------------------------------------

    def device_dir(self, device_name):
        path = os.path.join(self.fed_dir, _safe_name(device_name))
        os.makedirs(path, exist_ok=True)
        return path

    def device_events_path(self, device_name):
        return os.path.join(self.device_dir(device_name), "events.csv")

    def device_mirror_dir(self, device_name):
        path = os.path.join(self.device_dir(device_name), "mirror")
        os.makedirs(path, exist_ok=True)
        return path


    # --- config / state ---------------------------------------------------

    def write_config(self, config):
        _atomic_write_json(self.config_path, config)

    def write_state(self, state):
        """Persist resume state. Cheap enough to call on every change."""
        payload = dict(state)
        payload["session_root"] = self.root
        payload["updated_at"] = host_iso()
        _atomic_write_json(self.state_path, payload)

    def read_state(self):
        try:
            with open(self.state_path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def mark_closed(self):
        state = self.read_state()
        state["status"] = STATUS_CLOSED
        state["closed_at"] = host_iso()
        self.write_state(state)

    # --- clock sync -------------------------------------------------------

    def log_clock_sync(self, device_name, device_time, sent_at=None):
        """Record host time against the device RTC it reported back.

        ``offset_s`` is device minus host; a growing magnitude across a session
        is RTC drift, and it is what lets device-clock SD timestamps be mapped
        onto the host time base.
        """
        host_ts = sent_at if sent_at is not None else host_now()
        offset = ""
        if device_time is not None:
            offset = f"{device_time.timestamp() - host_ts:.3f}"
        _append_csv(
            self.clock_sync_path,
            ["host_time", "host_iso", "device", "device_time", "offset_s"],
            [f"{host_ts:.6f}", host_iso(host_ts), device_name,
             device_time.isoformat(sep=" ") if device_time else "", offset],
        )


class SessionLogger:
    """Narrative log plus a machine-readable interaction record.

    Buffers from the moment the tab opens, so when a recording starts the log
    already contains the lead-up (scan, connect, mode changes). Both sinks get
    every action: ``session_log.txt`` to read, ``interactions.csv`` to analyse.
    """

    FIELDS = ["host_time", "host_iso", "source", "device", "action", "detail", "result"]

    def __init__(self):
        self._buffered = []          # (values...) awaiting a session directory
        self._text = None
        self._csv_path = None

    def log(self, action, device="", detail="", source="user", result="ok"):
        """Record one action. ``source`` is user / scheduler / device / system."""
        ts = host_now()
        row = [f"{ts:.6f}", host_iso(ts), source, device, action, detail, result]
        line = f"{row[1]}  [{source}] {device + ': ' if device else ''}{action}" \
               + (f" — {detail}" if detail else "") \
               + ("" if result == "ok" else f" [{result}]")

        if self._text is not None:
            self._text.write(line + "\n")
            self._text.flush()
            _append_csv(self._csv_path, self.FIELDS, row)
        else:
            self._buffered.append((line, row))
        return line

    def attach(self, session):
        """Point the logger at a session, flushing anything buffered."""
        self._text = open(session.log_path, "a", encoding="utf-8")
        self._csv_path = session.interactions_path
        for line, row in self._buffered:
            self._text.write(line + "\n")
            _append_csv(self._csv_path, self.FIELDS, row)
        self._text.flush()
        self._buffered.clear()

    def detach(self):
        if self._text is not None:
            self._text.close()
            self._text = None
        self._csv_path = None


class DeviceEventLog:
    """Appends one row per behavioural event for a single device.

    Counts are the device's absolute totals, so the file stays correct across a
    reconnection even if events were missed while the link was down.
    """

    FIELDS = ["host_time", "host_iso", "device", "event",
              "device_time", "left", "right", "pellet"]

    def __init__(self, path, device_name):
        self.path = path
        self.device_name = device_name

    def append(self, event, host_ts=None):
        ts = host_ts if host_ts is not None else host_now()
        _append_csv(self.path, self.FIELDS, [
            f"{ts:.6f}",
            host_iso(ts),
            self.device_name,
            event.kind,
            event.device_time.isoformat(sep=" ") if event.device_time else "",
            event.counts.get("left", ""),
            event.counts.get("right", ""),
            event.counts.get("pellet", ""),
        ])

    def read_event_times(self, kind=None):
        """Host timestamps of logged events, for rebuilding a plot on resume."""
        times = []
        try:
            with open(self.path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if kind and row.get("event") != kind:
                        continue
                    try:
                        times.append(datetime.fromtimestamp(float(row["host_time"])))
                    except (KeyError, TypeError, ValueError):
                        continue
        except OSError:
            pass
        return times


def find_resumable_sessions(base_dir, max_age_hours=48):
    """Sessions under ``base_dir`` that were never closed cleanly, newest first.

    Anything older than ``max_age_hours`` is ignored — resuming a week-old
    session is far more likely to be a mistake than an intent.
    """
    found = []
    cutoff = host_now() - max_age_hours * 3600
    try:
        entries = os.listdir(base_dir)
    except OSError:
        return found

    for entry in entries:
        root = os.path.join(base_dir, entry)
        state_path = os.path.join(root, "session_state.json")
        if not entry.endswith(FOLDER_SUFFIX) or not os.path.isfile(state_path):
            continue
        try:
            with open(state_path, encoding="utf-8") as f:
                state = json.load(f)
        except (OSError, ValueError):
            continue
        if state.get("status") != STATUS_RUNNING:
            continue
        try:
            mtime = os.path.getmtime(state_path)
        except OSError:
            mtime = 0
        if mtime < cutoff:
            continue
        found.append((root, state, mtime))

    found.sort(key=lambda item: item[2], reverse=True)
    return [(root, state) for root, state, _ in found]


def _append_csv(path, fields, row):
    """Append a row, writing the header if the file is new. Flushed per row.

    Per-row flushing is deliberate: an interrupted session must leave every
    event that already happened on disk.
    """
    if path is None:
        return
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new:
            writer.writerow(fields)
        writer.writerow(row)
        f.flush()


# Windows refuses these as file or directory names, with or without an
# extension, and silently drops trailing dots. A lab that names a cage after the
# port it is on — "COM3" — would otherwise get an unwritable session folder on
# the machine that actually runs the experiments.
_WINDOWS_RESERVED = frozenset(
    ["con", "prn", "aux", "nul"]
    + [f"com{i}" for i in range(1, 10)]
    + [f"lpt{i}" for i in range(1, 10)])


def _safe_name(name):
    """Filesystem-safe version of a user-supplied device name."""
    cleaned = "".join(c if c.isalnum() or c in "-_. " else "_" for c in str(name)).strip()
    cleaned = cleaned.replace(" ", "_").rstrip(".")
    if cleaned.split(".")[0].lower() in _WINDOWS_RESERVED:
        cleaned += "_"
    return cleaned or "unnamed"
