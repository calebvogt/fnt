"""Continuous mirroring of a FED3's SD card to the host.

The failure this exists to prevent: a trial ends, the device is unreachable, and
the only copy of the data is on an SD card that has to be pulled by hand.

Rather than treating the SD card as an archive to be downloaded at the end, the
host keeps a **byte-exact running copy** of every ``FED*.CSV`` on the card. After
each behavioural event (debounced) and on a periodic tick, the mirror asks the
device for its file sizes and pulls only the bytes it does not already have.

Why incremental rather than "resend the whole CSV on every poke": a full resend
occupies the serial link for the entire file on every event, and it is precisely
the long uninterruptible transfer that wedged devices before. Range transfers
are short, resumable, and CRC-verified per chunk, so an interruption costs one
chunk rather than the session. The end state is the same — a complete CSV on the
host — but it stays complete continuously instead of only at the end.

Offsets live in ``mirror_state.json`` next to the mirrored files, so a mirror
survives an FNT crash and resumes from the last verified byte.
"""

import json
import os

from PyQt5.QtCore import QObject, QTimer, pyqtSignal

# Pull at most this often even if events are arriving continuously; a busy device
# would otherwise spend the whole session servicing transfers.
MIN_INTERVAL_MS = 15000

# Quiet period after an event before pulling, so a burst of pokes costs one
# transfer rather than one per poke.
EVENT_DEBOUNCE_MS = 3000

# Backstop pull for a device that is simply idle (no pokes to trigger a sync).
PERIODIC_MS = 120000


class DeviceMirror(QObject):
    """Maintains a local byte-exact copy of one device's SD-card logs."""

    progress = pyqtSignal(str)              # human-readable status
    failed = pyqtSignal(str)
    updated = pyqtSignal(str, int, int)     # filename, bytes_added, total_bytes

    def __init__(self, transfer, mirror_dir, parent=None):
        super().__init__(parent)
        self._transfer = transfer
        self.mirror_dir = mirror_dir
        self.state_path = os.path.join(mirror_dir, "mirror_state.json")
        self.offsets = self._load_state()
        self.enabled = True
        self.last_error = None
        self._syncing = False
        self._queue = []                    # [(filename, device_size)] pending this pass

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self.sync_now)

        self._periodic = QTimer(self)
        self._periodic.timeout.connect(self.sync_now)
        self._periodic.start(PERIODIC_MS)

        self._cooldown = QTimer(self)       # enforces MIN_INTERVAL_MS
        self._cooldown.setSingleShot(True)

    # --- triggers ---------------------------------------------------------

    def note_event(self):
        """Called on each behavioural event; schedules a debounced pull."""
        if self.enabled and not self._debounce.isActive():
            self._debounce.start(EVENT_DEBOUNCE_MS)

    def sync_now(self, force=False):
        """Pull anything new. No-op if a pull is already running or cooling down."""
        if not self.enabled or self._syncing:
            return
        if not force and self._cooldown.isActive():
            return
        if self._transfer.busy:
            # A user-initiated export is using the link; try again shortly.
            self._debounce.start(EVENT_DEBOUNCE_MS)
            return

        self._syncing = True
        self._cooldown.start(MIN_INTERVAL_MS)
        self._transfer.list_files(self._on_listed)

    def stop(self):
        self._debounce.stop()
        self._periodic.stop()

    # --- pull sequence ----------------------------------------------------

    def _on_listed(self, ok, data):
        if not ok:
            self._syncing = False
            self.last_error = str(data)
            self.failed.emit(f"mirror: could not list files ({data})")
            return

        self._queue = []
        for filename, device_size in data:
            have = self._local_size(filename)
            if have != self.offsets.get(filename):
                # Trust the file on disk over remembered state: it is the thing
                # we would actually be appending to.
                self.offsets[filename] = have
            if have < device_size:
                self._queue.append((filename, device_size))

        if not self._queue:
            self._syncing = False
            self.progress.emit("mirror: up to date")
            self._save_state()
            return

        self._pull_next()

    def _pull_next(self):
        if not self._queue:
            self._syncing = False
            self._save_state()
            self.progress.emit("mirror: up to date")
            return

        filename, device_size = self._queue[0]
        offset = self.offsets.get(filename, 0)
        self.progress.emit(
            f"mirror: {filename} {offset}/{device_size} bytes")
        self._transfer.download(
            filename, offset,
            lambda ok, payload, start=offset, f=filename: self._on_chunk(ok, payload, start, f))

    def _on_chunk(self, ok, payload, start_offset=0, filename=None):
        if self._queue:
            self._queue.pop(0)

        if not ok:
            self._syncing = False
            self.last_error = str(payload)
            # Offsets are unchanged, so the next pass retries the same range.
            self.failed.emit(f"mirror: {filename} failed ({payload})")
            return

        try:
            added = self._append(filename, start_offset, payload)
        except OSError as exc:
            self._syncing = False
            self.last_error = str(exc)
            self.failed.emit(f"mirror: could not write {filename} ({exc})")
            return

        self.offsets[filename] = start_offset + added
        self._save_state()
        if added:
            self.updated.emit(filename, added, self.offsets[filename])

        self._pull_next()

    # --- disk -------------------------------------------------------------

    def _path(self, filename):
        return os.path.join(self.mirror_dir, os.path.basename(filename))

    def _local_size(self, filename):
        try:
            return os.path.getsize(self._path(filename))
        except OSError:
            return 0

    def _append(self, filename, start_offset, payload):
        """Write ``payload`` at ``start_offset``, truncating any stale tail."""
        path = self._path(filename)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        mode = "r+b" if os.path.exists(path) and start_offset > 0 else "wb"
        with open(path, mode) as f:
            f.seek(start_offset)
            f.write(payload)
            f.truncate()
            f.flush()
            os.fsync(f.fileno())
        return len(payload)

    def _load_state(self):
        try:
            with open(self.state_path, encoding="utf-8") as f:
                state = json.load(f)
            return {k: int(v) for k, v in state.get("offsets", {}).items()}
        except (OSError, ValueError, TypeError):
            return {}

    def _save_state(self):
        tmp = f"{self.state_path}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"offsets": self.offsets}, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.state_path)
        except OSError:
            pass    # mirroring continues; offsets are re-derived from file sizes
