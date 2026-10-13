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

Two destinations
----------------
An earlier version pulled every file on the card into the session folder, with
its offsets stored inside that folder. Two things followed. Every session
re-downloaded the entire card from byte zero, because a fresh session started
with no offsets; and files were pulled in directory order, so the file actually
being written — the only one the running experiment cares about — was pulled
*last*, behind however many months of history the card had accumulated. On a
card with 23 files the live data was roughly a minute behind from the start, and
a session that ended sooner than that captured none of it.

So the mirror writes to two places:

``session_dir``
    The files this session owns — the log the device opened when recording
    started, plus any it rolls over to later. Pulled first on every pass, so the
    running experiment is never behind. Self-contained: the session folder holds
    this session's data and nothing else.

``archive_dir``
    Everything else on the card, kept once per device under the sessions root
    rather than once per session. Backfilled a few files at a time *after* the
    session files are current, so historical data still reaches the host without
    ever delaying live data.
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

# Historical files copied per pass. Bounded because a pass holds the link for its
# whole duration: draining a large backlog in one go would block the next
# event-triggered pull of the live file for as long as it took.
ARCHIVE_FILES_PER_PASS = 2

# A mirrored file with no more than this many lines is treated as a stub that the
# device is entitled to recycle. FED3's getFilename() deletes and reuses any log
# with fewer than three lines, so after NEW_TRIAL a "new" log can arrive under a
# name the host already has bytes for — with different content. Re-pulling from
# zero is right for a stub; a shrunken file with real data in it is an anomaly
# that must be reported rather than silently overwritten.
RECYCLABLE_STUB_LINES = 3


class _MirrorStore:
    """One directory of mirrored files, plus the byte offsets reached in each.

    Offsets live in ``mirror_state.json`` beside the files, so a mirror survives
    an FNT crash and resumes from the last verified byte.
    """

    def __init__(self, directory):
        self.dir = directory
        self.state_path = os.path.join(directory, "mirror_state.json")
        self.offsets = self._load()

    def path(self, filename):
        return os.path.join(self.dir, os.path.basename(filename))

    def local_size(self, filename):
        try:
            return os.path.getsize(self.path(filename))
        except OSError:
            return 0

    def reconcile(self, filename):
        """Trust the file on disk over remembered state, and return its size.

        The file is the thing that would actually be appended to, so a stale or
        missing offset must not be allowed to write into the middle of it.
        """
        have = self.local_size(filename)
        if have != self.offsets.get(filename):
            self.offsets[filename] = have
        return have

    def looks_recycled(self, filename):
        """Whether the local copy is a stub the device may legitimately reuse."""
        try:
            with open(self.path(filename), "rb") as f:
                return f.read().count(b"\n") <= RECYCLABLE_STUB_LINES
        except OSError:
            return True

    def reset(self, filename):
        """Forget a file so the next pull re-fetches it from byte zero."""
        self.offsets[filename] = 0
        try:
            os.remove(self.path(filename))
        except OSError:
            pass

    def append(self, filename, start_offset, payload):
        """Write ``payload`` at ``start_offset``, truncating any stale tail."""
        path = self.path(filename)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        mode = "r+b" if os.path.exists(path) and start_offset > 0 else "wb"
        with open(path, mode) as f:
            f.seek(start_offset)
            f.write(payload)
            f.truncate()
            f.flush()
            os.fsync(f.fileno())
        self.offsets[filename] = start_offset + len(payload)
        return len(payload)

    def _load(self):
        try:
            with open(self.state_path, encoding="utf-8") as f:
                state = json.load(f)
            return {k: int(v) for k, v in state.get("offsets", {}).items()}
        except (OSError, ValueError, TypeError):
            return {}

    def save(self):
        tmp = f"{self.state_path}.tmp"
        try:
            os.makedirs(self.dir, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"offsets": self.offsets}, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.state_path)
        except OSError:
            pass    # mirroring continues; offsets are re-derived from file sizes


class DeviceMirror(QObject):
    """Maintains a local byte-exact copy of one device's SD-card logs."""

    progress = pyqtSignal(str)              # human-readable status
    failed = pyqtSignal(str)
    updated = pyqtSignal(str, int, int)     # filename, bytes_added, total_bytes

    def __init__(self, transfer, session_dir, archive_dir, parent=None):
        super().__init__(parent)
        self._transfer = transfer
        self.session = _MirrorStore(session_dir)
        self.archive = _MirrorStore(archive_dir)

        # Filenames this session owns. Seeded by the device's reported current
        # file and extended whenever it rolls over to a new one.
        self.session_files = []
        self.current_file = None

        self.enabled = True
        self.last_error = None
        self._syncing = False
        self._queue = []                    # [(store, filename, device_size)]
        self._backlog = 0                   # archived files still to copy

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self.sync_now)

        self._periodic = QTimer(self)
        self._periodic.timeout.connect(self.sync_now)
        self._periodic.start(PERIODIC_MS)

        self._cooldown = QTimer(self)       # enforces MIN_INTERVAL_MS
        self._cooldown.setSingleShot(True)

    # --- session scope ----------------------------------------------------

    def adopt_current_file(self, filename):
        """Record the log the device is writing to now.

        Called with the filename from STATUS at session start and from
        ``NEW_TRIAL_STARTED`` when the device rolls over, so the set of files the
        session owns tracks the device rather than being guessed from names.
        """
        if not filename:
            return
        filename = filename.strip()
        if not filename:
            return
        self.current_file = filename
        if filename not in self.session_files:
            self.session_files.append(filename)

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

    @property
    def busy(self):
        """Whether a pull is in flight *or* waiting for the link to free up.

        The debounce timer counts: ``sync_now`` defers to a link that is busy
        with an export by re-arming it, without ever setting ``_syncing``. Read
        as idle, the final pull at session close returned immediately and the
        bytes written since the previous pull were left on the SD card.
        """
        return self._syncing or self._debounce.isActive()

    def stop(self):
        """Stop scheduling new pulls. Any pull already in flight still finishes."""
        self._debounce.stop()
        self._periodic.stop()
        self.enabled = False

    # --- pull sequence ----------------------------------------------------

    def _on_listed(self, ok, data):
        if not ok:
            self._syncing = False
            self.last_error = str(data)
            self.failed.emit(f"mirror: could not list files ({data})")
            return

        sizes = dict(data)

        # The live file leads, then the rest of this session's files, then a
        # bounded slice of the backlog. Ordering is the whole point: a pass that
        # started with the archive would leave the running experiment waiting.
        ordered = []
        if self.current_file in sizes:
            ordered.append(self.current_file)
        ordered += [f for f in self.session_files
                    if f in sizes and f not in ordered]

        self._queue = []
        for filename in ordered:
            if self._needs_pull(self.session, filename, sizes[filename]):
                self._queue.append((self.session, filename, sizes[filename]))

        archive_due = []
        for filename, device_size in data:
            if filename in ordered:
                continue
            if self._needs_pull(self.archive, filename, device_size):
                archive_due.append((self.archive, filename, device_size))
        self._queue += archive_due[:ARCHIVE_FILES_PER_PASS]

        if not self._queue:
            self._finish("mirror: up to date")
            return

        self._backlog = max(0, len(archive_due) - ARCHIVE_FILES_PER_PASS)
        self._pull_next()

    def _needs_pull(self, store, filename, device_size):
        """Whether ``filename`` has bytes the host does not have.

        A device file that is *smaller* than the local copy has been replaced,
        not appended to: FED3 recycles the name of any log with fewer than three
        lines, so a fresh trial can reopen a name the host already mirrored. Left
        alone, the host would keep the old content and later append the new log's
        tail onto it, producing a file that never existed on the card.
        """
        have = store.reconcile(filename)
        if have > device_size:
            if not store.looks_recycled(filename):
                self.failed.emit(
                    f"mirror: {filename} shrank on the device "
                    f"({have} local vs {device_size} remote); left untouched")
                return False
            store.reset(filename)
            self.progress.emit(f"mirror: {filename} was recycled, re-copying")
            return device_size > 0
        return have < device_size

    def _pull_next(self):
        if not self._queue:
            self._finish(
                "mirror: up to date" if not self._backlog else
                f"mirror: live data up to date "
                f"({self._backlog} archived files still to copy)")
            return

        store, filename, device_size = self._queue[0]
        offset = store.offsets.get(filename, 0)
        scope = "live" if filename == self.current_file else (
            "session" if store is self.session else "archive")
        self.progress.emit(
            f"mirror: {filename} {offset}/{device_size} bytes ({scope})")
        # ``start`` stays the third parameter: the transfer passes the offset the
        # device actually echoed back as a positional argument, which must
        # override the default rather than land in another slot.
        self._transfer.download(
            filename, offset,
            lambda ok, payload, start=offset, s=store, f=filename:
                self._on_chunk(ok, payload, s, start, f))

    def _on_chunk(self, ok, payload, store=None, start_offset=0, filename=None):
        if self._queue:
            self._queue.pop(0)

        if not ok:
            self._syncing = False
            self.last_error = str(payload)
            # Offsets are unchanged, so the next pass retries the same range.
            self.failed.emit(f"mirror: {filename} failed ({payload})")
            return

        try:
            added = store.append(filename, start_offset, payload)
        except OSError as exc:
            self._syncing = False
            self.last_error = str(exc)
            self.failed.emit(f"mirror: could not write {filename} ({exc})")
            return

        store.save()
        if added:
            self.updated.emit(filename, added, store.offsets[filename])

        self._pull_next()

    def _finish(self, message):
        self._syncing = False
        self.session.save()
        self.archive.save()
        self.progress.emit(message)
