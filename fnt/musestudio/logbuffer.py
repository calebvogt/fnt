"""In-app diagnostic log for MuseStudio.

Collects everything useful for troubleshooting into one buffer the user can
read and copy:

* Python ``logging`` records and uncaught exceptions (full traceback)
* Qt warnings (``qInstallMessageHandler``)
* **Native stdout/stderr**, teed at the file-descriptor level

That last one matters. The messages that actually explain a failed session —
liblsl's ``Stream transmission broke off ... re-connecting`` and OpenCV's
``camera failed to properly initialize`` — are written by C++ libraries
straight to fd 1/2. Replacing ``sys.stderr`` in Python never sees them, so we
duplicate the descriptor into a pipe and forward every line to both the real
terminal and this buffer.
"""

import os
import platform
import sys
import threading
import traceback
from collections import deque
from datetime import datetime

MAX_LINES = 4000


class LogBuffer:
    """Thread-safe ring buffer of timestamped log lines."""

    def __init__(self, maxlen=MAX_LINES):
        self._lines = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._installed = False
        self._tees = []

    # --- collection -------------------------------------------------------
    def add(self, text, source="app"):
        stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        for raw in str(text).rstrip().splitlines() or [""]:
            with self._lock:
                self._lines.append(f"{stamp}  [{source}] {raw}")

    def lines(self):
        with self._lock:
            return list(self._lines)

    def text(self):
        return "\n".join(self.lines())

    def clear(self):
        with self._lock:
            self._lines.clear()

    def count(self):
        with self._lock:
            return len(self._lines)

    # --- report -----------------------------------------------------------
    def report(self):
        """Log text with an environment header — what to paste when asking for help."""
        head = ["=== MuseStudio diagnostic report ===",
                f"generated : {datetime.now().isoformat(timespec='seconds')}",
                f"platform  : {platform.platform()}",
                f"python    : {sys.version.split()[0]}"]
        for mod in ("PyQt5.QtCore", "numpy", "scipy", "pyqtgraph", "mne_lsl",
                    "bleak", "sounddevice", "cv2", "pandas"):
            head.append(f"{mod:<10}: {_version_of(mod)}")
        head.append(f"log lines : {self.count()}")
        head.append("=" * 36)
        return "\n".join(head) + "\n" + self.text()

    # --- installation -----------------------------------------------------
    def install(self):
        """Hook Python logging, exceptions, Qt messages and the native fds."""
        if self._installed:
            return
        self._installed = True
        self._install_logging()
        self._install_excepthook()
        self._install_qt_handler()
        self._install_fd_tee(1, "stdout")
        self._install_fd_tee(2, "stderr")
        self.add("Diagnostic logging started", "log")

    def _install_logging(self):
        import logging

        buffer = self

        class _Handler(logging.Handler):
            def emit(self, record):
                try:
                    buffer.add(self.format(record), record.name.split(".")[0])
                except Exception:
                    pass

        handler = _Handler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        root = logging.getLogger()
        root.addHandler(handler)
        if root.level == logging.NOTSET or root.level > logging.INFO:
            root.setLevel(logging.INFO)

    def _install_excepthook(self):
        previous = sys.excepthook

        def hook(exc_type, exc, tb):
            self.add("UNCAUGHT EXCEPTION\n"
                     + "".join(traceback.format_exception(exc_type, exc, tb)),
                     "error")
            previous(exc_type, exc, tb)

        sys.excepthook = hook

    def _install_qt_handler(self):
        try:
            from PyQt5.QtCore import qInstallMessageHandler

            def handler(mode, context, message):
                self.add(message, "qt")

            qInstallMessageHandler(handler)
        except Exception:
            pass

    def _install_fd_tee(self, fd, label):
        """Redirect ``fd`` into a pipe whose lines land in this buffer.

        By default the terminal is left **quiet**: everything native libraries
        write to stdout/stderr (liblsl reconnect errors, OpenCV camera
        complaints, netinterfaces spam) is captured for the Session Logs window
        instead of scrolling the user's shell. Set the environment variable
        ``FNT_LOG_ECHO=1`` to also echo captured lines back to the terminal —
        useful when running from source and watching live.
        """
        echo = os.environ.get("FNT_LOG_ECHO", "") == "1"
        try:
            saved = os.dup(fd)          # kept so the fd's destination survives
            read_fd, write_fd = os.pipe()
            os.dup2(write_fd, fd)
            os.close(write_fd)
        except Exception:
            return   # sandboxed/redirected environment — skip silently

        def pump():
            with os.fdopen(read_fd, "rb", buffering=0) as reader:
                pending = b""
                while True:
                    try:
                        chunk = reader.read(4096)
                    except Exception:
                        break
                    if not chunk:
                        break
                    if echo:
                        try:
                            os.write(saved, chunk)
                        except Exception:
                            pass
                    pending += chunk
                    *complete, pending = pending.split(b"\n")
                    for line in complete:
                        text = line.decode("utf-8", "replace").rstrip()
                        if text:
                            self.add(text, label)

        thread = threading.Thread(target=pump, name=f"logtee-{label}", daemon=True)
        thread.start()
        self._tees.append((fd, saved, thread))


def _version_of(module_name):
    try:
        module = __import__(module_name, fromlist=["__version__"])
        return getattr(module, "__version__", "installed")
    except Exception as exc:
        return f"unavailable ({type(exc).__name__})"


# Process-wide buffer.
LOG = LogBuffer()
