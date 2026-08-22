"""Spoken guidance for MuseStudio.

Eyes-closed sessions are the normal case here, which makes on-screen
instructions useless once the session starts. This speaks them instead.

Utterances are queued and played by a background worker so the GUI never
blocks, and ``started``/``finished`` are emitted around each one so the host can
duck the binaural tone — otherwise the voice competes with the beats and you
miss the instruction.

Backends, in order of preference:

* macOS ``say`` — built in, good voices, no extra dependency
* ``pyttsx3`` — cross-platform, if installed
* nothing — every call becomes a silent no-op
"""

import queue
import shutil
import subprocess
import sys
import threading

from PyQt5.QtCore import QObject, pyqtSignal

_STOP = object()


def _macos_available():
    return sys.platform == "darwin" and shutil.which("say") is not None


def _pyttsx3_available():
    try:
        import pyttsx3  # noqa: F401
        return True
    except Exception:
        return False


def list_voices():
    """Available English voices as ``[(display_name, voice_id)]``."""
    if _macos_available():
        try:
            out = subprocess.run(["say", "-v", "?"], capture_output=True,
                                 text=True, timeout=5).stdout
        except Exception:
            return []
        voices = []
        for line in out.splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            locale = next((p for p in parts if "_" in p and len(p) == 5), None)
            if locale is None or not locale.startswith("en"):
                continue
            name = line[:line.index(locale)].strip()
            if name:
                voices.append((f"{name} ({locale})", name))
        return sorted(voices)
    if _pyttsx3_available():
        try:
            import pyttsx3
            engine = pyttsx3.init()
            found = [(v.name, v.id) for v in engine.getProperty("voices")]
            engine.stop()
            return found
        except Exception:
            return []
    return []


class Speaker(QObject):
    """Queued, non-blocking text-to-speech."""

    started = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.enabled = True
        self.voice = None
        self.rate = 175                     # words per minute
        self._queue = queue.Queue()
        self._proc = None
        self._proc_lock = threading.Lock()
        self._backend = ("macos" if _macos_available()
                         else "pyttsx3" if _pyttsx3_available() else None)
        self._thread = threading.Thread(target=self._run, name="speaker",
                                        daemon=True)
        self._thread.start()

    # --- capability -------------------------------------------------------
    def available(self):
        return self._backend is not None

    def backend_name(self):
        return {"macos": "macOS say", "pyttsx3": "pyttsx3"}.get(
            self._backend, "unavailable")

    # --- control ----------------------------------------------------------
    def say(self, text, interrupt=True):
        """Queue an utterance. ``interrupt`` drops anything still pending —
        during a session the newest instruction is the only relevant one."""
        if not text or not self.enabled or not self.available():
            return
        if interrupt:
            self.stop()
        self._queue.put(str(text))

    def stop(self):
        """Silence immediately and drop the queue."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        with self._proc_lock:
            proc = self._proc
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass

    def shutdown(self):
        self.stop()
        self._queue.put(_STOP)

    # --- worker -----------------------------------------------------------
    def _run(self):
        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            try:
                self.started.emit()
                self._speak_blocking(item)
            except Exception:
                pass
            finally:
                self.finished.emit()

    def _speak_blocking(self, text):
        if self._backend == "macos":
            cmd = ["say", "-r", str(int(self.rate))]
            if self.voice:
                cmd += ["-v", str(self.voice)]
            cmd.append(text)
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
            with self._proc_lock:
                self._proc = proc
            proc.wait()
            with self._proc_lock:
                self._proc = None
        elif self._backend == "pyttsx3":
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", int(self.rate))
            if self.voice:
                engine.setProperty("voice", self.voice)
            engine.say(text)
            engine.runAndWait()
            engine.stop()


def spoken_duration(seconds):
    """Turn a phase length into something natural to hear."""
    seconds = int(round(seconds or 0))
    if seconds <= 0:
        return ""
    if seconds < 60:
        return f"{seconds} seconds"
    minutes, rest = divmod(seconds, 60)
    unit = "minute" if minutes == 1 else "minutes"
    if rest == 0:
        return f"{minutes} {unit}"
    return f"{minutes} {unit} {rest} seconds"
