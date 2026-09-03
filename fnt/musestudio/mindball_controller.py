"""Live Mindball: EEG -> calm score -> ball, with telemetry and a ghost opponent.

Thin, like the flight controller: timers, a buffer, a CSV writer. All judgement
lives in :mod:`mindball` (game) and :mod:`meditation` (calm), which are pure and
testable without a headband.
"""

import csv
import glob
import json
import os

import numpy as np
from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from fnt.musestudio.meditation import MeditationIndex, UPDATE_HZ
from fnt.musestudio.mindball import (
    GhostOpponent, MindballGame, SyntheticOpponent,
)

ELECTRODES = ("TP9", "AF7", "AF8", "TP10")
# Telemetry rows are written once per physics step, so a stored trace plays back
# at the render rate rather than the analysis rate.
GHOST_HZ = 30.0
# Physics AND rendering both run here. Stepping the ball only at the 2 Hz
# analysis rate made it visibly jump twice a second — the calm estimate is
# genuinely slow, but the ball's motion need not inherit that. Integrating at
# 30 Hz against the most recent calm value gives continuous motion without
# pretending the measurement is faster than it is.
RENDER_FPS = 30


def load_ghost(recording_dir):
    """Build a ghost from a previous match's telemetry, if one exists."""
    path = os.path.join(recording_dir, "Data", "Analysis", "mindball_telemetry.csv")
    if not os.path.exists(path):
        return None
    try:
        vals = []
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                v = row.get("my_calm")
                if v not in (None, ""):
                    vals.append(float(v))
        # Reject traces that would make a meaningless opponent.
        #
        # A ghost is only as good as the session it came from, and early
        # sessions recorded a calm column that was zero while the app was still
        # calibrating. Replaying one of those produces an opponent with a median
        # calm of 0.018 that cannot lose — which looks like a win and teaches
        # nothing. Require real duration and a genuinely active trace.
        arr = np.asarray(vals, dtype=float)
        if len(arr) < int(30 * UPDATE_HZ):          # under ~30 s of play
            return None
        if float(np.median(arr)) < 0.05:            # essentially flat/dead
            return None
        if float((arr <= 1e-9).mean()) > 0.10:      # long dead stretches
            return None
        label = os.path.basename(recording_dir.rstrip("/"))[:19]
        # Ghost traces are written at the physics rate, which is what the
        # 'hz' here must match for time-indexed playback to run at real speed.
        return GhostOpponent(vals, hz=GHOST_HZ, label=f"you, {label[-6:]}")
    except Exception:  # noqa: BLE001
        return None


def latest_ghost(base_dir):
    """Most recent past match, so 'play yourself' needs no file picking."""
    for d in sorted(glob.glob(os.path.join(base_dir, "*_FNT_MuseStudio_recording")),
                    reverse=True):
        g = load_ghost(d)
        if g is not None:
            return g
    return None


class MindballController(QObject):
    frame = pyqtSignal(object, str, object)   # BallState, opponent label, history
    status = pyqtSignal(str)
    calibrating = pyqtSignal(float)
    finished = pyqtSignal(object)        # summary dict

    def __init__(self, parent=None):
        super().__init__(parent)
        self.index = MeditationIndex(ELECTRODES)
        self.game = None
        self._cols = {}
        self._t = 0.0
        self._csv = None
        self._writer = None
        self._announced = False
        self._calm = 0.0
        self._drowsy = False
        self._history = []           # (t, my_calm, their_calm) for the traces
        self._control = QTimer(self)
        self._control.setInterval(int(1000 / UPDATE_HZ))
        self._control.timeout.connect(self._measure)
        self._render = QTimer(self)
        self._render.setInterval(int(1000 / RENDER_FPS))
        self._render.timeout.connect(self._step_and_emit)

    # ---------------------------------------------------------------- setup
    def start(self, opponent=None, telemetry_dir=None, fs=256.0):
        # No calibration block. raw_calm() is scale-free, so the match starts as
        # soon as the analysis window fills (about 2 s) — as the original did.
        self.index = MeditationIndex(ELECTRODES, fs=fs)
        # No past match yet -> a synthetic partner, so the first game is
        # playable rather than blocked on having played before.
        self.game = MindballGame(opponent or SyntheticOpponent())
        self._t = 0.0
        self._announced = False
        self._calm = 0.0
        self._history = []
        if telemetry_dir:
            self._open_telemetry(telemetry_dir)
        self._control.start()
        self._render.start()
        self.status.emit("Reading your signal…")

    def stop(self):
        self._control.stop()
        self._render.stop()
        self._close()

    def is_running(self):
        return self._control.isActive()

    def set_channels(self, names):
        self._cols = {}
        upper = [str(n).upper() for n in (names or [])]
        for e in ELECTRODES:
            for i, n in enumerate(upper):
                if e in n:
                    self._cols[e] = i
                    break
        if len(self._cols) < len(ELECTRODES) and len(upper) >= 4:
            self._cols = {e: i for i, e in enumerate(ELECTRODES)}

    def add_eeg(self, names, data):
        if not self._cols:
            self.set_channels(names)
        if data is None or len(data) == 0:
            return
        for e, i in self._cols.items():
            if i < data.shape[1]:
                self.index.push(e, data[:, i])

    # ----------------------------------------------------------------- loop
    def _measure(self):
        """Update the calm estimate. Slow by nature — 2 Hz over 4 s windows."""
        if not self.index.ready():
            return
        calm, drowsy = self.index.raw_calm()
        if calm is None:
            return
        self._drowsy = drowsy
        # Dozing must not win, so a drowsy window contributes no calm at all.
        self._calm = 0.0 if drowsy else float(calm)
        if not self._announced:
            self._announced = True
            self.status.emit("Match on. The ball rolls away from whoever is calmer.")

    def _step_and_emit(self):
        if self.game is None:
            return
        dt = 1.0 / RENDER_FPS
        self._t += dt
        if not self._announced:
            self.status.emit("Reading your signal…")
            return
        st = self.game.step(self._calm, dt, t=self._t)
        self._history.append((st.t, st.my_calm, st.their_calm))
        if len(self._history) > int(RENDER_FPS * 180):
            self._history.pop(0)
        self._write(st)
        self.frame.emit(st, getattr(self.game.opponent, "label", "opponent"),
                        self._history)
        if st.winner:
            self.status.emit({"player": "You win.",
                              "opponent": "Your opponent wins.",
                              "draw": "Draw — neither of you could hold a lead."}
                             .get(st.winner, ""))
            summary = self.game.summary()
            self.stop()
            self.finished.emit(summary)

    # ------------------------------------------------------------ telemetry
    def _open_telemetry(self, out_dir):
        try:
            os.makedirs(out_dir, exist_ok=True)
            self._csv = open(os.path.join(out_dir, "mindball_telemetry.csv"),
                             "w", newline="")
            self._writer = csv.DictWriter(
                self._csv,
                fieldnames=["t", "position", "velocity", "my_calm", "their_calm",
                            "winner", "wraps", "drowsy"])
            self._writer.writeheader()
        except Exception as exc:  # noqa: BLE001
            self._csv = self._writer = None
            self.status.emit(f"Telemetry disabled: {exc}")

    def _write(self, st):
        if self._writer is None:
            return
        try:
            row = st.as_row()
            row["drowsy"] = int(self._drowsy)
            self._writer.writerow(row)
        except Exception:  # noqa: BLE001
            pass

    def _close(self):
        if self._csv is not None:
            try:
                self._csv.close()
            except Exception:  # noqa: BLE001
                pass
        self._csv = self._writer = None
