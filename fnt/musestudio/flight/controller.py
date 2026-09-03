"""Live bridge: EEG stream -> control pipeline -> craft -> view + telemetry.

Everything decision-making lives in :mod:`pipeline` and :mod:`sim`, which are
pure and testable. This module is deliberately thin -- it owns a QTimer, a
buffer, and a CSV writer, and nothing that needs a headband to verify.

The 30 fps render cap is not arbitrary. This project has already been bitten by
the GUI thread holding the GIL long enough to starve an audio callback with 23 ms
of slack; a render loop is the same hazard, larger. The renderer measures about
0.86 ms per frame, so 30 fps costs under 3% of the budget and leaves the LSL
reader thread room to breathe. Control updates run slower still, at 10 Hz, since
the underlying 2 s analysis window makes anything faster a redundant re-read of
mostly the same samples.
"""

import csv

import numpy as np
import os

from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from fnt.musestudio.flight.pipeline import ControlConfig, ControlPipeline, UPDATE_HZ
from fnt.musestudio.flight.sim import CraftConfig, CraftSim, FlightPhase

RENDER_FPS = 30
ELECTRODES = ("TP9", "AF7", "AF8", "TP10")

# A timed, cued flight. Free flight tells you nothing: without an instruction to
# compare against there is no way to separate "the pilot commanded a climb" from
# "alpha happened to drift up". Alternating CLIMB/DESCEND blocks turn the flight
# into a measurement — if altitude tracks the cue, control is real.
# How long the settle block may over-run while waiting for a usable baseline.
# Bounded so a headband that never gets clean contact still fails visibly
# instead of hanging on the first block forever.
_SETTLE_MAX_EXTRA_S = 60.0

FLIGHT_TEST = [
    (55, "settle", "Eyes open. Rest your gaze while the baseline builds. "
                   "This waits until the signal is good enough to fly on."),
    (20, "climb",  "Close your eyes and let yourself settle. Climb."),
    (20, "descend", "Open your eyes. Look around the room."),
    (20, "climb",  "Close your eyes again. Climb."),
    (20, "descend", "Open your eyes."),
    (20, "climb",  "Eyes closed. Last climb."),
    (15, "descend", "Open your eyes. Test complete."),
]


class FlightController(QObject):
    """Runs a flight. Feed it EEG chunks; it emits frames."""

    frame = pyqtSignal(object, object)        # CraftState, PipelineTrace
    altitude_tone = pyqtSignal(float)         # 0..1 normalized height, for audio
    cue = pyqtSignal(str, str)                # label, spoken instruction
    finished = pyqtSignal()
    status = pyqtSignal(str)
    phase_changed = pyqtSignal(str)           # FlightPhase value
    calibrating = pyqtSignal(float)           # 0..1 baseline progress

    def __init__(self, parent=None, config=None, craft=None):
        super().__init__(parent)
        self.cfg = config or ControlConfig()
        self.pipeline = ControlPipeline(ELECTRODES, self.cfg)
        self.craft = CraftSim(craft or CraftConfig())
        self._cols = {}
        self._last_trace = None
        self._last_phase = self.craft.state.phase
        self._tilt = 0.0
        self._t = 0.0
        self._csv = None
        self._writer = None
        self._announced_ready = False

        self._control = QTimer(self)
        self._control.setInterval(int(1000 / UPDATE_HZ))
        self._control.timeout.connect(self._tick)

        self._render = QTimer(self)
        self._render.setInterval(int(1000 / RENDER_FPS))
        self._render.timeout.connect(self._emit_frame)

    # ------------------------------------------------------------- lifecycle
    def start(self, telemetry_dir=None, schedule=None):
        self.pipeline = ControlPipeline(ELECTRODES, self.cfg)
        self.craft.reset()
        self._last_trace = None
        self._t = 0.0
        self._announced_ready = False
        self._tilt = 0.0
        self._schedule = list(schedule) if schedule else None
        self._sched_i = -1
        self._sched_end = 0.0
        self._cue_label = ""
        if telemetry_dir:
            self._open_telemetry(telemetry_dir)
        self._control.start()
        self._render.start()
        self.status.emit("Calibrating — eyes open, rest your gaze and hold still.")

    def stop(self):
        self._control.stop()
        self._render.stop()
        self._close_telemetry()

    def is_running(self):
        return self._control.isActive()

    # ---------------------------------------------------------------- inputs
    def set_channels(self, names):
        """Map electrode -> column index in the EEG stream.

        Falls back to the standard Muse order when labels do not match, the same
        way SynchronyAnalyzer does — OpenMuse has been seen to label channels
        generically, and a flight that silently reads the wrong columns is worse
        than one that refuses to start.
        """
        self._cols = {}
        upper = [str(n).upper() for n in (names or [])]
        for e in ELECTRODES:
            for i, n in enumerate(upper):
                if e in n:
                    self._cols[e] = i
                    break
        if len(self._cols) < len(ELECTRODES) and len(upper) >= 4:
            self._cols = {e: i for i, e in enumerate(ELECTRODES)}
            self.status.emit("EEG channel labels unrecognised — assuming standard "
                             "Muse order TP9, AF7, AF8, TP10.")

    def add_motion(self, names, data):
        """Head IMU -> steering tilt. NOT part of the cortical control path.

        Kept on a separate input from add_eeg so the two can never be confused:
        altitude comes from the cortex, heading from the neck, and the EEG
        artifact gate actively rejects the head movement that steers.
        """
        if data is None or len(data) == 0:
            return
        idx = None
        for i, n in enumerate(names or []):
            u = str(n).upper()
            if "GYRO_Y" in u or "GYRO_Z" in u:
                idx = i
                break
        if idx is None or idx >= data.shape[1]:
            return
        # Gyro is a rate; integrate lightly and decay, so a sustained tilt holds
        # a turn while a quick glance does not.
        rate = float(np.mean(data[:, idx]))
        self._tilt = float(np.clip(0.88 * self._tilt + rate / 90.0, -1.0, 1.0))

    def current_tilt(self):
        return self._tilt

    def add_eeg(self, names, data):
        if not self._cols:
            self.set_channels(names)
        if data is None or len(data) == 0:
            return
        for e, i in self._cols.items():
            if i < data.shape[1]:
                self.pipeline.push(e, data[:, i])

    # ----------------------------------------------------------------- loops
    def _tick(self):
        dt = 1.0 / UPDATE_HZ
        self._t += dt
        if not self.pipeline.ready():
            return
        trace = self.pipeline.tick(t=self._t)
        self._last_trace = trace

        if not self.pipeline.baseline_ready():
            self.calibrating.emit(self.pipeline.baseline_progress())
        elif not self._announced_ready:
            self._announced_ready = True
            self.calibrating.emit(1.0)
            self.status.emit("Baseline set — close your eyes and let yourself settle.")

        self._advance_schedule()
        # Hold the craft on the ground through the settle block. The baseline is
        # still forming there, so z is measured against a moving reference and
        # can drift high enough to arm — which is what launched the craft during
        # the eyes-OPEN block, sank it immediately, and left the pilot grounded
        # before the first climb cue ever arrived.
        armed = not (self._schedule and self._cue_label in ("", "settle"))
        self.craft.step(trace.thrust if armed else 0.0, dt, t=self._t,
                        tilt=self._tilt if armed else 0.0)
        phase = self.craft.state.phase
        if phase is not self._last_phase:
            self._last_phase = phase
            self.phase_changed.emit(phase.value)
            if phase is FlightPhase.AIRBORNE:
                self.status.emit("Airborne.")
            elif phase is FlightPhase.LANDED:
                self.status.emit(
                    f"Down — peak altitude {self.craft.state.peak_altitude:.0f}. "
                    "Climb again whenever you are ready.")
                # Deliberately NOT stop(). During a cued test the pilot is asked
                # to descend on purpose; ending the flight the first time they
                # succeed is precisely backwards. The schedule decides when the
                # run is over, nothing else.
                if not self._schedule:
                    self.stop()
        self._write_row(trace)

    def _advance_schedule(self):
        """Step the cued CLIMB/DESCEND sequence and stop at the end."""
        if not self._schedule:
            return
        if self._sched_i >= 0 and self._t < self._sched_end:
            return
        # Hold on the settle block until the baseline genuinely exists. The
        # first run wasted its opening climb AND descend cue -- both flat, z
        # pinned at 0.00 -- because only 1.23 electrodes were clean early on and
        # the baseline took ~65 s to form against a fixed 25 s block. Cueing a
        # pilot to fly a craft that cannot yet respond teaches them the control
        # does not work.
        if (self._sched_i >= 0
                and self._schedule[self._sched_i][1] == "settle"
                and not self.pipeline.baseline_ready()
                and self._t < self._sched_end + _SETTLE_MAX_EXTRA_S):
            return
        self._sched_i += 1
        if self._sched_i >= len(self._schedule):
            # Stop FIRST, then announce once. Emitting a status here made the
            # window speak "flight test complete", and _on_flight_finished then
            # spoke it a second time — two full utterances back to back, with
            # the screen still live behind them. That gap is what read as the
            # simulation carrying on after it had ended.
            self.stop()
            self.finished.emit()
            return
        # Leaving the settle block: fix the thrust mapping to this session's own
        # resting spread before any cue asks the pilot to fly.
        if (self._sched_i > 0
                and self._schedule[self._sched_i - 1][1] == "settle"):
            dz, fs_z = self.pipeline.calibrate_thresholds()
            self.status.emit(f"Calibrated: dead zone {dz:.1f} z, full scale {fs_z:.1f} z.")
            self._settle_z_n = len(self.pipeline._settle_z)

        dur, label, speech = self._schedule[self._sched_i]
        self._sched_end = self._t + dur
        self._cue_label = label
        self.cue.emit(label, speech)

    def current_cue(self):
        return self._cue_label

    def _emit_frame(self):
        self.frame.emit(self.craft.state, self._last_trace)
        # Audio is the pilot's ONLY feedback channel — the eyes are closed by
        # construction, since that is where the alpha comes from. Pitch tracks
        # altitude because it is the one sonification nobody has to be taught.
        self.altitude_tone.emit(
            max(0.0, min(1.0, self.craft.state.altitude / self.craft.cfg.ceiling)))

    # ------------------------------------------------------------- telemetry
    def _open_telemetry(self, out_dir):
        """One row per control tick, every pipeline stage preserved.

        The flight is flown with the eyes closed, so this file is the only way
        the pilot can afterwards ask why the craft did what it did. Writing the
        intermediate stages rather than just the thrust is what makes that
        answerable instead of a matter of trust.
        """
        try:
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, "flight_telemetry.csv")
            self._csv = open(path, "w", newline="")
            sample = self.pipeline.tick(t=0.0).as_row()
            sample.update(self.craft.state.as_row())
            sample["cue"] = ""
            sample["tilt"] = 0.0
            sample["heading"] = 0.0
            self._writer = csv.DictWriter(self._csv, fieldnames=list(sample))
            self._writer.writeheader()
            # That probe tick consumed a window; rebuild so the flight starts clean.
            self.pipeline = ControlPipeline(ELECTRODES, self.cfg)
        except Exception as exc:  # noqa: BLE001
            self._csv = None
            self._writer = None
            self.status.emit(f"Telemetry disabled: {exc}")

    def _write_row(self, trace):
        if self._writer is None:
            return
        try:
            row = trace.as_row()
            row.update(self.craft.state.as_row())
            row["cue"] = self._cue_label
            row["tilt"] = round(self._tilt, 4)
            row["heading"] = round(self.craft.state.heading, 4)
            self._writer.writerow(row)
        except Exception:  # noqa: BLE001
            pass

    def _close_telemetry(self):
        if self._csv is not None:
            try:
                self._csv.close()
            except Exception:  # noqa: BLE001
                pass
        self._csv = None
        self._writer = None
