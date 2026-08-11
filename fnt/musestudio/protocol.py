"""Guided experiment protocols for MuseStudio.

A protocol is an ordered list of phases. Each phase shows the user an
instruction and either runs for a fixed duration or waits for the user to
click Continue. On entering a phase the runner emits its actions (start
recording, calibrate baseline, turn audio on/off, stop recording), which the
host window carries out.

The first preset is a 10-minute eyes-closed alpha binaural-beat protocol:
2 min baseline (also calibrates), 6 min stimulation, 2 min recovery. Durations
live here as data, so tuning the trial is a one-line change.
"""

import time
from dataclasses import dataclass, field

from PyQt5.QtCore import QObject, QTimer, pyqtSignal


@dataclass
class Phase:
    name: str
    instruction: str                  # shown on screen
    duration: float | None            # seconds; None = wait for user Continue
    actions: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    gate: str = ""                    # e.g. "synced": advance early once condition holds
    ramp: dict = None                 # e.g. {"param": "heterodyne_offset", "to": 1.0}
    # Spoken version. Written for listening, not reading — no "click", no line
    # breaks, and it says how long the phase lasts since you can't see the timer.
    speech: str = ""

    def spoken(self):
        return self.speech or self.instruction.replace("\n", " ")


@dataclass
class Protocol:
    key: str
    name: str
    description: str
    phases: list


def _binaural_10min():
    return Protocol(
        key="hemisync1",
        name="Hemisphere Synchronization Protocol #1",
        description=("Eyes-closed 10 Hz (alpha) binaural-beat trial: "
                     "2 min baseline, 6 min stimulation, 2 min recovery."),
        phases=[
            Phase(
                "Set up",
                "Put on the Muse headband and adjust it until contact is good "
                "(watch the head map on the right). Put in your earphones, sit "
                "comfortably, and relax your jaw.\n\nClick Continue when you are ready.",
                None,
                speech="Put on the Muse headband and adjust it until contact is "
                       "good. Put in your earphones, sit comfortably, and relax "
                       "your jaw. Press continue when you are ready.",
            ),
            Phase(
                "Baseline",
                "Close your eyes, stay still, and breathe normally.\n\n"
                "We are measuring your resting brain rhythm for 2 minutes.",
                120,
                actions=["start_recording", "calibrate"],
                speech="Baseline. Close your eyes, stay still, and breathe "
                       "normally for two minutes while we measure your resting "
                       "rhythm.",
            ),
            Phase(
                "Stimulation",
                "Keep your eyes closed and relax.\n\nBinaural beats are now playing — "
                "let the sound settle. The tone becomes purer as your hemispheres "
                "synchronize.",
                360,
                actions=["audio_on"],
                params={"base": 200, "beat": 10, "closed_loop": True},
                speech="Stimulation. Keep your eyes closed and relax. The tones "
                       "are starting now. Let the sound settle for the next six "
                       "minutes.",
            ),
            Phase(
                "Recovery",
                "The tones are fading out.\n\nKeep your eyes closed and rest quietly "
                "for 2 more minutes.",
                120,
                actions=["audio_fade_out"],
                speech="Recovery. The tones are fading out. Keep your eyes "
                       "closed and rest quietly for two more minutes.",
            ),
            Phase(
                "Done",
                "Protocol complete — you can open your eyes and remove the headband.\n\n"
                "Your recording has been saved.",
                None,
                actions=["stop_recording"],
                speech="Protocol complete. You can open your eyes and remove the "
                       "headband. Your recording has been saved.",
            ),
        ],
    )


def _heterodyne():
    return Protocol(
        key="heterodyne1",
        name="Hemisphere Heterodyne Protocol #2",
        description=("Sync both hemispheres in alpha (monaural AM, closed-loop "
                     "gated), then ramp a small interhemispheric offset (0→1 Hz) "
                     "to induce a heterodyne, then recover."),
        phases=[
            Phase(
                "Set up",
                "Put on the Muse headband (good contact on the head map) and your "
                "earphones. Sit comfortably and relax your jaw.\n\nClick Continue "
                "when you are ready.",
                None,
                speech="Put on the Muse headband and check the contact is good. "
                       "Put in your earphones, sit comfortably, and relax your "
                       "jaw. Press continue when you are ready.",
            ),
            Phase(
                "Baseline",
                "Close your eyes, stay still, breathe normally.\n\nMeasuring your "
                "resting rhythm for 2 minutes.",
                120,
                actions=["start_recording", "calibrate"],
                speech="Baseline. Close your eyes, stay still, and breathe "
                       "normally for two minutes.",
            ),
            Phase(
                "Sync induction",
                "Eyes closed. Both ears pulse together at 10 Hz — let your "
                "hemispheres fall into sync. This stage ends automatically once "
                "you hold synchronization (up to 5 min).",
                300,                       # safety cap; ends early when synced
                actions=["audio_on"],
                params={"base": 200, "beat": 10, "closed_loop": True,
                        "mode": "monaural_am"},
                gate="synced",
                speech="Sync induction. Eyes closed. Both ears are now pulsing "
                       "together at ten hertz. Let your hemispheres fall into "
                       "sync. This stage ends on its own once you hold it.",
            ),
            Phase(
                "Heterodyne",
                "Eyes closed. One hemisphere is now drifting slightly faster — "
                "notice the slow pulsing between the two. Stay relaxed.",
                240,
                actions=["heterodyne_start"],
                ramp={"param": "heterodyne_offset", "to": 1.0},  # 0→1 Hz over the stage
                speech="Heterodyne. One side is now drifting slightly faster than "
                       "the other. Notice the slow pulsing between them. Stay "
                       "relaxed for the next four minutes.",
            ),
            Phase(
                "Recovery",
                "The tones are fading out.\n\nKeep your eyes closed and rest "
                "quietly for 2 minutes.",
                120,
                actions=["audio_fade_out"],
                speech="Recovery. The tones are fading out. Keep your eyes closed "
                       "and rest quietly for two minutes.",
            ),
            Phase(
                "Done",
                "Protocol complete — you can open your eyes and remove the "
                "headband.\n\nYour recording has been saved.",
                None,
                actions=["stop_recording"],
                speech="Protocol complete. You can open your eyes and remove the "
                       "headband. Your recording has been saved.",
            ),
        ],
    )


def _hemisync_probe_a():
    """Session 1 of the hemi-sync iteration: a controlled 9-minute probe.

    This is deliberately *not* "listen to beats and hope PLV rises". Without
    controls, a null result is uninterpretable — bad electrodes, an insensitive
    metric, and a genuinely ineffective stimulus all look identical. So the run
    carries three internal controls:

    1. **Eyes-open → eyes-closed** is a positive control. Alpha reliably rises
       on eye closure (Berger effect); if it doesn't, the recording is bad and
       nothing else in the session should be believed.
    2. **A matched control tone** (same carrier, both ears, no frequency
       difference) precedes the binaural block, so the primary contrast
       isolates *the beat* rather than "hearing a tone" or "paying attention".
    3. **Rest blocks bracket the intervention**, so a change that is really
       time-on-task or growing drowsiness shows up as rest-late ≈ binaural.

    Audio is open-loop throughout — closed-loop feedback would make the
    stimulus depend on the measurement, which is circular when the measurement
    is the thing being evaluated.
    """
    return Protocol(
        key="probe_a",
        name="Hemi-Sync Probe A  (9 min, controlled)",
        description=("Controlled probe: eyes-open → eyes-closed → control tone "
                     "→ 10 Hz binaural → rest. Open loop, for measurement."),
        phases=[
            Phase(
                "Set up",
                "Put on the headband and adjust until contact is good, put in "
                "your earphones, and sit comfortably.\n\nClick Continue when ready.",
                None,
                speech="Put on the headband and adjust it until the contact is "
                       "good. Put in your earphones and sit comfortably. Press "
                       "continue when you are ready.",
            ),
            Phase(
                "Eyes open",
                "Eyes OPEN. Rest your gaze on one spot and stay still.\n\n"
                "This is the control block — please don't close your eyes yet.",
                60,
                actions=["start_recording"],
                speech="Block one. Keep your eyes open and rest your gaze on one "
                       "spot. Stay still and breathe normally for one minute.",
            ),
            Phase(
                "Eyes closed rest",
                "Now CLOSE your eyes and rest. No sound this block.",
                90,
                actions=["calibrate"],
                speech="Block two. Now close your eyes and simply rest. There "
                       "will be no sound for the next ninety seconds.",
            ),
            Phase(
                "Control tone",
                "Eyes closed. A steady tone plays in both ears — no beat.",
                120,
                actions=["audio_control"],
                params={"base": 200},
                speech="Block three. Keep your eyes closed. A steady tone is "
                       "starting now. Just rest with it for two minutes.",
            ),
            Phase(
                "Binaural 10 Hz",
                "Eyes closed. 10 Hz binaural beat (200 / 210 Hz).\n\n"
                "Let the pulsing settle; don't try to force anything.",
                180,
                actions=["audio_on"],
                params={"base": 200, "beat": 10, "closed_loop": False},
                speech="Block four. The tone now has a ten hertz pulse in it. "
                       "Keep your eyes closed and let the pulsing settle. Don't "
                       "try to force anything. Three minutes.",
            ),
            Phase(
                "Rest again",
                "Eyes closed, sound fading. Rest exactly as in block two.",
                90,
                actions=["audio_fade_out"],
                speech="Block five. The sound is fading out. Keep your eyes "
                       "closed and rest for a final ninety seconds.",
            ),
            Phase(
                "Done",
                "Probe complete. Open your eyes — a short questionnaire follows.",
                None,
                actions=["stop_recording"],
                speech="Probe complete. You can open your eyes. There is a short "
                       "questionnaire on screen.",
            ),
        ],
    )


PROTOCOLS = {p.key: p for p in [_hemisync_probe_a(), _binaural_10min(),
                                _heterodyne()]}


class ProtocolRunner(QObject):
    """Steps a Protocol through its phases on a QTimer."""

    phase_started = pyqtSignal(object, int, int)   # phase, index, total
    tick = pyqtSignal(float, float)                # remaining, phase_duration
    finished = pyqtSignal()
    aborted = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._protocol = None
        self._i = -1
        self._phase_end = 0.0
        self._phase_dur = 0.0
        self._waiting = False
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._on_tick)

    def start(self, protocol):
        self._protocol = protocol
        self._i = -1
        self._enter_next()

    def is_running(self):
        return self._protocol is not None

    def waiting_for_user(self):
        return self._waiting

    def _enter_next(self):
        self._i += 1
        if self._protocol is None or self._i >= len(self._protocol.phases):
            self._timer.stop()
            self._protocol = None
            self.finished.emit()
            return
        phase = self._protocol.phases[self._i]
        self.phase_started.emit(phase, self._i, len(self._protocol.phases))
        if phase.duration is None:
            self._waiting = True
            self._timer.stop()
        else:
            self._waiting = False
            self._phase_dur = float(phase.duration)
            # Absolute end time -> countdown immune to QTimer jitter/drift.
            self._phase_end = time.monotonic() + self._phase_dur
            self.tick.emit(self._phase_dur, self._phase_dur)
            self._timer.start()

    def _on_tick(self):
        remaining = self._phase_end - time.monotonic()
        if remaining <= 0:
            self._timer.stop()
            self._enter_next()
        else:
            self.tick.emit(remaining, self._phase_dur)

    def advance(self):
        """User pressed Continue on a wait-for-user phase."""
        if self._waiting:
            self._waiting = False
            self._enter_next()

    def skip_to_next(self):
        """Advance a timed phase early (e.g. a gate condition was met)."""
        if self._protocol is not None and not self._waiting:
            self._timer.stop()
            self._enter_next()

    def abort(self):
        self._timer.stop()
        self._protocol = None
        self._waiting = False
        self.aborted.emit()
