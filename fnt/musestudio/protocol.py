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
    # What the eyes are doing: "fixate" (open, show a fixation target),
    # "closed" (dim the screen), or "" to infer it from the text.
    gaze: str = ""
    # Capability token this phase needs (e.g. "stereo_audio"). When the host
    # cannot supply it, the phase is skipped rather than run degraded.
    #
    # This exists because the available hardware should decide what data gets
    # collected. A block that tests left/right audio is meaningless over laptop
    # speakers, where both ears hear both channels -- running it anyway would
    # produce a row of data that looks comparable to a headphone session and is
    # not. Skipping is recorded in events.csv, so a later analysis can see that
    # the block was omitted rather than failed.
    requires: str = ""

    def spoken(self):
        return self.speech or self.instruction.replace("\n", " ")

    def gaze_mode(self):
        """Whether this phase wants a fixation target or a dark screen.

        Inferred from the wording when not set explicitly, so protocols written
        before this existed behave correctly without edits.
        """
        if self.gaze:
            return self.gaze
        text = f"{self.instruction} {self.speech}".lower()
        if "eyes open" in text or "keep your eyes open" in text:
            return "fixate"
        if "close your eyes" in text or "eyes closed" in text:
            return "closed"
        return ""


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


def _hemisync_probe_s():
    """Speakers-only sibling of Probe A — no headphones required.

    A binaural beat cannot survive open air: both tones reach both ears and
    physically sum, so the interaural difference that defines "binaural" is
    gone. But what that summation *produces* is an amplitude-modulated
    waveform whose envelope beats at the difference frequency — a monaural
    beat. The cochlea receives a real 10 Hz envelope, which drives the
    auditory steady-state response far more robustly than a binaural beat
    (whose beat exists only after binaural convergence in the brainstem).

    For hemisphere synchronization specifically that may be the *better*
    stimulus: both ears — and via bilateral auditory projections, both
    hemispheres — receive the identical physical rhythm. Two oscillators
    driven by one external clock are phase-locked to each other.

    Practical speaker constraints baked in: 440 Hz carrier (laptop speakers
    roll off steeply below ~300 Hz, so Probe A's 200 Hz would be thin), 100%
    modulation depth, and the set-up phase plays the tone while you adjust
    the volume, since speaker level varies with room and distance.

    Same control structure as Probe A, so the two probes are directly
    comparable: headphones/binaural vs speakers/physical-AM.
    """
    return Protocol(
        key="probe_s",
        name="Hemi-Sync Probe S  (speakers, 9 min)",
        description=("Speakers-only controlled probe: eyes-open → eyes-closed "
                     "→ steady tone → 10 Hz AM tone → rest. No headphones."),
        phases=[
            Phase(
                "Set up",
                "No headphones needed — this probe uses the laptop speakers.\n\n"
                "A steady tone is playing: set your volume to comfortable-but-"
                "clear, sit at arm's length from the laptop, then Continue.",
                None,
                actions=["audio_control"],
                params={"base": 440},
                speech="This session uses the laptop speakers. No headphones. "
                       "A steady tone is playing now. Set your volume so it is "
                       "comfortable but clearly audible, sit at about arm's "
                       "length from the laptop, and press continue when ready.",
            ),
            Phase(
                "Eyes open",
                "Eyes OPEN. Rest your gaze on one spot and stay still.\n\n"
                "Silence for this control block.",
                60,
                actions=["audio_off", "start_recording"],
                speech="Block one. Keep your eyes open and rest your gaze on "
                       "one spot. Stay still and breathe normally for one minute.",
            ),
            Phase(
                "Eyes closed rest",
                "Now CLOSE your eyes and rest. No sound this block.",
                90,
                actions=["calibrate"],
                speech="Block two. Now close your eyes and simply rest. No "
                       "sound for the next ninety seconds.",
            ),
            Phase(
                "Control tone",
                "Eyes closed. A steady unmodulated tone — no pulse.",
                120,
                actions=["audio_control"],
                params={"base": 440},
                speech="Block three. Keep your eyes closed. A steady tone is "
                       "starting now. Just rest with it for two minutes.",
            ),
            Phase(
                "AM tone 10 Hz",
                "Eyes closed. The same tone, now pulsing at 10 Hz.\n\n"
                "Let the pulsing settle; don't try to force anything.",
                180,
                actions=["audio_on"],
                params={"base": 440, "beat": 10, "closed_loop": False,
                        "mode": "monaural_am"},
                speech="Block four. The tone now pulses ten times per second. "
                       "Keep your eyes closed and let the pulsing settle. "
                       "Three minutes.",
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
                speech="Probe complete. You can open your eyes. There is a "
                       "short questionnaire on screen.",
            ),
        ],
    )


def _hemisync_probe_quick():
    """5-minute relaxation-oriented probe — the default for a new person.

    Tuned after session 1, where the subject rated discomfort 9/10 and showed
    no entrainment at all. The lesson was that comfort is a precondition for
    the effect rather than a courtesy, so this version uses the soft harmonic
    timbre at 40% modulation depth, a 220 Hz carrier (warm rather than the
    piercing 440 Hz), and a gentle fade into every tone.

    It still carries all three of Probe S's controls — eyes-open baseline,
    matched unmodulated control tone, closing rest block — so results remain
    directly comparable with the 9-minute runs. The trade is statistical: about
    half the analysis windows per block, so only larger effects will clear the
    noise. Get someone through this comfortably first; run the long version
    when something looks worth pinning down.
    """
    return Protocol(
        key="probe_quick",
        name="Relax + Hemi-Sync  (speakers, 5 min)",
        description=("Short, comfortable controlled probe: eyes-open → settle → "
                     "control tone → 10 Hz AM → rest. Soft timbre, 40% depth."),
        phases=[
            Phase(
                "Set up",
                "No headphones needed — this uses the laptop speakers.\n\n"
                "A soft tone is playing: set the volume comfortable-but-clear, "
                "sit at arm's length, then Continue.",
                None,
                actions=["audio_control"],
                params={"base": 220, "timbre": "soft", "depth": 0.40},
                speech="This session uses the laptop speakers. A steady tone is "
                       "playing now. Set the volume so it is comfortable but "
                       "clearly audible, and press continue when you are ready.",
            ),
            Phase(
                "Eyes open",
                "Eyes OPEN. Rest your gaze on one spot and stay still.",
                40,
                actions=["audio_off", "start_recording"],
                speech="Block one. Keep your eyes open, rest your gaze on one "
                       "spot, and stay still for forty seconds.",
            ),
            Phase(
                "Eyes closed rest",
                "Now CLOSE your eyes and rest. No sound this block.",
                60,
                actions=["calibrate"],
                speech="Block two. Close your eyes and let yourself settle. "
                       "Breathe normally. One minute of quiet.",
            ),
            Phase(
                "Control tone",
                "Eyes closed. The same soft tone, steady — no pulse.",
                70,
                actions=["audio_control"],
                params={"base": 220, "timbre": "soft", "depth": 0.40},
                speech="Block three. Keep your eyes closed. A steady tone is "
                       "starting now.",
            ),
            Phase(
                "AM tone 10 Hz",
                "Eyes closed. The same tone, now pulsing at 10 Hz.",
                90,
                actions=["audio_on"],
                params={"base": 220, "beat": 10, "closed_loop": False,
                        "mode": "monaural_am", "timbre": "soft",
                        "depth": 0.40},
                speech="Block four. A slow pulse is fading into the tone. "
                       "Keep your eyes closed, let your breathing settle, and "
                       "let the pulse carry you.",
            ),
            Phase(
                "Rest again",
                "Eyes closed, sound fading. Rest as in block two.",
                40,
                actions=["audio_fade_out"],
                speech="Block five. The sound is fading. Keep your eyes closed "
                       "and rest for a final forty seconds.",
            ),
            Phase(
                "Done",
                "Done. Open your eyes — a short questionnaire follows.",
                None,
                actions=["stop_recording"],
                speech="All done. You can open your eyes. There is a short "
                       "questionnaire on screen.",
            ),
        ],
    )



def _flight_calibration():
    """5-minute calibration for Flight Mode. Answers what M01's data cannot.

    Flight Mode flies on relative alpha normalized against the pilot's own
    baseline, and every constant in that control law -- dead zone, full-scale,
    artifact vetoes -- currently rests on a single 9-minute session in which
    both ear electrodes failed within two minutes and the subject was only
    half-engaged. This protocol exists to replace those numbers with real ones.

    Five blocks, each answering a specific open question:

    1. EYES OPEN (45 s) -- establishes the baseline.

       The length is measured, not guessed. It was 75 s, sized for M01's poor
       recording (1.04 clean windows per second on her best electrode). A clean
       subject is far better: CV01 measured 9.65/s on AF7 at 93% clean, with an
       EMG ratio of 0.03 against M01's 0.99. At that yield the flight
       controller's 100-window baseline is satisfied in ten seconds.

       45 s is therefore NOT set by window count -- it is set by independence.
       Overlapping 2 s windows are not separate observations, and resting alpha
       waxes and wanes over tens of seconds, so the block has to span several of
       those cycles to characterise the distribution rather than a single crest.
       45 s gives ~22 non-overlapping windows and covers several alpha cycles.
       Going much shorter buys comfort by making every later z-score noisier.

    2. EYES CLOSED, REST (75 s) -- the neutral flight state. Fixes how far above
       the eyes-open baseline simply closing the eyes puts the pilot, which sets
       where the craft trims out.

    3. EYES CLOSED, DEEPEN (60 s) -- THE decisive block. Can the pilot
       voluntarily push alpha ABOVE their own eyes-closed rest? If yes, Flight
       Mode is genuine continuous neurofeedback. If no, it is a one-shot
       eyes-open/eyes-closed switch dressed up as flying, and the honest move is
       to redesign around that rather than pretend.

    4. EYES CLOSED, ALERT (45 s) -- serial subtraction, eyes still closed. Tests
       the DOWN direction: can alpha be voluntarily suppressed without opening
       the eyes and without moving? Gives the control law a bidirectional range
       instead of relying on passive decay for descent.

    5. ARTIFACTS, DELIBERATE (45 s) -- blink, then clench, then move the head, on
       cue. A positive control for the rejection path: the veto thresholds are
       currently calibrated on one subject, and this measures directly whether
       they fire on THIS pilot's artifacts. It also gives every later analysis a
       labelled example of each contaminant.

    AUDIO POLICY: spoken guidance only, and no stimulus tone in any measurement
    block. The control signal is alpha, and a rhythmic sound is exactly the wrong
    thing to introduce while measuring it -- any block-to-block difference has to
    be attributable to the pilot's mental state rather than to a soundtrack. The
    Set-up phase plays one quiet reference tone so the subject can set a safe
    volume for the guidance, and after that the session is silent apart from the
    voice. Earbuds are strongly preferred: the subject's eyes are closed, so
    speech is their only channel, and headphones keep it off the room.
    """
    return Protocol(
        key="flight_cal",
        name="Flight Calibration  (4 min)",
        description=("Calibrates Flight Mode: eyes-open baseline, eyes-closed "
                     "rest, voluntary deepen, voluntary alert, and a labelled "
                     "artifact block. No audio stimulus."),
        phases=[
            Phase(
                "Set up",
                "1. Put in your earbuds. The status line shows which output the "
                "Mac will use — if it does not say headphones, switch it now.\n\n"
                "2. A quiet reference tone is playing. Raise your system volume "
                "until the spoken guidance is comfortably clear. It starts "
                "deliberately soft; do not set it loud.\n\n"
                "3. Settle the headband until all four contact dots are green — "
                "get the ear sensors onto bare skin, clipping hair back if "
                "needed.\n\n4. Sit comfortably, rest your hands in your lap, and "
                "unclench your jaw.\n\nClick Continue when ready.",
                None,
                actions=["audio_check"],
                speech="Put in your earbuds, and raise your volume until this "
                       "voice is comfortably clear. It is deliberately quiet to "
                       "start with. Then settle the headband until the contact "
                       "indicators are good, clipping any hair away from the ear "
                       "sensors. Sit comfortably, rest your hands, and let your "
                       "jaw hang loose. Press continue when you are ready.",
            ),
            Phase(
                "Eyes open",
                "Eyes OPEN. Rest your gaze on the cross and hold still.\n\n"
                "Try not to clench your jaw. Blink normally — don't suppress it.",
                45,
                actions=["start_recording"],
                gaze="fixate",
                speech="Block one. Keep your eyes open and rest your gaze on the "
                       "cross. Hold still, let your jaw stay loose, and blink "
                       "normally. The ring shows the time left.",
            ),
            Phase(
                "Eyes closed rest",
                "Close your eyes. Rest. Do nothing in particular.\n\n"
                "Don't try to relax or concentrate — just let your mind idle.",
                60,
                gaze="closed",
                speech="Block two. Close your eyes now and simply rest. Do not try "
                       "to relax and do not try to concentrate — let your mind "
                       "idle. One minute.",
            ),
            Phase(
                "Deepen",
                "Eyes still closed.\n\nNow go DEEPER — let yourself sink, soften "
                "your attention, and drift inward as far as you can.",
                60,
                gaze="closed",
                speech="Block three, and this is the important one. Keep your eyes "
                       "closed, and now go deeper. Let yourself sink. Soften your "
                       "attention and drift inward as far as you can, and hold it "
                       "there. Sixty seconds.",
            ),
            Phase(
                "Alert",
                "Eyes still closed — do NOT open them.\n\nNow become sharply "
                "alert: count backwards from 300 in steps of 7, silently and as "
                "fast as you can.",
                30,
                gaze="closed",
                speech="Block four. Keep your eyes closed, but now become sharply "
                       "alert. Silently count backwards from three hundred in "
                       "sevens, as quickly as you can. Do not move or speak. "
                       "Thirty seconds.",
            ),
            # Split into three separately-labelled 15 s phases rather than one
            # 45 s block with pauses in the script. Two reasons: spoken cues land
            # on time (a single string is read straight through in about 25 s,
            # so the pilot would be clenching during the head-turn window), and
            # each contaminant gets its own interval in events.csv, which is what
            # lets the analysis report a veto rate per artifact type instead of
            # one blended number.
            # Eye movement leads the artifact set because it is the contaminant
            # most likely to be mistaken for frontal signal: AF7/AF8 sit directly
            # over the eyes, so a gaze sweep drags the corneo-retinal dipole
            # across them. Having a labelled example makes it separable later.
            # It is deliberately NOT part of the eyes-open baseline, which has to
            # stay the cleanest reference in the session.
            Phase(
                "Artifact: eye movement",
                "Eyes OPEN.\n\nFollow the moving dot smoothly with your eyes.\n\n"
                "Keep your head still — move only your eyes.",
                12,
                gaze="pursuit",
                speech="Now open your eyes and follow the moving dot smoothly, "
                       "using only your eyes. Keep your head completely still.",
            ),
            Phase(
                "Artifact: blinks",
                "Eyes closed.\n\nBlink HARD, about once a second.",
                12,
                gaze="closed",
                speech="Close your eyes again. Now blink hard, about once a "
                       "second, starting now.",
            ),
            Phase(
                "Artifact: jaw clench",
                "Eyes closed.\n\nStop blinking. Now CLENCH your jaw hard, "
                "release, and clench again.",
                12,
                gaze="closed",
                speech="Stop blinking. Now clench your jaw hard, release, and "
                       "clench again, over and over.",
            ),
            Phase(
                "Artifact: head motion",
                "Eyes closed.\n\nUnclench. Now slowly turn your head "
                "left and right.",
                12,
                gaze="closed",
                speech="Unclench your jaw. Now slowly turn your head left and "
                       "right, and keep going until I tell you to stop.",
            ),
            # HEADPHONES ONLY. Skipped entirely on speakers, where both ears
            # hear both channels and a left/right block would produce data that
            # looks comparable to a headphone session but is not. Placed after
            # every measurement block so the core five minutes are identical
            # regardless of gear -- only this optional extra appears or does not.
            Phase(
                "Lateral audio check",
                "Eyes closed. Quiet tones will alternate between your ears.\n\n"
                "Do nothing — just listen.",
                24,
                gaze="closed",
                requires="stereo_audio",
                actions=["lateral_cues"],
                speech="One extra block, because you are wearing headphones. "
                       "Quiet tones will alternate between your left and right "
                       "ear. Do nothing at all — just listen with your eyes "
                       "closed.",
            ),
            Phase(
                "Done",
                "Stop moving and hold still.\n\nCalibration complete — you can "
                "open your eyes and take the headband off.\n\nThe recording has been saved.",
                None,
                actions=["stop_recording"],
                speech="And stop. Hold still. Calibration complete — you can open "
                       "your eyes and remove the headband. The recording has been "
                       "saved.",
            ),
        ],
    )


PROTOCOLS = {p.key: p for p in [_flight_calibration(), _hemisync_probe_quick(),
                                _hemisync_probe_a(), _hemisync_probe_s(),
                                _binaural_10min(), _heterodyne()]}


class ProtocolRunner(QObject):
    """Steps a Protocol through its phases on a QTimer."""

    phase_started = pyqtSignal(object, int, int)   # phase, index, total
    tick = pyqtSignal(float, float)                # remaining, phase_duration
    finished = pyqtSignal()
    aborted = pyqtSignal()
    phase_skipped = pyqtSignal(object, str)        # phase, missing capability

    def __init__(self, parent=None):
        super().__init__(parent)
        self._protocol = None
        self._i = -1
        self._phase_end = 0.0
        self._phase_dur = 0.0
        self._waiting = False
        self._capabilities = set()
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._on_tick)

    def set_capabilities(self, tokens):
        """Declare what the current hardware can do (see audio_output)."""
        self._capabilities = set(tokens or ())

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
        if phase.requires and phase.requires not in self._capabilities:
            self.phase_skipped.emit(phase, phase.requires)
            self._enter_next()
            return
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
