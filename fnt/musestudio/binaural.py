"""Binaural-beat generator for MuseStudio.

A binaural beat is produced by playing a slightly different pure tone in each
ear; the brain perceives a "beat" at the difference frequency. Here:

    left ear  = base (carrier) frequency
    right ear = base + beat frequency

Controls: base/carrier frequency (20-1500 Hz) and beat frequency (1-50 Hz).
Requires headphones — the effect depends on each ear hearing only one tone.

Audio is synthesized in real time with a phase-continuous ``sounddevice``
output stream, so frequency changes glide without clicks. Start/stop apply a
short gain ramp to avoid pops. The panel emits an ``event`` signal on
play/stop and on committed parameter changes so the host window can log the
protocol (with an LSL timestamp) alongside the Muse and video data.
"""

import json

import numpy as np
from scipy.signal import lfilter
from PyQt5.QtCore import Qt, QSettings, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QInputDialog,
    QLabel, QMessageBox, QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

SAMPLE_RATE = 44100
MAX_AMPLITUDE = 0.4        # headroom so summed stereo never clips


def _play_sequence(notes, volume=0.25):
    """Play ``[(freq_hz, duration_s, gap_s), ...]`` as one short cue.

    Best-effort and fire-and-forget; silently no-ops if no audio device.
    """
    try:
        import sounddevice as sd

        chunks = []
        for freq, dur, gap in notes:
            t = np.arange(int(dur * SAMPLE_RATE)) / SAMPLE_RATE
            env = np.clip(np.minimum(t / 0.01, (dur - t) / 0.02), 0, 1)
            chunks.append(np.sin(2 * np.pi * freq * t) * volume * env)
            if gap > 0:
                chunks.append(np.zeros(int(gap * SAMPLE_RATE)))
        tone = np.concatenate(chunks).astype(np.float32)
        sd.play(np.column_stack([tone, tone]), SAMPLE_RATE)
    except Exception:
        pass


# Three deliberately distinct cues, because with your eyes closed the sound is
# the only channel — a slipped electrode must never be mistaken for "you are
# not synchronizing well".
def play_cue(freq=880.0, dur=0.18, volume=0.25):
    """Single clear beep — a protocol phase has changed."""
    _play_sequence([(freq, dur, 0.0)], volume)


def play_alert(volume=0.3):
    """Low urgent double-beep — something is wrong (lost contact, lost stream)."""
    _play_sequence([(300.0, 0.14, 0.07), (300.0, 0.14, 0.0)], volume)


def play_resolved(volume=0.22):
    """Rising two-note — the problem just cleared."""
    _play_sequence([(520.0, 0.10, 0.02), (780.0, 0.14, 0.0)], volume)


def play_complete(volume=0.25):
    """Resolving three-note chime — the session is finished."""
    _play_sequence([(660.0, 0.16, 0.03), (880.0, 0.16, 0.03),
                    (1320.0, 0.30, 0.0)], volume)


def play_reward(volume=0.18):
    """Soft high chime — sustained target state reached."""
    _play_sequence([(1175.0, 0.22, 0.0)], volume)
BASE_MIN, BASE_MAX = 20, 1500
BEAT_MIN, BEAT_MAX = 1, 50

# Closed-loop feedback layers (peak contribution before master gain).
DETUNE_HZ = 4.0            # dissonant partial offset -> acoustic roughness
ROUGH_MAX = 0.5
NOISE_MAX = 0.35
REWARD_MAX = 0.35         # consonant perfect-fifth pad when synchronized

# Reward-on-sustain mode: hold the target state this long to earn a chime.
REWARD_LEVEL = 0.6
REWARD_HOLD_S = 3.0
REWARD_BLOOM_S = 2.5

# --- stimulus voicing ------------------------------------------------------
# Session 1 (M01) rated the stimulus 9/10 for discomfort — "the constant beep
# really irritated me" — and showed no 10 Hz entrainment. A subject bracing
# against a buzzer is not going to entrain, so comfort is a prerequisite for
# the effect, not a nicety. Three things were wrong: a bare sine at 440 Hz sits
# in a bright, piercing region; 100% modulation depth chops the sound on and
# off; and it started at full level instantly.
TIMBRES = [
    ("Soft tone", "soft"),      # harmonic-rich, organ-like — warm, not piercing
    ("Warm noise", "noise"),    # filtered noise, like distant rain
    ("Pure tone", "pure"),      # the original bare sine, kept for comparison
]
DEFAULT_TIMBRE = "soft"
# 40% depth still drives the auditory steady-state response but is far easier
# to sit with than the 100% on/off chopping used in session 1.
DEFAULT_MOD_DEPTH = 0.40
ONSET_SECONDS = 2.5             # gentle fade-in instead of a hard start


class BinauralPlayer:
    """Phase-continuous stereo sine generator (left = base, right = base+beat)."""

    def __init__(self):
        self.mode = "binaural"          # "binaural" or "monaural_am"
        self.left_freq = 200.0          # left ear tone (binaural) / carrier (AM)
        self.right_freq = 210.0         # right ear tone (binaural)
        self.am_left = 10.0             # left-ear modulation rate (AM mode)
        self.am_right = 10.0            # right-ear modulation rate (AM mode)
        self.timbre = DEFAULT_TIMBRE
        self.mod_depth = DEFAULT_MOD_DEPTH
        self._carrier_zi = np.zeros(1)  # noise-carrier filter state
        # Ramped gains (current -> target) for click-free changes.
        self._gain = self._t_gain = 0.0
        self._rough = self._t_rough = 0.0
        self._noise = self._t_noise = 0.0
        self._reward = self._t_reward = 0.0
        self._phase_l = 0.0
        self._phase_r = 0.0
        self._phase_aml = 0.0           # AM envelope phases
        self._phase_amr = 0.0
        self._phase_rough = 0.0
        self._phase_reward = 0.0
        self._noise_zi = np.zeros(1)   # one-pole lowpass state (brown-ish noise)
        self._stream = None

    def _ramp(self, current, target, n):
        return np.linspace(current, target, n, endpoint=False)

    def _carrier(self, idx, n):
        """The sound being modulated, per timbre.

        ``soft`` adds a fifth and an octave above the fundamental at falling
        amplitude. Laptop speakers roll off badly in the low end, so the
        partials also let a warm, low-sounding tone survive a small driver:
        the upper harmonics carry the timbre while the ear still hears the
        fundamental. ``noise`` low-passes white noise into something like
        distant rain, which is the least fatiguing option over minutes.
        """
        phase = self._phase_l
        w = 2 * np.pi * self.left_freq / SAMPLE_RATE
        if self.timbre == "noise":
            white = np.random.randn(n)
            out, self._carrier_zi = lfilter([0.08], [1, -0.92], white,
                                            zi=self._carrier_zi)
            return np.clip(out * 2.2, -1.0, 1.0)
        if self.timbre == "soft":
            # Strictly integer harmonics. A non-integer partial (a musical
            # fifth at 1.5x) beats against the fundamental at 0.5x — 110 Hz of
            # roughness for a 220 Hz carrier, which is the very harshness this
            # timbre exists to avoid. 2x and 3x share the fundamental's period
            # and simply thicken the tone.
            base = phase + w * idx
            return (np.sin(base)
                    + 0.35 * np.sin(2.0 * base)     # octave
                    + 0.15 * np.sin(3.0 * base)     # twelfth
                    ) / 1.50
        return np.sin(phase + w * idx)

    def _callback(self, outdata, frames, time_info, status):
        n = frames
        idx = np.arange(n)
        wl = 2 * np.pi * self.left_freq / SAMPLE_RATE
        wr = 2 * np.pi * self.right_freq / SAMPLE_RATE
        wrough = 2 * np.pi * (self.left_freq + DETUNE_HZ) / SAMPLE_RATE
        wreward = 2 * np.pi * (self.left_freq * 1.5) / SAMPLE_RATE  # perfect fifth

        if self.mode == "monaural_am":
            # Shared carrier (base), amplitude-modulated at each ear's own rate
            # -> more lateralized drive for interhemispheric heterodyning.
            carrier = self._carrier(idx, n)
            waml = 2 * np.pi * self.am_left / SAMPLE_RATE
            wamr = 2 * np.pi * self.am_right / SAMPLE_RATE
            # Envelope swings between (1 - depth) and 1 rather than 0 and 1, so
            # the rhythm is clearly present without the sound cutting out.
            d = float(np.clip(self.mod_depth, 0.0, 1.0))
            env_l = (1.0 - d) + d * (0.5 + 0.5 * np.sin(self._phase_aml + waml * idx))
            env_r = (1.0 - d) + d * (0.5 + 0.5 * np.sin(self._phase_amr + wamr * idx))
            core_l = carrier * env_l
            core_r = carrier * env_r
            self._phase_r = (self._phase_r + wr * n) % (2 * np.pi)  # keep advancing
            self._phase_aml = (self._phase_aml + waml * n) % (2 * np.pi)
            self._phase_amr = (self._phase_amr + wamr * n) % (2 * np.pi)
        else:
            # Binaural: each ear needs its own frequency, so the harmonic/noise
            # voicings (which share one carrier) don't apply here.
            core_l = np.sin(self._phase_l + wl * idx)
            core_r = np.sin(self._phase_r + wr * idx)
            self._phase_r = (self._phase_r + wr * n) % (2 * np.pi)
        # Phases advance whether or not the layer is audible, so switching a
        # layer on never produces a click.
        self._phase_l = (self._phase_l + wl * n) % (2 * np.pi)
        self._phase_rough = (self._phase_rough + wrough * n) % (2 * np.pi)
        self._phase_reward = (self._phase_reward + wreward * n) % (2 * np.pi)

        g_master = self._ramp(self._gain, self._t_gain, n)
        self._gain = self._t_gain

        # Only synthesize a feedback layer when it is actually audible now or
        # about to be. A plain tone (control block, open loop) previously still
        # cost three extra oscillators plus a filtered noise generator on every
        # callback; that work competes for the GIL with the GUI thread and is
        # what made the tone catch and stutter.
        shared = None

        def _add(component, gain_now, gain_target):
            nonlocal shared
            if gain_now <= 0.0 and gain_target <= 0.0:
                return
            ramp = self._ramp(gain_now, gain_target, n)
            shared = component * ramp if shared is None else shared + component * ramp

        if self._rough > 0.0 or self._t_rough > 0.0:
            _add(np.sin(self._phase_rough + wrough * idx), self._rough, self._t_rough)
        if self._reward > 0.0 or self._t_reward > 0.0:
            _add(np.sin(self._phase_reward + wreward * idx), self._reward, self._t_reward)
        if self._noise > 0.0 or self._t_noise > 0.0:
            white = np.random.randn(n)
            noise, self._noise_zi = lfilter([0.05], [1, -0.95], white,
                                            zi=self._noise_zi)
            _add(noise * 2.0, self._noise, self._t_noise)

        self._rough, self._noise = self._t_rough, self._t_noise
        self._reward = self._t_reward
        if shared is None:
            shared = 0.0
        left = np.clip((core_l + shared) * g_master, -1.0, 1.0)
        right = np.clip((core_r + shared) * g_master, -1.0, 1.0)
        outdata[:, 0] = left.astype(np.float32)
        outdata[:, 1] = right.astype(np.float32)

    def set_frequencies(self, left_freq, right_freq):
        self.left_freq = float(left_freq)
        self.right_freq = float(right_freq)

    def set_mode(self, mode):
        self.mode = "monaural_am" if mode == "monaural_am" else "binaural"

    def set_timbre(self, timbre):
        self.timbre = timbre if timbre in {t for _, t in TIMBRES} else DEFAULT_TIMBRE

    def set_mod_depth(self, depth):
        self.mod_depth = float(np.clip(depth, 0.0, 1.0))

    def set_am_freqs(self, left_hz, right_hz):
        self.am_left = float(left_hz)
        self.am_right = float(right_hz)

    def set_volume(self, volume01):
        """Set target amplitude from a 0..1 volume."""
        self._t_gain = float(np.clip(volume01, 0.0, 1.0)) * MAX_AMPLITUDE

    def set_roughness(self, x):
        self._t_rough = float(np.clip(x, 0.0, 1.0)) * ROUGH_MAX

    def set_noise(self, x):
        self._t_noise = float(np.clip(x, 0.0, 1.0)) * NOISE_MAX

    def set_reward(self, x):
        self._t_reward = float(np.clip(x, 0.0, 1.0)) * REWARD_MAX

    def is_playing(self):
        return self._stream is not None and self._t_gain > 0

    def play(self):
        if self._stream is None:
            import sounddevice as sd

            # A large block buys scheduling slack, and latency simply does not
            # matter for a continuous tone: nothing is triggered by it and
            # nobody is playing along. At 1024 frames the callback had ~23 ms to
            # run before an underrun clicked — easily lost when the GUI thread
            # holds the GIL through a plot redraw or a spectral update. 4096
            # frames raises that to ~93 ms.
            self._stream = sd.OutputStream(
                samplerate=SAMPLE_RATE, channels=2, dtype="float32",
                blocksize=4096, latency="high", callback=self._callback,
            )
            self._stream.start()

    def stop(self):
        """Ramp to silence but keep the stream open for an instant restart."""
        self._t_gain = 0.0

    def set_master_gain(self, gain):
        """Set the master target gain directly (used by timed fades)."""
        self._t_gain = float(gain)

    def close(self):
        self._t_gain = 0.0
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None


class BinauralPanel(QGroupBox):
    """UI for setting base/beat frequencies, volume, playback and presets."""

    # event dict: {event, base_hz, beat_hz, left_hz, right_hz, volume}
    # NB: not named "event" — that collides with QWidget.event().
    tone_event = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__("Binaural Beats  (use headphones)", parent)
        self.player = BinauralPlayer()
        self._volume = 0.5
        self._last_level = 0.0
        self._hold_start = None      # when the target state was first reached
        self._bloom_until = 0.0      # reward chime decay deadline
        self._duck = 1.0             # <1 while spoken guidance is talking
        self._fading = False
        self._fade_timer = QTimer(self)
        self._fade_timer.timeout.connect(self._fade_step)
        self._settings = QSettings("FNT", "MuseStudio")
        self._build_ui()
        self._apply_params(log=False)
        self._reload_presets()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        grid = QGridLayout(self)

        # Base / carrier frequency.
        base_tip = "Carrier pitch played to both ears (20–1500 Hz). The left ear gets this frequency."
        grid.addWidget(QLabel("Base (carrier) Hz"), 0, 0)
        self.base_slider = QSlider(Qt.Horizontal)
        self.base_slider.setRange(BASE_MIN, BASE_MAX)
        self.base_slider.setValue(200)
        self.base_slider.setToolTip(base_tip)
        grid.addWidget(self.base_slider, 0, 1)
        self.base_spin = QSpinBox()
        self.base_spin.setRange(BASE_MIN, BASE_MAX)
        self.base_spin.setValue(200)
        self.base_spin.setSuffix(" Hz")
        self.base_spin.setToolTip(base_tip)
        grid.addWidget(self.base_spin, 0, 2)

        # Beat frequency (the perceived difference).
        beat_tip = ("Difference between the ears (1–50 Hz) — the perceived beat and the "
                    "brain rhythm you're targeting (e.g. 10 Hz = alpha).")
        grid.addWidget(QLabel("Beat Hz"), 1, 0)
        self.beat_slider = QSlider(Qt.Horizontal)
        self.beat_slider.setRange(BEAT_MIN, BEAT_MAX)
        self.beat_slider.setValue(10)
        self.beat_slider.setToolTip(beat_tip)
        grid.addWidget(self.beat_slider, 1, 1)
        self.beat_spin = QSpinBox()
        self.beat_spin.setRange(BEAT_MIN, BEAT_MAX)
        self.beat_spin.setValue(10)
        self.beat_spin.setSuffix(" Hz")
        self.beat_spin.setToolTip(beat_tip)
        grid.addWidget(self.beat_spin, 1, 2)

        # Volume.
        grid.addWidget(QLabel("Volume"), 2, 0)
        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.setRange(0, 100)
        self.vol_slider.setValue(50)
        self.vol_slider.setToolTip("Playback loudness. In closed-loop mode this is scaled by your synchrony.")
        grid.addWidget(self.vol_slider, 2, 1)
        self.vol_label = QLabel("50%")
        grid.addWidget(self.vol_label, 2, 2)
        grid.setColumnStretch(1, 1)   # let sliders take the slack in a narrow column

        # Readout on its own line, then transport + presets stacked (narrow-friendly).
        self.readout = QLabel()
        self.readout.setWordWrap(True)
        self.readout.setStyleSheet("font-weight: bold; color: #4fc3f7;")
        grid.addWidget(self.readout, 3, 0, 1, 3)

        transport = QHBoxLayout()
        self.stimulus_combo = QComboBox()
        self.stimulus_combo.addItem("Binaural", "binaural")
        self.stimulus_combo.addItem("Monaural AM", "monaural_am")
        self.stimulus_combo.setToolTip(
            "Binaural: each ear a pure tone, beat = interaural difference (central).\n"
            "Monaural AM: each ear's carrier amplitude-modulated at its own rate "
            "(more lateralized; used for interhemispheric heterodyning)."
        )
        self.stimulus_combo.activated.connect(lambda _i: self._apply_params(log=True))
        transport.addWidget(self.stimulus_combo)
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.on_play_toggle)
        self.play_btn.setToolTip("Start/stop the tone. Use headphones for the effect to work.")
        transport.addWidget(self.play_btn)
        self.loop_check = QCheckBox("Closed-loop")
        self.loop_check.setToolTip(
            "Drive tone purity/volume from measured hemisphere synchrony:\n"
            "desynchronized = rough + noisy, synchronized = pure + full."
        )
        self.loop_check.toggled.connect(self._on_loop_toggled)
        transport.addWidget(self.loop_check)
        transport.addStretch()
        grid.addLayout(transport, 4, 0, 1, 3)

        voice = QHBoxLayout()
        voice.addWidget(QLabel("Timbre"))
        self.timbre_combo = QComboBox()
        for label, value in TIMBRES:
            self.timbre_combo.addItem(label, value)
        self.timbre_combo.setToolTip(
            "How the stimulus sounds. Comfort matters for the result, not just "
            "for politeness: a subject bracing against a harsh sound is not "
            "going to entrain to it.\n\n"
            "Soft tone — a fundamental with a fifth and octave above it; warm "
            "and organ-like, and it carries better on small laptop speakers.\n"
            "Warm noise — filtered noise, like distant rain. Least fatiguing "
            "over several minutes.\n"
            "Pure tone — a bare sine. Thin and piercing; kept for comparison "
            "with earlier sessions."
        )
        self.timbre_combo.activated.connect(lambda _i: self._apply_params(log=True))
        voice.addWidget(self.timbre_combo, stretch=1)

        voice.addWidget(QLabel("Depth"))
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(10, 100)
        self.depth_spin.setSuffix(" %")
        self.depth_spin.setValue(int(DEFAULT_MOD_DEPTH * 100))
        self.depth_spin.setToolTip(
            "How deeply the pulse modulates the sound.\n\n"
            "100% switches it fully on and off — the strongest drive, but it "
            "chops and quickly becomes irritating. Around 40% keeps the rhythm "
            "clearly audible while staying comfortable for several minutes, "
            "which is the trade this project needs."
        )
        self.depth_spin.valueChanged.connect(lambda _v: self._apply_params(log=False))
        voice.addWidget(self.depth_spin)

        self.preview_btn = QPushButton("Preview 8s")
        self.preview_btn.setToolTip(
            "Audition the current stimulus for eight seconds before committing "
            "someone to a whole session with it.")
        self.preview_btn.clicked.connect(self.on_preview)
        voice.addWidget(self.preview_btn)
        grid.addLayout(voice, 5, 0, 1, 3)

        fb = QHBoxLayout()
        fb.addWidget(QLabel("Feedback"))
        self.feedback_combo = QComboBox()
        self.feedback_combo.addItem("Reward on sustain", "reward")
        self.feedback_combo.addItem("Continuous", "continuous")
        self.feedback_combo.setToolTip(
            "How closed-loop feedback reaches you.\n\n"
            "Reward on sustain (best for eyes-closed meditation): the tone stays\n"
            "steady and a soft chime arrives only once you hold the target state\n"
            f"for ~{REWARD_HOLD_S:.0f}s. Quiet and non-distracting.\n\n"
            "Continuous: roughness and noise track the measure several times a\n"
            "second, so you always hear exactly where you are — more informative,\n"
            "but the constant change can pull your attention.\n\n"
            "Only applies when Closed-loop is ticked."
        )
        self.feedback_combo.activated.connect(
            lambda _i: self._emit_event("feedback_mode"))
        fb.addWidget(self.feedback_combo, stretch=1)
        grid.addLayout(fb, 6, 0, 1, 3)

        presets = QHBoxLayout()
        presets.addWidget(QLabel("Preset"))
        self.preset_combo = QComboBox()
        self.preset_combo.activated.connect(self._on_preset_selected)
        self.preset_combo.setToolTip("Load a saved base/beat/volume combination.")
        presets.addWidget(self.preset_combo, stretch=1)
        self.save_btn = QPushButton("Save…")
        self.save_btn.clicked.connect(self._on_save_preset)
        self.save_btn.setToolTip("Save the current base/beat/volume as a named preset.")
        presets.addWidget(self.save_btn)
        self.del_btn = QPushButton("Delete")
        self.del_btn.clicked.connect(self._on_delete_preset)
        self.del_btn.setToolTip("Delete the selected preset.")
        presets.addWidget(self.del_btn)
        grid.addLayout(presets, 7, 0, 1, 3)

        # Live updates (glide the audio) vs committed events (log to CSV).
        self.base_slider.valueChanged.connect(self.base_spin.setValue)
        self.base_spin.valueChanged.connect(self.base_slider.setValue)
        self.beat_slider.valueChanged.connect(self.beat_spin.setValue)
        self.beat_spin.valueChanged.connect(self.beat_slider.setValue)
        self.base_spin.valueChanged.connect(lambda _: self._apply_params(log=False))
        self.beat_spin.valueChanged.connect(lambda _: self._apply_params(log=False))
        self.vol_slider.valueChanged.connect(self._on_volume)
        for w in (self.base_slider, self.beat_slider, self.vol_slider):
            w.sliderReleased.connect(lambda: self._emit_event("param"))
        self.base_spin.editingFinished.connect(lambda: self._emit_event("param"))
        self.beat_spin.editingFinished.connect(lambda: self._emit_event("param"))

    # --------------------------------------------------------------- params
    def _params(self):
        base = self.base_spin.value()
        beat = self.beat_spin.value()
        return {
            "base_hz": base,
            "beat_hz": beat,
            "left_hz": float(base),
            "right_hz": float(base + beat),
            "volume": round(self._volume, 3),
        }

    def _apply_params(self, log=True):
        p = self._params()
        self.player.set_timbre(self.timbre_combo.currentData())
        self.player.set_mod_depth(self.depth_spin.value() / 100.0)
        if self.stimulus_combo.currentData() == "monaural_am":
            self.player.set_mode("monaural_am")
            self.player.set_frequencies(p["base_hz"], p["base_hz"])  # shared carrier
            self.player.set_am_freqs(p["beat_hz"], p["beat_hz"])     # isochronic both ears
            self.readout.setText(
                f"Monaural AM — carrier {p['base_hz']} Hz, both ears {p['beat_hz']} Hz"
            )
        else:
            self.player.set_mode("binaural")
            self.player.set_frequencies(p["left_hz"], p["right_hz"])
            self.readout.setText(
                f"Left {p['left_hz']:.0f} Hz   Right {p['right_hz']:.0f} Hz   "
                f"(beat {p['beat_hz']} Hz)"
            )
        if log:
            self._emit_event("param")

    def set_heterodyne_offset(self, delta_hz):
        """In monaural-AM mode, offset the right ear's rate by ``delta_hz`` from
        the left (drives the interhemispheric heterodyne)."""
        beat = self.beat_spin.value()
        self.player.set_mode("monaural_am")
        self.player.set_am_freqs(beat, beat + delta_hz)
        self.readout.setText(
            f"Heterodyne — L {beat:.1f} Hz   R {beat + delta_hz:.2f} Hz   "
            f"(Δ {delta_hz:.2f} Hz)"
        )

    def _on_volume(self, value):
        self._volume = value / 100.0
        self.vol_label.setText(f"{value}%")
        self.player.set_volume(self._volume)

    def on_play_toggle(self):
        if self.player.is_playing():
            self.player.stop()
            self.play_btn.setText("Play")
            self._emit_event("stop")
        else:
            self._push_volume()
            try:
                self.player.play()
            except Exception as exc:  # noqa: BLE001 - audio device may be missing
                QMessageBox.critical(self, "Audio error", str(exc))
                return
            self.play_btn.setText("Stop")
            if self.loop_check.isChecked():
                self.apply_synchrony(self._last_level)
            self._emit_event("play")

    def _push_volume(self):
        """Send the effective volume (user setting × duck) to the player."""
        self.player.set_volume(self._volume * self._duck)

    def set_ducked(self, ducked, level=0.25):
        """Drop the tone under spoken guidance so the voice is intelligible.

        Without this the beats sit at the same loudness as the instruction and
        you simply miss what was said — which matters most in exactly the
        situation this exists for: eyes closed, mid-session.
        """
        self._duck = level if ducked else 1.0
        if self.player.is_playing() and not self._fading:
            self._push_volume()

    def is_closed_loop(self):
        return self.loop_check.isChecked()

    def set_voicing(self, timbre=None, depth=None):
        """Point the timbre/depth controls at a protocol's chosen voicing.

        The widgets are updated so the panel keeps showing what the subject is
        actually hearing, but the values are also pushed to the player directly:
        setting a combo index in code does not emit ``activated``, and setting a
        spin box to the value it already holds emits nothing at all, so relying
        on the signals would silently skip the change.
        """
        if timbre:
            idx = self.timbre_combo.findData(timbre)
            if idx >= 0:
                self.timbre_combo.setCurrentIndex(idx)
            self.player.set_timbre(timbre)
        if depth:
            self.depth_spin.setValue(int(round(float(depth) * 100)))
            self.player.set_mod_depth(float(depth))

    def protocol_audio_on(self, base, beat, closed_loop=False, mode="binaural",
                          timbre=None, depth=None):
        """Set tone and start playback (used by the guided protocol runner)."""
        idx = self.stimulus_combo.findData(mode)
        if idx >= 0:
            self.stimulus_combo.setCurrentIndex(idx)
        self.set_voicing(timbre, depth)
        self.base_spin.setValue(int(base))
        self.beat_spin.setValue(int(beat))
        self._apply_params(log=False)
        self.loop_check.setChecked(bool(closed_loop))
        if not self.player.is_playing():
            self.on_play_toggle()
            self.fade_in()      # never start a subject at full level

    def on_preview(self):
        """Play the current stimulus briefly so it can be judged before use."""
        was_playing = self.player.is_playing()
        if not was_playing:
            self._apply_params(log=False)
            self.player.set_volume(self._volume)
            try:
                self.player.play()
            except Exception as exc:  # noqa: BLE001
                QMessageBox.critical(self, "Audio error", str(exc))
                return
            self.fade_in()
            QTimer.singleShot(8000, self.fade_out)
            self._emit_event("preview")

    def fade_in(self, seconds=ONSET_SECONDS):
        """Bring the tone up gently — a stimulus that starts at full level is
        startling, and the first seconds set how the whole session feels."""
        self._fade_timer.stop()
        self._fading = False
        self._fade_dir = 1
        self._fade_steps = max(1, int(seconds / 0.05))
        self._fade_i = 0
        self._fade_g0 = self._volume * self._duck * MAX_AMPLITUDE
        self.player.set_master_gain(0.0)
        self._fade_timer.start(50)

    def protocol_control_tone(self, base, timbre=None, depth=None):
        """Identical tone in both ears — a matched control with no beat.

        Same carrier, same loudness, same "there is a sound" experience, but no
        interaural frequency difference, so contrasting this against the
        binaural block isolates the beat itself.
        """
        idx = self.stimulus_combo.findData("binaural")
        if idx >= 0:
            self.stimulus_combo.setCurrentIndex(idx)
        self.set_voicing(timbre, depth)
        self.loop_check.setChecked(False)
        self.base_spin.setValue(int(base))
        self.player.set_mode("binaural")
        self.player.set_frequencies(base, base)      # zero beat
        self.readout.setText(f"Control tone — {int(base)} Hz both ears (no beat)")
        if not self.player.is_playing():
            self.on_play_toggle()
            self.fade_in()
        else:
            self._push_volume()
        self._emit_event("control_tone")

    def protocol_audio_off(self):
        self._fade_timer.stop()
        self._fading = False
        if self.player.is_playing():
            self.on_play_toggle()

    def fade_out(self, seconds=5.0):
        """Gradually ramp the tone to silence over ``seconds``, then stop."""
        if not self.player.is_playing() or self._fading:
            return
        self._fading = True
        self._fade_dir = -1
        self._fade_steps = max(1, int(seconds / 0.05))
        self._fade_i = 0
        self._fade_g0 = self.player._t_gain   # current master gain
        self._fade_timer.start(50)

    def _fade_step(self):
        self._fade_i += 1
        progress = min(1.0, self._fade_i / self._fade_steps)
        if getattr(self, "_fade_dir", -1) > 0:          # fading in
            self.player.set_master_gain(self._fade_g0 * progress)
            if progress >= 1.0:
                self._fade_timer.stop()
                self._fading = False
            return
        frac = 1.0 - progress
        if frac <= 0:
            self._fade_timer.stop()
            self._fading = False
            self.loop_check.setChecked(False)
            if self.player.is_playing():
                self.on_play_toggle()
            self._emit_event("fade_out")
            return
        self.player.set_master_gain(self._fade_g0 * frac)

    def _on_loop_toggled(self, on):
        if on:
            self.apply_synchrony(self._last_level)
            self._emit_event("closed_loop_on")
        else:
            # Restore pure manual tone at the user's set volume.
            self.player.set_roughness(0.0)
            self.player.set_noise(0.0)
            self.player.set_reward(0.0)
            self._push_volume()
            self._emit_event("closed_loop_off")

    def apply_synchrony(self, level):
        """Map a 0..1 synchrony level onto the feedback audio.

        Two styles, because they suit different goals:

        *Reward on sustain* (default for meditation) keeps a steady, unchanging
        tone and only adds a soft chime once you hold the target state — audio
        that changes four times a second is attention-grabbing, which is the
        opposite of what you want while settling.

        *Continuous* morphs roughness and noise with the measure in real time:
        more information per second, better for active training.
        """
        self._last_level = float(np.clip(level, 0.0, 1.0))
        if self._fading:   # don't fight an in-progress fade-out
            return
        if not (self.loop_check.isChecked() and self.player.is_playing()):
            return
        if self.feedback_combo.currentData() == "continuous":
            self._apply_continuous(self._last_level)
        else:
            self._apply_reward(self._last_level)

    def _apply_continuous(self, lv):
        self.player.set_roughness(1.0 - lv)
        self.player.set_noise((1.0 - lv) * 0.9)
        self.player.set_reward(lv)
        self.player.set_volume(self._volume * self._duck * (0.4 + 0.6 * lv))

    def _apply_reward(self, lv):
        import time as _time

        now = _time.monotonic()
        # Steady bed — nothing about the base tone tracks the measure.
        self.player.set_roughness(0.0)
        self.player.set_noise(0.0)
        self._push_volume()

        if lv >= REWARD_LEVEL:
            if self._hold_start is None:
                self._hold_start = now
            elif now - self._hold_start >= REWARD_HOLD_S and now >= self._bloom_until:
                self._bloom_until = now + REWARD_BLOOM_S
                self._hold_start = None          # re-arm for the next hold
                play_reward()
                self._emit_event("reward")
        else:
            self._hold_start = None

        # Bloom the consonant partial in, then let it decay away.
        if now < self._bloom_until:
            self.player.set_reward((self._bloom_until - now) / REWARD_BLOOM_S)
        else:
            self.player.set_reward(0.0)

    def _emit_event(self, kind):
        payload = {"event": kind, **self._params()}
        self.tone_event.emit(payload)

    def is_playing(self):
        return self.player.is_playing()

    def close_audio(self):
        self.player.close()

    # -------------------------------------------------------------- presets
    def _load_presets(self):
        return json.loads(self._settings.value("binaural_presets", "{}"))

    def _reload_presets(self):
        presets = self._load_presets()
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItem("— presets —", None)
        for name in sorted(presets):
            self.preset_combo.addItem(name, name)
        self.preset_combo.blockSignals(False)

    def _on_preset_selected(self, _index):
        name = self.preset_combo.currentData()
        if not name:
            return
        p = self._load_presets().get(name)
        if not p:
            return
        self.base_spin.setValue(int(p.get("base_hz", 200)))
        self.beat_spin.setValue(int(p.get("beat_hz", 10)))
        self.vol_slider.setValue(int(round(p.get("volume", 0.5) * 100)))
        self._apply_params(log=True)

    def _on_save_preset(self):
        p = self._params()
        default = f"{p['base_hz']}Hz base / {p['beat_hz']}Hz beat"
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name:", text=default)
        if not ok or not name.strip():
            return
        presets = self._load_presets()
        presets[name.strip()] = {
            "base_hz": p["base_hz"], "beat_hz": p["beat_hz"], "volume": p["volume"],
        }
        self._settings.setValue("binaural_presets", json.dumps(presets))
        self._reload_presets()
        self.preset_combo.setCurrentText(name.strip())

    def _on_delete_preset(self):
        name = self.preset_combo.currentData()
        if not name:
            return
        presets = self._load_presets()
        if name in presets:
            del presets[name]
            self._settings.setValue("binaural_presets", json.dumps(presets))
            self._reload_presets()
