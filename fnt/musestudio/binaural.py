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
from PyQt5.QtCore import Qt, QSettings, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QGridLayout, QGroupBox, QHBoxLayout, QInputDialog,
    QLabel, QMessageBox, QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget,
)

SAMPLE_RATE = 44100
MAX_AMPLITUDE = 0.4        # headroom so summed stereo never clips


def play_cue(freq=880.0, dur=0.18, volume=0.25):
    """Play a short beep to mark a protocol phase transition (eyes-closed use).

    Best-effort and fire-and-forget; silently no-ops if no audio device.
    """
    try:
        import sounddevice as sd

        t = np.arange(int(dur * SAMPLE_RATE)) / SAMPLE_RATE
        env = np.clip(np.minimum(t / 0.01, (dur - t) / 0.02), 0, 1)  # fade in/out
        tone = (np.sin(2 * np.pi * freq * t) * volume * env).astype(np.float32)
        sd.play(np.column_stack([tone, tone]), SAMPLE_RATE)
    except Exception:
        pass
BASE_MIN, BASE_MAX = 20, 1500
BEAT_MIN, BEAT_MAX = 1, 50

# Closed-loop feedback layers (peak contribution before master gain).
DETUNE_HZ = 4.0            # dissonant partial offset -> acoustic roughness
ROUGH_MAX = 0.5
NOISE_MAX = 0.35
REWARD_MAX = 0.35         # consonant perfect-fifth pad when synchronized


class BinauralPlayer:
    """Phase-continuous stereo sine generator (left = base, right = base+beat)."""

    def __init__(self):
        self.left_freq = 200.0
        self.right_freq = 210.0
        # Ramped gains (current -> target) for click-free changes.
        self._gain = self._t_gain = 0.0
        self._rough = self._t_rough = 0.0
        self._noise = self._t_noise = 0.0
        self._reward = self._t_reward = 0.0
        self._phase_l = 0.0
        self._phase_r = 0.0
        self._phase_rough = 0.0
        self._phase_reward = 0.0
        self._noise_zi = np.zeros(1)   # one-pole lowpass state (brown-ish noise)
        self._stream = None

    def _ramp(self, current, target, n):
        return np.linspace(current, target, n, endpoint=False)

    def _callback(self, outdata, frames, time_info, status):
        n = frames
        idx = np.arange(n)
        wl = 2 * np.pi * self.left_freq / SAMPLE_RATE
        wr = 2 * np.pi * self.right_freq / SAMPLE_RATE
        wrough = 2 * np.pi * (self.left_freq + DETUNE_HZ) / SAMPLE_RATE
        wreward = 2 * np.pi * (self.left_freq * 1.5) / SAMPLE_RATE  # perfect fifth

        core_l = np.sin(self._phase_l + wl * idx)
        core_r = np.sin(self._phase_r + wr * idx)
        rough = np.sin(self._phase_rough + wrough * idx)
        reward = np.sin(self._phase_reward + wreward * idx)
        self._phase_l = (self._phase_l + wl * n) % (2 * np.pi)
        self._phase_r = (self._phase_r + wr * n) % (2 * np.pi)
        self._phase_rough = (self._phase_rough + wrough * n) % (2 * np.pi)
        self._phase_reward = (self._phase_reward + wreward * n) % (2 * np.pi)

        # Brown-ish noise bed: one-pole lowpass of white noise (state carried).
        # Scaled to sit under the tones without hard-clipping the summed output.
        white = np.random.randn(n)
        noise, self._noise_zi = lfilter([0.05], [1, -0.95], white, zi=self._noise_zi)
        noise *= 2.0

        g_rough = self._ramp(self._rough, self._t_rough, n)
        g_noise = self._ramp(self._noise, self._t_noise, n)
        g_reward = self._ramp(self._reward, self._t_reward, n)
        g_master = self._ramp(self._gain, self._t_gain, n)
        self._rough, self._noise = self._t_rough, self._t_noise
        self._reward, self._gain = self._t_reward, self._t_gain

        shared = rough * g_rough + reward * g_reward + noise * g_noise
        left = np.clip((core_l + shared) * g_master, -1.0, 1.0)
        right = np.clip((core_r + shared) * g_master, -1.0, 1.0)
        outdata[:, 0] = left.astype(np.float32)
        outdata[:, 1] = right.astype(np.float32)

    def set_frequencies(self, left_freq, right_freq):
        self.left_freq = float(left_freq)
        self.right_freq = float(right_freq)

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

            self._stream = sd.OutputStream(
                samplerate=SAMPLE_RATE, channels=2, dtype="float32",
                blocksize=1024, callback=self._callback,
            )
            self._stream.start()

    def stop(self):
        """Ramp to silence but keep the stream open for an instant restart."""
        self._t_gain = 0.0

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

        # Readout + transport + presets.
        bottom = QHBoxLayout()
        self.readout = QLabel()
        self.readout.setStyleSheet("font-weight: bold; color: #4fc3f7;")
        bottom.addWidget(self.readout)
        bottom.addStretch()

        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.on_play_toggle)
        self.play_btn.setToolTip("Start/stop the binaural tone. Use headphones for the effect to work.")
        bottom.addWidget(self.play_btn)

        self.loop_check = QCheckBox("Closed-loop")
        self.loop_check.setToolTip(
            "Drive tone purity/volume from measured hemisphere synchrony:\n"
            "desynchronized = rough + noisy, synchronized = pure + full."
        )
        self.loop_check.toggled.connect(self._on_loop_toggled)
        bottom.addWidget(self.loop_check)

        bottom.addWidget(QLabel("Preset:"))
        self.preset_combo = QComboBox()
        self.preset_combo.setMinimumWidth(140)
        self.preset_combo.activated.connect(self._on_preset_selected)
        self.preset_combo.setToolTip("Load a saved base/beat/volume combination.")
        bottom.addWidget(self.preset_combo)
        self.save_btn = QPushButton("Save…")
        self.save_btn.clicked.connect(self._on_save_preset)
        self.save_btn.setToolTip("Save the current base/beat/volume as a named preset.")
        bottom.addWidget(self.save_btn)
        self.del_btn = QPushButton("Delete")
        self.del_btn.clicked.connect(self._on_delete_preset)
        self.del_btn.setToolTip("Delete the selected preset.")
        bottom.addWidget(self.del_btn)

        grid.addLayout(bottom, 3, 0, 1, 3)

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
        self.player.set_frequencies(p["left_hz"], p["right_hz"])
        self.readout.setText(
            f"Left {p['left_hz']:.0f} Hz   Right {p['right_hz']:.0f} Hz   "
            f"(beat {p['beat_hz']} Hz)"
        )
        if log:
            self._emit_event("param")

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
            self.player.set_volume(self._volume)
            try:
                self.player.play()
            except Exception as exc:  # noqa: BLE001 - audio device may be missing
                QMessageBox.critical(self, "Audio error", str(exc))
                return
            self.play_btn.setText("Stop")
            if self.loop_check.isChecked():
                self.apply_synchrony(self._last_level)
            self._emit_event("play")

    def is_closed_loop(self):
        return self.loop_check.isChecked()

    def protocol_audio_on(self, base, beat, closed_loop=False):
        """Set tone and start playback (used by the guided protocol runner)."""
        self.base_spin.setValue(int(base))
        self.beat_spin.setValue(int(beat))
        self._apply_params(log=False)
        self.loop_check.setChecked(bool(closed_loop))
        if not self.player.is_playing():
            self.on_play_toggle()

    def protocol_audio_off(self):
        if self.player.is_playing():
            self.on_play_toggle()

    def _on_loop_toggled(self, on):
        if on:
            self.apply_synchrony(self._last_level)
            self._emit_event("closed_loop_on")
        else:
            # Restore pure manual tone at the user's set volume.
            self.player.set_roughness(0.0)
            self.player.set_noise(0.0)
            self.player.set_reward(0.0)
            self.player.set_volume(self._volume)
            self._emit_event("closed_loop_off")

    def apply_synchrony(self, level):
        """Map a 0..1 synchrony level onto the feedback audio layers.

        Low synchrony -> rough + noisy + quieter; high -> pure + reward + full.
        """
        self._last_level = float(np.clip(level, 0.0, 1.0))
        if not (self.loop_check.isChecked() and self.player.is_playing()):
            return
        lv = self._last_level
        self.player.set_roughness(1.0 - lv)
        self.player.set_noise((1.0 - lv) * 0.9)
        self.player.set_reward(lv)
        self.player.set_volume(self._volume * (0.4 + 0.6 * lv))

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
