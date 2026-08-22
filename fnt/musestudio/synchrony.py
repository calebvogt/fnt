"""Interhemispheric synchrony analysis for MuseStudio closed-loop biofeedback.

Computes the Phase-Locking Value (PLV) between the Athena's homologous
left/right electrode pairs — frontal AF7<->AF8 and temporal TP9<->TP10 — in a
chosen band (alpha by default). PLV runs 0 (no phase locking) to 1 (locked).

A short rest baseline fixes the resting "floor"; the reported ``level`` is the
current PLV mapped relative to that floor into 0..1, so feedback rewards
increases over your own resting synchrony rather than an absolute value.

Caveats surfaced by the UI: Muse's shared Fpz reference inflates raw
interhemispheric coupling (so treat absolute PLV cautiously — relative change is
the meaningful signal); gamma is easily contaminated by muscle EMG; blinks and
clenches are gated out via an amplitude check.

Fed from the GUI thread (the EEG ``samples_ready`` chunks); a QTimer drives
computation, so no cross-thread state.
"""

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from PyQt5.QtCore import QObject, QTimer, pyqtSignal
from scipy.signal import butter, hilbert, sosfiltfilt

FS = 256                       # Muse EEG sample rate (Hz)
WINDOW_SEC = 2.0
UPDATE_MS = 250
ARTIFACT_P2P_UV = 250.0        # per-electrode peak-to-peak gate (µV)

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta": (13, 30),
    "gamma": (30, 50),
}

# (pair label, left electrode, right electrode)
PAIRS = [("frontal", "AF7", "AF8"), ("temporal", "TP9", "TP10")]
ELECTRODES = ["AF7", "AF8", "TP9", "TP10"]


@dataclass
class SynchronyMetrics:
    plv: dict = field(default_factory=dict)       # {"frontal": x, "temporal": y}
    plv_combined: float = 0.0
    level: float = 0.0                            # calibrated 0..1 (== combined if not calibrated)
    band: str = "alpha"
    band_power: dict = field(default_factory=dict)  # {electrode: power}
    drift_hz: float = 0.0                         # interhemispheric phase-drift rate (heterodyne)
    contact_ok: bool = False
    calibrated: bool = False


class SynchronyAnalyzer(QObject):
    metrics_updated = pyqtSignal(object)          # SynchronyMetrics
    baseline_progress = pyqtSignal(float)         # 0..1
    baseline_done = pyqtSignal(float)             # resting floor PLV
    status = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fs = float(FS)
        self._rebuild_buffers()
        self._col = {}                # electrode -> column index in EEG stream
        self._fallback_warned = False
        self._band = "alpha"
        self._sos = self._design(self._band)

        self._floor = 0.0
        self._calibrated = False
        self._baseline_active = False
        self._baseline_vals = []
        self._baseline_end = 0.0
        self._baseline_dur = 0.0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._compute)
        self._timer.start(UPDATE_MS)

    # ------------------------------------------------------------- config
    def _rebuild_buffers(self):
        self._win = int(WINDOW_SEC * self._fs)
        self._buffers = {e: deque(maxlen=self._win) for e in ELECTRODES}

    def _design(self, band):
        lo, hi = BANDS[band]
        return butter(4, [lo, hi], btype="band", fs=self._fs, output="sos")

    def set_sample_rate(self, fs):
        """Set the EEG sample rate from the stream (default assumes 256 Hz)."""
        if fs and fs > 1 and abs(fs - self._fs) > 0.5:
            self._fs = float(fs)
            self._rebuild_buffers()
            self._sos = self._design(self._band)

    def set_band(self, band):
        if band not in BANDS:
            return
        self._band = band
        self._sos = self._design(band)
        # Band change invalidates the resting floor.
        self._calibrated = False
        self._floor = 0.0
        self.status.emit(f"Band set to {band}; recalibrate baseline.")

    def start_baseline(self, duration=30.0):
        self._baseline_active = True
        self._baseline_vals = []
        self._baseline_dur = duration
        self._baseline_end = time.monotonic() + duration
        self.status.emit(f"Calibrating baseline ({int(duration)} s) — rest quietly…")

    def reset(self):
        for b in self._buffers.values():
            b.clear()
        self._col = {}
        self._fallback_warned = False
        self._calibrated = False
        self._floor = 0.0
        self._baseline_active = False

    # ------------------------------------------------------------- ingest
    def add_eeg(self, channel_names, data):
        """Append an EEG chunk. ``data`` is (n_samples, n_channels)."""
        if not self.ready():
            self._resolve_columns(channel_names)
        if data is None or len(data) == 0:
            return
        data = np.asarray(data)
        for e, c in self._col.items():
            if c < data.shape[1]:
                self._buffers[e].extend(data[:, c].tolist())

    def _resolve_columns(self, channel_names):
        """Map electrodes to EEG columns by name; fall back to the standard Muse
        channel order (TP9, AF7, AF8, TP10) if the labels aren't recognized."""
        upper = [str(n).upper() for n in channel_names]
        col = {}
        for e in ELECTRODES:
            for i, name in enumerate(upper):
                if e in name:
                    col[e] = i
                    break
        if len(col) == len(ELECTRODES):
            self._col = col
            return
        # Fallback: assume the first four channels are TP9, AF7, AF8, TP10.
        if len(upper) >= 4:
            self._col = {"TP9": 0, "AF7": 1, "AF8": 2, "TP10": 3}
            if not self._fallback_warned:
                self._fallback_warned = True
                self.status.emit(
                    "EEG channel labels unrecognized — assuming standard Muse "
                    "order (TP9, AF7, AF8, TP10)."
                )

    def ready(self):
        return len(self._col) == len(ELECTRODES)

    # ------------------------------------------------------------- compute
    def _compute(self):
        if not self.ready():
            return
        need = int(self._win * 0.6)
        if any(len(self._buffers[e]) < need for e in ELECTRODES):
            return

        sig = {}
        p2p = {}
        for e in ELECTRODES:
            x = np.fromiter(self._buffers[e], dtype=float, count=len(self._buffers[e]))
            x = x - np.mean(x)
            sig[e] = x
            p2p[e] = float(np.ptp(x))

        # High-passed contact check: the raw signal's DC offset and drift would
        # otherwise fail this test even on well-seated electrodes.
        from fnt.musestudio.dsp import contact_quality
        contact_ok = all(contact_quality(sig[e], self._fs, ARTIFACT_P2P_UV)
                         for e in ELECTRODES)

        # Band-limited analytic phase + power per electrode.
        phase = {}
        power = {}
        for e in ELECTRODES:
            filtered = sosfiltfilt(self._sos, sig[e])
            analytic = hilbert(filtered)
            phase[e] = np.angle(analytic)
            power[e] = float(np.mean(np.abs(analytic) ** 2))

        plv = {}
        drifts = []
        t = np.arange(len(sig[ELECTRODES[0]])) / self._fs
        for label, left, right in PAIRS:
            dphi = phase[left] - phase[right]
            plv[label] = float(np.abs(np.mean(np.exp(1j * dphi))))
            # Heterodyne rate: slope of the unwrapped phase difference (Hz).
            slope = np.polyfit(t, np.unwrap(dphi), 1)[0]
            drifts.append(slope / (2 * np.pi))
        combined = float(np.mean(list(plv.values()))) if plv else 0.0
        drift_hz = float(np.mean(drifts)) if drifts else 0.0

        # Baseline calibration collects combined PLV during a rest window.
        if self._baseline_active:
            if contact_ok:
                self._baseline_vals.append(combined)
            remaining = self._baseline_end - time.monotonic()
            self.baseline_progress.emit(
                float(np.clip(1.0 - remaining / max(self._baseline_dur, 1e-6), 0, 1))
            )
            if remaining <= 0:
                self._finish_baseline()

        level = self._to_level(combined)

        self.metrics_updated.emit(SynchronyMetrics(
            plv=plv, plv_combined=combined, level=level, band=self._band,
            band_power=power, drift_hz=drift_hz,
            contact_ok=contact_ok, calibrated=self._calibrated,
        ))

    def _finish_baseline(self):
        self._baseline_active = False
        if self._baseline_vals:
            # Resting floor = median resting PLV, clamped to a sane range.
            self._floor = float(np.clip(np.median(self._baseline_vals), 0.1, 0.8))
            self._calibrated = True
            self.baseline_done.emit(self._floor)
            self.status.emit(f"Baseline set (resting PLV {self._floor:.2f}).")
        else:
            self.status.emit("Baseline failed — poor contact throughout.")

    def _to_level(self, combined):
        """Map raw PLV to a calibrated 0..1 relative to the resting floor."""
        if not self._calibrated:
            return combined
        return float(np.clip((combined - self._floor) / (1.0 - self._floor), 0.0, 1.0))
