"""Real-time analysis feeding MuseStudio's frequency and hemodynamics views.

``BandPowerAnalyzer`` turns the EEG stream into per-channel band powers,
frontal alpha asymmetry and a rolling spectrogram.
``HemodynamicsAnalyzer`` turns the optics stream into relative optical-density
changes and a left/right prefrontal comparison.

Both are fed from the GUI thread with the same chunks the plots receive, and
both run their own QTimer, so there is no cross-thread state.
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np
from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from fnt.musestudio.dsp import (
    BAND_ORDER, RollingSpectrogram, alpha_asymmetry, band_powers,
    contact_quality, optical_density, relative_band_powers, scalp_channels,
    split_lateral,
)

EEG_WINDOW_SEC = 4.0        # 0.25 Hz resolution — enough for delta/theta
EEG_UPDATE_MS = 500
OPTICS_WINDOW_SEC = 30.0    # hemodynamics are slow
OPTICS_UPDATE_MS = 1000
HISTORY_POINTS = 240        # ~2 min of band history at 2 Hz
ARTIFACT_P2P_UV = 250.0


@dataclass
class BandMetrics:
    relative: dict = field(default_factory=dict)      # {channel: {band: 0..1}}
    absolute: dict = field(default_factory=dict)      # {channel: {band: power}}
    mean_relative: dict = field(default_factory=dict)  # {band: 0..1}
    alpha_asym: float = 0.0                            # ln(AF8 alpha) - ln(AF7 alpha)
    dominant: str = ""
    contact_ok: bool = False
    contact_per_channel: dict = field(default_factory=dict)   # {channel: bool}


@dataclass
class HemoMetrics:
    left_od: float = 0.0        # mean ΔOD, left optodes
    right_od: float = 0.0       # mean ΔOD, right optodes
    laterality: float = 0.0     # right - left (positive = more right absorption)
    ready: bool = False


class BandPowerAnalyzer(QObject):
    """Per-channel EEG band powers, alpha asymmetry and a spectrogram feed."""

    updated = pyqtSignal(object)          # BandMetrics
    spectrogram_updated = pyqtSignal(object)   # RollingSpectrogram

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fs = 256.0
        self._names = []
        self._all_names = []
        self._cols = {}
        self._buffers = {}
        self._history = {b: deque(maxlen=HISTORY_POINTS) for b in BAND_ORDER}
        self._hist_t = deque(maxlen=HISTORY_POINTS)
        self._t0 = None
        self.spectrogram = RollingSpectrogram(n_cols=HISTORY_POINTS)
        self.spectrogram_channel = 0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._compute)
        self._timer.start(EEG_UPDATE_MS)

    # --- configuration ----------------------------------------------------
    def configure(self, channel_names, fs=None):
        """Buffer only the real scalp electrodes.

        The EEG stream also carries AUX_1..AUX_4 (unconnected auxiliary inputs
        that read as noise). Including them would swamp the band mix and make
        the contact check fail permanently, which in turn silently empties the
        band history and blocks closed-loop feedback.
        """
        if fs:
            self._fs = float(fs)
        all_names = [str(n) for n in channel_names]
        names = scalp_channels(all_names) or all_names
        self._all_names = all_names
        if names != self._names:
            self._names = names
            # Map each kept channel back to its column in the incoming chunk.
            self._cols = {n: all_names.index(n) for n in names}
            maxlen = int(EEG_WINDOW_SEC * self._fs)
            self._buffers = {n: deque(maxlen=maxlen) for n in names}

    def set_spectrogram_channel(self, index):
        if index != self.spectrogram_channel:
            self.spectrogram_channel = int(index)
            self.spectrogram.reset()

    def history(self):
        """(times, {band: values}) for the band-power history plot."""
        t = np.fromiter(self._hist_t, dtype=float, count=len(self._hist_t))
        return t, {b: np.fromiter(v, dtype=float, count=len(v))
                   for b, v in self._history.items()}

    def reset(self):
        for b in self._buffers.values():
            b.clear()
        for h in self._history.values():
            h.clear()
        self._hist_t.clear()
        self._t0 = None
        self.spectrogram.reset()

    # --- ingest -----------------------------------------------------------
    def add_eeg(self, channel_names, data):
        if not self._buffers:
            self.configure(channel_names)
        data = np.asarray(data, dtype=float)
        if data.ndim == 1:
            data = data[:, None]
        for name, col in self._cols.items():
            if col < data.shape[1]:
                self._buffers[name].extend(data[:, col].tolist())

    # --- compute ----------------------------------------------------------
    def _compute(self):
        if not self._buffers:
            return
        need = int(EEG_WINDOW_SEC * self._fs * 0.5)
        arrays = {}
        for name, buf in self._buffers.items():
            if len(buf) < need:
                return
            arrays[name] = np.fromiter(buf, dtype=float, count=len(buf))

        # Judged on high-passed data — raw Muse EEG has a large DC offset, so a
        # peak-to-peak test on the raw signal rejects even good electrodes.
        per_channel = {n: contact_quality(a, self._fs, ARTIFACT_P2P_UV)
                       for n, a in arrays.items()}
        contact_ok = all(per_channel.values())

        absolute, relative = {}, {}
        for name, x in arrays.items():
            ab = band_powers(x, self._fs)
            absolute[name] = ab
            relative[name] = relative_band_powers(ab)

        mean_rel = {b: float(np.mean([relative[n][b] for n in relative]))
                    for b in BAND_ORDER}
        dominant = max(mean_rel, key=mean_rel.get) if mean_rel else ""

        asym = 0.0
        left = next((n for n in arrays if "AF7" in n.upper()), None)
        right = next((n for n in arrays if "AF8" in n.upper()), None)
        if left and right:
            asym = alpha_asymmetry(absolute[left]["alpha"], absolute[right]["alpha"])

        # History (only while contact is good, so artifacts don't poison it).
        if contact_ok:
            import time as _time
            now = _time.monotonic()
            if self._t0 is None:
                self._t0 = now
            self._hist_t.append(now - self._t0)
            for b in BAND_ORDER:
                self._history[b].append(mean_rel[b])

        # Spectrogram from the selected channel.
        names = list(arrays)
        idx = min(self.spectrogram_channel, len(names) - 1)
        self.spectrogram.push(arrays[names[idx]], self._fs)
        self.spectrogram_updated.emit(self.spectrogram)

        self.updated.emit(BandMetrics(
            relative=relative, absolute=absolute, mean_relative=mean_rel,
            alpha_asym=asym, dominant=dominant, contact_ok=contact_ok,
            contact_per_channel=per_channel,
        ))


class HemodynamicsAnalyzer(QObject):
    """Relative optical-density change per side from the optics stream.

    This is an *uncalibrated proxy*: converting to true HbO/HbR concentration
    needs the Athena's wavelength/optode mapping and differential pathlength
    factors, which are not published. ΔOD still tracks the direction and
    relative size of blood-volume change, which is what the laterality view
    reports — label it as a proxy wherever it is shown.
    """

    updated = pyqtSignal(object)          # HemoMetrics

    def __init__(self, parent=None):
        super().__init__(parent)
        self._fs = 64.0
        self._names = []
        self._buffers = {}
        self._baseline = {}
        self._left_idx, self._right_idx = [], []
        self.history = deque(maxlen=HISTORY_POINTS)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._compute)
        self._timer.start(OPTICS_UPDATE_MS)

    def configure(self, channel_names, fs=None):
        if fs:
            self._fs = float(fs)
        names = [str(n) for n in channel_names]
        if names != self._names:
            self._names = names
            maxlen = int(OPTICS_WINDOW_SEC * self._fs)
            self._buffers = {n: deque(maxlen=maxlen) for n in names}
            self._baseline = {}
            self._left_idx, self._right_idx = split_lateral(names)

    def reset(self):
        for b in self._buffers.values():
            b.clear()
        self._baseline = {}
        self.history.clear()

    def add_optics(self, channel_names, data):
        if not self._buffers:
            self.configure(channel_names)
        data = np.asarray(data, dtype=float)
        if data.ndim == 1:
            data = data[:, None]
        for i, name in enumerate(self._names):
            if i < data.shape[1]:
                self._buffers[name].extend(data[:, i].tolist())

    def _compute(self):
        if not self._buffers or not self._names:
            return
        need = int(self._fs * 3)
        arrays = {}
        for name, buf in self._buffers.items():
            if len(buf) < need:
                return
            arrays[name] = np.fromiter(buf, dtype=float, count=len(buf))

        # Baseline = median of the first full window we see per channel.
        if not self._baseline:
            self._baseline = {n: float(np.median(a)) for n, a in arrays.items()}

        od = {}
        for name, a in arrays.items():
            recent = a[-int(self._fs):]        # last ~1 s
            od[name] = float(np.mean(optical_density(recent, self._baseline[name])))

        def side_mean(indices):
            vals = [od[self._names[i]] for i in indices
                    if i < len(self._names) and self._names[i] in od]
            return float(np.mean(vals)) if vals else 0.0

        left = side_mean(self._left_idx)
        right = side_mean(self._right_idx)
        self.history.append((left, right))
        self.updated.emit(HemoMetrics(left_od=left, right_od=right,
                                      laterality=right - left, ready=True))
