import numpy as np
import soundfile as sf
import os
from scipy.signal import butter, filtfilt
import glob

#: Heterodyne playback defaults. 40 kHz matches the batch tool below and the
#: other FNT audio tools; the audible band is what a listener can resolve
#: without the playback carrying anything a sound card would have to discard.
HETERODYNE_LO_HZ = 40_000
HETERODYNE_AUDIO_BAND = (300.0, 12_000.0)
HETERODYNE_OUT_SR = 48_000


def heterodyne_signal(data, rate, carrier_freq=40000, lowcut=1000, highcut=50000):
    t = np.arange(len(data)) / rate
    carrier = np.cos(2 * np.pi * carrier_freq * t)
    mixed = data * carrier

    # Band-pass filter to isolate difference frequencies
    b, a = butter(4, [lowcut / (rate / 2), highcut / (rate / 2)], btype='band')
    filtered = filtfilt(b, a, mixed)

    return filtered


def heterodyne_for_playback(segment, rate, lo_hz=HETERODYNE_LO_HZ, *,
                            out_rate=HETERODYNE_OUT_SR,
                            band=HETERODYNE_AUDIO_BAND,
                            target_peak=0.6, max_gain=300.0,
                            fade_s=0.005, floor_hz=0.0,
                            match_level=False):
    """Ultrasound -> audible, for listening at real-time speed.

    Mixing with a local oscillator at ``lo_hz`` moves a component at ``f`` to
    ``|f - lo_hz|`` (and to ``f + lo_hz``, which the low-pass removes), so the
    calls within ``band[1]`` of the oscillator are heard as they happen --
    same timing, same rhythm, frequency sweeps heard as pitch sweeps. That is
    the point next to slowed-down playback, which keeps the full spectrum but
    stretches a bout until its rhythm is gone.

    Unlike :func:`heterodyne_signal` (kept as it is for the batch tool), this
    filters down to the audible band and resamples to ``out_rate`` before it
    is played: the batch version keeps everything up to 50 kHz at the
    recording's rate, which is mostly inaudible, and leaves the sound card's
    resampler to dispose of the rest -- where any ultrasound it does not fully
    reject folds back into the audible band as hiss.

    A heterodyne passes both sides of the oscillator: a call 5 kHz above
    ``lo_hz`` and one 5 kHz below are both heard at 5 kHz. Tuning ``lo_hz``
    just below the calls of interest keeps them on one side.

    ``floor_hz`` high-passes the recording before mixing, so only what lies
    above it is heard. Below ``lo_hz - band[1]`` it changes nothing -- the
    low-pass already keeps only ``lo_hz +/- band[1]`` (28-52 kHz at a 40 kHz
    LO) -- so it earns its keep between there and the LO: set it to the LO
    to hear only the side above it.

    Gain is set so the loud end of the segment sits at ``target_peak``, capped
    at ``max_gain`` so a silent window plays as near-silence rather than
    amplified hiss. With ``match_level`` it instead lands at the level of the
    input itself -- what slowed playback plays at -- so switching between the
    two in headphones does not jump: normalising made heterodyne ~31 dB
    louder than slowed playback of the same window. Short fades stop the cut
    edges from clicking. Returns float32 at ``out_rate``; empty input gives
    empty output.
    """
    from math import gcd
    from scipy.signal import butter as _butter, resample_poly, sosfiltfilt

    x = np.asarray(segment, dtype=np.float64).reshape(-1)
    if x.size < 16 or rate <= 0:
        return np.zeros(0, dtype=np.float32)
    x = x - x.mean()
    lo, hi = float(band[0]), float(band[1])
    nyq = rate / 2.0
    if not (0 < lo < hi < nyq) or not (0 < lo_hz < nyq):
        raise ValueError(f"band {band} / LO {lo_hz} Hz do not fit a "
                         f"{rate} Hz recording")
    floor_hz = float(floor_hz or 0.0)
    if floor_hz >= nyq:
        raise ValueError(f"floor {floor_hz} Hz does not fit a {rate} Hz "
                         "recording")
    # The level slowed playback would play this window at.
    ref_in = float(np.percentile(np.abs(x), 99.9))
    t = np.arange(x.size) / float(rate)
    carrier = np.cos(2.0 * np.pi * float(lo_hz) * t)
    # Low-pass to the audible band (drops the sum products and every
    # component further than ``hi`` from the oscillator), then a gentle
    # high-pass for the oscillator's own leak through DC and room rumble.
    sos = _butter(8, hi / nyq, btype="low", output="sos")
    sos_hp = _butter(2, lo / nyq, btype="high", output="sos")
    g = gcd(int(round(rate)), int(out_rate))

    def _mix_down(sig):
        y_ = sosfiltfilt(sos_hp, sosfiltfilt(sos, sig * carrier))
        return resample_poly(y_, int(out_rate) // g, int(round(rate)) // g)

    y = _mix_down(x)
    if y.size == 0:
        return np.zeros(0, dtype=np.float32)
    # Gain comes from the UNFLOORED mix, so the floor can only take sound
    # away. Set from the floored one, the gain would turn whatever the floor
    # removed straight back up to full level.
    # Percentile rather than max: one click would otherwise set the gain for
    # the whole window and leave the calls inaudible.
    ref = float(np.percentile(np.abs(y), 99.9))
    if floor_hz > 0:
        y = _mix_down(sosfiltfilt(_butter(8, floor_hz / nyq, btype="high",
                                          output="sos"), x))
    target = ref_in if match_level else target_peak
    gain = min(max_gain, target / ref) if ref > 0 else 0.0
    y = np.clip(y * gain, -1.0, 1.0)
    n_fade = min(int(fade_s * out_rate), y.size // 2)
    if n_fade > 0:
        ramp = np.linspace(0.0, 1.0, n_fade)
        y[:n_fade] *= ramp
        y[-n_fade:] *= ramp[::-1]
    return y.astype(np.float32)


def usv_batch_heterodyne(carrier_freq=40000):
    # Imported here so the playback function above can be used from a Qt app
    # without loading Tk.
    import tkinter as tk
    from tkinter import filedialog

    # File dialog for selecting directory
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select folder containing .wav files")

    if not folder_path:
        print("No folder selected.")
        return

    wav_files = sorted(glob.glob(os.path.join(folder_path, "*.wav")))

    if not wav_files:
        print("No .wav files found in the selected folder.")
        return

    print(f"Found {len(wav_files)} .wav files in:\n{folder_path}\n")

    for idx, wav_path in enumerate(wav_files, 1):
        print(f"Processing file {idx} of {len(wav_files)}: {os.path.basename(wav_path)}")

        # Load and convert to float32
        data, rate = sf.read(wav_path, dtype='float32')

        # Use only first channel if multi-channel
        if data.ndim > 1:
            data = data[:, 0]

        output = heterodyne_signal(data, rate, carrier_freq)

        # Construct output file path
        base, ext = os.path.splitext(wav_path)
        out_path = base + "_heterodyned.wav"

        sf.write(out_path, output, rate)

    print("\n All files processed.")

# Run the batch processor
if __name__ == "__main__":
    usv_batch_heterodyne()
