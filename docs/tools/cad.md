# Classic Audio Detector (CAD)

CAD finds ultrasonic vocalizations (USVs) with classic signal processing: no trained model. It draws a spectrogram of each recording, keeps the pixels that stand out above the background noise, groups them into calls, filters the calls by shape, and measures each one. You then review the calls on the spectrogram, accept or reject them, mark harmonics, and add or adjust calls by hand. Every change is saved straight to a CSV next to the recording.

Launch it from the FNT main GUI **Audio** tab. For a model-based detector that learns from your labels, see [MAD](../mad/overview.md).

## What CAD writes

| File | Where | When |
|---|---|---|
| `<recording>_FNT_CAD_detections.csv` | next to the WAV | after detection runs on the file, and rewritten in full after every edit (accept, reject, harmonic, add, delete, moving a box) |
| `<recording>_FNT.mad` | next to the WAV | only if you edit a call's mask |
| `~/.fnt/detection_profiles/<name>.json` | your user folder | when you save a custom detection profile |

The CSV has **one row per call**. A fundamental and each of its harmonics are separate rows, linked by `call_group_id`. CAD writes no summary or batch file: summarise the CSVs in your own analysis.

Older CAD files (`_cad.csv`, `_usv_dsp.csv`) are still read. CAD offers to rename them, and edits are saved under the name the file was loaded from. **Clear All** deletes the file's CSV.

## How a call is detected

1. **Chunks.** A recording longer than 20 s is processed in 10 s chunks with 0.5 s of overlap, so each chunk gets its own noise estimate. Where two chunks find the same call, the louder copy is kept.
2. **Band-pass and spectrogram.** The audio is band-pass filtered to the frequency band you set (4th-order Butterworth). It is then turned into a power spectrogram in dB with 512-sample frames, 384 samples of overlap, a 1024-point FFT and a Hann window.
3. **Threshold.** The noise floor is the 25th percentile of each frequency bin, per block of time when *Noise block* is set. A pixel belongs to a call when it is more than the **energy threshold** (default 10 dB) above that floor.
4. **Group.** Neighbouring above-threshold pixels are joined into regions. Small holes are closed with a 3×3 dilate and erode. A region is split where its frequency has a gap of 5 kHz or more, or where its energy dips deeply.
5. **Filter.** Each region is kept only if it passes the limits you set:
    - duration (minimum and maximum);
    - frequency bandwidth;
    - SNR;
    - tonality and spectral entropy;
    - minimum call frequency;
    - power;
    - sweep rate and contour jitter;
    - minimum inter-call interval.

    Calls separated by less than the **minimum gap** whose frequencies overlap are merged into one. Any limit set to 0 is off.
6. **Harmonics.** Two calls that overlap in time by at least half, where one's peak frequency is 2× or 3× the other's (±15%), are linked. The higher one is marked as a harmonic. Harmonics are labelled, never removed.

The full settings of each run are saved with every automatic row (`dsp_params_json`), so a CSV always records how it was made. The **Prairie Vole USVs** profile sets the following; you can save your own profiles.

| Setting | Value |
|---|---|
| Frequency band | 20–80 kHz |
| Duration | 15–500 ms |
| Maximum bandwidth | 40 kHz |
| Tonality | 0.20 |
| Minimum call frequency | 18 kHz |
| Minimum gap | 20 ms |
| Minimum bandwidth | 1 kHz |
| SNR | 8 dB |

## The statistics measured for each call

Every value is measured on the band-passed spectrogram, in dB relative to the recording, not calibrated sound pressure. The **call pixels** are the pixels of the region that passed the threshold. For a call you add or adjust by hand, they are every pixel inside its box (see [Calls you add or adjust by hand](#calls-you-add-or-adjust-by-hand)).

### Identity and timing

| Column | Units | Meaning |
|---|---|---|
| `call_number` | | 1, 2, 3 … in time order; renumbered at every save |
| `call_id` | | the call's permanent id within the file; links point at it |
| `start_seconds`, `stop_seconds` | s | first and last spectrogram frame of the call |
| `duration_ms` | ms | stop − start |
| `source` | | `dsp` = found by detection; `manual` = added by hand |
| `dsp_params_json` | | the detection settings of the run (blank on manual rows) |

### Frequency

| Column | Units | Meaning |
|---|---|---|
| `min_freq_hz`, `max_freq_hz` | Hz | lowest and highest frequency of the call pixels |
| `freq_bandwidth_hz` | Hz | max − min |
| `peak_freq_hz` | Hz | frequency of the single loudest call pixel |
| `mean_freq_hz` | Hz | power-weighted mean frequency of the call pixels |
| `peak_freq_1` … `peak_freq_5` | Hz | the call's frequency contour: the loudest frequency within the call's band at 5 evenly spaced moments from start to stop |

### Amplitude and energy

| Column | Units | Meaning |
|---|---|---|
| `mean_power_db` | dB | mean power of the call pixels |
| `max_power_db` | dB | power of the loudest pixel |
| `snr_db` | dB | `max_power_db` minus the noise floor over the call's frequency range |
| `tonality` | 0–1 | how pure the tone is: per frame, the share of the power within ±2 bins of the peak, averaged over the call. A whistle is near 1, broadband noise near 0 |
| `spectral_entropy` | 0–1 | normalised Shannon entropy of each frame's spectrum, averaged. Low = tonal, high = noise-like |

### Shape of the contour

Computed from the contour (`peak_freq_1` … `peak_freq_5`):

| Column | Units | Meaning |
|---|---|---|
| `mean_sweep_rate_khz_ms` | kHz/ms | mean absolute change in frequency between contour points per ms. Flat calls are near 0 |
| `max_sweep_rate_khz_ms` | kHz/ms | the steepest such change |
| `contour_smoothness` | kHz | mean absolute second difference of the contour. Higher = more jitter or frequency jumps |

### Review

| Column | Meaning |
|---|---|
| `status` | `pending` (not reviewed), `accepted` (**A**), `rejected` (**R**). Marking a harmonic also accepts it |
| `feature_source` | `measured` or `measured_mask` on calls re-measured after a hand edit |
| `freq_samples_manual` | True once you have dragged a contour point by hand (it is then never overwritten) |

For analysis, keep `status == "accepted"`. Calls still `pending` have not been reviewed.

### Harmonics

| Column | Meaning |
|---|---|
| `is_harmonic` | True for a harmonic (a call 2× or 3× another's peak frequency, overlapping it in time) |
| `call_group_id` | shared by a fundamental and its harmonics: use it to count a fundamental and its harmonics as one vocalization |
| `harmonic_ratio` | the ratio of the harmonic's peak frequency to the fundamental's (about 2 or 3; automatic harmonics only) |
| `harmonic_of_start` | the fundamental's start time (s; automatic harmonics only) |
| `harmonic_of`, `harmonic_rank`, `harmonic_of_call_number` | the fundamental's `call_id`, the overtone order and the fundamental's `call_number`, for harmonics linked by hand |

!!! warning "Use `call_group_id` for automatic harmonics"
    On harmonics found by detection, `harmonic_of` does not yet hold the fundamental's `call_id`. `harmonic_rank` and `harmonic_of_call_number` are not filled in either. `call_group_id` is correct for both automatic and hand-linked harmonics.

## Calls you add or adjust by hand

**+ Add Label (P)** places a new box in the middle of the view: 40 ± 4 kHz, 30 ms long (or 5% of the view, whichever is shorter). Drag its edges to fit the call. After you add a call, move or resize a box, type new limits, or paint its mask, CAD measures the call again. You get the **same set of columns** as a detected call:

- **From the box, as you set it:** `start_seconds`, `stop_seconds`, `min_freq_hz`, `max_freq_hz` and `duration_ms`.
- **From the sound inside the box:** peak and mean frequency, power, SNR, tonality, entropy, the contour and the sweep and smoothness measures. These come from a fresh spectrogram of the box plus 50 ms on each side. They use every pixel inside the box, or the painted mask, not only the pixels above the threshold.

Two consequences for analysis:

- A hand-drawn box includes some background, so its power, tonality and entropy are diluted compared with a detected call of the same sound.
- The SNR of a hand-edited call uses a noise floor from that short slice only, so it is not directly comparable with the SNR of a detected call.

Hand-added calls have no `call_group_id`, `is_harmonic` or `dsp_params_json` until you link them.

## Keyboard shortcuts

| Key | Action |
|---|---|
| **A** / **R** | accept / reject the selected call |
| **S** | skip to the next call without changing its status |
| **H** | mark the selected call as a harmonic |
| **D** | delete the selected call |
| **P** | add a call |
| **M** | paint the selected call's mask |
| **N** / **B** | next / previous call |
| **← / →** | pan the spectrogram |
| **↑ / ↓** | zoom in / out |
| **F** | show the detection filters over the spectrogram |
| **Q** | preview snapshot |
