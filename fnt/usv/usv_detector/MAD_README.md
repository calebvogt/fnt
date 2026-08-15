# MAD — Mask Audio Detector

MAD is FNT's tool for detecting ultrasonic vocalizations (USVs) by **pixel
segmentation** of the spectrogram. Instead of drawing boxes, you teach a model
to color in the exact pixels that belong to a call, then let it find calls in
new recordings. The workflow is **human-in-the-loop**: label a little, train,
let the model predict, and accept / reject / fix its predictions.

GUI entry point: `python fnt/usv/mad_pyqt.py`

---

## The two tabs

1. **Label & Train** — build the project. Paint/segment calls on a handful of
   files, confirm them (they become training examples), and train a U-Net.
2. **Inference (Deploy)** — point a trained model at your recordings. It writes
   per-call detections you review (accept / reject / adjust).

The loop you're meant to run: label a few calls → train → run inference →
correct the predictions → (optionally) feed corrections back in and retrain
until the model is accurate enough, then just deploy and review.

---

## How the model actually works (the important mental model)

The model is a **U-Net segmentation network**, not a box detector. For every
single pixel of the spectrogram it outputs a number from **0 to 1**: *"how
likely is this pixel part of a call?"* A full recording is therefore turned into
a giant **probability grid** the same size as the spectrogram (frequency bins ×
time frames — easily 500 × 1,000,000+ values).

A "call" is derived from that grid in two steps:

1. **Threshold** — keep pixels at or above the cutoff (default `0.5`).
2. **Blob extraction** — group the surviving connected pixels into islands.
   Each island = one detected call, with a bounding box, an area, and a
   confidence **score** (the mean probability inside it).

So the per-call **score** (e.g. `0.97`) and the per-call **mask shape** are both
*derived from* the probability grid. The grid is just the raw, unthresholded
intermediate.

---

## Storage layout (and the big performance decision)

The **CSV is canonical** for tabular output. Pixel data lives in HDF5 siblings.

Per recording `<wav>`:

- `<wav>_FNT_MAD_predictions.csv` — the **unified table of every call** on the
  file: hand-labels **and** model predictions, one row each. The first 16 columns
  mirror CAD's `_FNT_CAD_detections.csv` (so the two tools are cross-readable);
  the rest are MAD's richer per-call quantification. CAD's harmonic columns and
  `dsp_params_json` are omitted (MAD has no harmonic linker and trains a model).

  Every metric is computed by **one shared function** (`compute_call_metrics`)
  for both predictions (over the blob) and hand-labels (over the painted mask),
  so the two row types are directly comparable. Power/contour stats read the
  spectrogram dB, clipped to the project's `db_min…db_max`.

  **Identity & review**
  - `call_number` — 1…N display index, renumbered by onset time on every write.
  - `call_id` — stable join key to the h5 mask (int for predictions, string id
    for hand-labels).
  - `status` — `pending` (unreviewed) · `accepted` (hand-labels write accepted,
    and accepting a prediction in train mode also saves a training example) ·
    `rejected` (a **recorded** human "no", kept visible as an audit trail;
    *not* used as negative training data). **Delete** drops the row + mask
    entirely (no `deleted` status). The on-screen color of each status depends
    on the active spectrogram colormap — see *Review colors* below.
  - `source` — `prediction` or `label`.
  - `class` — call type (e.g. `USV`); `score` — mean model probability (`1.0`
    for hand-labels). Inference preserves existing label rows and replaces only
    the prediction rows.

  **Time / sequence**
  - `start_seconds`, `stop_seconds`, `duration_ms` — call onset, offset, length.
  - `inter_call_interval_ms` — gap from the previous call's offset (blank for the
    first); the basis for bout/sequence analysis.
  - `call_rate_hz` — local emission rate: calls whose onset falls within ±0.5 s
    of this one, per second.

  > **Time convention (changed).** Spectrogram frame `i` maps to
  > `nperseg/(2·sr) + i·(nperseg−noverlap)/sr` seconds — `scipy.signal.spectrogram`
  > centres its first frame half a window into the signal, and CAD reads its
  > times straight off that axis. MAD previously used `i·hop/sr`, dropping the
  > `nperseg/(2·sr)` origin term, so every exported onset sat early by ~1 ms at
  > `nperseg=512, sr=250 kHz` (two whole hops, and it scales with window size)
  > and did not line up with CAD's for the same call. Durations were unaffected
  > (both endpoints shifted equally); absolute onsets were not.
  >
  > **CSVs written before this change are offset by `nperseg/(2·sr)`.** Re-run
  > inference to regenerate them, or add that constant when comparing old files
  > against new ones, CAD output, or video/UWB timestamps. Stored mask crops
  > carry their own pixel offsets and are unaffected. `frames_to_seconds()` /
  > `seconds_to_frames()` in `mad_inference.py` are the single source of truth
  > for this conversion — use them rather than multiplying by `dt` by hand.

  **Frequency box & contour** (the contour is the peak-power frequency traced
  across each time column inside the mask)
  - `min_freq_hz`, `max_freq_hz` — frequency extent of the mask.
  - `peak_freq_hz` — frequency of the single loudest pixel.
  - `freq_bandwidth_hz` — `max − min`.
  - `start_freq_hz`, `end_freq_hz` — contour frequency at onset / offset
    (up-sweep vs down-sweep).
  - `mean_freq_hz`, `freq_std_hz` — mean and spread of the contour.
  - `freq_slope_hz_per_s` — net df/dt (least-squares fit); sweep direction/steep.
  - `freq_excursion_hz` — total frequency distance the contour travels
    (Σ|Δf|); separates simple tones from heavily modulated calls.
  - `num_freq_jumps` — count of abrupt frequency steps (adjacent-frame |Δf| >
    5 kHz); flags "step"/multi-component calls.
  - `sinuosity` — contour path length ÷ straight-line length (1.0 = straight);
    a wiggliness index for trills/complex calls.

  **Spectral shape / purity** — both computed **per time frame then averaged**
  over the full-frequency column (matching CAD's `dsp_detector`, so they're
  correct for frequency-modulated calls — a clean sweep reads as tonal, not
  noisy).
  - `spectral_centroid_hz` — intensity-weighted mean frequency over the call's
    pixels (energy "center of mass"; note this equals CAD's `mean_freq_hz`,
    whereas MAD's `mean_freq_hz` above is the contour mean — a naming nuance).
  - `spectral_entropy` — mean over frames of the Shannon entropy of each full
    frequency column's normalized power, ÷ log2(n_freq_bins); 0 = pure tone,
    1 = uniform/noisy.
  - `tonality` — mean over frames of the fraction of column energy within ±2
    bins of that frame's peak; →1 = pure tone, →0 = broadband. Good for
    rejecting non-USV noise.

  **Amplitude / quality** — power columns are read off the spectrogram **clipped
  to the project's `db_min…db_max`** (predictions can't exceed the model's
  normalized range), so a call saturating `db_max` reads as `db_max`; raise
  `db_max` if your calls are louder.
  - `max_power_db`, `mean_power_db` — loudest and mean dB over the call's pixels.
  - `total_energy_db` — summed power (dB) over the mask.
  - `snr_db` — `max_power_db` minus the local noise floor (median dB of the
    out-of-band rows at the call's time columns; CAD-style max − floor). A
    model-independent quality measure — good for filtering weak/false
    detections.
  - `peak_time_frac` — where the energy envelope peaks, 0–1 across the call
    (onset-loud vs offset-loud).
  - `amplitude_modulation` — envelope contrast `(max−min)/(max+min)` over the
    per-frame energy; 0 = flat, →1 = high contrast. A modulation-*depth* proxy
    (it does not measure modulation *rate*, and a single onset/offset ramp also
    raises it).

  **Morphology**
  - `area_pixels` — number of mask pixels (segmentation's "size").
  - `fill_ratio` — `area_pixels ÷ bounding-box area`; thin tonal calls fill
    little, broadband smears fill a lot.
  - `aspect_ratio` — bbox time-frames ÷ freq-bins (long-thin vs short-tall).

  **Provenance** (predictions only; blank for hand-labels)
  - `model_name` — checkpoint that produced the prediction.
  - `threshold`, `min_blob_pixels` — the inference settings used.
- `<wav>_FNT_masks.h5` — pixel data:
  - `/calls/<id>` — confirmed (human-labeled) call mask crops. Joined to the CSV
    by `call_id`.
  - `/pred_calls/<blob_id>` — **predicted call mask crops** (small,
    gzip-compressed uint8, with `f_off`/`t_off` offsets). Joined to the CSV by
    `call_id` (= the prediction's integer blob id).
  - root attr `n_pred_blobs` — cached prediction count, so file lists show
    counts without reading any pixel data.

Project-wide:

- `models/training_data/training_data.h5` — every confirmed labeling example
  (spec patch + mask + metadata) used for training.
- `.scratch/` — temporary masks/predictions for files you're **browsing in
  place** but haven't accepted a call on yet. Wiped on close (see below).

### Recordings are referenced, not copied

A project does **not** store your audio. `mad_project_info.json` holds a
`training_files` table of *references*: absolute path, basename, size, and a
cheap content fingerprint (size + first and last 1 MiB).

This is safe because of where training data actually lives. Every confirmed call
is baked into `training_data.h5` as a self-contained spectrogram patch + mask,
and `mad_training` reads only that store — **it never opens a wav**. The audio is
needed for exactly one thing: re-opening a file to look at it.

That is a much weaker dependency than SLEAP has on its videos. SLEAP stores frame
indices and coordinates, so a missing video means it cannot train at all; MAD's
model, labels and detections are all unaffected. So a missing recording is a
**soft** state, not an error:

- The Training Data row shows `⚠` and preview/playback are unavailable.
- Training, the example store, and every saved detection keep working.
- Batch inference silently skips missing files rather than aborting the run.

**Finding files again.** `mad_registry.resolve_entries` tries, cheapest first:
the stored path; then the same basename in a directory where another entry *did*
resolve (files move in groups, so a resolved sibling is the best hint) or in one
of the project's `source_folders`. A candidate is accepted only if the size — and
the fingerprint, when known — matches, so a same-named different recording is
never silently swapped in.

When automatic resolution fails, **File ▸ Locate Missing Recordings…** asks for
one file and infers the prefix change from it (`D:/exp/mic2/a.wav` →
`E:/data/exp/mic2/a.wav` implies `D:/` → `E:/data`), then repoints every sibling
that moved the same way. Fix one, fix two hundred. Prefix matching compares whole
path components, so a neighbouring `exp_backup/` tree is never dragged along, and
only files that are *currently missing* and whose rewritten path *exists* are
changed — a wrong guess is a no-op.

**Portability.** **File ▸ Pack Project (embed audio)…** copies every referenced
recording into `recordings/`, making the project fully self-contained for
archiving or handing to a collaborator. Packed files are marked `embedded=True`
and become project-owned: removing one from the Training Data set deletes it,
whereas removing a *referenced* file only unregisters it and never touches disk.

**Legacy projects.** Projects with `recordings/` copies still work. Those wavs are
adopted into the registry on open as `embedded=True`, so nothing changes for them
unless you delete the copies yourself.

### Why we do NOT store the full probability grid

We used to persist the entire probability grid (`/prob`) in each
`_FNT_masks.h5`. It existed for one feature: **re-thresholding** predictions
(slide the cutoff and watch calls appear/disappear) **without re-running the
model**.

That convenience was extremely expensive:

| | Full `/prob` grid | Per-call crops (`/pred_calls`) |
|---|---|---|
| Disk per file | **~900 MB** | **~9 MB** |
| Time to load on file switch | **~5 s** (decompress ~1.2 GB) | **~0.16 s** |

Every time you opened or switched to a file with predictions, MAD decompressed
the whole ~1 GB grid just to carve out the call shapes for display — even though
the shapes occupy a tiny fraction of it. That was the source of the multi-second
pinwheels when switching files.

**Decision:** MAD does **not** support re-thresholding. The grid is gone.
Instead, at inference time each call's small mask crop is saved once under
`/pred_calls`, and file switches read only those few MB. To use a different
threshold, **re-run inference** — it's fast, and re-running keeps detections
consistent with whatever model you've trained.

Rationale: MAD's goal is fast human-in-the-loop training and review. Inference
is cheap; re-thresholding stored probabilities is not worth ~100× the disk and
~30× the load time on the hot path.

### Legacy files migrate automatically

Old projects whose `_FNT_masks.h5` still contains a `/prob` grid are upgraded
the first time you open each file: MAD reads the grid once (the last slow load),
carves the per-call crops, drops the grid, and **repacks the file to reclaim the
disk** (~900 MB → ~9 MB). Every subsequent open is fast. You can also just
re-run inference to regenerate predictions in the new format.

> Implementation note: `h5py`'s `del` only unlinks a dataset; the bytes remain
> as slack in the file. `fnt_mask_store.delete_prob()` therefore *repacks*
> (copies everything except `/prob` into a fresh file and atomically replaces
> the original) so the disk is actually freed.

---

## Review colors

Overlay colors are **not fixed** — they're chosen per spectrogram colormap
(`_OVERLAY_PALETTES` in `mad_pyqt.py`). A fixed palette always collides with
some map: green confirmed masks disappear into viridis's green midtones and
yellow predictions disappear into its bright end. Each palette therefore draws
from hues the active map never produces, and every outline is stroked over a
contrast **halo** pen, which is what actually keeps it legible across the map's
full dark→bright range.

| Colormap | confirmed | pending | rejected |
|---|---|---|---|
| Viridis | white | magenta | red |
| Magma / Inferno | white | cyan | red |
| Grayscale | cyan | magenta | red |
| Grayscale Inverted | blue | purple | dark red |

Two rules keep it readable while switching maps: `rejected` stays red wherever
the map allows (it's an audit trail — stable semantics beat maximum pop), and
`confirmed` always takes the map's most contrasting hue, since that's the state
you scan for. The legend under **Labeling Tools** is generated from the active
palette, and the detections list, waveform-overview marks and per-file
`(A, P, R)` badges all read from the same source, so no surface can drift out
of sync.

---

## Inference options (what the dialog settings mean)

- **Probability threshold** (default `0.5`) — the per-pixel cutoff described
  above. Lower → more, fainter calls + more false positives. Higher → fewer,
  higher-confidence calls. Baked in at inference time (no re-thresholding).
- **Min blob pixels** (default `8`) — drop detections smaller than this; filters
  noise specks. Raise to suppress pinpoint false positives; lower to catch very
  short calls.
- **Preserve user-painted labels** — inference zeroes the probability in time
  columns you've already labeled, so predictions never overwrite confirmed
  calls.
- **Device** — `auto` picks CUDA (NVIDIA) / MPS (Apple Silicon) if available,
  else CPU.

## Training options

- **Encoder** — the U-Net's pretrained backbone. `resnet18` is a fast, solid
  default; larger backbones (`resnet50`) can be more accurate but are slower and
  need more labeled data.
- **Tile overlap fraction** — inference slides a fixed-width window across the
  recording; this is how much neighboring windows overlap so calls spanning a
  seam aren't cut. More overlap = cleaner seams, slightly slower.
- **Max epochs / Early-stop patience** — upper bound on training passes; it
  stops early once validation loss plateaus.
- **Batch size** — tiles processed at once. Larger = faster but more GPU memory;
  lower it on out-of-memory errors. No effect on final accuracy.
- **Learning rate** — weight-update step size; the `1e-3` default is usually
  fine.
- **Validation fraction** — share of labeled tiles held out to measure
  generalization and drive early stopping.

---

## Key modules

| File | Role |
|---|---|
| `mad_pyqt.py` | PyQt5 GUI (labeling, review, training/inference dialogs). |
| `mad_inference.py` | Run a checkpoint over a wav → CSV rows + per-call crops. |
| `mad_training.py` | Train the U-Net from confirmed examples. |
| `mad_examples.py` | Confirmed training-example store (`training_data.h5`). |
| `mad_dataset.py` | Spectrogram/tile helpers shared by training & inference. |
| `mad_project.py` | Project config / on-disk layout. |
| `mad_labels.py` | Sibling-path helpers (CSV naming, etc.). |
| `fnt_mask_store.py` | Shared HDF5 mask storage (CAD + MAD); `/calls`, `/pred_calls`, training store, repack. |
