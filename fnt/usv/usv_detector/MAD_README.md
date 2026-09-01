# MAD — Mask Audio Detector

MAD is FNT's tool for detecting ultrasonic vocalizations (USVs) by **pixel
segmentation** of the spectrogram. Instead of drawing boxes, you teach a model
to color in the exact pixels that belong to a call, then let it find calls in
new recordings. The workflow is **human-in-the-loop**: label a little, train,
let the model predict, and accept / reject / fix its predictions.

GUI entry point: `python fnt/usv/mad_pyqt.py`

---

## One list, one canvas

MAD follows SLEAP's project model: a project **points at** recordings where they
already live, the way a SLEAP project points at videos. There is a single
**Audio** list — every recording the project knows about — and a single
spectrogram canvas. Click a row to preview it, label it, and review its
detections.

Anything you confirm in that list trains the model. There is no separate
"training set" to curate: if a recording is in the list, its labels train; if you
don't want it to, take it out of the list.

The loop you're meant to run: add audio → label a few calls → train → run
inference → correct the predictions → (optionally) feed corrections back in and
retrain until the model is accurate enough, then just deploy and review.

**Running large batches** is the one thing that deliberately stays outside the
list. *Run Inference ▸ Folder* scans a folder tree once and streams the paths
straight to the worker: thousands of recordings don't become thousands of list
rows, and production audio you're not curating never joins the training set.

**A project is optional** — right up until you train. Add wavs, label them, load
a model from any trained project, run inference and review, all with no project
open (labels and detections save next to the audio either way). Training is what
needs a project, because the model, its checkpoints and the consolidated example
store have to live somewhere; the Run Training button offers to make one.

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

### The Audio list

One list, and everything in it is the project. There is no separate "session"
vs "training" set: any recording you add is labelable, and **the model trains on
every confirmed call across all of them**. Add files with **Add Folder…**
(offers the whole tree when there is one) or **Add Files…**.

What the rows mean:

- `name (a, p, r)` — accepted / pending / rejected call counts, colored to match
  the spectrogram overlay (see *Review colors*). A green ✓ means the file has
  recorded calls.
- **⚠ name** — the audio moved or its drive is not mounted. This is a **soft**
  state: labels, the training store, saved detections and batch inference are
  all unaffected; only preview and playback need the file. Selecting one blanks
  the canvas rather than leaving the previous file's detections on screen under
  the wrong name. Use **File ▸ Locate Missing Recordings…**, which infers the
  move from one file and repoints every sibling that moved with it.

Removing a row only *unregisters* it — the wav and its sibling csv/h5 stay where
they are, because MAD never owned them. Only project-owned copies (a legacy
`recordings/` file, or one made by **Pack Project**) are deleted, and the prompt
says so explicitly.

### Recordings are referenced, not copied

A project does **not** store your audio. `mad_project_info.json` holds an
`audio_files` table of *references*: absolute path, basename, size, and a
cheap content fingerprint (size + first and last 1 MiB).

This is safe because of where training data actually lives. Every confirmed call
is baked into `training_data.h5` as a self-contained spectrogram patch + mask,
and `mad_training` reads only that store — **it never opens a wav**. The audio is
needed for exactly one thing: re-opening a file to look at it.

That is a much weaker dependency than SLEAP has on its videos. SLEAP stores frame
indices and coordinates, so a missing video means it cannot train at all; MAD's
model, labels and detections are all unaffected. So a missing recording is a
**soft** state, not an error:

- The Audio row shows `⚠` and preview/playback are unavailable.
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
and become project-owned: removing one from the Audio list deletes it (and says
so first), whereas removing a *referenced* file only unregisters it and never
touches disk.

**Legacy projects.** Projects with `recordings/` copies still work. Those wavs are
adopted into the registry on open as `embedded=True`, so nothing changes for them
unless you delete the copies yourself.

Projects written before the lists were merged kept two entries —
`training_files` (the curated training set) and `audio_files` (the working
session list, plain path strings). They are folded into one `audio_files`
registry on open, registered entries first, and nothing is dropped. Recordings
that only ever sat in the session list now train the model, since that is what
being in the list means.

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

## Batch runs at scale

The target workflow is one model applied to thousands of recordings, so the
batch path is built around three properties.

**Recursive folder targets.** 24/7 multi-mic sets are nested
(experiment / mic / day), so folder scans walk the tree. The **Folder** target in
Run Inference is deliberately *not* routed through the Audio list: adding 3,000
recordings there would mean 3,000 list widgets plus a sibling-CSV stat per row
before anything runs — and would enrol production audio into the project's
training set, since everything in the list trains. The folder is scanned once and
paths stream to the worker.

**Resumable runs.** Two independent mechanisms, answering different questions:

- `batch_runs/<timestamp>/manifest.jsonl` — the run's own append-only log, one
  JSON line per file, flushed immediately. A crash at file 2,800 of 3,000 leaves
  a manifest valid up to 2,799; a torn final line is skipped on read.
- **Provenance skipping** asks whether a recording already carries detections
  from *this* model at *these* settings. That lives in the recording's own CSV
  (`model_name`, `threshold`, `min_blob_pixels`), so it survives losing the
  manifest and works across runs when two folders overlap.

Only settings that change *detections* invalidate prior work — `batch_size`,
`amp` and `device` deliberately do not, so raising the batch size never forces a
redo. Zero-detection files write no prediction rows and so can't be proven done
from the CSV alone; the manifest covers them, which matters because quiet files
dominate 24/7 recordings.

**Bounded memory.** `infer_probability_mask` never materializes the whole
probability grid as float32. The output grid is uint8 (probability x 255), the
Hann weight accumulator is 1-D (it never varied along frequency), and the float32
accumulator is per-chunk with the un-finalized tail carried across chunk
boundaries — so the result is identical to whole-grid accumulation with no seam
artifacts. Measured peak RSS for a 100 s @ 250 kHz slice: **1240 MB -> 95 MB**.

`mad analyze` has the same behavior headless (`--no-resume`, `--run-dir`,
`--log-root`, `--batch-size`, `--no-amp`) for HPC or overnight runs.

---

## Knowing when a model is ready

### The train/val split holds out whole recordings

Validation is only a measurement if the model has never seen the data. A
segmentation pipeline makes that easy to get wrong: one long call is sliced into
several tiles, and neighbouring tiles overlap, so shuffling *tiles* into train
and val puts near-duplicate — sometimes literally the same — pixels on both
sides. On a representative label set (3 recordings, 120 calls, long calls
spanning 3 tiles), a tile-level shuffle left **100%** of validation tiles coming
from a recording the model trained on and **55%** from a call it had already
seen other tiles of. Validation Dice then measures memorization.

MAD splits by **group**, strongest level the labels support:

| Level | What's held out | When it's used |
|---|---|---|
| `file` | whole recordings | labels on ≥2 recordings — **the only one that answers "will this work on a new recording?"** |
| `call` | whole calls, recordings shared | all labels on one recording |
| `tile` | nothing — train and val overlap | a single labeled call |

The level is reported next to the score everywhere it appears (training log, run
summary, CLI, and the checkpoint itself), and anything below `file` prints a
warning, because a Dice quoted without that caveat is how an over-fitted model
gets written up as a working one. `split_seed` (default 42) is recorded in the
run summary, so a reported number can be reproduced exactly.

The practical consequence: **label at least two recordings** before trusting a
validation number, and more if your recordings differ in mic, animal, or noise
floor — that variation is exactly what the held-out file is there to test.

### Call-level evaluation

Training reports validation **Dice** — a *pixel* score on *tiles*. It does not
answer the question that decides whether to spend hours of compute: how many real
calls will this find, and how much junk will I have to reject?

**Evaluate Model…** answers that at call level. It matches predicted blobs
against hand-labeled calls (time and frequency IoU, greedy by score, one-to-one —
the same accounting a human does) and reports precision / recall / F1 across a
threshold sweep. Because every prediction stores its own score, the whole curve
comes from **one** inference pass: run once at a permissive threshold, then
re-score the same detections at each cutoff. Picking a threshold stops being a
guess. Use held-out labeled files — scoring on training files flatters the model.

---

## Reviewing at scale

After a large run the bottleneck is human attention, not compute.

- **Run Summary** (Ctrl+B) — per-file results from the manifest: which
  recordings have calls, how many, calls/min. Reads no audio, so it opens
  instantly. Double-click to review a file.
- **Detection Gallery** (G) — a contact sheet of mask crops; click tiles to
  cycle accept -> reject, then Apply. This is what the per-call crop storage
  buys: a page is a few hundred KB and touches no audio. Decisions commit
  through the same paths as one-at-a-time review, so training examples, CSV
  status and undo behave identically.
- **Min score slider** — hides pending predictions below a confidence.
  Call-level re-thresholding for free, with no re-run, using the stored score.
  It never hides accepted or rejected calls (a decision is not a guess), and the
  count label always reports how many are hidden.

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
| `mad_training.py` | Train the U-Net from confirmed examples; grouped (leak-free) train/val split. |
| `mad_examples.py` | Confirmed training-example store (`training_data.h5`). |
| `mad_dataset.py` | Spectrogram/tile helpers shared by training & inference. |
| `mad_project.py` | Project config / on-disk layout; folds pre-merge two-list projects into one `audio_files` registry. |
| `mad_registry.py` | Referenced-recording registry: fingerprints, re-resolving moved files. |
| `mad_labels.py` | Sibling-path helpers (CSV naming, etc.). |
| `fnt_mask_store.py` | Shared HDF5 mask storage (CAD + MAD); `/calls`, `/pred_calls`, training store, repack. |
