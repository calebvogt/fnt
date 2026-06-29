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
  - `status` — `pending` (yellow, unreviewed) · `accepted` (green; hand-labels
    write accepted, and accepting a prediction in train mode also saves a
    training example) · `rejected` (a **recorded** human "no", kept visible red
    as an audit trail; *not* used as negative training data). **Delete** drops
    the row + mask entirely (no `deleted` status).
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

### Browse-in-place ingestion (finding calls in big recording sets)

When recording 24/7 you may have hundreds of wavs and not know which contain
USVs. Adding a folder to **Label & Train** does **not** copy those files into
the project. Instead they're **browsed in place**:

- Their masks/predictions are redirected to `<project>/.scratch/` (so the
  original recording folders stay clean — no stray `.h5`/`.csv` siblings). This
  redirect is implemented as a path override consulted by
  `fnt_mask_store.masks_sibling_path` / `mad_labels.pred_csv_sibling_path`.
- A file is **graduated** — copied into `recordings/`, with its scratch masks
  moved alongside and the file recorded in the project — only when you **accept
  a call** on it (hand-label + confirm, or change a prediction to accepted, via
  `_ensure_wav_in_project`). Reject/pending/delete do **not** graduate a file.
- On close, `.scratch/` is discarded, so files you never accepted leave no
  trace. Browsing is **session-only**: reopening the project shows just the
  graduated files; re-add the folder to keep searching.

So you can point MAD at 500 raw recordings, run post-training inference, accept
calls in the handful that have USVs, and only those few wavs (plus their masks)
ever land in the project — no bulk copying of files you'll discard.

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
