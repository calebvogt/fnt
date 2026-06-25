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

- `<wav>_FNT_MAD_predictions.csv` — one row per predicted call: box (start/stop
  s, min/max Hz), area, **score**, and review **status**:
  - `pending` — not yet reviewed (shown yellow).
  - `accepted` — confirmed by the user; in train mode also saved as a training
    example (shown blue / removed from the pending queue).
  - `rejected` — a **recorded** human "no": kept visible shaded **red** and
    labeled "Reject" as an audit trail of what the labeler dismissed and why
    (informs a later user of the accept criteria). Rejected masks are *not* used
    as explicit negative training data — training only consumes accepted
    examples — but the record is valuable for reproducibility.
  - **Delete** removes a detection entirely and leaves no trace — its CSV row
    and stored crop are both dropped (no `deleted` status is written). Use
    Delete for genuine noise you don't want recorded; use Reject to record a
    decision.
- `<wav>_FNT_masks.h5` — pixel data:
  - `/calls/<id>` — confirmed (human-labeled) call mask crops.
  - `/pred_calls/<blob_id>` — **predicted call mask crops** (small,
    gzip-compressed uint8, with `f_off`/`t_off` offsets). Joined to the CSV by
    `blob_id`.
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
