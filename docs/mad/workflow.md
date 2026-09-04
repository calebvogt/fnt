# MAD Workflow — from a new project to a finished batch

This is the practical, step-by-step path. It says what to click and why, and
where the traps are. For *how the pieces work* (storage layout, the CSV columns,
the normalization rule, resume semantics) see the technical reference in
[`fnt/usv/usv_detector/MAD_README.md`](https://github.com/calebvogt/fnt/blob/dev/fnt/usv/usv_detector/MAD_README.md).

The loop you are running, end to end:

```
label a few calls → train → run on your labeled files → accept/reject
       ↑                                                       │
       └──── add breadth, retrain ◄──── evaluate ◄─────────────┘
                                          │
                                   good enough? → batch run over everything
```

Most of the time goes into the loop, not the batch. The batch is the easy part
once the model is right.

---

## 0. Before you start

Three checks, once per machine.

**GPU.** Open MAD and go to **Help ▸ Check GPU / CUDA setup…**. It should say
*GPU ready*. If it says PyTorch is CPU-only, it prints the exact `pip` command for
your driver — run it in the `fnt` env and restart. Do not skip this: on CPU,
inference runs at roughly realtime, which makes a 3,000-file batch a matter of
months rather than hours.

**Acoustic settings.** `nperseg / noverlap / nfft` and the dB range are fixed
for the life of a project and its models. The defaults (512 / 384 / 1024,
−100…−20 dB) are right for 250 kHz rodent USV recordings. Changing them later
means re-labeling, so decide now.

**Normalization.** `db_norm` is `fixed` by default. `per_file` normalizes each
recording to its own range and generalizes better across trials with different
gain and noise floors — but a label saved under one rule cannot be used under
the other, and training will refuse a mixed store. **Decide before you label.**
Reasonable default: stay `fixed`, get labeling breadth first, revisit if
cross-trial performance disappoints.

---

## 1. Create a project

**File ▸ New Project…** Pick a folder that will hold the project metadata,
the example store and the model checkpoints. It will *not* hold your audio.

A project references recordings where they already live, the way a SLEAP
project points at videos. That keeps projects small (kilobytes, not hundreds of
GB) and means one source of truth per recording. If you ever need a
self-contained copy — to archive it or hand it to someone — **File ▸ Pack
Project** copies the audio in.

---

## 2. Add audio

**Add Folder…** or **Add Files…** in the *Audio* panel. Add Folder offers the
whole subtree when there is one, which is what you want for a nested
`experiment / mic / day` layout.

Everything in the list is labelable, and **every confirmed call in the list
trains the model**. There is no separate training set to manage: if a recording
is in the list, its labels train; take it out and they stop.

Do not add your whole production set here. Add the handful you intend to label.
Large batches run through the **Folder** inference target in step 8, which never
touches this list.

A **⚠** row means the audio has moved or its drive is not mounted. Nothing is
broken — labels, the training store and saved detections are all fine, only
preview needs the file. **File ▸ Locate Missing Recordings…** repoints one file
and fixes every sibling that moved with it.

---

## 3. Label your first calls

Click a recording. Zoom with `↑`/`↓` or the wheel; pan with `←`/`→`. A 0.3–0.5 s
window is comfortable for USVs.

Three tools, one key each:

| Key | Tool | Use it for |
|---|---|---|
| **M** | SAM | Click a call, get a proposed mask. Right-click to add a negative point. This is the fast path. |
| **P** | Paint | Brush over pixels by hand. Wheel resizes the brush. |
| **E** | Eraser | Trim a proposal. |

**Enter** confirms the pending mask and saves it as a training example.
**Esc** / **Clear** discards it. **U** undoes the last one.

**What to label — decide once, write it down, hold to it.** The model learns
the boundary you draw; an inconsistent boundary teaches it that the boundary
is ambiguous, and you get unstable low-confidence predictions in exactly that
regime. A defensible policy for counting vocalizations:

- **Label every fundamental you can reliably trace**, including faint and
  distant ones. Skipping faint calls under-counts animals far from the mic.
- **Do not label harmonics.** Harmonic visibility tracks amplitude and
  distance, so labeling them turns one vocalization into 1–N detections and
  makes the count partly a measure of proximity to the microphone. Your
  training tiles span the full frequency axis, so an unpainted harmonic sitting
  above a painted fundamental is actively taught as *not a call*.
- Pick a concrete faint-call floor (e.g. "a contour I can follow across ≥3
  columns") and keep it. The CSV records `snr_db` per call, so you can filter
  later; labeling low preserves that option, skipping does not.

Aim for **breadth over depth**: thirty calls from each of twenty recordings
across your trials beats a thousand calls from three. Generalization, not
capacity, is what limits a first model.

---

## 4. Train a first model

In *Run Training*, open **Training settings** if you want to change them —
you usually don't for a first model. The defaults are
sensible:

- **U-Net + resnet18** for a first model. Move to resnet34/50 or **HRNet** (crisp
  thin contours) once you have a few thousand labels.
- **Max epochs 100, patience 20** — early stopping keeps the best checkpoint.
- **Augmentation on.** It is the biggest lever against overfitting a small set.
- **Device auto** picks the GPU.

Click **Run Training**. The live graph opens in its own window. If **Then
run inference on** is set to anything but *Nothing*, those recordings are
analyzed automatically on the new weights when training finishes — using the
detection settings (probability threshold, minimum blob size) from *Run
Inference*, so set those first if you want something other than the defaults.
The model lands in the *Model* dropdown, selected.

**Read the split line in the log.** Training holds out *whole recordings* for
validation when it can (`split_level: file`), so the reported score means
"works on a recording it never saw". With labels on only one recording it falls
back to holding out calls (`call`) or tiles (`tile`), and says so — those
numbers are optimistic. With very unbalanced per-file label counts the split can
hold out most of your data; check `n_val_tiles` rather than trusting the score.

---

## 5. Run on your labeled files and review

In *Run Inference*, tick **Audio list** with **All**, then click
**Run Inference**. (To predict as soon as training finishes instead, set
*Run Training*'s **Then run inference on** dropdown before you train.)
Predictions appear as pending (colored per the legend under *Labeling Tools*;
colors follow the color map so they stay visible).

Review with the keyboard:

| Key | Action |
|---|---|
| **A** | Accept — saves it as a training example |
| **R** | Reject — keeps a visible record, **and stores it as a hard negative** |
| **D** | Delete — removes every trace |
| **S** | Skip |
| **N / B** | Next / Back |
| **Ctrl+Z** | Undo (covers the example store too) |
| **G** | Gallery — a grid of the pending calls, click to mark, Apply |

**Reject is worth the keystroke.** A rejection is exactly a false positive a
human identified, and it is fed back to the model as a negative example. This is
also how harmonic suppression gets taught under the policy in step 3: reject the
harmonics the model finds, and it learns to stop.

**Q** runs the selected model on just the visible window — useful for spot
checks without a full run.

The **Min score** slider hides pending calls below a confidence. It never hides
accepted or rejected calls, and the count label always says how many are hidden.
The completion prompt will not claim a file is done while the filter is hiding
calls.

---

## 6. Iterate

Add a few recordings from a *different* trial, channel or time of day. Run the
current model on them, correct it, retrain. Two or three rounds of this is
normally where a model goes from "finds the obvious ones" to usable.

Where to spend labeling time: sort the Detections list by **Score** and work the
middle — calls near the decision threshold are where a label changes the model
most. Confidently right and confidently wrong calls teach it little.

---

## 7. Evaluate before you scale

**Evaluate Model…** (next to *Load Project Models*) scores the model at
**call level** — precision, recall, F1 across a sweep of thresholds, from one
inference pass. Training reports pixel Dice on tiles, which cannot tell you how
many real calls you will catch or how much junk you will reject.

Read it carefully:

- Use recordings you have **fully reviewed**. An unreviewed prediction is not
  ground truth, so re-detecting one scores as a false positive. The dialog says
  when precision is understated for this reason and by how many calls.
- Use recordings the model was **not trained on**, or the numbers flatter it.
- The best-F1 threshold is highlighted; **Use Selected Threshold** copies it
  into the inference settings. Raise it when false positives cost you review
  time, lower it when missed calls matter more.

---

## 8. The batch run

Tick **Folder**, click **Choose Folder…**, point it at the production tree. It
scans once (recursively) and streams paths to the worker; thousands of files
never become thousands of list rows.

Settings that matter:

- **Threshold / Min blob pixels** — from step 7.
- **Batch size 8, mixed precision on** — measured optimum on a 4070 Ti-class GPU;
  larger batches do not help this model.
- **Re-detect from scratch** off — a normal run keeps hand-labels and reviewed
  calls and only regenerates pending predictions.

Then **Run Inference**. Every finished file is logged immediately, so a crash or
a deliberate stop loses nothing: run it again and it offers to **resume**,
skipping files already analyzed *by this model at these settings*. Retrain,
re-run the same folder, and it correctly analyzes everything again.

For an overnight run without the GUI:

```bash
mad analyze --model <project>/models/<run>/weights.pt --input <folder> --device cuda
```

It resumes the same way (`--no-resume` to force a redo).

**Batch Run Summary** (`Ctrl+B`) is the triage view: per-file detection counts,
calls per minute, scan speed. Sort by detections, filter to files with calls or
to failures, double-click a row to open it for review.

---

## 9. What you get

A run records itself in `<recording>_FNT.mad` beside the wav — masks,
detections, review status and harmonic assignments in one HDF5 file. That store
is the record; nothing else is written during a run.

**File ▸ Export Annotations to CSV…** turns it into one standalone
`<recording>_FNT_MAD_annotations.csv` per wav — the same per-file profile SLEAP
and DLC use. Export when you are done reviewing, not before: the CSV is a
snapshot, and re-exporting after more review simply overwrites it. There is
deliberately no aggregate table; concatenate what you need in your analysis
code.

Every row is one call: onset/offset, frequency box and contour metrics (peak,
bandwidth, slope, excursion, sinuosity), spectral shape (centroid, entropy,
tonality), power and SNR, sequence metrics (inter-call interval, local call
rate), the model score, and `status` (`pending` / `accepted` / `rejected`).
Columns are documented in the technical reference.


---

## Gotchas

- **"GPU present, but PyTorch is CPU-only."** Step 0. It is the single biggest
  factor in how long everything takes.
- **Training refuses to start: "more than one dB normalization rule."** Labels
  were saved under both `fixed` and `per_file`. Pick one and re-label the
  minority; a saved patch cannot be re-normalized.
- **Validation score looks great but the model is poor on new files.** Check
  `split_level`. If it is `call` or `tile`, the score was measured on
  recordings the model trained on. Label a second recording.
- **A file shows ⚠.** Its audio moved. Training and detections are unaffected;
  use *Locate Missing Recordings…* to restore preview.
- **Resume skipped files it should have analyzed / analyzed files it should
  have skipped.** Resume keys on model *and* threshold *and* min-blob *and* the
  merge options. Change any of them and prior results are correctly not reused.
- **Accept-All on a file with the score filter set** only accepts the visible
  calls. That is intentional; the count label tells you how many are hidden.
