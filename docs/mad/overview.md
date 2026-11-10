# Mask Audio Detector (MAD)

MAD detects and classifies vocalizations using a custom deep-learning pipeline. Rather than hand-tuned DSP thresholds, users label call regions directly on the spectrogram — with the help of a Segment Anything Model — and train a segmentation model that learns to find and classify calls automatically.

Launch from the FNT main GUI **Audio** tab → **Mask Audio Detector (MAD)**.

## How it works

- **Spectrogram generation** — raw audio is converted to a time–frequency spectrogram (short-time Fourier transform) at the recording's native sample rate (up to 250+ kHz).
- **Interactive labeling with SAM2** — the user clicks on a vocalization and a Segment Anything Model 2 (SAM2) proposes a pixel-level mask for that call, dramatically accelerating annotation compared to manual painting.
- **Manual refinement** — brush and eraser tools correct SAM2 proposals. Labels follow a **pending → confirmed / rejected** workflow to keep training data clean.
- **Training-data storage** — confirmed call masks are stored as rasterized examples in a shared HDF5 store, paired with the corresponding spectrogram patches.
- **Semantic segmentation model** — labeled examples train a **U-Net** (ResNet18 encoder, via `segmentation-models-pytorch`) to perform pixel-level call detection on spectrograms.
- **Batch inference** — the trained model runs across recordings, producing per-pixel prediction masks that are thresholded and converted to discrete call detections with time, frequency, duration, and class metadata.
- **Iterative refinement** — predictions are reviewed in the GUI, where the user confirms or corrects them, feeding new examples back into the training set.

## Project structure

MAD follows SLEAP's project model: a project **points at** recordings where they already live, the way a SLEAP project points at videos. The project folder holds those path references, the training-data store, and trained model checkpoints — never copies of your audio, which for 250 kHz recordings would run to hundreds of MB per file. Detections are saved in a `.mad` file next to each recording.

There is a single **Audio** list: every recording the project knows about. Anything you confirm in it trains the model — there is no separate training set to curate. Large production runs stay outside the list via the **Folder** inference target, so recordings you are not curating never join the training set.

A project is optional until you train. You can add wavs, label them, load a model from any trained project, run inference and review the results with no project open — labels and detections save beside the audio either way. Training needs a project, because the model and its example store have to live somewhere.

Because recordings are referenced rather than copied, a recording that moves is a **soft** state: the row is flagged, but training, the example store and every saved detection keep working, and **File ▸ Locate Missing Recordings…** repoints a whole moved tree from one file. **File ▸ Pack Project** is the escape hatch — it copies the audio in, making the project self-contained for archiving or sharing.

## Detection output

Each recording's masks, detections and review status live in its `<recording>_FNT.mad` file beside the wav. **File ▸ Export Annotations to CSV…** turns that into a standalone `<recording>_FNT_MAD_annotations.csv` per recording — the same per-file profile SLEAP and DeepLabCut use for video. For analysis across a run, **File ▸ Export…** (or `mad export`) writes folder-wide tables instead: one row per recording, one per detection and one per labelled call, with optional spectrogram images of the labelled calls. See [Exporting tables](export.md). There is no interchange export: third-party formats (Raven, Audacity) are upkeep the project does not carry. If a downstream tool is ever needed, compatibility gets built then.

## Consecutive-detection merging

A long call can surface as several adjacent blobs — split across inference tile
seams or broken by a brief sub-threshold dip. Optional **consecutive merging**
stitches these back into one detection: blobs that are close enough in time
(and, by default, overlap in frequency) are combined into a single call with a
union bounding box and an area-weighted confidence. The frequency-overlap gate
keeps calls that are adjacent in time but separated in frequency — such as a
harmonic and its fundamental — distinct. Enable it via the CLI
(`--merge-consecutive`) or the `MADInferenceConfig.merge_consecutive` flag.

## Joining fragments after inference

A faint call can cross the model's threshold in pieces. If inference ran
without merging, each piece is stored as its own detection, which inflates
call counts and splits shape measures. `mad join-fragments` joins them in the
`.mad` files afterwards. It needs no model, GPU or audio.

```bash
mad join-fragments recordings/ --gap-ms 10 --max-df-khz 10
mad join-fragments recordings/ --dry-run      # report only
mad join-fragments recordings/ --undo         # put the pieces back
```

It takes `.mad` files, wav files (their `.mad`) or folders (walked to the
bottom). Two detections A (the earlier) and B are joined when all of these
hold:

1. Both are pending: no label or rejection names them, and no label's box
   overlaps them. Reviewed calls are never changed.
2. Both are fundamentals (`harmonic_n` 1).
3. The gap from A's last frame to B's first is at most `--gap-ms` (default
   10). The limit is rounded to whole frames as inference rounds it, so
   10 ms is 20 frames (10.24 ms). Pieces that overlap in time pass.
4. B's start frequency is within `--max-df-khz` (default 5) of A's end
   frequency. Each is the brightness ridge (the brightest path through the
   mask, as stored in `/pred_ridge`) averaged over 3 frames (`--end-frames`)
   at that end.
5. If they overlap in time by more than half the shorter one
   (`--concurrent-fraction`), their frequency bands must also overlap. This
   keeps a harmonic, or a second animal's call, separate.

`--band-rule` also applies inference's own band test: sequential pieces'
bands must overlap or lie within `--band-max-gap-bins` (default 8, about
2 kHz).

A detection joins at most one piece before it and one after it, best match
first, so a call in three pieces comes back as one. The joined call is the
union of the pieces' masks. The gaps between them stay empty. It keeps the
earliest piece's id, and records `joined_from` (the piece ids,
comma-separated) and `n_pieces`. Its score is the pieces' scores weighted by
area. Its ridge is recomputed. Harmonic grouping is then recomputed for the
recording, but stored only on the joined calls and the detections whose
harmonic call id named one of the pieces. Every other detection and every
label keeps its grouping.

Before a detection is first changed, its original entries are copied into
`/unjoined` in the same file. Running the command again with other settings
starts from those originals. `--undo` restores them exactly and removes the
join's root attributes (`join_gap_s`, `join_max_df_khz`, `joined_at`,
`join_fnt_version` and the others). A joined call that has been reviewed or
deleted since the join is left as it is, with its pieces still kept in
`/unjoined`. If the recording is analysed again after a join, the copies are
out of date: they are dropped and never restored over the new detections.
Run `mad wheel` again after a join, since its rows are per detection.

The MAD window opens a joined file as usual. Joined calls are ordinary
pending detections.

**Choosing the settings.** On the 2021_8x8 pair queue (57 neighbouring
detections judged by eye on 2026-11-09: 31 one call, 21 two calls, 5
unclear), 10 ms and 10 kHz joined 22 of the 31 one-call pairs and 2 of the
21 two-call pairs. Inference's band rule at 10 ms joined 20 and 9. A 5 kHz
limit missed many pieces of steep sweeps, whose ends are several kHz apart
even across a short gap.

## Command-line interface

MAD's train / analyze / embeddings pipeline is also available headless as the
`mad` console script, for scripting into batch and HPC workflows:

```bash
# Run a trained model over a folder (recursive), merging split calls
mad analyze --model weights.pt --input recordings/ --merge-consecutive

# Train a model from a MAD project directory
mad train --project my_project/ --epochs 30 --device cuda

# Export per-detection embeddings for clustering / similarity analysis
mad embeddings --model weights.pt --input recordings/ --out embeddings.npz

# Folder-wide tables of recordings and detections (see Exporting tables)
mad export --input recordings/ --out tables/

# Add stored spectrograms, then brightness ridges, to existing .mad files
mad add-specs --input recordings/
mad add-ridge --input recordings/ --model <project>/models/<run>

# Join detections that are pieces of one call (reversible with --undo)
mad join-fragments recordings/ --gap-ms 10 --max-df-khz 10

# Flag running-wheel squeaks among the stored detections
mad wheel --input recordings/ --workers 3 --low-priority

# Inference with only its progress window on screen; it quits when the run ends
python -m fnt.usv.mad_pyqt --project my_project/ --infer <project>/models/<run> --inference-window-only
# ...and stop that run from the command line: it finishes the current file and quits, no prompt
mad stop --project my_project/
```

Run `mad <command> --help` for the full option list. A run started again on the same
project skips the recordings it already analysed at the same settings.

## Running-wheel squeaks

A running wheel near a microphone thumps in the low register (2-15 kHz) at a
steady beat, about 7 per second, while a mouse runs. It can also squeak in
the upper register, and the model can take those squeaks for calls. They are
short and narrow, often near 105 kHz but also lower, and they come back at
the thump beat. `mad wheel` finds them from the audio alone; no video is
needed.

For each detection it looks at 2 s of audio either side:

1. **The beat** is the autocorrelation peak of the 2-15 kHz energy between 2
   and 20 Hz. It takes the shortest strong lag, so a subharmonic (3 beats)
   is not taken for the beat.
2. **The comb** of a band is how much its autocorrelation peaks at 1, 2 and 3
   beats and dips half-way between them. The rule computes it for the low
   band, and for the band of the detection's own mask.
3. **A detection is flagged** when all of these hold:
    - the low band's comb is at least 0.15 (the wheel beats cleanly);
    - its own band's comb is at least 0.25;
    - the low band's autocorrelation at the beat is at least 0.2;
    - it is shorter than 0.6 of a beat.

A mouse can call while it runs, so running alone flags nothing. A call that
lasts several beats, or a bout with its own rhythm, does not comb at the
wheel's beat and is kept. Requiring the low band's own comb keeps dense call
bouts: at about 8 calls per second they are close to the wheel's rhythm, but
the mouse's own movement noise has no clean beat.

The rule was calibrated on the 2021_8x8 labels. It flagged:

- 1 of 1,515 accepted calls;
- 40 % of rejected C57 detections;
- 75 % of a hand-checked squeak train.

The results go to `/wheel` in each `.mad`, and nothing else in the file
changes. `/wheel` holds:

- one row per detection: `blob_id`, `squeak` (0/1), `comb`, `low_comb`,
  `period_s`, `low_acf` and `dur_beats`;
- a running track in 4-s windows every 0.5 s: `track_t_s`,
  `track_period_s`, `track_acf`, `track_comb` and `track_running`.

A recording is redone automatically after a new inference run. Detections
are never deleted. To leave squeaks out of an analysis, drop the rows with
`squeak` = 1.

## Call embeddings

The `embeddings` command extracts a fixed-length feature vector for each saved
detection by pushing its spectrogram patch through the trained model's encoder
and global-average-pooling the deepest feature map (512 dimensions for a
ResNet18 encoder). The output `.npz` holds the embeddings alongside each call's
source file, id, time bounds, and class — ready for clustering calls by
similarity, surfacing novel call types, or comparing repertoires across animals.

## Requirements

- **FFmpeg** — required for ADPCM and other non-PCM WAV formats. See [Installation](../installation.md).
- **PyTorch** — required for model training and inference; a CUDA-capable GPU is strongly recommended. See [GPU Support](../installation.md#gpu-support-optional).
