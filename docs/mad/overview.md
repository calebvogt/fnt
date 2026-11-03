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

# Flag running-wheel squeaks among the stored detections
mad wheel --input recordings/ --workers 3 --low-priority
```

Run `mad <command> --help` for the full option list.

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
