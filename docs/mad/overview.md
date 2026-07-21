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

MAD organizes work into a project folder holding session audio references, the training-data store, and trained model checkpoints. Detections are saved as a sibling CSV next to each recording, cross-readable with the Classic Audio Detector (CAD) output.

## Exporting detections

Beyond the native CSV, MAD exports detections to two interchange formats used widely in bioacoustics. Both are available from **File → Export Detections** and operate on the currently loaded file (rejected detections are dropped, and the remainder are sorted by onset time):

- **Raven selection table** (`*.selections.txt`) — a tab-delimited table that opens directly on a sound in [Raven Pro](https://www.ravensoundsoftware.com/). Call type maps to the Annotation column and model confidence to a Score column.
- **Audacity label track** (`*.labels.txt`) — an Audacity label file using the extended frequency-label format, so labels land on the spectrogram at the correct frequency band.

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
# Run a trained model over a folder, merging split calls, exporting Raven tables
mad analyze --model weights.pt --input recordings/ \
    --merge-consecutive --export raven

# Train a model from a MAD project directory
mad train --project my_project/ --epochs 30 --device cuda

# Export per-detection embeddings for clustering / similarity analysis
mad embeddings --model weights.pt --input recordings/ --out embeddings.npz
```

Run `mad <command> --help` for the full option list.

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
