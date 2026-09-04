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

MAD follows SLEAP's project model: a project **points at** recordings where they already live, the way a SLEAP project points at videos. The project folder holds those path references, the training-data store, and trained model checkpoints — never copies of your audio, which for 250 kHz recordings would run to hundreds of MB per file. Detections are saved as a sibling CSV next to each recording, cross-readable with the Classic Audio Detector (CAD) output.

There is a single **Audio** list: every recording the project knows about. Anything you confirm in it trains the model — there is no separate training set to curate. Large production runs stay outside the list via the **Folder** inference target, so recordings you are not curating never join the training set.

A project is optional until you train. You can add wavs, label them, load a model from any trained project, run inference and review the results with no project open — labels and detections save beside the audio either way. Training needs a project, because the model and its example store have to live somewhere.

Because recordings are referenced rather than copied, a recording that moves is a **soft** state: the row is flagged, but training, the example store and every saved detection keep working, and **File ▸ Locate Missing Recordings…** repoints a whole moved tree from one file. **File ▸ Pack Project** is the escape hatch — it copies the audio in, making the project self-contained for archiving or sharing.

## Detection output

Every recording gets its own standalone `<wav>_FNT_MAD_annotations.csv` sibling holding that file's complete detection table — the same per-file profile SLEAP and DeepLabCut use for video. There is no aggregate run-level table and no interchange export: MAD is deliberately self-contained for both analysis and review, and third-party formats (Raven, Audacity) are upkeep the project does not carry. If a downstream tool is ever needed, compatibility gets built then.

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
