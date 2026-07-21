# Audio / USV

Tools for processing, detecting, and classifying ultrasonic vocalizations (USVs) and other audio signals. Supports high-sample-rate recordings (up to 250+ kHz) commonly used in rodent vocal communication research.

Launch any tool from the FNT main GUI **Audio** tab.

## Mask Audio Detector (MAD)

Project-based spectrogram labeling, training, and inference tool. See the [MAD Overview](../mad/overview.md) for a full description of the pipeline.

- **Interactive spectrogram viewer** — pan, zoom, and adjust frequency range across the full recording
- **Paint-based labeling** — brush and SAM2-assisted tools to paint call masks directly on the spectrogram
- **Label workflow** — pending → confirmed or rejected annotations
- **Training** — train a U-Net segmentation model on labeled examples within the GUI
- **Inference** — run trained models on new recordings; review and correct predictions
- **Project system** — organizes session audio, training data, and models in a single project folder
- **Export** — detections to CSV, Raven selection tables, and Audacity label tracks

## Classic Audio Detector (CAD)

DSP-based detection and labeling tool.

- **Spectrogram-based detection** using configurable DSP parameters (threshold, frequency band, minimum duration)
- **Manual labeling** — review detections and assign class labels
- **Random Forest classifier** — train and run classifiers on labeled call features
- **Harmonic linking** — associate fundamental frequencies with their harmonics
- **Batch processing** — run detection across folders of recordings

## Compress Audio Files

Batch WAV compression for archival storage — convert to FLAC (lossless) or other formats with folder-based batch processing, preserving sample rate and bit depth.

## Trim Audio File

Interactive audio trimming with spectrogram visualization, visual start/end selection, frequency filtering, and heterodyne playback for previewing ultrasonic content in the audible range.

## USV Heterodyne Processing

Batch conversion of ultrasonic recordings to audible heterodyne signals — mixes the recording with a carrier frequency (default 40 kHz) to shift ultrasonic content into the audible range, with band-pass filtering to isolate the difference frequencies.

## Audio format support

The module uses **soundfile** for standard WAV formats and falls back to **FFmpeg** for exotic encodings (e.g., ADPCM). FFmpeg must be installed and on PATH for full format support — see [Installation](../installation.md).
