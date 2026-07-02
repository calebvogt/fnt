# Audio / USV Module

Tools for processing, detecting, and classifying ultrasonic vocalizations (USVs) and other audio signals. Supports high-sample-rate recordings (up to 250+ kHz) commonly used in rodent vocal communication research.

## Tools

### Mask Audio Detector (MAD)

Launch from FNT main GUI: **Audio** tab → **Mask Audio Detector (MAD)**

Project-based spectrogram labeling, training, and inference tool:

- **Interactive spectrogram viewer** — pan, zoom, and adjust frequency range across the full recording
- **Paint-based labeling** — use brush and SAM-assisted tools to paint call masks directly on the spectrogram
- **Label workflow** — pending (yellow) → confirmed (green) or rejected (red) annotations
- **Training** — train a lightweight CNN model on labeled examples within the GUI
- **Inference** — run trained models on new recordings; review and correct predictions
- **Project system** — organizes session audio, training data, and models in a single project folder
- **HDF5 mask storage** — labeled examples stored in a shared `.h5` store for efficient access

### Classic Audio Detector (CAD)

Launch from FNT main GUI: **Audio** tab → **Classic Audio Detector (CAD)**

DSP-based detection and labeling tool:

- **Spectrogram-based detection** using configurable DSP parameters (threshold, frequency band, minimum duration)
- **Manual labeling** — review detections and assign class labels
- **Random Forest classifier** — train and run classifiers on labeled call features
- **Harmonic linking** — associate fundamental frequencies with their harmonics
- **Batch processing** — run detection across folders of recordings

### Compress Audio Files

Launch from FNT main GUI: **Audio** tab → **Compress Audio Files**

Batch WAV compression for archival storage:

- Convert WAV files to FLAC (lossless) or other compressed formats
- Folder-based batch processing
- Preserves original sample rate and bit depth

### Trim Audio File

Launch from FNT main GUI: **Audio** tab → **Trim Audio File**

Interactive audio trimming with spectrogram visualization:

- Spectrogram preview of the full recording
- Select start/end points visually
- Frequency filtering options
- Heterodyne playback for previewing ultrasonic content in the audible range

### USV Heterodyne Processing

Launch from FNT main GUI: **Audio** tab → **USV Heterodyne Processing**

Batch conversion of ultrasonic recordings to audible heterodyne signals:

- Mixes the recording with a carrier frequency (default 40 kHz) to shift ultrasonic content into the audible range
- Band-pass filtering to isolate the difference frequencies
- Batch processes all `.wav` files in a folder

## Audio Format Support

The module uses **soundfile** for standard WAV formats and falls back to **FFmpeg** for exotic encodings (e.g., ADPCM). FFmpeg must be installed and on PATH for full format support.

## Requirements

- **FFmpeg** — required for ADPCM and other non-PCM WAV formats. On Windows, install system-wide from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/). On macOS/Linux, `conda install -c conda-forge ffmpeg` works.
- **PyTorch** — optional, enables GPU-accelerated spectrogram computation and MAD model training/inference.
