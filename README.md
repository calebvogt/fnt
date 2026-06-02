# FieldNeuroToolbox (fnt)

Preprocessing and analysis toolbox for neurobehavioral data.

Authored by Caleb C. Vogt, PhD in extensive collaboration with Anthropic's Claude (especially Opus 4.6).

This is a broad-based toolbox, with many different kinds of tools within it. The Video Tracking tools in particular were inspired/influenced by my experience using various academic softwares, including SLEAP, DeepLabCut, idTracker, UMATracker, and LabGym. Without exposure to these other softwares, I would not have been able to intuit the specific style of workflow utilized in the various FNT tracking tools (Simple Tracker and Mask Tracker).

## Getting Started

### Standalone Executables (Recommended)

The easiest way to use FieldNeuroToolbox is by downloading the pre-built standalone executable for your operating system. **You do not need to install Python or Anaconda to use this version.**

**Requirements for the Executable:**

- **ffmpeg**: Must be installed and added to your system PATH for video processing features to work.

1. Navigate to the **Releases** page on the GitHub repository.
2. Download the latest release `.zip` or `.tar.gz` for your operating system (Windows, macOS, or Linux).
3. Extract the downloaded archive.
4. Run the extracted `fnt` executable to launch the GUI.

_Note: For GPU acceleration in SAM-based video tracking, you will need an NVIDIA GPU with CUDA drivers installed on your system. If you want full control over your Python environment or want to modify the code, see the [Development Installation](#development-installation) section at the bottom._

## Tools

The GUI is organized into tabbed categories. Each tool launches in its own window and can run in parallel with others.

### Video

- **Video PreProcessing** — Batch video preprocessing: downsampling, re-encoding, format conversion, frame rate adjustment, grayscale conversion, CLAHE enhancement, and audio removal. Supports customizable quality, resolution, codec, and encoding presets.
- **Video Trim and Crop** — Interactive video trimming and cropping with batch processing. Set start position, duration, and draw crop regions with per-video configuration.
- **Video Concatenation** — Concatenate multiple videos within directories using FFmpeg with progress tracking.
- **Behavior Scoring Studio** — Manual behavioral annotation with ethogram definition, custom behavior categories with keyboard shortcuts, video playback controls, timeline visualization, and CSV/JSON export.
- **Simple Tracker** — Fast CPU-only tracking using background subtraction (MOG2) for static camera setups. Multi-object centroid tracking with Hungarian matching, single-animal mode, and batch processing with CSVs, plots, and tracked video output.
- **Mask Tracker (SAM2)** — Instance segmentation annotation and training. Annotate frames with manual or AI-assisted masks using SAM2, then train Mask R-CNN models for automated tracking. Generates bounding boxes, instance masks, and trajectory data.
- **ROI Tool** — Post-processing tool for tracking data (SLEAP or Mask Tracker). Define regions of interest, analyze spatial occupancy, and generate occupancy timeseries and summary statistics. Supports Open Field Test, Light-Dark Box, and custom ROI configurations.
- **SLEAP Inference** — Run SLEAP pose estimation inference with optional tracking on videos. Supports top-down and bottom-up models with configurable tracking parameters.
- **SLEAP File Conversion** — Batch convert SLEAP `.slp` prediction files to CSV and HDF5 analysis formats.
- **SLEAP Re-tracking** — Re-run tracking on existing SLEAP predictions without re-running inference.
- **SLEAP Video Rendering** — Create tracked videos from existing `.slp` files without re-running inference.
- **LabGym Training Image Generator** — Extract frames from videos for LabGym training datasets with configurable frame interval and sampling methods.

### USV (Ultrasonic Vocalization)

- **Classic Audio Detector (CAD)** — DSP-based USV detection and labeling. Load audio files, apply signal processing for peak detection, perform manual ground-truthing, and train Random Forest classifiers for automated labeling. Spectrogram-based UI with waveform display.
- **Deep Audio Detector (DAD)** — YOLO-based ML model training and inference for USV detection. Project-based workflow: create projects, load audio, label manually, train YOLO models, and run inference with spectrogram visualization.
- **Mask Audio Detector (MAD)** — Pixel-level segmentation-based USV detection. Paint-based annotation of spectrograms with brush/eraser tools, U-Net model training, and blob-review inference workflow.
- **USV Heterodyne Processing** — Batch heterodyne processing for ultrasonic recordings. Mixes audio with a 40 kHz carrier and band-pass filters to shift ultrasonic content to the audible range.
- **Audio Trimming** — Trim audio files with spectrogram visualization and frequency filtering.
- **WAV Compression** — Compress WAV files for storage using FLAC or other codecs with batch processing.

### UWB (Ultra-Wideband)

- **UWB PreProcessing** — Preprocess and export UWB tracking data from UWB receivers. Parse proprietary database formats, extract position/time data, handle timezone conversions, and export to CSV.

### RFID

- **RFID Preprocessing** — Universal RFID preprocessing pipeline: raw data to movement bouts, GBI (group-by-individual) matrices, social networks, edgelists, displacement detection, and Hinde index calculation. Configurable zone templates.

### WiFP (Wireless Fiber Photometry)

- **Process .doric Files** — Batch process Doric `.doric` (HDF5) fiber photometry files. Auto-detect 470 nm signal and 405/415 nm isosbestic channels, calculate ΔF/F with isosbestic correction, synchronize with behavior video timestamps, and export CSV and combined overlay video.
- **Explore .doric Structure** — View the internal HDF5 structure of `.doric` files to inspect datasets, groups, and metadata.

### Imaging

- **CZI Viewer** — View and process Zeiss CZI multi-channel microscopy images. False-color channels, adjust brightness/contrast/gamma/sharpness, background subtraction, annotations, and export to PNG/TIFF.
- **Image Quantification** — Cell counting and colocalization analysis for CZI images. Multi-channel particle detection with watershed separation, colocalization metrics, ROI-based density measurement, and CSV export.

### FED3

- **FED Sync & Tracking** — Interface for managing Feeding Experimentation Device (FED3) data over serial connections. Add devices, configure auto-sync schedules, and visualize pellet retrieval data over time with real-time plots.

### Utilities

- **File Splitter** — Split large files to meet GitHub's 50 MB limit. Smart CSV splitting (row-based, preserves headers) or binary splitting.
- **Data Transfer** — Copy data files to a destination folder with optional auto-splitting, recursive directory traversal, preserved folder structure, and optional file curation via tree-view explorer.

## Development Installation

If you intend to modify the code or prefer managing your own Python environment, use this method.

### Prerequisites

- Git
- Anaconda or Miniconda
- ffmpeg (installed and added to your system PATH)
- Package dependencies are listed in `pyproject.toml`

### Windows Installation

Clone the fnt GitHub repo and install the package in editable mode. **Python 3.12 or newer is required** (3.13 recommended; the codebase uses Python 3.12+ syntax):

```bash
conda create --name fnt python=3.13
conda activate fnt
cd path\to\fnt
pip install -e .
```

#### GPU Support for Video Tracking (Recommended)

For SAM-based video tracking, GPU acceleration provides ~50x speedup. Install PyTorch with CUDA support:

```bash
conda activate fnt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Note**: This requires an NVIDIA GPU with CUDA support. The tracker will work on CPU but will be significantly slower (~10s per SAM frame vs ~0.1s on GPU).

To verify GPU detection:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

You should see `CUDA available: True` and your GPU name.

### Mac Installation

Open terminal:

```bash
conda create --name fnt python=3.13
conda activate fnt
cd path/to/fnt
pip install -e .
```

### Launch the GUI

After installation, launch the FieldNeuroToolbox GUI:

```bash
conda activate fnt
fnt
```

### Updating

Because the package is installed in editable mode, updating is simple:

```bash
cd path/to/fnt
git pull
```

That's it — the updated code is immediately available the next time you run `fnt`.

**Note:** If the update includes changes to `pyproject.toml` (e.g., new dependencies), you will need to re-run the install:

```bash
conda activate fnt
pip install -e .
```
