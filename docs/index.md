---
hide:
  - navigation
---

# FieldNeuroToolbox (FNT)

Preprocessing and analysis toolbox for field and laboratory neurobehavioral data. FNT is open source and free for research and hobby use. It is released as-is and grows alongside our own research.

**Bugs and feature requests:** please [open an issue](https://github.com/calebvogt/fnt/issues).

## What's inside

FNT bundles GUI tools for the common data types of field and laboratory neuroethology, in one desktop app that runs on an ordinary laptop:

<div class="grid cards" markdown>

- :material-waveform: **[Audio / USV](tools/audio.md)**

    Ultrasonic vocalization detection, classification and audio preprocessing, including the Classic Audio Detector (CAD).

- :material-brain: **[Mask Audio Detector (MAD)](mad/overview.md)**

    Label calls on the spectrogram with SAM, train a segmentation model, and run it over thousands of recordings.

- :material-video: **[Video Processing](tools/video-processing.md)**

    Batch preprocessing, trimming, concatenation and manual behavioral scoring.

- :material-target: **[Video Tracking](tools/video-tracking.md)**

    SAM-based interactive animal tracking for behavioral assays.

- :material-shape-outline: **[Mask Tracker](tools/mask-tracker.md)**

    Label animals with SAM, train a YOLO segmentation model, review its predictions and run it over whole videos.

- :material-map-marker-distance: **[UWB Tracking](tools/uwb.md)**

    Ultra-wideband positions to cleaned, smoothed tracks, contact bouts, chases and region occupancy.

- :material-antenna: **[RFID Tracking](tools/rfid.md)**

    Biomark reader downloads to cleaned reads, bouts, zone holders and co-occupancy, with QC reports.

- :material-microscope: **[Imaging](tools/imaging.md)**

    Zeiss CZI microscopy viewing, processing and cell quantification.

</div>

## Quick start

The easiest way to use FNT is the pre-built standalone executable; no Python or Anaconda is needed.

1. Download the latest release for your OS from the [Releases page](https://github.com/calebvogt/fnt/releases).
2. Extract the archive.
3. Run the `fnt` executable.

See [Installation](installation.md) for the standalone requirements (ffmpeg) and for the development installation if you want to modify the code.

## Tutorials

Video tutorials are posted to [this YouTube playlist](https://youtube.com/playlist?list=PLY8yLegR_viXsARZr460L3ZHRp3HZsOOC).

## Citing FNT

If you use FNT in your research, please cite it. See [Citing and license](about.md) for the citation, the license and the attribution FNT's license asks of copies.
