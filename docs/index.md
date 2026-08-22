# FieldNeuroToolbox (fnt)

Preprocessing and analysis toolbox for field and laboratory neurobehavioral data. This software is fully open source and made available freely to the research and hobbyist community. It is released "as-is" and continues to evolve alongside our own research needs.

/ **Bugs and feature requests:** please [open an issue](https://github.com/calebvogt/fnt/issues).

## What's inside

FNT bundles a suite of GUI tools spanning the common data modalities in field and laboratory neuroethology:

<div class="grid cards" markdown>

- :material-waveform: **[Audio / USV](tools/audio.md)**
  Ultrasonic vocalization detection, classification, and audio preprocessing — including the Mask Audio Detector (MAD) and Classic Audio Detector (CAD).

- :material-video: **[Video Processing](tools/video-processing.md)**
  Batch preprocessing, trimming, concatenation, and manual behavioral scoring.

- :material-target: **[Video Tracking](tools/video-tracking.md)**
  SAM-based interactive animal tracking for behavioral assays.

- :material-map-marker-distance: **[UWB Tracking](tools/uwb.md)**
  Ultra-wideband positional data preprocessing and proximity analysis.

- :material-microscope: **[Imaging](tools/imaging.md)**
  Zeiss CZI microscopy viewing, processing, and cell quantification.

</div>

## Quick start

The easiest way to use FNT is the pre-built standalone executable — no Python or Anaconda required.

1. Download the latest release for your OS from the [Releases page](https://github.com/calebvogt/fnt/releases).
2. Extract the archive.
3. Run the `fnt` executable.

See [Installation](installation.md) for the standalone requirements (ffmpeg) and for the development installation if you'd like to modify the code.

## Tutorials

Video tutorials are posted to [this YouTube playlist](https://youtube.com/playlist?list=PLY8yLegR_viXsARZr460L3ZHRp3HZsOOC). Subscribe for updates.

## Citation

If you use FNT in your research, please cite the repository:

```bibtex
@software{fnt,
  title  = {FieldNeuroToolbox (fnt)},
  author = {Vogt, Caleb C. and Contributors},
  url    = {https://github.com/calebvogt/fnt}
}
```
