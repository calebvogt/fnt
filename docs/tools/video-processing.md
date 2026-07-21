# Video Processing

Video preprocessing, editing, and behavioral annotation tools. All video operations use **FFmpeg** under the hood.

Launch any tool from the FNT main GUI **Video** tab.

## Video PreProcessing

Batch video preprocessing combining downsampling, re-encoding, and format conversion:

- **Resolution control** — downsample to a target resolution
- **Re-encoding** — convert between codecs (H.264, H.265, etc.)
- **Format conversion** — change container format (`.mp4`, `.avi`, `.mkv`)
- **Quality presets** — CRF-based quality control
- **Batch processing** — process entire folders with progress tracking

## Video Trim and Crop

Interactive trimming and cropping with visual preview:

- **Scrubber-based navigation** — preview any frame
- **Set start/stop times** or specify a duration
- **Polygon crop regions** — draw arbitrary crop shapes on the preview frame
- **Batch processing** — configure settings for multiple videos, then process all at once

## Video Concatenation

Join multiple video files within directories — automatic ordering by filename, FFmpeg concat demuxer for lossless joining of compatible streams, with progress tracking.

## Behavior Scoring Studio

Manual behavioral annotation tool for ethogram-based video scoring:

- **Define ethograms** — custom behavior categories with color coding
- **Video playback** with frame-accurate scrubbing and adjustable speed
- **Timeline visualization** — view and edit annotations on a temporal timeline
- **Keyboard shortcuts** for rapid scoring
- **Export** annotations to CSV and JSON

## Requirements

- **FFmpeg** must be installed and on PATH — see [Installation](../installation.md).
- Video playback and frame extraction use **OpenCV**, installed automatically with fnt.
