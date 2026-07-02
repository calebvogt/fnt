# Video Processing Module

Video preprocessing, editing, and behavioral annotation tools. All video operations use **FFmpeg** under the hood.

## Tools

### Video PreProcessing

Launch from FNT main GUI: **Video** tab → **Video PreProcessing**

Batch video preprocessing combining downsampling, re-encoding, and format conversion:

- **Resolution control** — downsample to a target resolution
- **Re-encoding** — convert between codecs (H.264, H.265, etc.)
- **Format conversion** — change container format (`.mp4`, `.avi`, `.mkv`)
- **Quality presets** — CRF-based quality control
- **Batch processing** — process entire folders of videos with progress tracking

### Video Trim and Crop

Launch from FNT main GUI: **Video** tab → **Video Trim and Crop**

Interactive trimming and cropping with visual preview:

- **Scrubber-based navigation** — preview any frame in the video
- **Set start/stop times** or specify a duration
- **Polygon crop regions** — draw arbitrary crop shapes on the preview frame
- **Batch processing** — configure trim/crop settings for multiple videos, then process all at once
- Outputs trimmed and cropped videos via FFmpeg

### Video Concatenation

Launch from FNT main GUI: **Video** tab → **Video Concatenation**

Join multiple video files within directories:

- Select folders containing video segments to concatenate
- Automatic ordering by filename
- FFmpeg concat demuxer for lossless joining of compatible streams
- Progress tracking with FFmpeg output display

### Behavior Scoring Studio

Launch from FNT main GUI: **Video** tab → **Behavior Scoring Studio**

Manual behavioral annotation tool for ethogram-based video scoring:

- **Define ethograms** — create custom behavior categories with color coding
- **Video playback** with frame-accurate scrubbing and adjustable speed
- **Timeline visualization** — view and edit annotations on a temporal timeline
- **Keyboard shortcuts** for rapid scoring
- **Export** annotations to CSV and JSON for downstream analysis

## Requirements

- **FFmpeg** must be installed and on PATH. On Windows, install system-wide from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/). On macOS/Linux, `conda install -c conda-forge ffmpeg` works.
- Video playback and frame extraction use **OpenCV** (`cv2`), installed automatically with fnt.
