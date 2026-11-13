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
- **Safe to stop** — each video is encoded to `<name>.mp4.partial` and renamed only once it is complete. **Stop**, a closed window or a crash throws away the video in progress (a leftover `.partial` is ignored), and the next run with skip-existing encodes it again; an existing output that can't be read is re-encoded too.
- **Full speed in the background** — on Windows each ffmpeg is opted out of power throttling, which otherwise holds a windowless encode on the efficiency cores once the display turns off (about 40% slower on an i9-14900K).
- **Damaged videos keep their real length, lost frames black** — on by default (*Keep damaged videos at their real length*). See below.

### Lost frames in mp4, mkv and other videos with timestamps

When frames are lost in a video that carries its own timestamps (a recorder that dropped frames, or a damaged stretch), the timestamps jump. The encode keeps the video at its real length across the jump, but on its own FFmpeg fills the jump with copies of the last frame before it: a still picture that a tracker would read as an animal sitting still. Video PreProcessing first reads the video's timestamps (no decoding) and finds every jump of 2.5 frame intervals or more; the encode draws those frames **black**, and `proc/recovered_frames/<name>_frames.csv` lists them (`placement` = `timestamps`), as for AVIs. Time lost in a way the timestamps don't show (a recorder that writes one frame interval after another whatever happened) can't be seen in the file; check such videos against the camera's burned-in clock.

### Damaged AVIs

Security recorders (Viewtron, iDVR) write AVI files that can lose blocks of data, for example when a copy off the recorder goes wrong: stretches of the file are zeroed, or overwritten by another file. FFmpeg then either cannot open the file (its header is gone) or encodes it with the lost frames simply left out. The second case is easy to miss: the video comes out shorter than the recording, and after every break it runs ahead of the camera's clock.

When an AVI fails, comes out more than a few seconds short, or has lost its header, Video PreProcessing rebuilds it from the frames that survive:

- every surviving frame goes back at its real time, using the AVI's own index; frames whose index entries are lost too are placed between the nearest listed ones (within a few seconds, checked against the camera clock);
- lost frames are written **black**, so the video has the recording's full length and frame *n* is at *n / fps* from the start;
- a frame with part of its data lost is left out rather than shown half-decoded, and so are the frames that depend on it, up to the next keyframe;
- if the header is lost, its settings (frame rate, size, codec, number of chunks) are taken from an intact recording of about the same size in the same folder;
- `proc/recovered_frames/<name>_frames.csv` lists each stretch of the output: `first_frame`, `last_frame`, `start_s`, `end_s`, `content` (`recovered` or `black`) and `placement` (`index` for exact, `estimated` when the index entries were lost). Leave the black stretches out of analyses.

A file can be recovered on its own from the command line, with the same default settings:

```bash
python -m fnt.videoProcessing.avi_recovery "path/to/damaged.avi" --map-only
python -m fnt.videoProcessing.avi_recovery "path/to/damaged.avi"
```

`--map-only` reports how much survives without writing anything; `--reference <intact.avi>` names the recording to borrow header settings from. H.264 and HEVC recordings are supported. Burned-in `.smi` timestamps are not added to a rebuilt video, but these cameras burn their own clock into the picture.

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
