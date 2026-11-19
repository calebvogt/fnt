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
- **Videos that need no re-encoding are copied** — on by default (*Copy videos that need no re-encoding*; `--no-stream-copy` at launch turns it off). See below.

### Copying instead of re-encoding

Many recorders already write what the encode would make. A video that is H.264 or H.265 (8-bit 4:2:0), at the target size and frame rate, gray when *Convert to Grayscale* is on (three frames checked), and without lost frames (mp4/mkv: no jumps in its timestamps; AVI: the copy must keep the recording's length and header) is copied into the mp4 as it is, audio dropped: seconds per file instead of minutes, and no detail lost. Anything else is encoded as usual, and the Status Log says why (`re-encoding <name>: it has colour`, `...: its frame rate is 25/1`, ...). Copies are larger than x265 encodes: about the camera's own video bitrate.

Measured 2026-11-16 on VoleCosm NVR recordings (H.265 1080p, 30 fps, gray, keyframe every 2 s, no B-frames): a 2-h AVI copies in ~2 s (plus staging); the copy is ~75% of the AVI (its 128 kbps PCM audio dropped), against ~40% for x265 ultrafast CRF 20, which takes 16-26 min per file. In OpenCV the copy has the recording's exact frame count and every seek lands on the right frame; in Mask Tracker it tracks at the same speed as the x265 encode, with the same tracks and detections matching within the usual jitter between encodes, and against a lossless reference its detections match better than x265's (F1 0.97 vs 0.91). NVENC (HEVC or AV1) gave no smaller files than x265 at the same quality and was barely faster.

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

## Camera Grid

Puts several cameras of one recorder (NVR) side by side in one video, every cell at the same moment, for scoring. Each camera's files are placed on one wall clock from the day stamp in their names and their durations; where a camera has no footage its cell is black and the others carry on.

### Following the burnt-in clocks

The file timeline puts each camera within a second or two, but that error wanders over a trial (0–2 s between cameras on VoleCosm VC-T005, with the lead changing from day to day), so one fixed offset per camera cannot hold the cameras together. The command-line grid reads every camera's burnt-in clock (`YYYY/MM/DD HH:MM:SS`, bottom right) every minute and moves each camera's frames by its measured error, so all cells show the same clock time:

```
fnt-grid <trial folder> --day 1          # or python -m fnt.videoProcessing.camera_grid_cli
python -m fnt.videoProcessing.camera_grid_cli <trial folder> --day 2026-02-07 --day 2026-02-08
python -m fnt.videoProcessing.camera_grid_cli <trial folder> --all-days
```

- **Input**: a trial folder with one sub-folder per camera (`cam1`…`cam5`, or `RZ_1`…`RZ_4` and `overhead`); cameras are named by the `Camera<N>` in their file names.
- **Layout**: by default the mesocosm 3×3: cam2 top left, cam3 top right, cam5 centre, cam1 bottom left, cam4 bottom right. `--layout "cam2,,cam3;,cam5,;cam1,,cam4"` sets another (rows split by `;`, cells by `,`).
- **Days**: one video per calendar day, midnight to midnight (`<trial>_<date>_day<N>_grid.mp4` in `<trial>/grid/`, day 1 being the first day with footage), black where a camera has no footage. The NVR's "no video" black before the cameras come on (and after they go off) is left out: the video starts a minute before the first readable clock and ends a couple after the last (`--keep-black` keeps the whole day).
- **Large clock errors**: a camera whose clock sits far from its file timeline (two minutes on VoleCosm VC-T010 cam1-2; an hour on the day after a daylight-saving change, whose 25-hour files spill into the next day) is gathered over a correspondingly wider stretch, so its footage still fills the day. Files joined across a small gap in the timeline (each day's first file is stamped 00:00:00) carry that gap in their correction.
- **Clock font**: learned from the footage on the first run (no OCR package): at times the file timeline knows the date, hour and minute, the digits on screen are labelled from it.
- **Reading to the frame**: the frame on which the seconds digit changes marks the true second, so each reading is good to about a frame (33 ms at 30 fps). Each file's error is fitted as a straight line through its readings, misreads left out.
- **Progress window**: run from a console, the command opens a window with a row per day (its stage, a progress bar, the video time encoded, speed and time left; the clock check when done) and a bar for all days. It only reads the logs the run writes, so closing it doesn't stop anything, and `python -m fnt.videoProcessing.camera_grid_progress <trial folder> --all-days` (or `--queue days.csv --root <folder>` for days of several trials) opens it again on runs already going. `--window` / `--no-window` force it on or off (off when the command's output isn't a console, e.g. runs started by a queue that shows one window for all of them).
- **Check**: after encoding, the clocks are read back off the finished grid every minute (`--check-every`), and the log reports the share of times all cells agree within 1 and 3 frames, and the worst.
- **Files** in `<trial>/grid/clock/`: `<trial>_<date>_clock_readings.csv` (every reading), `_clock_model.csv` (each file's fitted error), `_grid_check.csv` (the check, per camera, with the spread in frames), and the ffmpeg log. Readings already taken are reused; a day already encoded is skipped unless `--overwrite`.
- **Speed**: one 1080p day takes about 3–4 h to encode, limited by ffmpeg's single-threaded filter chain rather than by decoding (decoding on the GPU was slower, and 20 fps saved only 13%). `--parallel N` runs N days at once, each reading its clocks and then encoding; one encode uses about 10 cores, so 3 at a time suits a 32-core machine.
- Options: `--workers` (processes reading clocks), `--codec libx264|h264_nvenc`, `--crf`, `--width/--height` (default 1920×1080; 3840×2160 keeps 60% of each corner camera's detail against 30%, encodes about 2.2× slower and takes about 3.5× the space), `--minutes N` (encode only the first N minutes, for a test), `--from HH:MM` (start a clip at that time of day; with `--minutes`, named `..._day<N>_<HHMM>_grid.mp4`).
- `--clips` also writes each camera alone (`..._<camera>.mp4`, `--clip-size`, default 1920×1080), cut by the same clock sync as the grid: frame n of a clip is the same moment as frame n of the grid, and each clip's clock is checked like the grid's. Use them to track each camera and draw the tracks back onto the grid; `--no-grid` writes the clips only.
- **Tracked grid**: `python -m fnt.videoProcessing.tracked_grid <identity.json> [--size 4k|1080] [--workers 8] [--frames A:B] [--hold N]` draws the tracked animals onto the grid, built from those clips: each animal's mask in its colour, its midline keypoints, its name over its head on a black box, readable on any background (from Mask Tracker's `mtt_identity`, which the json was run with; `?` = unsure, grey = unnamed) and a legend of who is in which camera. The clip is split into parts drawn in parallel and joined without re-encoding, with a window showing each part; a minute of 4K takes about 2 minutes with 8 workers. `--hold N` keeps each analysed frame's masks on screen for tracks made on every Nth frame.

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
