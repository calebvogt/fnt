"""Build the FFmpeg commands that turn synchronised camera tracks into a mosaic.

Kept free of Qt so the command construction and the size/time estimates can be
tested directly.

HOW THE MOSAIC IS BUILT
-----------------------
One pass per output chunk. A black canvas of the full output size is created,
then each camera's footage is scaled to its cell and overlaid at that cell's
position, shifted to its true offset within the chunk::

    color=black:s=WxH          -> the canvas
    [n:v] trim, setpts=PTS-STARTPTS+T/TB, scale=cw:ch   -> one run of footage
    [bg][run] overlay=x:y:enable='between(t,T,T+D)'     -> placed in its cell

The consequence worth spelling out: GAPS NEED NO SPECIAL HANDLING. Where a
camera has no footage nothing is overlaid, so the black canvas shows through
and the cell simply goes dark while every other camera keeps its own time. No
black filler files, no codec matching, no concat trickery. Verified on real
footage across a 40-minute dropout: the affected cell reads pixel-mean 0.0
while its neighbours continue undisturbed.

Each contiguous run of footage becomes one input. A camera normally contributes
one or two runs per day, so a five-camera daily chunk is a handful of inputs
rather than the hundred-plus that feeding every segment separately would need.
"""

import datetime
import time
import os
import subprocess

# Measured on a real FULL-DAY chunk -- 1080p 30fps, five cameras of static
# surveillance footage read off the network share: 8.14x realtime at
# 0.183 Mbps. Calibrating on a short window instead would mislead badly; a
# 10-minute window measured 9.2x because it spans few segments, whereas a
# 24h chunk spans the whole day's files.
#
# That input count is why segments are merged (see collect_runs). Feeding each
# segment separately put ~64 inputs in one graph and measured 2.13x -- the same
# work at less than a third the speed.
#
# Held slightly under the measurement as margin. Estimates only: actual size
# tracks how much the scene moves.
_REF = {
    "libx264": {"mbps": 0.20, "speed": 8.0},      # crf 23, veryfast
    "h264_nvenc": {"mbps": 0.21, "speed": 10.5},  # p4, cq 28
}
_REF_PIXELS = 1920 * 1080
_REF_FPS = 30
_REF_CRF = 23


class GridLayout:
    """A rows x cols arrangement of cameras, with empty cells allowed."""

    def __init__(self, rows=3, cols=3, assignments=None):
        self.rows = rows
        self.cols = cols
        # {(row, col): camera_name}; absent cells render black
        self.assignments = dict(assignments or {})

    def place(self, row, col, camera):
        self.clear_camera(camera)
        self.assignments[(row, col)] = camera

    def clear_camera(self, camera):
        for cell, name in list(self.assignments.items()):
            if name == camera:
                del self.assignments[cell]

    def clear_cell(self, row, col):
        self.assignments.pop((row, col), None)

    def cell_of(self, camera):
        for cell, name in self.assignments.items():
            if name == camera:
                return cell
        return None

    @property
    def cameras(self):
        return list(self.assignments.values())

    def resize(self, rows, cols):
        """Change the grid, dropping any assignment that no longer fits."""
        self.rows, self.cols = rows, cols
        self.assignments = {(r, c): n for (r, c), n in self.assignments.items()
                            if r < rows and c < cols}

    def copy(self):
        """An independent copy, for snapshotting a job onto a queue."""
        return GridLayout(self.rows, self.cols, dict(self.assignments))

    @staticmethod
    def filled(cameras, rows, cols):
        """Place `cameras` left to right, top to bottom, on a rows x cols grid.

        Cameras beyond the grid's capacity are left out, which is a legitimate
        way to export only some of them. Rearranging afterwards is a drag away.
        """
        layout = GridLayout(rows, cols)
        cells = [(r, c) for r in range(rows) for c in range(cols)]
        layout.assignments = dict(zip(cells, list(cameras)[:len(cells)]))
        return layout


class EncodeSettings:
    """Output options for a mosaic encode."""

    def __init__(self, width=1920, height=1080, fps=30, codec="libx264",
                 crf=23, preset="veryfast", chunk_mode="daily",
                 show_camera_labels=True, show_no_signal=True,
                 show_clock=False, cell_gap=0,
                 remove_audio=True, audio_source=None, grayscale=False,
                 stage_output_locally=True):
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        self.crf = crf
        self.preset = preset
        # "daily" | "continuous" | "both". "both" writes the daily files and
        # then joins them, rather than encoding the trial a second time.
        self.chunk_mode = chunk_mode
        self.show_camera_labels = show_camera_labels
        self.show_no_signal = show_no_signal
        self.show_clock = show_clock
        self.cell_gap = cell_gap
        self.remove_audio = remove_audio
        # Which camera's audio to keep when audio is retained. A mosaic has
        # several soundtracks and no meaningful way to merge them, so one
        # camera is chosen rather than mixing them into mush.
        self.audio_source = audio_source
        self.grayscale = grayscale
        # Encode to local disk and move the finished file to a network
        # destination, rather than writing across the share for hours. Does
        # not make the encode faster -- the sources are read over the network
        # either way and the output bitrate is tiny -- but it shrinks the
        # window in which a dropped share destroys work from the whole encode
        # down to one file copy.
        self.stage_output_locally = stage_output_locally

    def copy(self):
        """An independent copy, so a queued job keeps the settings it was
        added with even after the controls move on."""
        clone = EncodeSettings()
        clone.__dict__.update(self.__dict__)
        return clone

    def cell_size(self, layout):
        """Pixel size of one grid cell, rounded to even numbers for yuv420p."""
        w = (self.width // layout.cols) & ~1
        h = (self.height // layout.rows) & ~1
        return max(2, w), max(2, h)


#: Grid shapes offered in the UI, smallest first, so the first one that holds
#: every camera is also the tightest fit.
GRID_SHAPES = ((1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 4), (4, 4), (5, 5))


def best_grid(n_cameras, shapes=GRID_SHAPES):
    """Smallest offered grid that holds `n_cameras`.

    Detecting sixteen cameras should land on 4x4 without the user resizing
    first; detecting five should not waste a 5x5.
    """
    for rows, cols in shapes:
        if rows * cols >= n_cameras:
            return rows, cols
    return shapes[-1]


def _esc(text):
    """Escape text for use inside a drawtext filter."""
    return (str(text).replace("\\", "\\\\").replace(":", "\\:")
            .replace("'", "").replace("%", "\\%"))


# Windows ffmpeg builds usually ship without fontconfig, so drawtext cannot
# resolve a font by name -- it aborts with "Cannot load default config file"
# and takes the process down with an access violation. An explicit fontfile is
# required, and its drive-letter colon has to be escaped for the filtergraph
# parser.
_FONT_CANDIDATES = (
    r"C:/Windows/Fonts/arial.ttf",
    r"C:/Windows/Fonts/segoeui.ttf",
    r"C:/Windows/Fonts/verdana.ttf",
    r"C:/Windows/Fonts/consola.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
)


def find_font():
    """Path to a usable TrueType font, or None if drawtext can't be used."""
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


def _font_arg(font_path):
    """`fontfile=...:` fragment for drawtext, or '' when no font was found."""
    if not font_path:
        return ""
    return f"fontfile='{font_path.replace(':', chr(92) + ':')}':"


def _no_window():
    """Keep ffprobe from flashing a console window on Windows."""
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}
    return {}


def has_audio(path):
    """True if `path` carries at least one audio stream."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=60, **_no_window())
        return bool(r.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return False


def collect_runs(track, window_start, window_end, min_seconds=0.4, join_tol=1.5):
    """Contiguous stretches of `track`'s footage inside a window.

    Segments that butt up against each other are MERGED into one run, so a
    camera that split its day into 21 files becomes a single entry -- and, at
    encode time, a single ffmpeg input -- while a genuine dropout ends the run.

    This merging is what makes full-day chunks practical. Feeding every segment
    separately put ~64 inputs into one filtergraph and measured 2.1x realtime;
    merging collapses that to roughly one input per camera.

    Each run carries the ordered file list it spans, where to seek within the
    first file, and where the run belongs in the output window.
    """
    # A manual calibration shifts where this camera's footage sits on the
    # shared clock. Applied here so every downstream placement honours it.
    # A positive offset shows LATER footage at a given moment, which means
    # pulling this camera's segments EARLIER on the shared timeline -- the
    # mirror of what CameraTrack.locate does to the requested time.
    shift = datetime.timedelta(seconds=getattr(track, "clock_offset", 0.0) or 0.0)

    runs = []
    for seg in track.segments:
        seg_start, seg_end = seg.start - shift, seg.end - shift
        if seg_end <= window_start or seg_start >= window_end:
            continue
        a = max(seg_start, window_start)
        b = min(seg_end, window_end)
        if (b - a).total_seconds() < min_seconds:
            continue

        # Continue the previous run when this segment starts where that one
        # ended; anything else means footage is missing in between.
        if runs and abs((a - runs[-1]["_end"]).total_seconds()) <= join_tol:
            runs[-1]["paths"].append(seg.path)
            runs[-1]["duration"] += (b - a).total_seconds()
            runs[-1]["_end"] = b
            continue

        runs.append({
            "paths": [seg.path],
            "seek": (a - seg_start).total_seconds(),
            "offset": (a - window_start).total_seconds(),
            "duration": (b - a).total_seconds(),
            "_end": b,
        })
    for run in runs:
        run.pop("_end", None)
        run["path"] = run["paths"][0]      # convenience for single-file runs
    return runs


def write_concat_list(paths, list_path):
    """Write an ffmpeg concat-demuxer list file and return its path.

    Lets a run spanning many segment files enter the graph as ONE input.
    Single quotes are escaped per the demuxer's rules.
    """
    quote = "'"
    with open(list_path, "w", encoding="utf-8") as fh:
        for p in paths:
            escaped = str(p).replace(quote, quote + chr(92) + quote + quote)
            fh.write("file " + quote + escaped + quote + "\n")
    return list_path


def build_chunk_command(tracks, layout, settings, window_start, window_end,
                        output_path, seek_pad=2.0, work_dir=None):
    """FFmpeg command for one output chunk.

    `tracks` maps camera name -> CameraTrack. `work_dir`, when given, is where
    concat list files are written for runs spanning several segment files;
    without it such runs fall back to their first file only, which is fine for
    short windows (previews, samples) that sit inside one segment.

    Returns (cmd, run_count).
    """
    duration = (window_end - window_start).total_seconds()
    cw, ch = settings.cell_size(layout)

    # Gather every run that has to be drawn, tagged with its destination cell.
    placements = []
    for (row, col), cam in sorted(layout.assignments.items()):
        track = tracks.get(cam)
        if track is None:
            continue
        for run in collect_runs(track, window_start, window_end):
            run = dict(run)
            run["x"] = col * cw
            run["y"] = row * ch
            run["camera"] = cam
            placements.append(run)

    cmd = ["ffmpeg", "-y", "-nostdin", "-hwaccel", "none"]
    for i, run in enumerate(placements):
        # Fast seek to just before the cut, then trim precisely in the graph:
        # a bare -ss lands on the previous keyframe and would desynchronise
        # this cell from the others.
        pre = max(0.0, run["seek"] - seek_pad)
        src, pre_args = run["path"], []
        if len(run["paths"]) > 1 and work_dir:
            src = write_concat_list(
                run["paths"], os.path.join(work_dir, f"run{i:03d}.txt"))
            pre_args = ["-f", "concat", "-safe", "0"]
        cmd += pre_args + ["-ss", f"{pre:.3f}",
                           "-t", f"{run['duration'] + seek_pad * 2:.3f}",
                           "-i", src]

    parts = [f"color=black:s={settings.width}x{settings.height}"
             f":r={settings.fps}:d={duration:.3f}[bg]"]

    for i, run in enumerate(placements):
        fine = min(seek_pad, run["seek"])
        parts.append(
            f"[{i}:v]trim=start={fine:.3f}:duration={run['duration']:.3f},"
            f"setpts=PTS-STARTPTS+{run['offset']:.3f}/TB,"
            f"scale={cw}:{ch},fps={settings.fps}[c{i}]")

    prev = "bg"
    for i, run in enumerate(placements):
        label = f"v{i}"
        parts.append(
            f"[{prev}][c{i}]overlay={run['x']}:{run['y']}"
            f":enable='between(t,{run['offset']:.3f},"
            f"{run['offset'] + run['duration']:.3f})'[{label}]")
        prev = label

    prev = _append_overlays(parts, prev, tracks, layout, settings,
                            window_start, cw, ch, duration, find_font())

    if settings.grayscale:
        # Applied once to the finished mosaic rather than per cell -- one
        # filter instead of one per camera, and identical output.
        parts.append(f"[{prev}]format=gray[gray]")
        prev = "gray"

    parts.append(f"[{prev}]null[vout]")

    audio_label = _append_audio(parts, placements, settings, seek_pad)

    cmd += ["-filter_complex", ";".join(parts), "-map", "[vout]"]
    if audio_label:
        cmd += ["-map", audio_label, "-c:a", "aac", "-b:a", "128k"]
    else:
        cmd += ["-an"]
    cmd += _encoder_args(settings)
    cmd += ["-t", f"{duration:.3f}", "-movflags", "+faststart", output_path]
    return cmd, len(placements)


def _append_audio(parts, placements, settings, seek_pad):
    """Build the audio chain from one camera, or return None for a silent file.

    The chosen camera's runs are delayed to their true offsets and mixed. They
    never overlap, so mixing simply lays them on one timeline with silence
    across that camera's dropouts -- keeping the audio in step with the video
    rather than letting it slide.
    """
    if settings.remove_audio:
        return None
    wanted = settings.audio_source
    runs = [(i, r) for i, r in enumerate(placements)
            if wanted is None or r["camera"] == wanted]
    if not runs:
        return None
    # A source with no audio stream would abort the whole encode.
    if not has_audio(runs[0][1]["path"]):
        return None

    labels = []
    for i, run in runs:
        fine = min(seek_pad, run["seek"])
        delay = int(round(run["offset"] * 1000))
        parts.append(
            f"[{i}:a]atrim=start={fine:.3f}:duration={run['duration']:.3f},"
            f"asetpts=PTS-STARTPTS,adelay=delays={delay}:all=1[a{i}]")
        labels.append(f"[a{i}]")

    if len(labels) == 1:
        parts.append(f"{labels[0]}anull[aout]")
    else:
        parts.append("".join(labels) +
                     f"amix=inputs={len(labels)}:dropout_transition=0:"
                     f"normalize=0[aout]")
    return "[aout]"


def _append_overlays(parts, prev, tracks, layout, settings, window_start,
                     cw, ch, duration, font_path=None):
    """Add the optional burnt-in annotations, each independently toggleable.

    Without a usable font drawtext would crash the whole encode, so every
    annotation is skipped rather than risking the run.
    """
    if font_path is None:
        return prev
    font = _font_arg(font_path)
    font_small = max(12, ch // 16)

    if settings.show_no_signal:
        # Drawn UNDER nothing -- it sits on the canvas, so any overlaid footage
        # covers it. A cell only reads "NO SIGNAL" when nothing was drawn there,
        # which is exactly a dropout. Without this a black cell is ambiguous:
        # a scorer cannot tell a camera outage from an empty arena.
        for (row, col), cam in sorted(layout.assignments.items()):
            track = tracks.get(cam)
            if track is None:
                continue
            for gap in track.gaps:
                g0 = (gap.start - window_start).total_seconds()
                g1 = g0 + gap.duration
                if g1 <= 0 or g0 >= duration:
                    continue
                parts.append(
                    f"[{prev}]drawtext={font}text='NO SIGNAL':"
                    f"fontcolor=red:fontsize={max(16, ch // 9)}:"
                    f"x={col * cw}+({cw}-text_w)/2:y={row * ch}+({ch}-text_h)/2:"
                    f"enable='between(t,{max(0.0, g0):.3f},{min(duration, g1):.3f})'"
                    f"[ns{row}{col}{int(g0)}]")
                prev = f"ns{row}{col}{int(g0)}"

    if settings.show_camera_labels:
        for (row, col), cam in sorted(layout.assignments.items()):
            parts.append(
                f"[{prev}]drawtext={font}text='{_esc(cam)}':"
                f"fontcolor=yellow:fontsize={font_small}:"
                f"box=1:boxcolor=black@0.45:boxborderw=4:"
                f"x={col * cw}+8:y={row * ch}+8[lb{row}{col}]")
            prev = f"lb{row}{col}"

    if settings.show_clock:
        # A single large wall-clock driven by the reconstructed timeline. The
        # per-camera burnt-in clocks are tiny once scaled into a cell, so this
        # gives a scorer one legible time reference.
        #
        # It must TICK, not just stamp the chunk's start time. `pts:localtime`
        # renders each frame's presentation time added to an epoch base, so the
        # clock advances with playback. Colons inside the expression are escaped
        # because the filtergraph parser would otherwise read them as argument
        # separators.
        epoch = int(time.mktime(window_start.timetuple()))
        clock = (r"%{pts\:localtime\:" + str(epoch) + r"\:%Y-%m-%d %H\\\:%M\\\:%S}")
        parts.append(
            f"[{prev}]drawtext={font}text='{clock}':"
            f"fontcolor=white:fontsize={max(20, ch // 7)}:"
            f"box=1:boxcolor=black@0.55:boxborderw=8:"
            f"x=(w-text_w)/2:y=h-text_h-12[clk]")
        prev = "clk"

    return prev


def _encoder_args(settings):
    if settings.codec == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p4",
                "-cq", str(settings.crf), "-pix_fmt", "yuv420p"]
    if settings.codec == "hevc_nvenc":
        return ["-c:v", "hevc_nvenc", "-preset", "p4",
                "-cq", str(settings.crf), "-pix_fmt", "yuv420p"]
    return ["-c:v", settings.codec, "-preset", settings.preset,
            "-crf", str(settings.crf), "-pix_fmt", "yuv420p",
            "-threads", "0"]


def build_concat_command(files, output_path, list_path):
    """Join finished chunks into one file WITHOUT re-encoding.

    The daily files already carry the exact stream the full-trial video needs,
    so the concat demuxer with ``-c copy`` just rewrites the container. That
    turns the second output from another multi-hour encode into a file copy.
    Only valid because every chunk came from one EncodeSettings and therefore
    shares codec, resolution and framerate.
    """
    write_concat_list(files, list_path)
    return ["ffmpeg", "-y", "-nostdin",
            "-f", "concat", "-safe", "0", "-i", list_path,
            "-c", "copy", "-movflags", "+faststart",
            "-fflags", "+genpts", output_path]


def estimate(settings, layout, total_seconds, n_cameras):
    """Rough output size and encode time for the chosen settings.

    Anchored on measurements of real mesocosm footage rather than theory, then
    scaled for resolution, framerate and quality. Static surveillance scenes
    compress far better than typical video, so a generic bitrate table would be
    badly wrong here. Still an estimate: actual size tracks how much the scene
    moves.
    """
    ref = _REF.get(settings.codec, _REF["libx264"])

    pixel_scale = (settings.width * settings.height) / _REF_PIXELS
    # Bitrate grows sub-linearly with framerate: successive frames of a mostly
    # static scene are cheap to predict.
    fps_scale = (settings.fps / _REF_FPS) ** 0.6
    # ~12% per CRF step is the usual rule of thumb for x264.
    crf_scale = 1.12 ** (_REF_CRF - settings.crf)
    # More live cells means more of the canvas actually changes.
    cam_scale = max(0.35, n_cameras / 5.0) ** 0.7

    mbps = ref["mbps"] * pixel_scale * fps_scale * crf_scale * cam_scale
    size_bytes = mbps * 1e6 / 8 * total_seconds
    if settings.chunk_mode == "both":
        # The joined trial video is a second copy of the same footage.
        size_bytes *= 2

    speed = ref["speed"] / max(0.2, pixel_scale) / max(0.2, (settings.fps / _REF_FPS))
    if settings.preset in ("medium", "slow", "slower"):
        speed *= 0.45
    elif settings.preset in ("fast",):
        speed *= 0.8
    encode_seconds = total_seconds / max(0.05, speed)
    if settings.chunk_mode == "both":
        # Joining is a stream copy, not an encode; allow for reading and
        # rewriting the footage once at a rough disk/share rate.
        encode_seconds += (size_bytes / 2) / (80e6)

    return {
        "bitrate_mbps": mbps,
        "size_bytes": size_bytes,
        "encode_seconds": encode_seconds,
        "realtime_factor": speed,
    }


def human_size(n_bytes):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024 or unit == "TB":
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024


def human_time(seconds):
    seconds = int(max(0, seconds))
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, _ = divmod(rem, 60)
    if d:
        return f"{d}d {h}h"
    if h:
        return f"{h}h {m}m"
    return f"{m}m"


def chunk_filename(prefix, window_start, chunk_mode, index, total):
    """Output name for a chunk: dated per day, indexed otherwise.

    "both" names its per-day files exactly as "daily" does -- they are the same
    files, and the joined trial video is written separately under the plain
    continuous name.
    """
    if chunk_mode in ("daily", "both"):
        return f"{prefix}_{window_start:%Y%m%d}_grid.mp4"
    if total > 1:
        return f"{prefix}_part{index + 1:02d}_grid.mp4"
    return f"{prefix}_grid.mp4"
