#!/usr/bin/env python3
"""
Video Concatenation Tool - PyQt5 Implementation

Concatenate multiple video files within directories using FFmpeg.
Features batch processing with progress tracking and FFmpeg output display.
"""

import os
import sys
import subprocess
import glob
import csv
import random
import re as _re
from datetime import datetime as _datetime, timedelta as _timedelta
from pathlib import Path

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGridLayout, QPushButton, QLabel, QFileDialog, QMessageBox,
        QProgressBar, QTextEdit, QGroupBox, QFrame, QScrollArea, QLineEdit,
        QComboBox, QSpinBox, QCheckBox, QDialog, QSizePolicy, QRadioButton
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QWaitCondition, QRect, QPointF
    from PyQt5.QtGui import (QFont, QImage, QPixmap, QPainter, QPen, QColor,
                              QBrush, QPolygonF)
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    print("PyQt5 not available. Please install with: pip install PyQt5")
    sys.exit(1)


# ---------------------------------------------------------------------------
# LocalModels directory (shared with SAM2 checkpoint manager)
# ---------------------------------------------------------------------------
_FNT_REPO_ROOT = Path(__file__).parent.parent.parent
_LOCAL_MODELS_DIR = _FNT_REPO_ROOT / "LocalModels"
_EASYOCR_MODEL_DIR = _LOCAL_MODELS_DIR / "easyocr"


def _ensure_gitignore_entry(repo_root: Path, entry: str):
    """Add an entry to .gitignore if it doesn't already exist."""
    gitignore = repo_root / ".gitignore"
    if gitignore.exists():
        content = gitignore.read_text()
        if entry in content:
            return
        if not content.endswith("\n"):
            content += "\n"
        content += f"{entry}\n"
        gitignore.write_text(content)
    else:
        gitignore.write_text(f"{entry}\n")


def _preprocess_ocr_crop(crop_img, engine="easyocr",
                          text_color="light_on_dark"):
    """Preprocess a cropped ROI image for OCR.

    Both engines receive: grayscale → 3x upscale → binarise → ensure
    dark-text-on-white-background → pad with white border.

    Parameters
    ----------
    crop_img : numpy array
        The raw ROI crop (BGR or grayscale).
    engine : str
        ``"easyocr"`` uses Otsu threshold; ``"tesseract"`` uses adaptive
        Gaussian threshold.
    text_color : str
        ``"light_on_dark"`` — bright text on a dark background (most DVRs).
        ``"dark_on_light"`` — dark text on a bright background.
        Used to deterministically invert the image so the output is
        always dark-text-on-white for the OCR engine.
    """
    import cv2

    if len(crop_img.shape) == 3:
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop_img.copy()

    h, w = gray.shape
    gray = cv2.resize(gray, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)

    if engine == "easyocr":
        # Light Gaussian blur to reduce compression noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        # Otsu threshold — clean global binarisation
        _, binary = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Adaptive threshold — handles uneven illumination
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 10)

    # Ensure dark text on white background.
    # After thresholding, light-on-dark text produces a mostly-black
    # image with white text → invert.  Dark-on-light is already correct.
    if text_color == "light_on_dark":
        # White text was thresholded to white (255) on black (0).
        # Invert so text becomes dark on white background.
        binary = cv2.bitwise_not(binary)

    # Add white border padding — helps OCR engines detect text edges
    pad = 20
    binary = cv2.copyMakeBorder(
        binary, pad, pad, pad, pad,
        cv2.BORDER_CONSTANT, value=255)

    return binary


# ---------------------------------------------------------------------------
# Timestamp format definitions for format-aware OCR post-processing
# ---------------------------------------------------------------------------
# Each entry maps a human-readable label to the digit-group layout.
# "groups" is a list of (num_digits, separator_after) tuples.
# The parser strips all non-digit chars from the raw OCR text, checks that
# the total digit count matches, then re-inserts the correct separators.
TIMESTAMP_FORMATS = {
    "YYYY/MM/DD HH:MM:SS": {
        "groups": [(4, "/"), (2, "/"), (2, " "), (2, ":"), (2, ":"), (2, "")],
        "total_digits": 14,
    },
    "MM/DD/YYYY HH:MM:SS": {
        "groups": [(2, "/"), (2, "/"), (4, " "), (2, ":"), (2, ":"), (2, "")],
        "total_digits": 14,
    },
    "DD/MM/YYYY HH:MM:SS": {
        "groups": [(2, "/"), (2, "/"), (4, " "), (2, ":"), (2, ":"), (2, "")],
        "total_digits": 14,
    },
    "YYYY/MM/DD HH:MM": {
        "groups": [(4, "/"), (2, "/"), (2, " "), (2, ":"), (2, "")],
        "total_digits": 12,
    },
}


def _parse_timestamp_by_format(raw_text, fmt_key=None):
    """Parse a raw OCR string into a formatted timestamp.

    If *fmt_key* is given and matches a key in TIMESTAMP_FORMATS, the
    parser extracts **only the digits** from the raw OCR output and
    re-formats them according to the expected layout.  This makes the
    result immune to separator confusion (``/`` vs ``.`` vs ``:``).

    Falls back to a flexible regex if no format is specified or if the
    digit count doesn't match.

    Returns the formatted string, or empty string on failure.
    """
    # --- Format-aware path: extract digits, re-insert separators ---
    if fmt_key and fmt_key in TIMESTAMP_FORMATS:
        fmt = TIMESTAMP_FORMATS[fmt_key]
        digits = _re.sub(r'\D', '', raw_text)
        if len(digits) == fmt["total_digits"]:
            parts = []
            pos = 0
            for n_digits, sep in fmt["groups"]:
                parts.append(digits[pos:pos + n_digits])
                parts.append(sep)
                pos += n_digits
            return "".join(parts)
        # If digit count is off (e.g. OCR dropped/added a digit),
        # fall through to the regex fallback.

    # --- Regex fallback (separator-agnostic) ---
    m = _re.search(
        r'(\d{4})[/\-.\s](\d{2})[/\-.\s](\d{2})[/\-.\s]*(\d{2})[:\-.\s](\d{2})[:\-.\s](\d{2})',
        raw_text)
    if m:
        return "{}/{}/{} {}:{}:{}".format(*m.groups())
    return ""


def _temporal_consistency_check(results, sample_interval_sec=1, max_jump_sec=None):
    """Detect and correct temporally inconsistent OCR timestamps.

    Parameters
    ----------
    results : list of (frame_idx, raw_text, parsed_str)
        The OCR results in frame order.
    sample_interval_sec : int
        Expected real-world seconds between consecutive samples.
    max_jump_sec : float or None
        Maximum allowable jump between adjacent timestamps.  Defaults to
        ``5 × sample_interval_sec`` — generous enough to tolerate minor
        jitter while catching gross OCR errors (year/month digit flips).

    Returns
    -------
    corrected : list of (frame_idx, raw_text, parsed_str, corrected_str)
        A copy of *results* with an extra element per row.  *corrected_str*
        equals *parsed_str* when no correction was needed, or the
        interpolated replacement when the original was flagged.
    n_corrected : int
        Number of timestamps that were replaced.
    """
    if max_jump_sec is None:
        max_jump_sec = max(5 * sample_interval_sec, 10)

    # Datetime parse helpers — handle the two human-readable layouts we emit
    _DT_FMTS = [
        "%Y/%m/%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%Y/%m/%d %H:%M",
    ]

    def _to_dt(s):
        """Try to parse a formatted timestamp string into a datetime."""
        if not s:
            return None
        for fmt in _DT_FMTS:
            try:
                return _datetime.strptime(s, fmt)
            except ValueError:
                continue
        return None

    def _from_dt(dt, ref_fmt_str):
        """Format a datetime back to the same layout as *ref_fmt_str*."""
        # Detect which format the reference used by trying all known formats
        for fmt in _DT_FMTS:
            try:
                _datetime.strptime(ref_fmt_str, fmt)
                return dt.strftime(fmt)
            except ValueError:
                continue
        # Fallback — use the first format
        return dt.strftime(_DT_FMTS[0])

    n = len(results)
    if n < 3:
        # Not enough data to do meaningful consistency checks
        return [(r[0], r[1], r[2], r[2]) for r in results], 0

    # Build parallel datetime array
    dts = [_to_dt(r[2]) for r in results]

    # Pass 1: Identify outlier indices.
    # An outlier is a parsed timestamp that jumps by more than max_jump_sec
    # from BOTH its predecessor and successor (if they exist and parsed OK).
    outlier = [False] * n
    for i in range(n):
        if dts[i] is None:
            continue  # unparsed — nothing to correct

        # Find nearest valid predecessor
        prev_dt = None
        prev_gap = 0  # how many samples back
        for j in range(i - 1, -1, -1):
            prev_gap = i - j
            if dts[j] is not None and not outlier[j]:
                prev_dt = dts[j]
                break

        # Find nearest valid successor
        next_dt = None
        next_gap = 0
        for j in range(i + 1, n):
            next_gap = j - i
            if dts[j] is not None:
                next_dt = dts[j]
                break

        # Check jump from predecessor
        bad_prev = False
        if prev_dt is not None:
            expected_delta = _timedelta(seconds=sample_interval_sec * prev_gap)
            actual_delta = abs((dts[i] - prev_dt).total_seconds())
            if actual_delta > max_jump_sec * prev_gap:
                bad_prev = True

        # Check jump to successor
        bad_next = False
        if next_dt is not None:
            actual_delta = abs((next_dt - dts[i]).total_seconds())
            if actual_delta > max_jump_sec * next_gap:
                bad_next = True

        # Only flag if jump is bad relative to BOTH neighbors (or only one
        # neighbor exists and it's bad).  This avoids flagging the valid
        # timestamps around a genuine time gap.
        if prev_dt is None and next_dt is None:
            continue
        if prev_dt is not None and next_dt is not None:
            if bad_prev and bad_next:
                outlier[i] = True
        elif prev_dt is not None and bad_prev:
            # Only predecessor available — be cautious, flag it
            outlier[i] = True
        elif next_dt is not None and bad_next:
            outlier[i] = True

    # Pass 2: Interpolate corrections for outliers using nearest good neighbors
    corrected = []
    n_corrected = 0
    for i in range(n):
        frame_idx, raw_text, parsed = results[i]
        if not outlier[i]:
            corrected.append((frame_idx, raw_text, parsed, parsed))
            continue

        # Find nearest valid predecessor & successor for interpolation
        prev_dt, prev_idx = None, None
        for j in range(i - 1, -1, -1):
            if dts[j] is not None and not outlier[j]:
                prev_dt, prev_idx = dts[j], j
                break
        next_dt, next_idx = None, None
        for j in range(i + 1, n):
            if dts[j] is not None and not outlier[j]:
                next_dt, next_idx = dts[j], j
                break

        interp_dt = None
        ref_fmt = parsed  # for formatting the output

        if prev_dt is not None and next_dt is not None:
            # Linear interpolation between neighbors
            total_span = (next_dt - prev_dt).total_seconds()
            frac = (i - prev_idx) / (next_idx - prev_idx)
            interp_dt = prev_dt + _timedelta(seconds=total_span * frac)
        elif prev_dt is not None:
            gap = i - prev_idx
            interp_dt = prev_dt + _timedelta(seconds=sample_interval_sec * gap)
        elif next_dt is not None:
            gap = next_idx - i
            interp_dt = next_dt - _timedelta(seconds=sample_interval_sec * gap)

        if interp_dt is not None:
            corrected_str = _from_dt(interp_dt, ref_fmt)
            corrected.append((frame_idx, raw_text, parsed, corrected_str))
            n_corrected += 1
        else:
            corrected.append((frame_idx, raw_text, parsed, parsed))

    return corrected, n_corrected


class ConcatenationWorker(QThread):
    """Worker thread for video concatenation to avoid blocking the GUI"""
    progress_update = pyqtSignal(str)  # status message
    folder_progress = pyqtSignal(int, int)  # current folder, total folders
    ffmpeg_output = pyqtSignal(str)  # FFmpeg output lines
    finished = pyqtSignal(bool, str)  # success, final message
    # Signal to ask user whether to skip/move corrupt file(s) or cancel.
    # Emits (message, file_paths) — full paths so GUI can move files if needed.
    # GUI must call worker.respond_to_prompt("skip", "move", or "cancel").
    user_decision_needed = pyqtSignal(str, list)  # message, list of full file paths

    def __init__(self, input_dirs, output_filename, sort_order="default", instance_id=1,
                 enable_preprocessing=False, preprocess_settings=None,
                 enable_chunking=False, chunk_duration_minutes=60,
                 enable_ocr=False, ocr_roi=None, ocr_source_resolution=None,
                 ocr_engine="easyocr", ocr_decoder="greedy",
                 ocr_timestamp_format=None,
                 ocr_sample_interval_sec=60,
                 ocr_text_color="light_on_dark"):
        super().__init__()
        self.input_dirs = input_dirs
        # output_filename can be a str (same name for all folders) or a
        # dict mapping folder_path -> filename (per-folder naming).
        self._output_filename_default = output_filename if isinstance(output_filename, str) else "concatenated_output.mp4"
        self._output_filename_map = output_filename if isinstance(output_filename, dict) else {}
        self.sort_order = sort_order
        self.instance_id = instance_id
        self.should_stop = False
        # Preprocessing settings
        self.enable_preprocessing = enable_preprocessing
        self.preprocess_settings = preprocess_settings or {}
        # Chunking settings
        self.enable_chunking = enable_chunking
        self.chunk_duration_minutes = chunk_duration_minutes
        # OCR settings
        self.enable_ocr = enable_ocr
        self.ocr_roi = ocr_roi                          # (x, y, w, h) in source video coords
        self.ocr_source_resolution = ocr_source_resolution  # (w, h) of the video the ROI was drawn on
        self.ocr_engine = ocr_engine                    # "easyocr" or "tesseract"
        self.ocr_decoder = ocr_decoder                  # "greedy" or "beamsearch" (EasyOCR only)
        self.ocr_timestamp_format = ocr_timestamp_format  # key from TIMESTAMP_FORMATS or None
        self.ocr_sample_interval_sec = ocr_sample_interval_sec  # seconds between OCR samples
        self.ocr_text_color = ocr_text_color              # "light_on_dark" or "dark_on_light"
        # Mutex/condition for blocking the worker until the user responds
        self._decision_mutex = QMutex()
        self._decision_cond = QWaitCondition()
        self._decision_result = None  # "skip", "move", or "cancel"

    def _get_output_filename(self, folder_path):
        """Return the output filename for the given folder."""
        return self._output_filename_map.get(
            folder_path, self._output_filename_default)

    def stop(self):
        """Stop the processing"""
        self.should_stop = True
        # Wake up the worker if it's waiting on a user decision so it can exit
        self._decision_mutex.lock()
        self._decision_result = "cancel"
        self._decision_cond.wakeAll()
        self._decision_mutex.unlock()

    # ------------------------------------------------------------------
    # Cross-thread user decision helpers
    # ------------------------------------------------------------------
    def respond_to_prompt(self, action: str):
        """Called from the GUI thread after the user clicks a dialog button.
        *action* must be one of: "skip", "move", or "cancel"."""
        self._decision_mutex.lock()
        self._decision_result = action
        self._decision_cond.wakeAll()
        self._decision_mutex.unlock()

    def _ask_user_decision(self, message: str, file_paths: list) -> str:
        """Emit a signal so the GUI can show a dialog, then block until the
        user responds.  Returns "skip", "move", or "cancel"."""
        self._decision_mutex.lock()
        self._decision_result = None
        self._decision_mutex.unlock()

        # Tell the GUI to show the dialog (runs on the main thread)
        self.user_decision_needed.emit(message, file_paths)

        # Block until respond_to_prompt() is called
        self._decision_mutex.lock()
        while self._decision_result is None:
            self._decision_cond.wait(self._decision_mutex)
        result = self._decision_result
        self._decision_mutex.unlock()
        return result

    def sort_viewtron_files(self, video_files):
        """
        Sort ViewTron DVR files in chronological order.
        ViewTron naming: Base_YYYYMMDDHHMMSS.ext, then Base_YYYYMMDDHHMMSS(001).ext, etc.
        """
        import re

        def viewtron_sort_key(filepath):
            filename = os.path.basename(filepath)
            match = re.search(r'_(\d{14})(?:\((\d+)\))?', filename)
            if match:
                timestamp = match.group(1)
                sequence = match.group(2)
                if sequence is None:
                    sequence = -1
                else:
                    sequence = int(sequence)
                return (timestamp, sequence)
            else:
                return (filename, 0)

        return sorted(video_files, key=viewtron_sort_key)

    # ------------------------------------------------------------------
    # ffprobe helpers
    # ------------------------------------------------------------------
    def _probe_video(self, filepath):
        """Validate a single video file with ffprobe.
        Returns (ok: bool, info_or_error: str)."""
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_name,width,height,duration,nb_frames",
                "-show_entries", "format=duration,size",
                "-of", "default=noprint_wrappers=1",
                filepath
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return False, result.stderr.strip()
            return True, result.stdout.strip()
        except FileNotFoundError:
            return True, "ffprobe not found — skipping validation"
        except subprocess.TimeoutExpired:
            return False, "ffprobe timed out (file may be corrupt)"
        except Exception as e:
            return False, str(e)

    def _get_video_resolution(self, filepath):
        """Get video resolution (width, height) via ffprobe. Returns (w, h) or None."""
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=s=x:p=0",
                filepath
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split("x")
                if len(parts) == 2:
                    return int(parts[0]), int(parts[1])
        except Exception:
            pass
        return None

    def _try_repair_video(self, filepath, folder_path):
        """Attempt to repair a corrupt video by re-muxing it with ffmpeg.
        Returns the path to the repaired file, or None on failure."""
        base = os.path.basename(filepath)
        name, ext = os.path.splitext(base)
        repaired_path = os.path.join(folder_path, f"{name}_repaired{ext}")

        self.progress_update.emit(f"🔧 Attempting to repair: {base}")
        cmd = [
            "ffmpeg", "-y",
            "-err_detect", "ignore_err",
            "-i", filepath,
            "-c", "copy",
            "-movflags", "+faststart",
            repaired_path
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if proc.returncode == 0 and os.path.exists(repaired_path) and os.path.getsize(repaired_path) > 0:
                self.progress_update.emit(f"✅ Repair succeeded: {os.path.basename(repaired_path)}")
                return repaired_path
            else:
                self.progress_update.emit(f"❌ Repair failed for {base}")
                if proc.stderr:
                    for err_line in proc.stderr.strip().splitlines()[-3:]:
                        self.ffmpeg_output.emit(f"  repair stderr: {err_line}")
                # Clean up failed repair
                try:
                    os.remove(repaired_path)
                except Exception:
                    pass
                return None
        except Exception as e:
            self.progress_update.emit(f"❌ Repair error for {base}: {e}")
            try:
                os.remove(repaired_path)
            except Exception:
                pass
            return None

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------
    def run(self):
        """Main processing function"""
        try:
            total_folders = len(self.input_dirs)
            successful = 0
            failed = 0

            for idx, input_dir in enumerate(self.input_dirs, 1):
                if self.should_stop:
                    self.finished.emit(False, "Processing stopped by user.")
                    return

                self.folder_progress.emit(idx, total_folders)
                self.progress_update.emit(f"Processing folder {idx}/{total_folders}: {input_dir}")

                success = self.concatenate_folder(input_dir)

                if success:
                    successful += 1
                else:
                    failed += 1

            if self.should_stop:
                self.finished.emit(False, "Processing stopped by user.")
            elif failed > 0:
                msg = f"Concatenation finished with errors. {successful} folder(s) succeeded, {failed} failed."
                self.finished.emit(False, msg)
            else:
                msg = f"Concatenation complete! Processed {successful} folder(s) successfully."
                self.finished.emit(True, msg)

        except Exception as e:
            self.finished.emit(False, f"Error during processing: {str(e)}")

    # ------------------------------------------------------------------
    # Core concat logic
    # ------------------------------------------------------------------
    def _run_ffmpeg_concat(self, video_files, folder_path, output_file):
        """Run the ffmpeg concat command on a list of video files.

        When chunking is enabled the segment muxer is used *inline* so that
        chunk files are written directly — no intermediate mega-file is
        created.

        Returns (success: bool, last_stderr_lines: list[str]).
        """
        list_file = os.path.join(folder_path, "concat_list.txt")
        with open(list_file, "w") as fp:
            for video in video_files:
                rel_path = os.path.basename(video)
                fp.write(f"file '{rel_path}'\n")

        chunking = (self.enable_chunking and self.chunk_duration_minutes > 0)
        chunk_sec = self.chunk_duration_minutes * 60 if chunking else 0

        if self.enable_preprocessing:
            ps = self.preprocess_settings
            codec = ps.get("codec", "libx264")
            preset = ps.get("preset", "ultrafast")
            crf = ps.get("crf", 20)
            frame_rate = ps.get("frame_rate", 30)
            grayscale = ps.get("grayscale", False)
            remove_audio = ps.get("remove_audio", True)
            resolution = ps.get("resolution", "1080p")

            if resolution == "1080p":
                width, height = 1920, 1080
            elif resolution == "720p":
                width, height = 1280, 720
            else:  # 480p
                width, height = 854, 480

            video_filters = [f"fps={frame_rate}"]
            video_filters.append(
                f"scale={width}:{height}:force_original_aspect_ratio=decrease:eval=frame")
            video_filters.append(
                f"pad={width}:{height}:-1:-1:color=black")
            if grayscale:
                video_filters.append("format=gray")

            command = [
                "ffmpeg", "-y",
                # ---- Maximum error resilience (match VLC tolerance) ----
                # DVR recordings commonly have timestamp jumps, truncated
                # frames, and minor container issues between segments.
                # VLC plays them fine; these flags make FFmpeg equally
                # tolerant during re-encoding.
                "-err_detect", "ignore_err",
                "-analyzeduration", "200M",
                "-probesize", "200M",
                "-f", "concat", "-safe", "0", "-i", list_file,
                "-c:v", codec,
                "-preset", preset,
                "-crf", str(crf),
                "-pix_fmt", "yuv420p",
                "-vf", ",".join(video_filters),
                "-avoid_negative_ts", "make_zero",
                "-max_muxing_queue_size", "10000000",
                "-fflags", "+genpts+discardcorrupt+igndts",
                "-vsync", "vfr",
            ]
            if chunking:
                command.extend([
                    "-force_key_frames", f"expr:gte(t,n_forced*{chunk_sec})",
                ])
            if remove_audio:
                command.append("-an")
            else:
                command.extend(["-c:a", "aac", "-b:a", "128k"])
        else:
            command = [
                "ffmpeg", "-y",
                "-err_detect", "ignore_err",
                "-analyzeduration", "200M",
                "-probesize", "200M",
                "-f", "concat",
                "-safe", "0",
                "-i", list_file,
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                "-avoid_negative_ts", "make_zero",
                "-fflags", "+genpts+discardcorrupt+igndts",
                "-vsync", "vfr",
                "-an",
            ]
            if chunking:
                command.extend([
                    "-force_key_frames", f"expr:gte(t,n_forced*{chunk_sec})",
                ])

        # --- Output: segment muxer (chunking) or single file ---
        if chunking:
            base_name, ext = os.path.splitext(output_file)
            chunk_pattern = f"{base_name}_part%03d{ext}"
            command.extend([
                "-f", "segment",
                "-segment_time", str(chunk_sec),
                "-segment_start_number", "1",
                "-reset_timestamps", "1",
                chunk_pattern,
            ])
            if self.enable_preprocessing:
                self.progress_update.emit(
                    f"Encoding directly into {self.chunk_duration_minutes}-min chunks...")
            else:
                self.progress_update.emit(
                    f"Concatenating directly into {self.chunk_duration_minutes}-min chunks...")
        else:
            command.extend(["-movflags", "+faststart"])
            command.append(output_file)

        # Track which chunks have already had OCR run on them so that
        # _post_concat can skip them.
        self._ocr_completed_chunks = set()

        # For inline chunking + OCR: monitor the output directory so we
        # can run OCR on each chunk the moment FFmpeg finalises it.
        # We throttle the glob check to avoid hammering the filesystem
        # (especially on network drives) — check every ~30 seconds of
        # wall-clock time rather than on every FFmpeg output line.
        import time as _time
        ocr_inline = (chunking and self.enable_ocr
                      and self.ocr_roi is not None)
        if ocr_inline:
            chunk_base_name, chunk_ext = os.path.splitext(output_file)
            chunk_dir = os.path.dirname(output_file)
            chunk_glob_pattern = os.path.join(
                chunk_dir,
                f"{os.path.basename(chunk_base_name)}_part*{chunk_ext}")
            known_chunks = set()       # chunk paths we've already seen
            _last_glob_time = _time.monotonic()
            _GLOB_INTERVAL = 30        # seconds between filesystem checks

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=folder_path
        )

        last_lines = []       # rolling buffer for failure diagnostics
        error_lines = []      # non-progress lines (actual errors/warnings)
        for line in process.stdout:
            if self.should_stop:
                process.terminate()
                try:
                    os.remove(list_file)
                except Exception:
                    pass
                return False, ["Stopped by user"]
            line = line.strip()
            if line:
                self.ffmpeg_output.emit(line)
                last_lines.append(line)
                if len(last_lines) > 50:
                    last_lines.pop(0)
                # Capture non-progress lines separately — these contain
                # the actual error messages that explain failures.
                if not line.startswith("frame=") and not line.startswith("size="):
                    error_lines.append(line)
                    if len(error_lines) > 30:
                        error_lines.pop(0)

            # --- Inline OCR: periodically check for newly completed chunks ---
            if ocr_inline:
                now = _time.monotonic()
                if now - _last_glob_time >= _GLOB_INTERVAL:
                    _last_glob_time = now
                    current_chunks = set(glob.glob(chunk_glob_pattern))
                    new_chunks = current_chunks - known_chunks
                    if new_chunks and known_chunks:
                        # A new chunk appeared → the *previous* chunks in
                        # known_chunks that haven't been OCR'd are finalised.
                        ready = sorted(
                            known_chunks - self._ocr_completed_chunks)
                        for chunk_path in ready:
                            if self.should_stop:
                                break
                            self.progress_update.emit(
                                f"🔎 Running OCR on completed chunk: "
                                f"{os.path.basename(chunk_path)}")
                            self._run_ocr_pass(chunk_path)
                            self._ocr_completed_chunks.add(chunk_path)
                    known_chunks = current_chunks

        process.wait()
        success = process.returncode == 0

        # --- Inline OCR: process the final chunk (only if FFmpeg succeeded,
        #     otherwise the last chunk file is incomplete/corrupt) ---
        if ocr_inline and success:
            # One final glob to catch any chunks written since the last check
            current_chunks = set(glob.glob(chunk_glob_pattern))
            known_chunks |= current_chunks
            remaining = sorted(known_chunks - self._ocr_completed_chunks)
            for chunk_path in remaining:
                if self.should_stop:
                    break
                self.progress_update.emit(
                    f"🔎 Running OCR on final chunk: "
                    f"{os.path.basename(chunk_path)}")
                self._run_ocr_pass(chunk_path)
                self._ocr_completed_chunks.add(chunk_path)

        try:
            os.remove(list_file)
        except Exception:
            pass

        # Prefer error_lines for diagnostics (actual errors, not progress);
        # fall back to last_lines if nothing non-progress was captured.
        diagnostic_lines = error_lines if error_lines else last_lines
        return success, diagnostic_lines

    # ------------------------------------------------------------------
    # Fast per-file decode test
    # ------------------------------------------------------------------
    def _decode_test_video(self, filepath):
        """Quick-test a single video by decoding it to null output.
        This catches issues that ffprobe misses (corrupt frames, bad
        timestamps, codec errors).
        Returns (ok: bool, error_summary: str)."""
        try:
            cmd = [
                "ffmpeg", "-v", "error",
                "-i", filepath,
                "-f", "null",
                "-"
            ]
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=120  # generous per-file timeout
            )
            stderr = proc.stderr.strip()
            if proc.returncode != 0 or stderr:
                # Grab last few error lines
                err_lines = stderr.splitlines()[-5:] if stderr else ["non-zero exit code"]
                return False, "; ".join(err_lines)
            return True, ""
        except subprocess.TimeoutExpired:
            return False, "decode test timed out"
        except Exception as e:
            return False, str(e)

    def _get_failure_file_index(self, video_files, last_ffmpeg_lines):
        """Parse ffmpeg output to estimate which source file caused failure.

        Parses the last ffmpeg stderr lines for frame= and time= values,
        then maps them to a file index via cumulative frame counts or durations.
        Returns the 0-based index into video_files, or None if indeterminate.
        """
        import re

        total = len(video_files)

        # ------ Parse last ffmpeg output for frame/time ------
        failure_frame = None
        failure_time_sec = None
        for line in reversed(last_ffmpeg_lines):
            if failure_frame is None:
                fm = re.search(r'frame=\s*(\d+)', line)
                if fm:
                    failure_frame = int(fm.group(1))
            if failure_time_sec is None:
                tm = re.search(r'time=(\d+):(\d+):(\d+(?:\.\d+)?)', line)
                if tm:
                    h, mi, s = float(tm.group(1)), float(tm.group(2)), float(tm.group(3))
                    failure_time_sec = h * 3600 + mi * 60 + s
                else:
                    tm2 = re.search(r'time=(\d+(?:\.\d+)?)\s', line)
                    if tm2:
                        failure_time_sec = float(tm2.group(1))
            if failure_frame is not None and failure_time_sec is not None:
                break

        # ------ Map to file index via frame counts (preferred) ------
        estimated_idx = None

        if failure_frame is not None:
            self.progress_update.emit(
                f"📍 Failure occurred at output frame {failure_frame:,}"
                + (f" (~{failure_time_sec:.1f}s)" if failure_time_sec else "")
                + ". Mapping to source file...")

            cumulative_frames = 0
            for i, vf in enumerate(video_files):
                if self.should_stop:
                    return None
                fc = self._get_frame_count(vf)
                if fc is None:
                    fc = 0
                cumulative_frames += fc
                if cumulative_frames >= failure_frame:
                    estimated_idx = i
                    local_frame = failure_frame - (cumulative_frames - fc)
                    self.progress_update.emit(
                        f"   ➜ File #{i+1}/{total}: {os.path.basename(vf)}")
                    self.progress_update.emit(
                        f"     Failure at approximately frame {local_frame:,} "
                        f"within this file (file has {fc:,} frames)")
                    break

            if estimated_idx is None:
                self.progress_update.emit(
                    f"   ⚠️ Frame {failure_frame:,} exceeds cumulative count "
                    f"({cumulative_frames:,}). Failure likely in the last file.")
                estimated_idx = total - 1

        elif failure_time_sec is not None:
            self.progress_update.emit(
                f"📍 Failure occurred at ~{failure_time_sec:.1f}s in the output. "
                "Mapping to source file via duration...")
            cumulative = 0.0
            for i, vf in enumerate(video_files):
                dur = self._get_duration(vf)
                if dur is None:
                    dur = 0.0
                cumulative += dur
                if cumulative >= failure_time_sec:
                    estimated_idx = i
                    self.progress_update.emit(
                        f"   ➜ Estimated failure around file #{i+1}/{total}: "
                        f"{os.path.basename(vf)}")
                    break

        return estimated_idx

    def _find_bad_videos_fast(self, video_files, folder_path, last_ffmpeg_lines):
        """Identify the problematic video after a concat failure.

        Strategy:
        1.  Parse the last ffmpeg output to map failure to a file index.
        2.  Decode-test only the estimated failure file (and its immediate
            neighbor, since concat failures often occur at file boundaries).
        3.  If no file is detectably corrupt, blame the estimated failure
            file anyway — the issue is likely a container/timestamp
            discontinuity that only manifests during concatenation, not
            within a single-file decode test.

        Returns a list of (index, filepath, error) tuples for bad files.
        """
        total = len(video_files)
        bad_files = []

        estimated_idx = self._get_failure_file_index(
            video_files, last_ffmpeg_lines)

        if estimated_idx is None:
            # No estimate at all — test just the first file as a last resort
            self.progress_update.emit(
                "🔍 No frame/time estimate available. "
                "Testing first file only...")
            vf = video_files[0]
            self.progress_update.emit(
                f"  Testing [1/{total}]: {os.path.basename(vf)}...")
            ok, err = self._decode_test_video(vf)
            if not ok:
                bad_files.append((0, vf, err))
            else:
                # Blame it anyway — skip so concat can continue
                bad_files.append((
                    0, vf,
                    "concat failure (file decodes OK individually)"))
            return bad_files

        # Decode-test only the failure file and its immediate neighbor
        test_indices = [estimated_idx]
        if estimated_idx + 1 < total:
            test_indices.append(estimated_idx + 1)

        self.progress_update.emit(
            f"🔍 Decode-testing file #{estimated_idx + 1}"
            + (f" and #{estimated_idx + 2}" if len(test_indices) > 1 else "")
            + f" (around failure point)...")

        for i in test_indices:
            if self.should_stop:
                return bad_files
            vf = video_files[i]
            self.progress_update.emit(
                f"  Testing [{i+1}/{total}]: {os.path.basename(vf)}...")
            ok, err = self._decode_test_video(vf)
            if not ok:
                bad_files.append((i, vf, err))
                self.progress_update.emit(
                    f"  ❌ FAILED: {os.path.basename(vf)}")
                self.progress_update.emit(f"     Error: {err}")

        # If neither file is individually corrupt, the failure is likely
        # a container/timestamp discontinuity at the file boundary.
        # Blame the estimated failure file so recovery can skip past it.
        if not bad_files:
            vf = video_files[estimated_idx]
            self.progress_update.emit(
                f"  ⚠️ No individual file corruption detected. "
                f"Failure is likely at the file boundary — "
                f"flagging #{estimated_idx + 1}: {os.path.basename(vf)}")
            bad_files.append((
                estimated_idx, vf,
                "concat failure at file boundary "
                "(file decodes OK individually)"))

        return bad_files

    def _get_frame_count(self, filepath):
        """Get video frame count via ffprobe. Returns int or None."""
        try:
            # Try nb_frames from stream (fastest — reads header only)
            cmd = [
                "ffprobe", "-v", "quiet",
                "-select_streams", "v:0",
                "-show_entries", "stream=nb_frames",
                "-of", "default=noprint_wrappers=1:nokey=1",
                filepath
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                val = result.stdout.strip()
                if val and val != "N/A":
                    return int(val)
            # Fallback: count packets (slower but reliable)
            cmd2 = [
                "ffprobe", "-v", "quiet",
                "-select_streams", "v:0",
                "-count_packets",
                "-show_entries", "stream=nb_read_packets",
                "-of", "default=noprint_wrappers=1:nokey=1",
                filepath
            ]
            result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=30)
            if result2.returncode == 0:
                val2 = result2.stdout.strip()
                if val2 and val2 != "N/A":
                    return int(val2)
        except Exception:
            pass
        return None

    def _get_duration(self, filepath):
        """Get video duration in seconds via ffprobe. Returns float or None."""
        try:
            cmd = [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                filepath
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except Exception:
            pass
        return None

    def concatenate_folder(self, folder_path):
        """Concatenate all videos in a single folder with validation and auto-repair."""
        try:
            VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".MP4", ".mkv", ".flv", ".wmv", ".m4v")

            video_files_set = set()
            for ext in VIDEO_EXTENSIONS:
                found_files = glob.glob(os.path.join(folder_path, f"*{ext}"))
                video_files_set.update(found_files)

            video_files = list(video_files_set)

            if self.sort_order == "viewtron":
                video_files = self.sort_viewtron_files(video_files)
            else:
                video_files = sorted(video_files)

            if not video_files:
                self.progress_update.emit(f"⚠️ No video files found in {os.path.basename(folder_path)}")
                return False

            self.progress_update.emit(f"Found {len(video_files)} video file(s)")

            # Repair map for incremental recovery (files repaired on-the-fly)
            repaired_map = {}

            # --- Resolution consistency check (skip when preprocessing normalizes resolution) ---
            if self.enable_preprocessing:
                ps = self.preprocess_settings
                res_label = ps.get("resolution", "1080p")
                self.progress_update.emit(
                    f"✅ Preprocessing enabled — all files will be scaled to {res_label}. "
                    f"Skipping resolution consistency check.")
            else:
                self.progress_update.emit("Checking resolution consistency across files...")
                resolution_map = {}  # (width, height) -> [filepath, ...]
                for vf in video_files:
                    if self.should_stop:
                        self._cleanup_repaired(repaired_map)
                        return False
                    res = self._get_video_resolution(vf)
                    if res is not None:
                        resolution_map.setdefault(res, []).append(vf)

                if len(resolution_map) > 1:
                    # Mixed resolutions detected — report clearly
                    self.progress_update.emit("=" * 50)
                    self.progress_update.emit(
                        f"⚠️ MIXED RESOLUTIONS DETECTED — {len(resolution_map)} "
                        f"different resolutions found:")
                    for (w, h), files in sorted(resolution_map.items(),
                                                key=lambda x: len(x[1]), reverse=True):
                        self.progress_update.emit(
                            f"   {w}x{h}: {len(files)} file(s)")
                        # Show up to 5 example filenames for each resolution
                        for f in files[:5]:
                            self.progress_update.emit(
                                f"      • {os.path.basename(f)}")
                        if len(files) > 5:
                            self.progress_update.emit(
                                f"      ... and {len(files) - 5} more")

                    self.progress_update.emit("")
                    self.progress_update.emit(
                        "FFmpeg's concat demuxer requires all files to have the same "
                        "resolution. Please re-run the FNT Video PreProcessing Tool "
                        "on these files with a consistent resolution setting, or "
                        "enable preprocessing in the Advanced Options.")
                    self.progress_update.emit("=" * 50)

                    # Find the minority resolution files (fewer files = likely the outliers)
                    sorted_res = sorted(resolution_map.items(),
                                        key=lambda x: len(x[1]), reverse=True)
                    majority_res = sorted_res[0][0]
                    mismatched_files = []
                    for (w, h), files in sorted_res[1:]:
                        mismatched_files.extend(files)

                    mismatch_msg = (
                        f"{len(mismatched_files)} file(s) have a different resolution "
                        f"than the majority ({majority_res[0]}x{majority_res[1]}). "
                        f"These files need to be re-preprocessed."
                    )
                    action = self._ask_user_decision(mismatch_msg, mismatched_files)
                    if action == "cancel" or self.should_stop:
                        self.progress_update.emit("❌ Concatenation cancelled — resolve resolution mismatch first.")
                        self._cleanup_repaired(repaired_map)
                        return False
                    # User chose skip or move — remove mismatched files
                    for mf in mismatched_files:
                        verb = "Moved" if action == "move" else "Skipping"
                        self.progress_update.emit(
                            f"   {verb}: {os.path.basename(mf)}")
                    video_files = [vf for vf in video_files if vf not in set(mismatched_files)]

                    if not video_files:
                        self.progress_update.emit("❌ No files remain after removing mismatched resolutions.")
                        self._cleanup_repaired(repaired_map)
                        return False

                    self.progress_update.emit(
                        f"Continuing with {len(video_files)} file(s) at {majority_res[0]}x{majority_res[1]}")
                elif len(resolution_map) == 1:
                    res = list(resolution_map.keys())[0]
                    self.progress_update.emit(
                        f"✅ All files have consistent resolution: {res[0]}x{res[1]}")
                # else: couldn't determine resolution — proceed anyway

            # --- Phase 2: Attempt full concatenation ---
            output_dir = os.path.join(folder_path, "concatenated_output")
            os.makedirs(output_dir, exist_ok=True)
            folder_filename = self._get_output_filename(folder_path)
            output_file = os.path.join(output_dir, folder_filename)
            if os.path.exists(output_file):
                counter = 1
                base_name, ext = os.path.splitext(folder_filename)
                while os.path.exists(output_file):
                    output_file = os.path.join(output_dir, f"{base_name}_{counter}{ext}")
                    counter += 1
                self.progress_update.emit(f"Output file exists, using: {os.path.basename(output_file)}")

            self.progress_update.emit(f"Concatenating {len(video_files)} videos...")
            ok, last_lines = self._run_ffmpeg_concat(video_files, folder_path, output_file)

            if ok:
                chunking = (self.enable_chunking
                            and self.chunk_duration_minutes > 0)
                if not chunking:
                    self.progress_update.emit(
                        f"✅ Successfully created: "
                        f"{os.path.basename(output_file)}")
                final_files = self._post_concat(output_file)
                self._cleanup_repaired(repaired_map)
                return True

            # --- Phase 3: Incremental failure recovery ---
            self.progress_update.emit("=" * 50)
            self.progress_update.emit(
                "❌ Full concatenation failed. "
                "Starting incremental recovery...")

            if last_lines:
                self.progress_update.emit(
                    "FFmpeg error/warning output:")
                for ln in last_lines[-15:]:
                    self.progress_update.emit(f"  {ln}")

            success = self._incremental_concat_with_recovery(
                video_files, folder_path, output_file, repaired_map,
                initial_last_lines=last_lines)
            self._cleanup_repaired(repaired_map)
            if success:
                self._post_concat(output_file)
            return success

        except Exception as e:
            self.progress_update.emit(f"❌ Error processing {folder_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def _post_concat(self, output_file):
        """Discover chunk files (if chunking wrote them inline) and run OCR.

        When chunking is enabled, the normal encode path writes chunks
        directly via the segment muxer — no intermediate file exists.
        This method discovers those chunk files.

        For the incremental-recovery fallback (which produces a single
        merged file), we fall back to ``_chunk_output`` to split it.

        Returns the list of final output video files.
        """
        output_files = [output_file]

        if self.enable_chunking and self.chunk_duration_minutes > 0:
            base_name, ext = os.path.splitext(output_file)
            chunk_dir = os.path.dirname(output_file)
            chunk_base = os.path.basename(base_name)
            chunks = sorted(
                glob.glob(os.path.join(chunk_dir, f"{chunk_base}_part*{ext}")))
            if chunks:
                # Normal path: inline chunking already created the files
                self.progress_update.emit(f"✅ Encoded {len(chunks)} chunks:")
                for chunk in chunks:
                    self.progress_update.emit(
                        f"   📁 {os.path.basename(chunk)}")
                output_files = chunks
            elif os.path.exists(output_file):
                # Recovery fallback: a single merged file was produced —
                # split it now with stream copy (no re-encoding).
                fallback_chunks = self._chunk_output(output_file)
                if fallback_chunks:
                    output_files = fallback_chunks

        # OCR if enabled — skip chunks already processed inline during
        # encoding (tracked in self._ocr_completed_chunks).
        if self.enable_ocr and self.ocr_roi is not None:
            already_done = getattr(self, "_ocr_completed_chunks", set())
            for vf in output_files:
                if self.should_stop:
                    break
                if vf in already_done:
                    continue
                self._run_ocr_pass(vf)

        return output_files

    def _cleanup_repaired(self, repaired_map):
        """Remove any temporary repaired video files."""
        for original, repaired in repaired_map.items():
            try:
                if os.path.exists(repaired):
                    os.remove(repaired)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Chunking helper
    # ------------------------------------------------------------------
    def _chunk_output(self, output_file):
        """Split the concatenated output into time-based chunks using stream copy.

        Returns a list of chunk file paths on success, or an empty list if
        chunking was skipped or failed (the original file is preserved).
        """
        duration = self._get_duration(output_file)
        if duration is None:
            self.progress_update.emit(
                "⚠️ Could not determine video duration. Skipping chunking.")
            return []

        chunk_seconds = self.chunk_duration_minutes * 60

        if duration <= chunk_seconds:
            self.progress_update.emit(
                f"Video duration ({duration:.0f}s) is shorter than chunk size "
                f"({self.chunk_duration_minutes} min). No chunking needed.")
            return []

        num_chunks = int(duration // chunk_seconds) + (
            1 if duration % chunk_seconds > 0 else 0)
        self.progress_update.emit(
            f"Splitting into ~{num_chunks} chunks of "
            f"{self.chunk_duration_minutes} minutes...")

        base_name, ext = os.path.splitext(output_file)
        chunk_pattern = f"{base_name}_part%03d{ext}"

        command = [
            "ffmpeg", "-y",
            "-i", output_file,
            "-c", "copy",
            "-f", "segment",
            "-segment_time", str(chunk_seconds),
            "-segment_start_number", "1",
            "-reset_timestamps", "1",
            chunk_pattern
        ]

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        for line in process.stdout:
            if self.should_stop:
                process.terminate()
                return []
            line = line.strip()
            if line:
                self.ffmpeg_output.emit(line)

        process.wait()

        if process.returncode == 0:
            # Count and report chunks
            chunk_dir = os.path.dirname(output_file)
            chunk_base = os.path.basename(base_name)
            chunks = sorted(
                glob.glob(os.path.join(chunk_dir, f"{chunk_base}_part*{ext}")))
            self.progress_update.emit(f"✅ Split into {len(chunks)} chunks:")
            for chunk in chunks:
                self.progress_update.emit(
                    f"   📁 {os.path.basename(chunk)}")
            # Remove the full concatenated file now that chunks exist
            try:
                os.remove(output_file)
                self.progress_update.emit(
                    f"Removed full file: {os.path.basename(output_file)}")
            except Exception:
                pass
            return chunks
        else:
            self.progress_update.emit(
                "⚠️ Chunking failed. Full concatenated file preserved.")
            return []

    # ------------------------------------------------------------------
    # OCR timestamp extraction
    # ------------------------------------------------------------------
    def _map_roi_to_output(self, video_file):
        """Map the ROI from source-video coordinates to output-video coordinates.

        When preprocessing changes the resolution, the ROI drawn on the
        original frame must be transformed to match the output layout
        (scale + letterbox padding).  Returns (x, y, w, h) in output coords.
        """
        import cv2

        if self.ocr_roi is None:
            return None
        rx, ry, rw, rh = self.ocr_roi

        if not self.enable_preprocessing or self.ocr_source_resolution is None:
            return self.ocr_roi  # no resolution change

        src_w, src_h = self.ocr_source_resolution

        # Determine target resolution from preprocessing settings
        ps = self.preprocess_settings
        res = ps.get("resolution", "1080p")
        if res == "1080p":
            tgt_w, tgt_h = 1920, 1080
        elif res == "720p":
            tgt_w, tgt_h = 1280, 720
        else:
            tgt_w, tgt_h = 854, 480

        # Same transform as the ffmpeg scale+pad filter
        scale = min(tgt_w / src_w, tgt_h / src_h)
        scaled_w = int(src_w * scale)
        scaled_h = int(src_h * scale)
        pad_x = (tgt_w - scaled_w) // 2
        pad_y = (tgt_h - scaled_h) // 2

        ox = int(rx * scale) + pad_x
        oy = int(ry * scale) + pad_y
        ow = int(rw * scale)
        oh = int(rh * scale)
        return (ox, oy, ow, oh)

    def _init_ocr_engine(self):
        """Lazily initialise the OCR engine.  Called once per worker run.

        For EasyOCR this creates an ``easyocr.Reader`` stored in
        ``self._easyocr_reader``.  For Tesseract it simply verifies
        that ``pytesseract`` is importable.
        """
        if self.ocr_engine == "easyocr":
            try:
                import easyocr  # noqa: F811
            except ImportError:
                self.progress_update.emit(
                    "⚠️ easyocr not installed — skipping OCR.  "
                    "Install with:  pip install easyocr")
                return False

            # Ensure LocalModels/easyocr folder exists + .gitignore entry
            _EASYOCR_MODEL_DIR.mkdir(parents=True, exist_ok=True)
            _ensure_gitignore_entry(_FNT_REPO_ROOT, "LocalModels/")

            use_gpu = False
            try:
                import torch
                if torch.cuda.is_available():
                    use_gpu = True
            except ImportError:
                pass

            self.progress_update.emit(
                f"🔤 Initialising EasyOCR (GPU={'yes' if use_gpu else 'no'}, "
                f"decoder={self.ocr_decoder})...")
            self._easyocr_reader = easyocr.Reader(
                ["en"],
                gpu=use_gpu,
                model_storage_directory=str(_EASYOCR_MODEL_DIR),
                verbose=False,
            )
            return True

        else:  # tesseract
            try:
                import pytesseract  # noqa: F811
                pytesseract.get_tesseract_version()
            except ImportError:
                self.progress_update.emit(
                    "⚠️ pytesseract not installed — skipping OCR.  "
                    "Install with:  pip install pytesseract")
                return False
            except Exception:
                self.progress_update.emit(
                    "⚠️ Tesseract binary not found on system — skipping OCR.")
                return False
            return True

    def _ocr_read_text(self, processed_img):
        """Run OCR on a preprocessed image and return the raw text string."""
        if self.ocr_engine == "easyocr":
            # The ROI is already tightly cropped to the timestamp, so we
            # tune CRAFT detection to treat the whole image as one text
            # region rather than splitting it into sub-boxes:
            #   - width_ths=2.0: merge boxes within 2× char width
            #   - text_threshold=0.3: lower confidence for text detection
            #   - low_text=0.3: lower threshold for text boundary
            #   - paragraph=True: merge into single output string
            results = self._easyocr_reader.readtext(
                processed_img,
                detail=0,
                decoder=self.ocr_decoder,
                paragraph=True,
                width_ths=2.0,
                text_threshold=0.3,
                low_text=0.3,
            )
            return " ".join(results).strip()
        else:
            import pytesseract
            config = "--psm 7 -c tessedit_char_whitelist=0123456789/:-.  "
            return pytesseract.image_to_string(
                processed_img, config=config).strip()

    def _run_ocr_pass(self, video_file):
        """Run OCR on sampled frames and write a timestamp CSV alongside the video."""
        import cv2

        # Lazy-init the engine on first call
        if not hasattr(self, "_ocr_engine_ready"):
            self._ocr_engine_ready = self._init_ocr_engine()
        if not self._ocr_engine_ready:
            return

        roi = self._map_roi_to_output(video_file)
        if roi is None:
            self.progress_update.emit("⚠️ No OCR ROI set — skipping OCR.")
            return

        engine_label = "EasyOCR" if self.ocr_engine == "easyocr" else "Tesseract"
        self.progress_update.emit(
            f"🔎 OCR pass ({engine_label}): {os.path.basename(video_file)}")

        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            self.progress_update.emit(
                f"⚠️ Could not open video for OCR: {os.path.basename(video_file)}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_interval = max(1, int(fps * self.ocr_sample_interval_sec))

        est_samples = total_frames // sample_interval if sample_interval else 0
        self.progress_update.emit(
            f"   Sampling every {self.ocr_sample_interval_sec}s "
            f"({sample_interval} frames @ {fps:.1f} fps, "
            f"~{est_samples} samples)")

        rx, ry, rw, rh = roi

        csv_path = os.path.splitext(video_file)[0] + "_timestamps.csv"
        results = []
        success_count = 0
        frame_idx = 0

        while frame_idx < total_frames:
            if self.should_stop:
                cap.release()
                return

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                frame_idx += sample_interval
                continue

            # Crop & preprocess
            crop = frame[ry:ry + rh, rx:rx + rw]
            if crop.size == 0:
                frame_idx += sample_interval
                continue

            processed = _preprocess_ocr_crop(
                crop, engine=self.ocr_engine,
                text_color=self.ocr_text_color)

            try:
                raw_text = self._ocr_read_text(processed)
            except Exception:
                raw_text = ""

            parsed = _parse_timestamp_by_format(raw_text, self.ocr_timestamp_format)
            results.append((frame_idx, raw_text, parsed))
            if parsed:
                success_count += 1

            # Progress every 200 samples
            if len(results) % 200 == 0:
                self.progress_update.emit(
                    f"  OCR: {len(results)} samples  "
                    f"(frame {frame_idx}/{total_frames})")

            frame_idx += sample_interval

        cap.release()

        # ----- Temporal consistency check -----
        if len(results) >= 3:
            self.progress_update.emit("  🔍 Running temporal consistency check...")
            corrected_results, n_fixed = _temporal_consistency_check(
                results, sample_interval_sec=self.ocr_sample_interval_sec)
            if n_fixed > 0:
                self.progress_update.emit(
                    f"  ⚠️ Corrected {n_fixed} temporally inconsistent "
                    f"timestamp(s) via interpolation")
        else:
            corrected_results = [
                (r[0], r[1], r[2], r[2]) for r in results]
            n_fixed = 0

        # Write CSV (includes corrected column)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame_number", "timestamp_raw",
                "timestamp_parsed", "timestamp_corrected"])
            writer.writerows(corrected_results)

        total_samples = len(corrected_results)
        rate = (success_count / total_samples * 100) if total_samples > 0 else 0
        correction_note = (
            f", {n_fixed} corrected" if n_fixed > 0 else "")
        self.progress_update.emit(
            f"✅ OCR complete: {os.path.basename(csv_path)}  "
            f"({total_samples} samples, {rate:.0f}% parsed{correction_note})")

    # ------------------------------------------------------------------
    # Incremental failure recovery helpers
    # ------------------------------------------------------------------
    def _salvage_partial_output(self, partial_path, salvaged_path, prefix_files):
        """Salvage a partial/crashed ffmpeg output by re-muxing and trimming.

        When ffmpeg crashes at file N, the partial .mp4 contains files [0..N-1]
        plus some frames from file N. This method re-muxes the partial file to
        fix its container, then trims it to the total duration of prefix_files
        (files [0..N-1]) so that no content from the crash file leaks through.

        Returns True if salvaged_path was created and is valid.
        """
        if not os.path.exists(partial_path) or os.path.getsize(partial_path) < 1024:
            return False

        temp_dir = os.path.dirname(salvaged_path)
        remuxed_path = os.path.join(temp_dir, "_salvage_remux.mp4")

        try:
            # Step A: Re-mux partial file to fix container structure
            size_gb = os.path.getsize(partial_path) / (1024 ** 3)
            # Scale timeout: 120s base + 60s per GB (network drives can be slow)
            salvage_timeout = max(600, int(120 + size_gb * 60))
            self.progress_update.emit(
                f"💾 Salvaging partial output ({size_gb:.2f} GB, timeout {salvage_timeout}s)...")
            cmd = [
                "ffmpeg", "-y",
                "-err_detect", "ignore_err",
                "-i", partial_path,
                "-c", "copy",
                "-movflags", "+faststart",
                remuxed_path
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=salvage_timeout)
            if proc.returncode != 0 or not os.path.exists(remuxed_path) \
                    or os.path.getsize(remuxed_path) == 0:
                self.progress_update.emit("   Salvage re-mux failed.")
                return False

            # Step B: Calculate clean trim boundary
            prefix_duration = 0.0
            for f in prefix_files:
                dur = self._get_duration(f)
                if dur:
                    prefix_duration += dur
            if prefix_duration <= 0:
                self.progress_update.emit(
                    "   Cannot determine prefix duration — skipping trim.")
                # Use remuxed file as-is (may have slight overlap)
                os.rename(remuxed_path, salvaged_path)
                return True

            # Subtract 0.1s safety margin to guarantee no leaked crash-file content
            trim_duration = max(0.1, prefix_duration - 0.1)

            # Step C: Trim to clean boundary
            cmd_trim = [
                "ffmpeg", "-y",
                "-i", remuxed_path,
                "-t", f"{trim_duration:.3f}",
                "-c", "copy",
                "-movflags", "+faststart",
                salvaged_path
            ]
            proc2 = subprocess.run(cmd_trim, capture_output=True, text=True, timeout=salvage_timeout)
            try:
                os.remove(remuxed_path)
            except Exception:
                pass

            if proc2.returncode != 0 or not os.path.exists(salvaged_path) \
                    or os.path.getsize(salvaged_path) == 0:
                self.progress_update.emit("   Salvage trim failed.")
                return False

            # Step D: Validate
            ok, _ = self._probe_video(salvaged_path)
            if not ok:
                self.progress_update.emit("   Salvaged file failed validation.")
                try:
                    os.remove(salvaged_path)
                except Exception:
                    pass
                return False

            salvaged_gb = os.path.getsize(salvaged_path) / (1024 ** 3)
            self.progress_update.emit(
                f"   ✅ Salvaged {salvaged_gb:.2f} GB (trimmed to {trim_duration:.1f}s)")
            return True

        except Exception as e:
            self.progress_update.emit(f"   Salvage error: {e}")
            for tmp in (remuxed_path, salvaged_path):
                try:
                    os.remove(tmp)
                except Exception:
                    pass
            return False

    def _join_segments_streamcopy(self, segment_paths, output_file):
        """Join pre-encoded segments using stream copy (no re-encoding).

        All segments share identical encoding parameters (libx264 crf 18 medium
        yuv420p), so stream copy produces a lossless join near-instantly.
        Returns True on success.
        """
        if len(segment_paths) == 1:
            # Single segment — just move it to the output path
            try:
                if os.path.abspath(segment_paths[0]) != os.path.abspath(output_file):
                    import shutil
                    shutil.move(segment_paths[0], output_file)
                return True
            except Exception as e:
                self.progress_update.emit(f"   Failed to move segment: {e}")
                return False

        # Create concat list with absolute paths
        list_file = output_file + ".segments_list.txt"
        try:
            with open(list_file, "w") as fp:
                for seg in segment_paths:
                    abs_path = os.path.abspath(seg)
                    fp.write(f"file '{abs_path}'\n")

            command = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", list_file,
                "-c", "copy",
                "-movflags", "+faststart",
                output_file
            ]
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            for line in process.stdout:
                if self.should_stop:
                    process.terminate()
                    return False
                line = line.strip()
                if line:
                    self.ffmpeg_output.emit(line)
            process.wait()
            return process.returncode == 0
        except Exception as e:
            self.progress_update.emit(f"   Segment join error: {e}")
            return False
        finally:
            try:
                os.remove(list_file)
            except Exception:
                pass

    def _cleanup_temp_dir(self, temp_dir):
        """Remove the temporary segment directory and all its contents."""
        try:
            if os.path.isdir(temp_dir):
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

    def _incremental_concat_with_recovery(self, video_files, folder_path,
                                          output_file, repaired_map,
                                          initial_last_lines):
        """Incrementally concatenate videos with failure recovery.

        Instead of restarting from scratch on failure, this method:
        1. Salvages the partial output (already-encoded content) as a segment
        2. Identifies and repairs the problematic file(s)
        3. Continues concatenating only the remaining files
        4. Repeats on subsequent failures
        5. Joins all segments with stream copy at the end

        Parameters:
            video_files: Full list of video files to concatenate
            folder_path: Directory containing the videos
            output_file: Final output file path
            repaired_map: Dict tracking original→repaired file mappings
            initial_last_lines: Last ffmpeg output from the Phase 2 failure
        """
        MAX_RETRIES = 10
        temp_dir = os.path.join(folder_path, "_fnt_concat_temp")
        os.makedirs(temp_dir, exist_ok=True)

        segments = []        # Completed segment file paths
        remaining = list(video_files)
        skipped_files = []   # Basenames of files that were skipped
        attempt = 0
        first_iteration = True

        try:
            while remaining and attempt < MAX_RETRIES:
                if self.should_stop:
                    return False

                attempt += 1

                # ------ Run concat or use initial failure data ------
                if first_iteration:
                    # Use the Phase 2 partial output + failure data
                    first_iteration = False
                    ok = False
                    last_lines = initial_last_lines
                    segment_output = output_file  # the Phase 2 partial file
                else:
                    segment_output = os.path.join(
                        temp_dir, f"segment_{len(segments):03d}.mp4")
                    self.progress_update.emit(
                        f"{'=' * 50}\n"
                        f"Recovery attempt #{attempt - 1}: "
                        f"{len(remaining)} files remaining, "
                        f"{len(segments)} segment(s) completed")
                    ok, last_lines = self._run_ffmpeg_concat(
                        remaining, folder_path, segment_output)

                if ok:
                    segments.append(segment_output)
                    remaining = []
                    break

                # ------ Failure: salvage → identify → repair → continue ------
                if last_lines:
                    self.progress_update.emit(
                        "FFmpeg error/warning output:")
                    for ln in last_lines[-15:]:
                        self.progress_update.emit(f"  {ln}")

                # Step A: Determine failure file index within 'remaining'
                failure_idx = self._get_failure_file_index(
                    remaining, last_lines)

                if failure_idx is None:
                    # Can't parse failure point — run full decode scan
                    self.progress_update.emit(
                        "Could not determine failure point from ffmpeg output.")
                    detected_bad = self._find_bad_videos_fast(
                        remaining, folder_path, last_lines)
                    if not detected_bad:
                        self.progress_update.emit(
                            "⚠️ No bad files found. Cannot recover incrementally.")
                        break
                    failure_idx = detected_bad[0][0]
                else:
                    detected_bad = None  # will run decode test below

                # Step B: Salvage partial output (files before failure_idx)
                if failure_idx > 0:
                    partial_exists = (os.path.exists(segment_output)
                                     and os.path.getsize(segment_output) > 1024)
                    if partial_exists:
                        salvaged_path = os.path.join(
                            temp_dir, f"salvaged_{len(segments):03d}.mp4")
                        prefix_files = remaining[:failure_idx]
                        if self._salvage_partial_output(
                                segment_output, salvaged_path, prefix_files):
                            segments.append(salvaged_path)
                            self.progress_update.emit(
                                f"   Saved progress: {failure_idx} file(s) "
                                f"→ segment {len(segments)}")
                        else:
                            # Fallback: re-encode the known-good prefix
                            self.progress_update.emit(
                                "   Salvage failed. Re-encoding prefix...")
                            prefix_out = os.path.join(
                                temp_dir, f"prefix_{len(segments):03d}.mp4")
                            ok_p, _ = self._run_ffmpeg_concat(
                                prefix_files, folder_path, prefix_out)
                            if ok_p:
                                segments.append(prefix_out)
                                self.progress_update.emit(
                                    f"   Re-encoded {failure_idx} file(s) "
                                    f"→ segment {len(segments)}")
                            else:
                                self.progress_update.emit(
                                    "   ⚠️ Could not save progress for "
                                    "files before the failure point.")

                # Clean up the failed partial output
                if segment_output == output_file:
                    # Phase 2 partial — clean up from the main output path
                    try:
                        if os.path.exists(output_file):
                            os.remove(output_file)
                    except Exception:
                        pass
                else:
                    try:
                        if os.path.exists(segment_output):
                            os.remove(segment_output)
                    except Exception:
                        pass

                # Step C: Find and repair bad files around failure point
                if detected_bad is None:
                    detected_bad = self._find_bad_videos_fast(
                        remaining, folder_path, last_lines)

                bad_indices = set()
                if detected_bad:
                    self.progress_update.emit(
                        f"🔴 IDENTIFIED {len(detected_bad)} "
                        f"problematic file(s):")
                    for idx, filepath, err in detected_bad:
                        self.progress_update.emit(
                            f"   #{idx + 1}: {os.path.basename(filepath)}")
                        self.progress_update.emit(
                            f"         Error: {err}")

                    for idx, filepath, err in detected_bad:
                        if self.should_stop:
                            return False
                        # Skip repair for boundary failures — the file
                        # decodes fine individually; re-muxing won't help.
                        if "decodes OK individually" in err:
                            self.progress_update.emit(
                                f"   ⏭️ Skipping repair (boundary issue): "
                                f"{os.path.basename(filepath)}")
                            bad_indices.add(idx)
                            continue
                        repaired = self._try_repair_video(
                            filepath, folder_path)
                        if repaired:
                            repaired_map[filepath] = repaired
                            remaining[idx] = repaired
                        else:
                            bad_indices.add(idx)

                # Handle unrepairable files
                if bad_indices:
                    unrepairable = [remaining[i]
                                    for i in sorted(bad_indices)]
                    action = self._ask_user_decision(
                        f"{len(unrepairable)} file(s) could not be "
                        f"repaired during recovery.",
                        unrepairable)
                    if action == "cancel" or self.should_stop:
                        self.progress_update.emit(
                            "❌ Cancelled by user during recovery.")
                        return False
                    for uf in unrepairable:
                        skipped_files.append(os.path.basename(uf))
                    # Remove unrepairable from remaining
                    remaining = [f for i, f in enumerate(remaining)
                                 if i not in bad_indices]
                    # Adjust failure_idx for removed files before it
                    bad_before = len(
                        [i for i in bad_indices if i < failure_idx])
                    failure_idx -= bad_before

                # Step D: Advance past the salvaged/encoded prefix
                if failure_idx > 0:
                    remaining = remaining[failure_idx:]

            # ------ All iterations done — produce final output ------
            if not segments:
                self.progress_update.emit(
                    "❌ No segments could be produced.")
                return False

            if remaining:
                self.progress_update.emit(
                    f"⚠️ Reached maximum recovery attempts ({MAX_RETRIES}). "
                    f"{len(remaining)} file(s) could not be processed.")

            # Join segments
            if len(segments) == 1 and not remaining:
                # Single segment is the complete output
                import shutil
                if os.path.abspath(segments[0]) != os.path.abspath(output_file):
                    shutil.move(segments[0], output_file)
                self.progress_update.emit(
                    f"✅ Successfully created: "
                    f"{os.path.basename(output_file)}")
            else:
                self.progress_update.emit(
                    f"🔗 Joining {len(segments)} segment(s) with "
                    f"stream copy (no re-encoding)...")
                if not self._join_segments_streamcopy(
                        segments, output_file):
                    self.progress_update.emit(
                        "❌ Failed to join segments.")
                    return False
                self.progress_update.emit(
                    f"✅ Successfully created (after recovery): "
                    f"{os.path.basename(output_file)}")

            if skipped_files:
                self.progress_update.emit(
                    f"⚠️ NOTE: Output is missing footage from: "
                    f"{', '.join(skipped_files)}")

            return True

        finally:
            self._cleanup_temp_dir(temp_dir)


# ======================================================================
# OCR ROI Selection Dialog
# ======================================================================

class FrameLabel(QLabel):
    """QLabel subclass that allows drawing a rectangle via mouse drag."""
    roi_changed = pyqtSignal(tuple)  # (x, y, w, h) in display coordinates

    def __init__(self):
        super().__init__()
        self._start = None
        self._current = None
        self.roi_rect = None  # (x, y, w, h) in display coords
        self.setCursor(Qt.CrossCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._start = event.pos()
            self._current = event.pos()
            self.roi_rect = None
            self.update()

    def mouseMoveEvent(self, event):
        if self._start is not None:
            self._current = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._start is not None:
            end = event.pos()
            x1, y1 = min(self._start.x(), end.x()), min(self._start.y(), end.y())
            x2, y2 = max(self._start.x(), end.x()), max(self._start.y(), end.y())
            w, h = x2 - x1, y2 - y1
            if w > 5 and h > 5:
                self.roi_rect = (x1, y1, w, h)
                self.roi_changed.emit(self.roi_rect)
            self._start = None
            self._current = None
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        pen = QPen(QColor(0, 120, 212), 2, Qt.SolidLine)
        painter.setPen(pen)
        painter.setBrush(QColor(0, 120, 212, 40))
        if self._start is not None and self._current is not None:
            painter.drawRect(QRect(self._start, self._current).normalized())
        elif self.roi_rect is not None:
            x, y, w, h = self.roi_rect
            painter.drawRect(x, y, w, h)
        painter.end()


class OCRRoiSelector(QDialog):
    """Dialog for selecting the OCR ROI region on a video frame."""

    VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".MP4", ".mkv", ".flv", ".wmv", ".m4v")

    def __init__(self, video_files, parent=None, ocr_engine="easyocr",
                 ocr_decoder="greedy"):
        super().__init__(parent)
        self.video_files = video_files
        self.video_size = None       # (width, height) of the actual video
        self.display_scale = 1.0
        self.roi_video_coords = None  # (x, y, w, h) in video pixel coordinates
        self._current_frame = None    # numpy array (BGR)
        self._total_frames = 0
        self._ocr_engine = ocr_engine      # "easyocr" or "tesseract"
        self._ocr_decoder = ocr_decoder    # "greedy" or "beamsearch"
        self.timestamp_format = None       # key from TIMESTAMP_FORMATS or None
        self.text_color = "light_on_dark"  # default; updated from combo on preview/accept

        self.setWindowTitle("Select OCR Timestamp Region")
        self.setMinimumSize(750, 550)
        self.setStyleSheet("""
            QDialog { background-color: #2b2b2b; color: #cccccc; }
            QLabel { color: #cccccc; background-color: transparent; }
            QPushButton {
                background-color: #0078d4; color: white; border: none;
                padding: 8px 16px; border-radius: 4px; font-weight: bold;
            }
            QPushButton:hover { background-color: #106ebe; }
            QPushButton:disabled { background-color: #3f3f3f; color: #888888; }
        """)
        self._build_ui()
        self._load_frame(frame_idx=0)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        instr = QLabel(
            "Draw a rectangle around the timestamp text in the video frame below.\n"
            "The OCR engine will read text from this region for every output video.")
        instr.setWordWrap(True)
        instr.setStyleSheet("color: #999999; margin-bottom: 6px;")
        layout.addWidget(instr)

        # Frame display
        self.frame_label = FrameLabel()
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setMinimumSize(640, 360)
        self.frame_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.frame_label.roi_changed.connect(self._on_roi_changed)
        layout.addWidget(self.frame_label, stretch=1)

        # Info label
        self.roi_info = QLabel("ROI: not set — click and drag on the frame above")
        self.roi_info.setStyleSheet("margin-top: 4px;")
        layout.addWidget(self.roi_info)

        # --- Timestamp format row ---
        fmt_row = QHBoxLayout()

        self.fmt_check = QCheckBox("Expected timestamp format:")
        self.fmt_check.setChecked(True)
        self.fmt_check.setToolTip(
            "When enabled, the OCR post-processor extracts only the\n"
            "digit characters from the raw OCR output and re-inserts\n"
            "the correct separators based on the expected format.\n\n"
            "This eliminates separator confusion (e.g. OCR reading\n"
            "'.' instead of ':') — only the digits matter.\n\n"
            "Uncheck for free-form OCR output without reformatting.")
        self.fmt_check.toggled.connect(self._on_fmt_toggled)
        fmt_row.addWidget(self.fmt_check)

        self.fmt_combo = QComboBox()
        for label in TIMESTAMP_FORMATS:
            self.fmt_combo.addItem(label, label)
        self.fmt_combo.setToolTip(
            "Select the timestamp format burned into your DVR video.\n"
            "The separators (/ : - .) don't matter — the parser only\n"
            "uses the digit positions and group ordering.")
        self.fmt_combo.setFixedWidth(220)
        self.fmt_combo.setStyleSheet(
            "QComboBox { background-color: #3c3c3c; border: 1px solid #555; "
            "color: #cccccc; padding: 3px 6px; }"
            "QComboBox QAbstractItemView { background-color: #3c3c3c; "
            "color: #cccccc; selection-background-color: #0078d4; }")
        fmt_row.addWidget(self.fmt_combo)

        fmt_row.addSpacing(20)

        text_color_label = QLabel("Text color:")
        text_color_label.setStyleSheet("color: #cccccc;")
        fmt_row.addWidget(text_color_label)

        self.text_color_combo = QComboBox()
        self.text_color_combo.addItem("Light on dark", "light_on_dark")
        self.text_color_combo.addItem("Dark on light", "dark_on_light")
        self.text_color_combo.setToolTip(
            "Select the appearance of the timestamp text in the video.\n\n"
            "Light on dark: white/bright text on a dark background\n"
            "(most common for DVR/NVR overlays).\n\n"
            "Dark on light: dark text on a bright background.\n\n"
            "This tells the preprocessor how to produce a clean\n"
            "dark-text-on-white image for the OCR engine.")
        self.text_color_combo.setFixedWidth(140)
        self.text_color_combo.setStyleSheet(
            "QComboBox { background-color: #3c3c3c; border: 1px solid #555; "
            "color: #cccccc; padding: 3px 6px; }"
            "QComboBox QAbstractItemView { background-color: #3c3c3c; "
            "color: #cccccc; selection-background-color: #0078d4; }")
        fmt_row.addWidget(self.text_color_combo)

        fmt_row.addStretch()

        layout.addLayout(fmt_row)

        # --- OCR preview row: processed image + text result ---
        preview_row = QHBoxLayout()

        self.processed_img_label = QLabel()
        self.processed_img_label.setFixedHeight(50)
        self.processed_img_label.setMinimumWidth(150)
        self.processed_img_label.setAlignment(Qt.AlignCenter)
        self.processed_img_label.setStyleSheet(
            "border: 1px solid #555; background-color: #1e1e1e; padding: 2px;")
        self.processed_img_label.setToolTip(
            "This shows the preprocessed ROI image that is fed to the\n"
            "OCR engine. The image is binarised (black & white) with\n"
            "dark text on a white background — this is what the OCR\n"
            "engine actually sees.")
        preview_row.addWidget(self.processed_img_label)

        preview_text_col = QVBoxLayout()
        self.ocr_raw_label = QLabel("Raw OCR: —")
        self.ocr_raw_label.setStyleSheet("color: #999999;")
        preview_text_col.addWidget(self.ocr_raw_label)

        self.ocr_parsed_label = QLabel("Parsed: —")
        preview_text_col.addWidget(self.ocr_parsed_label)

        preview_row.addLayout(preview_text_col, stretch=1)
        layout.addLayout(preview_row)

        # Buttons
        btn_row = QHBoxLayout()

        random_btn = QPushButton("Show Random Frame")
        random_btn.setToolTip(
            "Load a random frame from any video in the folder set.\n"
            "Use this if the timestamp text is not visible on the\n"
            "default first frame (e.g. camera was off at the start).")
        random_btn.clicked.connect(self._show_random_frame)
        btn_row.addWidget(random_btn)

        preview_btn = QPushButton("Preview OCR")
        preview_btn.setToolTip(
            "Run OCR on the current ROI to verify it reads correctly\n"
            "before committing to the full video pass.")
        preview_btn.clicked.connect(self._preview_ocr)
        btn_row.addWidget(preview_btn)

        btn_row.addStretch()

        self.ok_btn = QPushButton("OK")
        self.ok_btn.setEnabled(False)
        self.ok_btn.clicked.connect(self.accept)
        btn_row.addWidget(self.ok_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        layout.addLayout(btn_row)

    def _on_fmt_toggled(self, checked):
        """Enable/disable the format combo when the checkbox is toggled."""
        self.fmt_combo.setEnabled(checked)

    # ------------------------------------------------------------------
    # Frame loading
    # ------------------------------------------------------------------
    def _load_frame(self, frame_idx=0, video_index=0):
        """Load a frame from the specified video file and display it."""
        import cv2

        if not self.video_files:
            return
        video_index = max(0, min(video_index, len(self.video_files) - 1))
        video_path = self.video_files[video_index]
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.roi_info.setText(
                f"Error: could not open {os.path.basename(video_path)}")
            return

        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = max(0, min(frame_idx, self._total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            self.roi_info.setText("Error: could not read frame")
            return

        self._current_frame = frame
        h, w = frame.shape[:2]
        self.video_size = (w, h)

        # Scale to fit dialog
        max_w, max_h = 950, 560
        self.display_scale = min(max_w / w, max_h / h, 1.0)
        disp_w = int(w * self.display_scale)
        disp_h = int(h * self.display_scale)

        display = cv2.resize(frame, (disp_w, disp_h))
        display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        qimg = QImage(display.data, disp_w, disp_h,
                       disp_w * 3, QImage.Format_RGB888)
        self.frame_label.setPixmap(QPixmap.fromImage(qimg))
        self.frame_label.setFixedSize(disp_w, disp_h)

        # Preserve existing ROI across frame changes if possible
        if self.roi_video_coords is not None:
            vx, vy, vw, vh = self.roi_video_coords
            dx = int(vx * self.display_scale)
            dy = int(vy * self.display_scale)
            dw = int(vw * self.display_scale)
            dh = int(vh * self.display_scale)
            self.frame_label.roi_rect = (dx, dy, dw, dh)
        else:
            self.frame_label.roi_rect = None

        self.frame_label.update()
        self.adjustSize()

    def _show_random_frame(self):
        """Load a random frame from a random video in the folder set."""
        if not self.video_files:
            return
        vid_idx = random.randint(0, len(self.video_files) - 1)
        # We need to probe this video's frame count
        import cv2
        cap = cv2.VideoCapture(self.video_files[vid_idx])
        if not cap.isOpened():
            # Fallback to first video frame 0
            self._load_frame(0, video_index=0)
            return
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        frame_idx = random.randint(0, max(0, total - 1)) if total > 0 else 0
        self._load_frame(frame_idx, video_index=vid_idx)

    # ------------------------------------------------------------------
    # ROI handling
    # ------------------------------------------------------------------
    def _on_roi_changed(self, display_roi):
        dx, dy, dw, dh = display_roi
        s = self.display_scale
        vx, vy = int(dx / s), int(dy / s)
        vw, vh = int(dw / s), int(dh / s)
        self.roi_video_coords = (vx, vy, vw, vh)
        self.ok_btn.setEnabled(True)
        vid_w, vid_h = self.video_size
        self.roi_info.setText(
            f"ROI: ({vx}, {vy}) to ({vx + vw}, {vy + vh})  "
            f"[{vw} x {vh} px]   (video: {vid_w} x {vid_h})")

    # ------------------------------------------------------------------
    # OCR preview
    # ------------------------------------------------------------------
    def _preview_ocr(self):
        import cv2

        if self._current_frame is None or self.roi_video_coords is None:
            self.ocr_raw_label.setText("Raw OCR: draw an ROI first")
            self.ocr_parsed_label.setText("Parsed: —")
            return

        vx, vy, vw, vh = self.roi_video_coords
        crop = self._current_frame[vy:vy + vh, vx:vx + vw]
        if crop.size == 0:
            self.ocr_raw_label.setText("Raw OCR: ROI is empty")
            self.ocr_parsed_label.setText("Parsed: —")
            return

        text_color = self.text_color_combo.currentData()
        processed = _preprocess_ocr_crop(
            crop, engine=self._ocr_engine, text_color=text_color)

        # --- Display the processed image ---
        disp = processed.copy()
        if len(disp.shape) == 2:
            # Grayscale → RGB for QImage
            disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2RGB)
        # Scale to fit the preview label (max height 46px, preserving aspect)
        ph, pw = disp.shape[:2]
        target_h = 46
        scale = target_h / ph
        disp = cv2.resize(disp, (int(pw * scale), target_h),
                           interpolation=cv2.INTER_AREA)
        qimg = QImage(disp.data, disp.shape[1], disp.shape[0],
                       disp.shape[1] * 3, QImage.Format_RGB888)
        self.processed_img_label.setPixmap(QPixmap.fromImage(qimg))
        self.processed_img_label.setFixedWidth(disp.shape[1] + 4)

        # --- Run OCR ---
        try:
            if self._ocr_engine == "easyocr":
                import easyocr
                _EASYOCR_MODEL_DIR.mkdir(parents=True, exist_ok=True)
                _ensure_gitignore_entry(_FNT_REPO_ROOT, "LocalModels/")
                if not hasattr(self, "_easyocr_reader"):
                    use_gpu = False
                    try:
                        import torch
                        if torch.cuda.is_available():
                            use_gpu = True
                    except ImportError:
                        pass
                    self._easyocr_reader = easyocr.Reader(
                        ["en"], gpu=use_gpu,
                        model_storage_directory=str(_EASYOCR_MODEL_DIR),
                        verbose=False)
                results = self._easyocr_reader.readtext(
                    processed, detail=0,
                    decoder=self._ocr_decoder,
                    paragraph=True,
                    width_ths=2.0,
                    text_threshold=0.3,
                    low_text=0.3)
                raw_text = " ".join(results).strip()
            else:
                import pytesseract
                config = "--psm 7 -c tessedit_char_whitelist=0123456789/:-.  "
                raw_text = pytesseract.image_to_string(
                    processed, config=config).strip()

            # Show raw OCR text
            self.ocr_raw_label.setText(f'Raw OCR: "{raw_text}"')
            self.ocr_raw_label.setStyleSheet(
                "color: #00cc66;" if raw_text else "color: #ff6666;")

            # Parse with format if enabled
            fmt_key = None
            if self.fmt_check.isChecked():
                fmt_key = self.fmt_combo.currentData()
            parsed = _parse_timestamp_by_format(raw_text, fmt_key)
            self.timestamp_format = fmt_key  # store for caller

            if parsed:
                self.ocr_parsed_label.setText(f'Parsed: "{parsed}"')
                self.ocr_parsed_label.setStyleSheet("color: #00cc66;")
            else:
                digits = _re.sub(r'\D', '', raw_text)
                expected = TIMESTAMP_FORMATS[fmt_key]["total_digits"] if fmt_key else "?"
                self.ocr_parsed_label.setText(
                    f'Parsed: FAILED — got {len(digits)} digits '
                    f'(expected {expected})')
                self.ocr_parsed_label.setStyleSheet("color: #ff6666;")

        except ImportError:
            pkg = "easyocr" if self._ocr_engine == "easyocr" else "pytesseract"
            self.ocr_raw_label.setText(
                f"Raw OCR: {pkg} not installed (pip install {pkg})")
            self.ocr_raw_label.setStyleSheet("color: #ff6666;")
            self.ocr_parsed_label.setText("Parsed: —")
        except Exception as e:
            self.ocr_raw_label.setText(f"Raw OCR: Error — {e}")
            self.ocr_raw_label.setStyleSheet("color: #ff6666;")
            self.ocr_parsed_label.setText("Parsed: —")

    @classmethod
    def find_video_files(cls, directories, sort_order="default"):
        """Find all video files in the given directories."""
        files = set()
        for d in directories:
            for ext in cls.VIDEO_EXTENSIONS:
                files.update(glob.glob(os.path.join(d, f"*{ext}")))
        return sorted(files)


class PerFolderNamingDialog(QDialog):
    """Dialog for assigning custom output filenames to each folder."""

    def __init__(self, folders, default_filename, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Output Filenames")
        self.setMinimumWidth(600)
        self.setStyleSheet("""
            QDialog { background-color: #1e1e2e; color: #e0e0e0; }
            QLabel { color: #e0e0e0; }
            QLineEdit {
                background-color: #2a2a3e; color: #e0e0e0;
                border: 1px solid #555; border-radius: 3px; padding: 4px;
            }
            QPushButton {
                background-color: #3a3a5e; color: #e0e0e0;
                border: 1px solid #555; border-radius: 3px; padding: 6px 16px;
            }
            QPushButton:hover { background-color: #4a4a7e; }
            QRadioButton { color: #e0e0e0; spacing: 6px; }
            QRadioButton::indicator {
                width: 14px; height: 14px;
                border: 2px solid #888; border-radius: 8px;
                background-color: #2a2a3e;
            }
            QRadioButton::indicator:checked {
                background-color: #66b3ff; border-color: #66b3ff;
            }
        """)

        layout = QVBoxLayout(self)

        header = QLabel(
            "You have multiple folders selected with chunking enabled.\n"
            "Choose the same output name for all folders, or enter a\n"
            "custom base name per folder (chunks will be appended as "
            "_partXXX.mp4).")
        header.setWordWrap(True)
        layout.addWidget(header)

        # --- "Use same name" row ---
        same_row = QHBoxLayout()
        self._same_radio = QRadioButton("Use the same name for all folders:")
        self._same_radio.setChecked(True)
        self._same_radio.toggled.connect(self._toggle_mode)
        same_row.addWidget(self._same_radio)

        base, ext = os.path.splitext(default_filename)
        self._same_edit = QLineEdit(base)
        self._same_edit.setToolTip("Base filename (extension added automatically)")
        same_row.addWidget(self._same_edit)

        ext_label = QLabel(ext or ".mp4")
        same_row.addWidget(ext_label)
        layout.addLayout(same_row)

        # --- "Custom per folder" section ---
        custom_row = QHBoxLayout()
        self._custom_radio = QRadioButton("Custom name per folder:")
        custom_row.addWidget(self._custom_radio)
        layout.addLayout(custom_row)

        # Folder → line-edit grid
        self._folder_edits = {}
        grid = QGridLayout()
        for i, folder in enumerate(folders):
            folder_label = QLabel(os.path.basename(folder))
            folder_label.setToolTip(folder)
            grid.addWidget(folder_label, i, 0)

            edit = QLineEdit(base)
            edit.setEnabled(False)  # disabled until custom mode
            grid.addWidget(edit, i, 1)

            ext_lbl = QLabel(ext or ".mp4")
            grid.addWidget(ext_lbl, i, 2)

            self._folder_edits[folder] = edit

        self._folder_grid_widget = QWidget()
        self._folder_grid_widget.setLayout(grid)
        layout.addWidget(self._folder_grid_widget)

        # --- Buttons ---
        btn_row = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        self._ext = ext or ".mp4"
        self._folders = folders

    def _toggle_mode(self, same_checked):
        """Enable/disable the per-folder edits based on radio selection."""
        self._same_edit.setEnabled(same_checked)
        for edit in self._folder_edits.values():
            edit.setEnabled(not same_checked)

    def get_filename_map(self):
        """Return a dict mapping folder_path -> filename (with extension).

        If the user chose "same for all", returns a plain str instead.
        """
        if self._same_radio.isChecked():
            name = self._same_edit.text().strip() or "concatenated_output"
            return name + self._ext
        # Per-folder mode
        result = {}
        for folder in self._folders:
            edit = self._folder_edits[folder]
            name = edit.text().strip() or "concatenated_output"
            result[folder] = name + self._ext
        return result


class VideoConcatenationGUI(QMainWindow):
    """Main GUI window for video concatenation"""

    # Class variable to track instance count
    instance_count = 0
    
    def __init__(self):
        super().__init__()
        
        # Increment instance counter and set unique ID
        VideoConcatenationGUI.instance_count += 1
        self.instance_id = VideoConcatenationGUI.instance_count
        
        self.selected_dirs = []
        self.worker = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle(f"Video Concatenation Tool #{self.instance_id} - FieldNeuroToolbox")
        self.setGeometry(200 + (self.instance_id - 1) * 50, 200 + (self.instance_id - 1) * 50, 900, 700)
        self.setMinimumSize(700, 600)
        
        # Set application style - Dark Mode
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #cccccc;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #cccccc;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                min-height: 20px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #3f3f3f;
                color: #888888;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3f3f3f;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 8px;
                color: #cccccc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QLabel {
                color: #cccccc;
                background-color: transparent;
            }
            QLineEdit {
                padding: 5px;
                border: 1px solid #3f3f3f;
                border-radius: 3px;
                background-color: #1e1e1e;
                color: #cccccc;
            }
            QSpinBox, QComboBox {
                padding: 5px;
                border: 1px solid #3f3f3f;
                border-radius: 3px;
                background-color: #1e1e1e;
                color: #cccccc;
            }
            QComboBox QAbstractItemView {
                background-color: #1e1e1e;
                color: #cccccc;
                selection-background-color: #0078d4;
                border: 1px solid #3f3f3f;
            }
            QCheckBox {
                color: #cccccc;
                spacing: 8px;
            }
            QTextEdit {
                background-color: #1e1e1e;
                border: 1px solid #3f3f3f;
                color: #cccccc;
            }
            QProgressBar {
                border: 1px solid #3f3f3f;
                border-radius: 3px;
                text-align: center;
                background-color: #1e1e1e;
                color: #cccccc;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
            }
            QScrollArea {
                border: none;
                background-color: #2b2b2b;
            }
            QFrame {
                background-color: #2b2b2b;
                border-color: #3f3f3f;
            }
        """)
        
        # Generate arrow PNGs at runtime for cross-platform QSpinBox/QComboBox
        # arrows (CSS border-triangles don't render in Qt on Windows).
        import tempfile
        arrow_dir = tempfile.mkdtemp(prefix="fnt_concat_arrows_")
        for name, points in [("up", [(0, 5), (4, 0), (8, 5)]),
                              ("dn", [(0, 0), (4, 5), (8, 0)])]:
            img = QImage(8, 6, QImage.Format_ARGB32)
            img.fill(QColor(0, 0, 0, 0))
            p = QPainter(img)
            p.setRenderHint(QPainter.Antialiasing)
            p.setBrush(QBrush(QColor(255, 255, 255)))
            p.setPen(Qt.NoPen)
            poly = QPolygonF([QPointF(x, y) for x, y in points])
            p.drawPolygon(poly)
            p.end()
            img.save(os.path.join(arrow_dir, f"{name}.png"))
        _up = os.path.join(arrow_dir, "up.png").replace("\\", "/")
        _dn = os.path.join(arrow_dir, "dn.png").replace("\\", "/")
        self._arrow_style = (
            "QSpinBox::up-button, QSpinBox::down-button {"
            "  background-color: #3f3f3f; border: none; width: 16px;"
            "}"
            "QSpinBox::up-button:hover, QSpinBox::down-button:hover {"
            "  background-color: #0078d4;"
            "}"
            f"QSpinBox::up-arrow {{ image: url({_up}); width: 8px; height: 6px; }}"
            f"QSpinBox::down-arrow {{ image: url({_dn}); width: 8px; height: 6px; }}"
            "QComboBox::drop-down {"
            "  subcontrol-origin: padding; subcontrol-position: center right;"
            "  width: 20px; border: none; background-color: #3f3f3f;"
            "  border-top-right-radius: 3px; border-bottom-right-radius: 3px;"
            "}"
            "QComboBox::drop-down:hover { background-color: #0078d4; }"
            f"QComboBox::down-arrow {{ image: url({_dn}); width: 8px; height: 6px; }}"
        )
        # Apply arrow style on top of the main stylesheet
        self.setStyleSheet(self.styleSheet() + self._arrow_style)

        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Header
        self.create_header(layout)
        
        # Main content in scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        scroll_widget.setLayout(scroll_layout)
        
        # Directory selection
        self.create_directory_selection(scroll_layout)
        
        # Output options
        self.create_output_options(scroll_layout)
        
        # Control buttons
        self.create_control_buttons(scroll_layout)
        
        # Progress section
        self.create_progress_section(scroll_layout)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
    
    def create_header(self, layout):
        """Create header section"""
        header_frame = QFrame()
        header_frame.setFrameStyle(QFrame.Box | QFrame.Raised)
        header_frame.setStyleSheet("background-color: #1e1e1e; padding: 15px; border: 1px solid #3f3f3f;")
        
        header_layout = QVBoxLayout()
        header_frame.setLayout(header_layout)
        
        title = QLabel(f"Video Concatenation Tool #{self.instance_id}")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setStyleSheet("color: #0078d4; background-color: transparent;")
        header_layout.addWidget(title)
        
        subtitle = QLabel("Join multiple video files together using FFmpeg concat")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setFont(QFont("Arial", 10))
        subtitle.setStyleSheet("color: #999999; font-style: italic; background-color: transparent;")
        header_layout.addWidget(subtitle)
        
        layout.addWidget(header_frame)
    
    def create_directory_selection(self, layout):
        """Create directory selection section"""
        group = QGroupBox("Input Directories")
        group_layout = QVBoxLayout()
        
        # Instructions
        instructions = QLabel("Select directories containing video files to concatenate (.mp4, .avi, .mov, .mkv, etc.)\nVideos in each directory will be concatenated into a single output file.")
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #999999; margin-bottom: 10px;")
        group_layout.addWidget(instructions)
        
        # Directory list display
        self.dir_list_label = QLabel("No directories selected")
        self.dir_list_label.setStyleSheet("border: 1px solid #3f3f3f; padding: 10px; background-color: #1e1e1e; min-height: 60px; color: #cccccc;")
        self.dir_list_label.setWordWrap(True)
        group_layout.addWidget(self.dir_list_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.add_dir_btn = QPushButton("Add Directory")
        self.add_dir_btn.clicked.connect(self.add_directory)
        button_layout.addWidget(self.add_dir_btn)
        
        self.clear_dirs_btn = QPushButton("Clear All")
        self.clear_dirs_btn.clicked.connect(self.clear_directories)
        button_layout.addWidget(self.clear_dirs_btn)
        
        button_layout.addStretch()
        group_layout.addLayout(button_layout)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
    
    def create_output_options(self, layout):
        """Create output options section with advanced preprocessing and chunking"""
        group = QGroupBox("Output Options")
        group_layout = QVBoxLayout()

        # --- Basic options in a grid ---
        basic_grid = QGridLayout()

        # Row 0: Output filename
        basic_grid.addWidget(QLabel("Output Filename:"), 0, 0)

        self.output_filename_edit = QLineEdit()
        self.output_filename_edit.setText("concatenated_output.mp4")
        self.output_filename_edit.setPlaceholderText("Enter output filename...")
        self.output_filename_edit.setToolTip(
            "Filename for the concatenated output video.\n\n"
            "The file is saved in a 'concatenated_output/' subfolder within\n"
            "each selected input directory. If a file with this name already\n"
            "exists, a numbered suffix is added automatically.")
        basic_grid.addWidget(self.output_filename_edit, 0, 1)

        info_label = QLabel("💡 Files saved in concatenated_output/ subfolder")
        info_label.setStyleSheet("color: #999999; font-style: italic;")
        basic_grid.addWidget(info_label, 0, 2)

        # Row 1: Sort order
        basic_grid.addWidget(QLabel("Sort Order:"), 1, 0)

        self.sort_order_combo = QComboBox()
        self.sort_order_combo.addItems([
            "Default (Alphabetical)",
            "ViewTron DVR (Chronological)"
        ])
        self.sort_order_combo.setCurrentIndex(0)
        self.sort_order_combo.setToolTip(
            "How to order videos before concatenation.\n\n"
            " - Default (Alphabetical): Standard alphabetical sorting by\n"
            "   filename. Works for most naming conventions.\n\n"
            " - ViewTron DVR (Chronological): Parses ViewTron's timestamp-\n"
            "   based naming convention (Base_YYYYMMDDHHMMSS.ext, then\n"
            "   Base_YYYYMMDDHHMMSS(001).ext for continuation files) and\n"
            "   sorts in true chronological order.\n\n"
            "The sort order is preserved in the final output and in\n"
            "chunk numbering when chunking is enabled.")
        basic_grid.addWidget(self.sort_order_combo, 1, 1)

        sort_info_label = QLabel("ℹ️ Choose ViewTron for DVR recordings")
        sort_info_label.setStyleSheet("color: #999999; font-style: italic;")
        basic_grid.addWidget(sort_info_label, 1, 2)

        basic_grid.setColumnStretch(1, 1)
        group_layout.addLayout(basic_grid)

        # --- Advanced Options toggle button ---
        self.advanced_btn = QPushButton("Show Advanced Options ▼")
        self.advanced_btn.clicked.connect(self.toggle_advanced_options)
        self.advanced_btn.setStyleSheet("""
            QPushButton {
                background-color: #3f3f3f;
                color: #cccccc;
                text-align: left;
                padding-left: 10px;
            }
            QPushButton:hover {
                background-color: #4f4f4f;
            }
        """)
        group_layout.addWidget(self.advanced_btn)

        # --- Advanced options frame (initially hidden) ---
        self.advanced_frame = QFrame()
        self.advanced_frame.setVisible(False)
        self.advanced_frame.setStyleSheet(
            "QFrame { border: 1px solid #3f3f3f; background-color: #1e1e1e; padding: 10px; }")
        advanced_layout = QVBoxLayout()
        self.advanced_frame.setLayout(advanced_layout)

        # ---- Preprocessing section ----
        self.preprocess_check = QCheckBox("Enable Advanced Output Options")
        self.preprocess_check.setChecked(False)
        self.preprocess_check.setToolTip(
            "Configure output video settings (frame rate, resolution, grayscale,\n"
            "codec, speed preset, quality). Use this to go directly from raw DVR\n"
            "footage to a downsampled, processed concatenated output.\n\n"
            "When disabled, videos are concatenated with default settings\n"
            "(libx264, CRF 18, no resolution/FPS change).")
        self.preprocess_check.toggled.connect(self._toggle_preprocessing)
        advanced_layout.addWidget(self.preprocess_check)

        self.preprocess_frame = QFrame()
        self.preprocess_frame.setEnabled(False)
        self.preprocess_frame.setStyleSheet(
            "QFrame { border: none; padding: 0px; margin-left: 20px; }")
        pp_grid = QGridLayout()
        self.preprocess_frame.setLayout(pp_grid)

        # Frame Rate
        pp_grid.addWidget(QLabel("Frame Rate (fps):"), 0, 0)
        self.pp_frame_rate_spin = QSpinBox()
        self.pp_frame_rate_spin.setRange(1, 120)
        self.pp_frame_rate_spin.setValue(20)
        self.pp_frame_rate_spin.setToolTip(
            "Target frame rate for the output video.\n\n"
            "Lowering the frame rate is one of the most effective ways to reduce\n"
            "total file size and speed up downstream processing (e.g. tracking),\n"
            "since fewer frames means less data to process per second of video.\n\n"
            "For behavioral tracking, 15-20 fps is often sufficient. The original\n"
            "DVR frame rate (often 25-30 fps) can be safely reduced without\n"
            "losing meaningful behavioral events.")
        pp_grid.addWidget(self.pp_frame_rate_spin, 0, 1)

        # Resolution
        pp_grid.addWidget(QLabel("Resolution:"), 1, 0)
        self.pp_resolution_combo = QComboBox()
        self.pp_resolution_combo.addItems([
            "1080p (1920x1080)", "720p (1280x720)", "480p (854x480)"
        ])
        self.pp_resolution_combo.setCurrentText("1080p (1920x1080)")
        self.pp_resolution_combo.setToolTip(
            "Target output resolution. Videos are scaled to fit within this\n"
            "frame size while preserving aspect ratio (letterboxed if needed).\n\n"
            "Higher resolution preserves spatial detail that is important for\n"
            "accurate pose estimation and tracking. For most tracking workflows,\n"
            "reducing frame rate has a larger impact on processing speed than\n"
            "reducing resolution, so prefer keeping resolution high.\n\n"
            " - 1080p: Best spatial detail, largest files\n"
            " - 720p: Good balance of detail and file size\n"
            " - 480p: Smallest files, may lose fine detail")
        pp_grid.addWidget(self.pp_resolution_combo, 1, 1)

        # Grayscale
        self.pp_grayscale_check = QCheckBox("Convert to Grayscale")
        self.pp_grayscale_check.setChecked(True)
        self.pp_grayscale_check.setToolTip(
            "Convert videos to single-channel grayscale.\n\n"
            "Reduces file size by ~30-50%% compared to color. Most tracking\n"
            "and pose estimation models work equally well on grayscale video.\n"
            "Recommended for IR/night-vision DVR footage which is already\n"
            "effectively grayscale.")
        pp_grid.addWidget(self.pp_grayscale_check, 2, 0, 1, 2)

        # Remove audio
        self.pp_remove_audio_check = QCheckBox("Remove Audio")
        self.pp_remove_audio_check.setChecked(True)
        self.pp_remove_audio_check.setToolTip(
            "Strip the audio track from the output video.\n\n"
            "Recommended for behavioral analysis workflows where audio is not\n"
            "needed. Reduces file size slightly and avoids audio sync issues\n"
            "that can occur when concatenating clips with different audio\n"
            "formats or sample rates.")
        pp_grid.addWidget(self.pp_remove_audio_check, 3, 0, 1, 2)

        # Video Codec
        pp_grid.addWidget(QLabel("Video Codec:"), 4, 0)
        self.pp_codec_combo = QComboBox()
        self.pp_codec_combo.addItems([
            "libx265 (H.265/HEVC)", "libx264 (H.264/AVC)"
        ])
        self.pp_codec_combo.setCurrentText("libx265 (H.265/HEVC)")
        self.pp_codec_combo.setToolTip(
            "Video compression codec.\n\n"
            " - H.265 (HEVC): ~30-50%% smaller files than H.264 at the same\n"
            "   visual quality. Slower to encode but produces significantly\n"
            "   smaller output. Best when storage is a concern.\n\n"
            " - H.264 (AVC): Faster to encode and more widely compatible\n"
            "   with older software/hardware. Produces larger files.\n\n"
            "Both codecs are lossily compressed — the CRF setting below\n"
            "controls the quality/size tradeoff within the chosen codec.")
        pp_grid.addWidget(self.pp_codec_combo, 4, 1)

        # Speed Preset
        pp_grid.addWidget(QLabel("Speed Preset:"), 5, 0)
        self.pp_preset_combo = QComboBox()
        self.pp_preset_combo.addItems([
            "ultrafast", "superfast", "veryfast", "faster",
            "fast", "medium", "slow", "slower", "veryslow"
        ])
        self.pp_preset_combo.setCurrentText("ultrafast")
        self.pp_preset_combo.setToolTip(
            "Controls the encoding speed vs. compression efficiency tradeoff.\n\n"
            "All presets produce the SAME visual quality at a given CRF value.\n"
            "Slower presets spend more CPU time finding better compression,\n"
            "resulting in smaller files — but the video looks identical.\n\n"
            " - ultrafast: ~5x faster encoding, files ~2x larger\n"
            " - veryfast/fast: Good middle ground\n"
            " - medium: FFmpeg default, balanced\n"
            " - slow/veryslow: Best compression, smallest files,\n"
            "   but encoding takes significantly longer\n\n"
            "Recommendation: Use 'ultrafast' when processing many hours of\n"
            "DVR footage and storage is not a bottleneck. Use 'medium' or\n"
            "'slow' when you need to minimize file sizes for long-term storage.")
        pp_grid.addWidget(self.pp_preset_combo, 5, 1)

        # CRF Quality
        pp_grid.addWidget(QLabel("CRF Quality:"), 6, 0)
        self.pp_crf_combo = QComboBox()
        self.pp_crf_combo.addItems([
            "10 (Best)", "15 (High)", "20 (Good)",
            "25 (Medium)", "30 (Low)"
        ])
        self.pp_crf_combo.setCurrentText("20 (Good)")
        self.pp_crf_combo.setToolTip(
            "Constant Rate Factor — controls visual quality vs. file size.\n\n"
            "CRF is a logarithmic scale where lower = better quality:\n"
            " - 10: Near-lossless. Very large files. Only needed if you plan\n"
            "   to re-encode the output again later.\n"
            " - 15: Visually indistinguishable from the original for most\n"
            "   content. Good for archival.\n"
            " - 20: Slight quality loss on close inspection but excellent\n"
            "   for tracking and analysis. Recommended default.\n"
            " - 25: Noticeable softening. Still usable for tracking but\n"
            "   fine details (whiskers, paw digits) may be lost.\n"
            " - 30: Significant compression artifacts. Only use when\n"
            "   minimizing file size is the top priority.\n\n"
            "NOTE: CRF interacts with the speed preset — a slower preset at\n"
            "the same CRF produces smaller files with identical quality.")
        pp_grid.addWidget(self.pp_crf_combo, 6, 1)

        advanced_layout.addWidget(self.preprocess_frame)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background-color: #3f3f3f; border: none; max-height: 1px;")
        advanced_layout.addWidget(sep)

        # ---- Chunking section ----
        self.chunk_check = QCheckBox("Enable Chunking")
        self.chunk_check.setChecked(False)
        self.chunk_check.setToolTip(
            "Split the concatenated output into smaller time-based chunks.\n\n"
            "Useful when the full concatenation would produce a very large\n"
            "single file (e.g. 24 hours of DVR footage). Chunks are split\n"
            "using stream copy (no re-encoding) so this step is nearly\n"
            "instant. Chunk files are numbered sequentially to preserve\n"
            "the original chronological sort order.")
        self.chunk_check.toggled.connect(self._toggle_chunking)
        advanced_layout.addWidget(self.chunk_check)

        self.chunk_frame = QFrame()
        self.chunk_frame.setEnabled(False)
        self.chunk_frame.setStyleSheet(
            "QFrame { border: none; padding: 0px; margin-left: 20px; }")
        chunk_grid = QGridLayout()
        self.chunk_frame.setLayout(chunk_grid)

        chunk_grid.addWidget(QLabel("Chunk Duration (min):"), 0, 0)
        self.chunk_duration_spin = QSpinBox()
        self.chunk_duration_spin.setRange(1, 1440)
        self.chunk_duration_spin.setValue(60)
        self.chunk_duration_spin.setToolTip(
            "Duration of each chunk in minutes.\n\n"
            "Common values:\n"
            " - 30 min: Manageable file sizes, easy to review\n"
            " - 60 min: Good balance for long recordings\n"
            " - 120+ min: Fewer files, larger per chunk\n\n"
            "The last chunk may be shorter than the specified duration\n"
            "if the total video length is not evenly divisible.")
        chunk_grid.addWidget(self.chunk_duration_spin, 0, 1)

        chunk_info = QLabel(
            "ℹ️ Chunks are numbered sequentially "
            "(e.g. _part001, _part002, ...)")
        chunk_info.setStyleSheet("color: #999999; font-style: italic;")
        chunk_info.setWordWrap(True)
        chunk_grid.addWidget(chunk_info, 1, 0, 1, 2)

        advanced_layout.addWidget(self.chunk_frame)

        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet(
            "background-color: #3f3f3f; border: none; max-height: 1px;")
        advanced_layout.addWidget(sep2)

        # ---- OCR section ----
        self.ocr_check = QCheckBox("Enable OCR Timestamp Export")
        self.ocr_check.setChecked(False)
        self.ocr_check.setToolTip(
            "Extract burned-in timestamps from the output video using OCR.\n\n"
            "Many DVR/NVR systems overlay a date-time stamp directly on the\n"
            "video (e.g. '2026/04/29 13:36:16'). This option reads that text\n"
            "from a user-defined region and exports a CSV file alongside each\n"
            "output video mapping frame numbers to timestamps.\n\n"
            "EasyOCR (default) is a pure-Python neural-network OCR engine\n"
            "that requires no external system binaries. Models (~30 MB) are\n"
            "auto-downloaded to the LocalModels/ folder on first use.\n\n"
            "Tesseract is a classic OCR engine that requires both the\n"
            "pytesseract Python package and the Tesseract binary.\n\n"
            "When enabled you will be asked to draw a rectangle around the\n"
            "timestamp region on a sample video frame.")
        self.ocr_check.toggled.connect(self._toggle_ocr)
        advanced_layout.addWidget(self.ocr_check)

        self.ocr_frame = QFrame()
        self.ocr_frame.setVisible(False)
        self.ocr_frame.setStyleSheet(
            "QFrame { border: none; padding: 0px; margin-left: 20px; }")
        ocr_inner_layout = QVBoxLayout()
        self.ocr_frame.setLayout(ocr_inner_layout)

        # OCR engine + decoder row
        ocr_settings_row = QHBoxLayout()

        ocr_engine_label = QLabel("OCR Engine:")
        ocr_engine_label.setStyleSheet("color: #cccccc;")
        ocr_settings_row.addWidget(ocr_engine_label)

        self.ocr_engine_combo = QComboBox()
        self.ocr_engine_combo.addItem("EasyOCR (Recommended)", "easyocr")
        self.ocr_engine_combo.addItem("Tesseract", "tesseract")
        self.ocr_engine_combo.setToolTip(
            "EasyOCR: Pure-Python neural-network OCR. No system binaries\n"
            "needed — just pip install easyocr. Models are downloaded\n"
            "automatically to LocalModels/easyocr/ on first use (~30 MB).\n"
            "Uses GPU automatically if available (PyTorch + CUDA).\n\n"
            "Tesseract: Classic rule-based OCR engine. Requires both the\n"
            "pytesseract Python package AND the Tesseract-OCR system binary\n"
            "installed separately (brew/apt/Windows installer).")
        self.ocr_engine_combo.setFixedWidth(200)
        self.ocr_engine_combo.currentIndexChanged.connect(
            self._on_ocr_engine_changed)
        ocr_settings_row.addWidget(self.ocr_engine_combo)

        ocr_settings_row.addSpacing(12)

        self.ocr_decoder_label = QLabel("Speed:")
        self.ocr_decoder_label.setStyleSheet("color: #cccccc;")
        ocr_settings_row.addWidget(self.ocr_decoder_label)

        self.ocr_decoder_combo = QComboBox()
        self.ocr_decoder_combo.addItem("Fast (Greedy)", "greedy")
        self.ocr_decoder_combo.addItem("Accurate (Beam Search)", "beamsearch")
        self.ocr_decoder_combo.setToolTip(
            "Controls the EasyOCR text decoder strategy.\n\n"
            "Fast (Greedy): Picks the best character at each step.\n"
            "Very fast, usually sufficient for clean DVR timestamps.\n\n"
            "Accurate (Beam Search): Considers multiple candidate\n"
            "sequences and picks the overall best. Slower but may\n"
            "improve accuracy on noisy or low-resolution text.")
        self.ocr_decoder_combo.setFixedWidth(180)
        ocr_settings_row.addWidget(self.ocr_decoder_combo)

        ocr_settings_row.addStretch()
        ocr_inner_layout.addLayout(ocr_settings_row)

        # OCR sample rate row
        ocr_rate_row = QHBoxLayout()

        ocr_rate_label = QLabel("Sample interval (sec):")
        ocr_rate_label.setStyleSheet("color: #cccccc;")
        ocr_rate_row.addWidget(ocr_rate_label)

        self.ocr_sample_interval_spin = QSpinBox()
        self.ocr_sample_interval_spin.setRange(1, 300)
        self.ocr_sample_interval_spin.setValue(60)
        self.ocr_sample_interval_spin.setSuffix("s")
        self.ocr_sample_interval_spin.setFixedWidth(80)
        self.ocr_sample_interval_spin.setToolTip(
            "How often to sample a frame for OCR (in seconds).\n\n"
            "60s (default): one sample per minute — fast and\n"
            "sufficient for time-aligning multi-hour recordings.\n\n"
            "5–10s: good when you need finer timestamp references\n"
            "without full per-second density.\n\n"
            "1s: reads every unique timestamp — DVR clocks update\n"
            "once per second, so this captures every change. Best\n"
            "for precise frame-to-timestamp mapping but slow.\n\n"
            "The actual frame skip is calculated from the video's\n"
            "real FPS, so this works regardless of preprocessing\n"
            "frame rate settings.")
        ocr_rate_row.addWidget(self.ocr_sample_interval_spin)

        ocr_rate_info = QLabel("")
        ocr_rate_info.setStyleSheet("color: #999999; font-style: italic;")
        self._ocr_rate_info_label = ocr_rate_info

        def _update_rate_info(val):
            if val == 1:
                ocr_rate_info.setText("every timestamp change")
            else:
                samples_per_hour = 3600 // val
                ocr_rate_info.setText(f"~{samples_per_hour} samples/hour")

        self.ocr_sample_interval_spin.valueChanged.connect(_update_rate_info)
        _update_rate_info(1)
        ocr_rate_row.addWidget(ocr_rate_info)

        ocr_rate_row.addStretch()
        ocr_inner_layout.addLayout(ocr_rate_row)

        # ROI info + redraw row
        ocr_roi_row = QHBoxLayout()

        self.ocr_roi_label = QLabel("ROI: not set")
        self.ocr_roi_label.setStyleSheet("color: #999999;")
        ocr_roi_row.addWidget(self.ocr_roi_label, stretch=1)

        self.ocr_redraw_btn = QPushButton("Select ROI")
        self.ocr_redraw_btn.setToolTip(
            "Open the ROI selector to draw/redraw the OCR region.\n"
            "Configure OCR engine and speed settings above first.")
        self.ocr_redraw_btn.clicked.connect(self._open_roi_selector)
        self.ocr_redraw_btn.setStyleSheet(
            "background-color: #3f3f3f; color: #cccccc; padding: 4px 12px;")
        ocr_roi_row.addWidget(self.ocr_redraw_btn)

        ocr_inner_layout.addLayout(ocr_roi_row)

        advanced_layout.addWidget(self.ocr_frame)

        group_layout.addWidget(self.advanced_frame)

        group.setLayout(group_layout)
        layout.addWidget(group)

        # Instance state for OCR
        self._ocr_roi = None               # (x, y, w, h) in video coords
        self._ocr_source_resolution = None  # (w, h) of the video the ROI was drawn on
        self._ocr_timestamp_format = None  # key from TIMESTAMP_FORMATS or None
        self._ocr_text_color = "light_on_dark"  # "light_on_dark" or "dark_on_light"

    def toggle_advanced_options(self):
        """Toggle visibility of advanced options"""
        is_visible = self.advanced_frame.isVisible()
        self.advanced_frame.setVisible(not is_visible)
        if is_visible:
            self.advanced_btn.setText("Show Advanced Options ▼")
        else:
            self.advanced_btn.setText("Hide Advanced Options ▲")

    def _toggle_preprocessing(self, checked):
        """Enable/disable preprocessing controls based on checkbox state"""
        self.preprocess_frame.setEnabled(checked)

    def _toggle_chunking(self, checked):
        """Enable/disable chunking controls based on checkbox state"""
        self.chunk_frame.setEnabled(checked)

    def _on_ocr_engine_changed(self, index):
        """Show/hide the decoder dropdown based on OCR engine selection."""
        engine = self.ocr_engine_combo.currentData()
        is_easyocr = (engine == "easyocr")
        self.ocr_decoder_label.setVisible(is_easyocr)
        self.ocr_decoder_combo.setVisible(is_easyocr)

    def _toggle_ocr(self, checked):
        """Handle OCR checkbox toggle — check deps and open ROI selector."""
        if not checked:
            self.ocr_frame.setVisible(False)
            return

        # Must have directories selected to load a sample frame
        if not self.selected_dirs:
            QMessageBox.warning(
                self, "No Directories Selected",
                "Please add at least one input directory first so a video\n"
                "frame can be loaded for OCR region selection.")
            self.ocr_check.setChecked(False)
            return

        # Find video files
        video_files = OCRRoiSelector.find_video_files(self.selected_dirs)
        if not video_files:
            QMessageBox.warning(
                self, "No Videos Found",
                "No video files were found in the selected directories.")
            self.ocr_check.setChecked(False)
            return

        # Check OCR engine availability
        engine = self.ocr_engine_combo.currentData()
        if engine == "easyocr":
            if not self._check_easyocr_available():
                self.ocr_check.setChecked(False)
                return
        else:
            if not self._check_tesseract_available():
                self.ocr_check.setChecked(False)
                return

        # Show the OCR settings panel — user can configure engine, decoder,
        # sample rate, then click "Select ROI" when ready.
        self.ocr_frame.setVisible(True)

    def _check_easyocr_available(self):
        """Check if easyocr is importable; offer to install if missing.
        Returns True if available (or just installed)."""
        try:
            import easyocr  # noqa: F401
            return True
        except ImportError:
            reply = QMessageBox.question(
                self, "EasyOCR Not Installed",
                "The easyocr package is required for OCR.\n\n"
                "Would you like to install it now?\n"
                "(pip install easyocr — this may take a minute)\n\n"
                "Note: EasyOCR models (~30 MB) will be downloaded\n"
                "to LocalModels/easyocr/ on first use.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                return self._pip_install_package("easyocr")
            return False

    def _check_tesseract_available(self):
        """Check pytesseract package + Tesseract binary.
        Returns True if both are available."""
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return True
        except ImportError:
            QMessageBox.warning(
                self, "pytesseract Not Installed",
                "The pytesseract Python package is required for OCR.\n\n"
                "Install with:  pip install pytesseract\n\n"
                "You also need the Tesseract OCR binary installed on\n"
                "your system (tesseract-ocr).")
            return False
        except Exception:
            QMessageBox.warning(
                self, "Tesseract Not Found",
                "The Tesseract OCR binary was not found on your system.\n\n"
                "Install Tesseract:\n"
                "  Windows: download from github.com/UB-Mannheim/tesseract\n"
                "  macOS:   brew install tesseract\n"
                "  Linux:   sudo apt install tesseract-ocr")
            return False

    def _pip_install_package(self, package_name):
        """Attempt to pip-install a package. Returns True on success."""
        import subprocess as _sp
        self.log_message(f"Installing {package_name}...")
        try:
            proc = _sp.run(
                [sys.executable, "-m", "pip", "install", package_name],
                capture_output=True, text=True, timeout=300)
            if proc.returncode == 0:
                self.log_message(f"✅ {package_name} installed successfully.")
                QMessageBox.information(
                    self, "Installation Complete",
                    f"{package_name} was installed successfully.")
                return True
            else:
                self.log_message(f"❌ Failed to install {package_name}:")
                self.log_message(proc.stderr[-500:] if proc.stderr else "(no output)")
                QMessageBox.warning(
                    self, "Installation Failed",
                    f"Could not install {package_name}.\n\n"
                    f"Try manually:  pip install {package_name}\n\n"
                    f"Error: {proc.stderr[-200:] if proc.stderr else 'unknown'}")
                return False
        except Exception as e:
            QMessageBox.warning(
                self, "Installation Error",
                f"Error running pip: {e}")
            return False

    def _open_roi_selector(self):
        """Open the OCR ROI selection dialog."""
        video_files = OCRRoiSelector.find_video_files(self.selected_dirs)
        if not video_files:
            QMessageBox.warning(self, "No Videos",
                                "No video files found in selected directories.")
            return

        try:
            import cv2
        except ImportError:
            QMessageBox.warning(
                self, "OpenCV Not Installed",
                "OpenCV (cv2) is required for OCR frame preview.\n"
                "Install with:  pip install opencv-python")
            self.ocr_check.setChecked(False)
            return

        engine = self.ocr_engine_combo.currentData()
        decoder = self.ocr_decoder_combo.currentData()
        dialog = OCRRoiSelector(video_files, parent=self,
                                ocr_engine=engine, ocr_decoder=decoder)
        # If we already have an ROI, pre-populate it
        if self._ocr_roi is not None:
            dialog.roi_video_coords = self._ocr_roi
            dialog.ok_btn.setEnabled(True)

        if dialog.exec_() == QDialog.Accepted and dialog.roi_video_coords is not None:
            self._ocr_roi = dialog.roi_video_coords
            self._ocr_source_resolution = dialog.video_size
            self._ocr_timestamp_format = dialog.timestamp_format
            self._ocr_text_color = dialog.text_color_combo.currentData()
            vx, vy, vw, vh = self._ocr_roi
            self.ocr_roi_label.setText(
                f"ROI: ({vx}, {vy}) to ({vx + vw}, {vy + vh})  [{vw} x {vh} px]")
            self.ocr_roi_label.setStyleSheet("color: #00cc66;")
            self.ocr_redraw_btn.setText("Redraw ROI")
            self.ocr_frame.setVisible(True)
        else:
            # User cancelled — keep settings visible if OCR is checked
            if self._ocr_roi is None and not self.ocr_check.isChecked():
                self.ocr_frame.setVisible(False)
    
    def create_control_buttons(self, layout):
        """Create control buttons section"""
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Concatenation")
        self.start_btn.clicked.connect(self.start_concatenation)
        self.start_btn.setEnabled(False)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_concatenation)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        button_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def create_progress_section(self, layout):
        """Create progress section"""
        group = QGroupBox("Progress")
        group_layout = QVBoxLayout()
        
        # Folder progress
        self.folder_progress_label = QLabel("Ready to start...")
        group_layout.addWidget(self.folder_progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        group_layout.addWidget(self.progress_bar)
        
        # Status log
        status_label = QLabel("Status Log:")
        status_label.setStyleSheet("font-weight: bold; margin-top: 10px; color: #cccccc;")
        group_layout.addWidget(status_label)
        
        self.status_log = QTextEdit()
        self.status_log.setMaximumHeight(100)
        self.status_log.setReadOnly(True)
        self.status_log.setStyleSheet("background-color: #1e1e1e; border: 1px solid #3f3f3f; color: #cccccc;")
        group_layout.addWidget(self.status_log)

        self.copy_status_btn = QPushButton("Copy Status Log Output")
        self.copy_status_btn.setStyleSheet(
            "background-color: #3f3f3f; color: #cccccc; padding: 4px 12px; border: 1px solid #555555;")
        self.copy_status_btn.clicked.connect(self.copy_status_log)
        group_layout.addWidget(self.copy_status_btn)

        # FFmpeg output log
        ffmpeg_label = QLabel("FFmpeg Output:")
        ffmpeg_label.setStyleSheet("font-weight: bold; margin-top: 10px; color: #cccccc;")
        group_layout.addWidget(ffmpeg_label)
        
        self.ffmpeg_log = QTextEdit()
        self.ffmpeg_log.setMaximumHeight(150)
        self.ffmpeg_log.setReadOnly(True)
        self.ffmpeg_log.setStyleSheet("background-color: #1e1e1e; border: 1px solid #3f3f3f; color: #cccccc; font-family: 'Courier New', monospace; font-size: 9px;")
        group_layout.addWidget(self.ffmpeg_log)
        
        group.setLayout(group_layout)
        layout.addWidget(group)
    
    def add_directory(self):
        """Add a directory to the concatenation list"""
        directory = QFileDialog.getExistingDirectory(
            self, 
            "Select Directory with Video Files",
            "", 
            QFileDialog.ShowDirsOnly
        )
        
        if directory and directory not in self.selected_dirs:
            self.selected_dirs.append(directory)
            self.update_directory_display()
            self.start_btn.setEnabled(len(self.selected_dirs) > 0)
    
    def clear_directories(self):
        """Clear all selected directories"""
        self.selected_dirs.clear()
        self.update_directory_display()
        self.start_btn.setEnabled(False)
    
    def update_directory_display(self):
        """Update the directory list display"""
        if not self.selected_dirs:
            self.dir_list_label.setText("No directories selected")
        else:
            dir_list = "\n".join([f"• {d}" for d in self.selected_dirs])
            self.dir_list_label.setText(f"Selected directories ({len(self.selected_dirs)}):\n{dir_list}")
    
    def start_concatenation(self):
        """Start video concatenation"""
        if not self.selected_dirs:
            QMessageBox.warning(self, "Warning", "Please select at least one directory.")
            return

        # Get output filename
        output_filename = self.output_filename_edit.text().strip()
        if not output_filename:
            QMessageBox.warning(self, "Warning", "Please enter an output filename.")
            return

        # Ensure .mp4 extension
        if not output_filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            output_filename += '.mp4'

        # Multi-folder + chunking: offer per-folder naming
        enable_chunking = self.chunk_check.isChecked()
        if len(self.selected_dirs) > 1 and enable_chunking:
            dlg = PerFolderNamingDialog(
                self.selected_dirs, output_filename, parent=self)
            if dlg.exec_() != QDialog.Accepted:
                return
            naming_result = dlg.get_filename_map()
            if isinstance(naming_result, dict):
                output_filename = naming_result  # per-folder dict
            else:
                output_filename = naming_result  # str — same for all

        # Disable controls
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.add_dir_btn.setEnabled(False)
        self.clear_dirs_btn.setEnabled(False)

        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Get sort order
        sort_order = "viewtron" if self.sort_order_combo.currentIndex() == 1 else "default"

        # Get preprocessing settings
        enable_preprocessing = self.preprocess_check.isChecked()
        preprocess_settings = None
        if enable_preprocessing:
            codec_text = self.pp_codec_combo.currentText()
            codec = codec_text.split()[0]  # "libx265" or "libx264"
            crf_text = self.pp_crf_combo.currentText()
            crf = int(crf_text.split()[0])  # Extract number
            resolution_text = self.pp_resolution_combo.currentText()
            resolution = resolution_text.split()[0]  # "1080p" or "720p"
            preprocess_settings = {
                "frame_rate": self.pp_frame_rate_spin.value(),
                "grayscale": self.pp_grayscale_check.isChecked(),
                "remove_audio": self.pp_remove_audio_check.isChecked(),
                "codec": codec,
                "preset": self.pp_preset_combo.currentText(),
                "crf": crf,
                "resolution": resolution,
            }

        # Get chunking settings (enable_chunking already set above for the naming dialog)
        chunk_duration_minutes = self.chunk_duration_spin.value()

        # Clear logs
        self.status_log.clear()
        self.ffmpeg_log.clear()
        self.log_message("Starting video concatenation...")
        if isinstance(output_filename, dict):
            self.log_message("Output filenames (per folder):")
            for folder, fname in output_filename.items():
                self.log_message(
                    f"  {os.path.basename(folder)} → {fname}")
        else:
            self.log_message(f"Output filename: {output_filename}")
        self.log_message(f"Sort order: {self.sort_order_combo.currentText()}")
        if enable_preprocessing:
            self.log_message(
                f"Output settings: {preprocess_settings['frame_rate']} fps, "
                f"{preprocess_settings['resolution']}, "
                f"{preprocess_settings['codec']}, "
                f"Preset: {preprocess_settings['preset']}, "
                f"CRF: {preprocess_settings['crf']}, "
                f"Grayscale: {preprocess_settings['grayscale']}, "
                f"Remove Audio: {preprocess_settings['remove_audio']}")
        else:
            self.log_message("Output settings: default (libx264, CRF 18)")
        if enable_chunking:
            self.log_message(f"Chunking: {chunk_duration_minutes} min per chunk")
        else:
            self.log_message("Chunking: off")

        # OCR settings
        enable_ocr = self.ocr_check.isChecked() and self._ocr_roi is not None
        ocr_engine = self.ocr_engine_combo.currentData()
        ocr_decoder = self.ocr_decoder_combo.currentData()
        if enable_ocr:
            vx, vy, vw, vh = self._ocr_roi
            engine_label = self.ocr_engine_combo.currentText()
            decoder_label = self.ocr_decoder_combo.currentText() if ocr_engine == "easyocr" else "N/A"
            fmt_label = self._ocr_timestamp_format or "auto-detect"
            interval = self.ocr_sample_interval_spin.value()
            self.log_message(
                f"OCR: enabled — {engine_label}, decoder={decoder_label}, "
                f"sample every {interval}s, format={fmt_label}, "
                f"ROI ({vx},{vy}) to ({vx+vw},{vy+vh})")
        else:
            self.log_message("OCR: off")

        # Start worker thread
        self.worker = ConcatenationWorker(
            self.selected_dirs, output_filename, sort_order, self.instance_id,
            enable_preprocessing=enable_preprocessing,
            preprocess_settings=preprocess_settings,
            enable_chunking=enable_chunking,
            chunk_duration_minutes=chunk_duration_minutes,
            enable_ocr=enable_ocr,
            ocr_roi=self._ocr_roi,
            ocr_source_resolution=self._ocr_source_resolution,
            ocr_engine=ocr_engine,
            ocr_decoder=ocr_decoder,
            ocr_timestamp_format=self._ocr_timestamp_format,
            ocr_sample_interval_sec=self.ocr_sample_interval_spin.value(),
            ocr_text_color=self._ocr_text_color)
        self.worker.progress_update.connect(self.log_message)
        self.worker.folder_progress.connect(self.update_folder_progress)
        self.worker.ffmpeg_output.connect(self.log_ffmpeg_output)
        self.worker.finished.connect(self.concatenation_finished)
        self.worker.user_decision_needed.connect(self._handle_user_decision)
        self.worker.start()
    
    def stop_concatenation(self):
        """Stop video concatenation"""
        if self.worker:
            self.log_message("Stopping concatenation...")
            self.worker.stop()
            self.stop_btn.setEnabled(False)
    
    def update_folder_progress(self, current, total):
        """Update folder progress"""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.folder_progress_label.setText(f"Processing folder {current} of {total}")
    
    def copy_status_log(self):
        """Copy the status log contents to the clipboard"""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.status_log.toPlainText())
        self.copy_status_btn.setText("Copied!")
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(2000, lambda: self.copy_status_btn.setText("Copy Status Log Output"))

    def log_message(self, message):
        """Add message to status log"""
        self.status_log.append(message)
        # Auto-scroll to bottom
        cursor = self.status_log.textCursor()
        cursor.movePosition(cursor.End)
        self.status_log.setTextCursor(cursor)
    
    def log_ffmpeg_output(self, output_line):
        """Add FFmpeg output to log"""
        self.ffmpeg_log.append(output_line)
        # Auto-scroll to bottom
        cursor = self.ffmpeg_log.textCursor()
        cursor.movePosition(cursor.End)
        self.ffmpeg_log.setTextCursor(cursor)
    
    def _handle_user_decision(self, message, file_paths):
        """Show a dialog asking the user whether to skip, move, or cancel.
        Called on the GUI thread via the worker's user_decision_needed signal.
        *file_paths* contains full paths so we can move files if requested."""
        import shutil

        file_names = "\n".join([f"  • {os.path.basename(f)}" for f in file_paths])
        full_message = (
            f"{message}\n\n"
            f"Affected file(s):\n{file_names}\n\n"
            "Choose an action:\n"
            "• Move && Continue — move file(s) to a 'corrupted_video' subfolder "
            "and continue concatenation\n"
            "• Skip && Continue — skip these file(s) and continue\n"
            "• Cancel — stop the concatenation process"
        )
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Corrupt Video File")
        msg_box.setIcon(QMessageBox.Warning)
        msg_box.setText(full_message)
        move_btn = msg_box.addButton("Move && Continue", QMessageBox.AcceptRole)
        skip_btn = msg_box.addButton("Skip && Continue", QMessageBox.AcceptRole)
        cancel_btn = msg_box.addButton("Cancel", QMessageBox.RejectRole)
        msg_box.setDefaultButton(move_btn)
        msg_box.exec_()

        clicked = msg_box.clickedButton()
        if clicked == move_btn:
            # Move corrupt files to corrupted_video/ subfolder
            for fp in file_paths:
                try:
                    parent_dir = os.path.dirname(fp)
                    corrupt_dir = os.path.join(parent_dir, "corrupted_video")
                    os.makedirs(corrupt_dir, exist_ok=True)
                    dest = os.path.join(corrupt_dir, os.path.basename(fp))
                    shutil.move(fp, dest)
                    self.log_message(
                        f"  📁 Moved to corrupted_video/: {os.path.basename(fp)}")
                except Exception as e:
                    self.log_message(
                        f"  ⚠️ Failed to move {os.path.basename(fp)}: {e}")
            self.worker.respond_to_prompt("move")
        elif clicked == skip_btn:
            self.worker.respond_to_prompt("skip")
        else:
            self.worker.respond_to_prompt("cancel")

    def concatenation_finished(self, success, message):
        """Handle concatenation completion"""
        self.log_message(message)

        # Re-enable controls
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.add_dir_btn.setEnabled(True)
        self.clear_dirs_btn.setEnabled(True)

        if success:
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.warning(self, "Concatenation Failed", message)
    
    def closeEvent(self, event):
        """Handle window close event"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "Concatenation in Progress",
                "Concatenation is still running. Are you sure you want to close?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.worker.stop()
                self.worker.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def video_concatenate():
    """Launch the PyQt5 video concatenation GUI"""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    window = VideoConcatenationGUI()
    window.show()
    
    return window


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoConcatenationGUI()
    window.show()
    sys.exit(app.exec_())
