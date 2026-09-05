"""Parent-side launcher: runs each FNT tool in its own OS process.

``ToolProcessManager`` spawns tool windows through :mod:`fnt.tool_host`, keeps
track of the children, notices when one dies abnormally, and offers to reopen
it. See :mod:`fnt.tool_host` for why the tools are isolated at all.

Typical use from the main window::

    self.tools = ToolProcessManager(self)
    self.tools.launch("uwb_preprocessing")
"""

import os
import subprocess
import sys

from PyQt5.QtCore import QObject, QProcess, pyqtSignal
from PyQt5.QtWidgets import QMessageBox

from fnt.tool_host import crash_log_path


class ToolSpec:
    """How to start one tool window in a host process."""

    def __init__(self, key, module, attr, title, multi=True, post_show=None):
        self.key = key
        self.module = module
        self.attr = attr
        self.title = title
        # multi=False -> focus/reuse rather than opening a second copy.
        self.multi = multi
        # Zero-arg window method to call once the window is on screen.
        self.post_show = post_show


# Every windowed FNT tool. `attr` is a class (or zero-arg factory) that
# fnt.tool_host instantiates inside the child process.
TOOL_REGISTRY = {
    spec.key: spec for spec in [
        # --- Behavior / acquisition ---
        ToolSpec("abma_designer", "fnt.abma.gui.abma_main_pyqt",
                 "ABMAWindow", "ABMA Designer",
                 post_show="show_start_dialog"),   # New / Open project chooser
        ToolSpec("rfid_preprocessing", "fnt.rfid.rfid_preprocessing_pyqt",
                 "RFIDPreprocessingWindow", "RFID PreProcessing"),
        # NOTE: the FED Processing tool is deliberately NOT registered here. It
        # still opens in-process from fnt.gui_pyqt.run_fed_processing and is
        # owned separately; isolating it is that owner's call.
        ToolSpec("musestudio", "fnt.musestudio.musestudio_pyqt",
                 "MuseStudioWindow", "Muse Studio"),

        # --- Video processing ---
        ToolSpec("video_trim", "fnt.videoProcessing.video_trim_pyqt",
                 "VideoTrimTool", "Video Trim and Crop"),
        ToolSpec("video_concatenate", "fnt.videoProcessing.video_concatenate_pyqt",
                 "VideoConcatenationGUI", "Video Concatenation"),
        ToolSpec("video_processing", "fnt.videoProcessing.videoProcessing",
                 "VideoProcessingGUI", "Video PreProcessing"),
        ToolSpec("behavior_scoring", "fnt.videoProcessing.behavior_scoring_studio_pyqt",
                 "BehaviorScoringStudioWindow", "Behavior Scoring Studio"),
        ToolSpec("camera_grid", "fnt.videoProcessing.camera_grid_pyqt",
                 "CameraGridWindow", "Camera Grid"),

        # --- SLEAP / pose ---
        ToolSpec("sleap_inference", "fnt.sleapProcessing.sleap_inference_tool_pyqt",
                 "VideoInferenceWindow", "SLEAP Video Inference"),
        ToolSpec("sleap_render_videos", "fnt.sleapProcessing.batch_render_videos_pyqt",
                 "RenderVideosWindow", "Render Videos"),
        ToolSpec("roi_tool", "fnt.sleapProcessing.sleap_roi_tool_pyqt",
                 "ROIToolGUI", "SLEAP ROI Tool"),
        ToolSpec("generate_training_images", "fnt.labgym.generate_training_images_pyqt",
                 "GenerateTrainingImagesWindow", "Generate Training Images"),

        # --- Tracking ---
        ToolSpec("mask_tracker", "fnt.videoTracking.mask_tracker_gui",
                 "MaskTrackerWindow", "SAM2 Mask Tracker"),
        ToolSpec("mask_pose_tracker", "fnt.videoTracking.mask_pose_tracker_gui",
                 "MaskPoseTrackerGUI", "Mask Pose Tracker"),
        ToolSpec("simple_tracker", "fnt.videoTracking.simple_tracker_gui_v2",
                 "SimpleTrackerGUI", "Simple Tracker"),

        # --- USV / audio ---
        ToolSpec("classic_audio_detector", "fnt.usv.classic_audio_detector",
                 "ClassicAudioDetectorWindow", "Classic Audio Detector"),
        ToolSpec("mask_audio_detector", "fnt.usv.mad_pyqt",
                 "MADMainWindow", "Mask Audio Detector"),
        ToolSpec("audio_trim", "fnt.usv.audio_trim_pyqt",
                 "AudioTrimWindow", "Audio Trim"),
        ToolSpec("compress_wavs", "fnt.usv.compress_wavs_pyqt",
                 "CompressWavsWindow", "Compress WAVs"),

        # --- UWB ---
        ToolSpec("uwb_preprocessing", "fnt.uwb.uwb_preprocessing_pyqt",
                 "UWBQuickVisualizationWindow", "UWB PreProcessing"),

        # --- Imaging ---
        ToolSpec("czi_viewer", "fnt.imaging.czi_viewer_pyqt",
                 "CZIViewerWindow", "CZI Viewer"),
        ToolSpec("image_quantification", "fnt.imaging.quantification_pyqt",
                 "QuantificationToolWindow", "Image Quantification"),

        # --- Misc ---
        ToolSpec("github_csv_transfer", "fnt.gitProcessing.github_csv_transfer_pyqt",
                 "GitHubCSVTransferWindow", "GitHub CSV Transfer"),
        ToolSpec("doric_processor", "fnt.DoricFP.doric_processor_pyqt",
                 "DoricProcessorWindow", "Doric Processor"),
    ]
}


def _child_python():
    """Interpreter to run children with — pythonw.exe on Windows (no console)."""
    exe = sys.executable
    if os.name == "nt":
        cand = os.path.join(os.path.dirname(exe), "pythonw.exe")
        if os.path.exists(cand):
            return cand
    return exe


class ToolProcessManager(QObject):
    """Spawns tool windows as child processes and watches them.

    Uses QProcess rather than subprocess so completion arrives as a signal on
    the Qt event loop — no polling, and the parent UI stays responsive.
    """

    tool_started = pyqtSignal(str)          # tool key
    tool_exited = pyqtSignal(str, int)      # tool key, exit code

    def __init__(self, parent_window):
        super().__init__(parent_window)
        self.window = parent_window
        self._procs = {}          # QProcess -> ToolSpec
        self._suppress_notice = False

    # --- launching -------------------------------------------------------

    def launch(self, key):
        """Start ``key`` in its own process. Returns True if it spawned."""
        spec = TOOL_REGISTRY.get(key)
        if spec is None:
            QMessageBox.critical(self.window, "Unknown tool",
                                 f"No tool registered under {key!r}.")
            return False

        if not spec.multi:
            for proc, running in self._procs.items():
                if running.key == key and proc.state() != QProcess.NotRunning:
                    QMessageBox.information(
                        self.window, spec.title,
                        f"{spec.title} is already open.")
                    return False

        # Frozen builds can't run `python -m`; fall back to in-process so the
        # button still works (without isolation).
        if getattr(sys, "frozen", False):
            return self._launch_in_process(spec)

        proc = QProcess(self)
        proc.setProgram(_child_python())
        cmd_args = [
            "-m", "fnt.tool_host", spec.module, spec.attr, "--title", spec.title,
        ]
        if spec.post_show:
            cmd_args += ["--post-show", spec.post_show]
        proc.setArguments(cmd_args)

        env = proc.processEnvironment()
        from PyQt5.QtCore import QProcessEnvironment
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")  # avoid cp1252 console crashes
        proc.setProcessEnvironment(env)

        # Send child stdout/stderr to a per-tool log. Writing to a file (rather
        # than a pipe we never drain) avoids the child blocking on a full pipe
        # buffer once it has printed ~64KB.
        log_dir = os.path.join(os.path.expanduser("~"), ".fnt", "logs")
        try:
            os.makedirs(log_dir, exist_ok=True)
            proc.setStandardOutputFile(
                os.path.join(log_dir, f"{spec.key}.log"), QProcess.Append)
            proc.setStandardErrorFile(
                os.path.join(log_dir, f"{spec.key}.log"), QProcess.Append)
        except Exception:
            pass

        proc.finished.connect(
            lambda code, status, p=proc: self._on_finished(p, code, status))
        proc.errorOccurred.connect(
            lambda err, p=proc: self._on_error(p, err))

        self._procs[proc] = spec
        proc.start()

        if not proc.waitForStarted(10000):
            self._procs.pop(proc, None)
            QMessageBox.critical(
                self.window, "Error",
                f"Failed to launch {spec.title}:\n{proc.errorString()}")
            return False

        self.tool_started.emit(key)
        return True

    def _launch_in_process(self, spec):
        """Frozen-build fallback: open the window inside this process."""
        try:
            from fnt.tool_host import load_window
            win = load_window(spec.module, spec.attr)
            win.setWindowTitle(spec.title)
            win.show()
            if not hasattr(self, "_inproc_windows"):
                self._inproc_windows = []
            self._inproc_windows.append(win)
            if spec.post_show:
                getattr(win, spec.post_show)()
            return True
        except Exception as e:
            QMessageBox.critical(self.window, "Error",
                                 f"{spec.title} failed to open:\n{e}")
            return False

    # --- monitoring ------------------------------------------------------

    def _on_error(self, proc, err):
        if err == QProcess.FailedToStart:
            spec = self._procs.pop(proc, None)
            if spec:
                QMessageBox.critical(
                    self.window, "Error",
                    f"Failed to start {spec.title}:\n{proc.errorString()}")

    def _on_finished(self, proc, code, status):
        spec = self._procs.pop(proc, None)
        if spec is None:
            return
        self.tool_exited.emit(spec.key, code)

        crashed = (status == QProcess.CrashExit) or (code != 0)
        if crashed and not self._suppress_notice:
            self._report_crash(spec, code)

    def _report_crash(self, spec, code):
        """Tell the user which tool died and offer to reopen it."""
        box = QMessageBox(self.window)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Tool closed unexpectedly")
        box.setText(f"{spec.title} closed unexpectedly.")
        box.setInformativeText(
            f"Exit code {code}. Your other FNT tools are unaffected and any "
            f"work in them is still running.\n\n"
            f"Crash details: {crash_log_path()}")
        reopen = box.addButton("Reopen", QMessageBox.AcceptRole)
        box.addButton("Dismiss", QMessageBox.RejectRole)
        box.exec_()
        if box.clickedButton() is reopen:
            self.launch(spec.key)

    # --- shutdown --------------------------------------------------------

    def running_tools(self):
        """Titles of tools still alive."""
        return [spec.title for proc, spec in self._procs.items()
                if proc.state() != QProcess.NotRunning]

    def terminate_all(self, wait_ms=3000):
        """Ask every child to close, then kill whatever is left."""
        self._suppress_notice = True   # quitting is not a crash
        for proc in list(self._procs):
            try:
                if proc.state() != QProcess.NotRunning:
                    proc.terminate()
            except Exception:
                pass
        for proc in list(self._procs):
            try:
                if proc.state() != QProcess.NotRunning and \
                        not proc.waitForFinished(wait_ms):
                    proc.kill()
            except Exception:
                pass
        self._procs.clear()
