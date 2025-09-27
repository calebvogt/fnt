#!/usr/bin/env python3
"""
FieldNeuroToolbox (FNT) - PyQt Main GUI Application

A professional GUI interface for neurobehavioral data preprocessing and analysis.

ARCHITECTURE:
- PyQt5 main interface for professional appearance and organization
- Individual processing functions use their existing tkinter GUIs (for now)
- This hybrid approach gives us a modern launcher while preserving familiar workflows
- Future: Individual tools will be gradually converted to PyQt as needed

Inspired by SLEAP, DeepLabCut, and other open-source neuroscience tools.
"""

import sys
import os
from pathlib import Path
import webbrowser

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, 
        QHBoxLayout, QGridLayout, QPushButton, QLabel, QStatusBar, 
        QMessageBox, QGroupBox, QTextEdit, QSplitter, QFrame,
        QScrollArea, QSizePolicy
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt5.QtGui import QFont, QIcon, QPalette, QColor
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    print("PyQt5 not available. Please install with: pip install PyQt5")
    sys.exit(1)


class WorkerThread(QThread):
    """Worker thread for running functions without blocking the GUI"""
    finished = pyqtSignal(bool, str)  # success, message
    status_update = pyqtSignal(str)
    
    def __init__(self, func, func_name):
        super().__init__()
        self.func = func
        self.func_name = func_name
    
    def run(self):
        try:
            self.status_update.emit(f"Running {self.func_name}...")
            self.func()
            self.finished.emit(True, f"{self.func_name} completed successfully")
        except Exception as e:
            self.finished.emit(False, f"{self.func_name} failed: {str(e)}")


class FNTMainWindow(QMainWindow):
    """Main PyQt GUI window for FieldNeuroToolbox"""
    
    def __init__(self):
        super().__init__()
        self.current_worker = None
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("FieldNeuroToolbox (FNT) v0.1")
        self.setGeometry(100, 100, 1000, 700)
        self.setMinimumSize(800, 600)
        
        # Set application style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QTabWidget::pane {
                border: 1px solid #c0c0c0;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e1e1e1;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 2px solid #007acc;
            }
            QPushButton {
                background-color: #007acc;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
            QPushButton:pressed {
                background-color: #004578;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #c0c0c0;
                border-radius: 5px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Header
        self.create_header(layout)
        
        # Tab widget
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # Create tabs
        self.create_video_tab()
        self.create_sleap_tab()
        self.create_usv_tab()
        self.create_uwb_tab()
        self.create_utilities_tab()
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - FieldNeuroToolbox initialized")
        
        # Center the window
        self.center_window()
    
    def center_window(self):
        """Center the window on the screen"""
        screen = QApplication.desktop().screenGeometry()
        size = self.geometry()
        self.move(
            int((screen.width() - size.width()) / 2),
            int((screen.height() - size.height()) / 2)
        )
    
    def create_header(self, layout):
        """Create the header section with title and description"""
        header_frame = QFrame()
        header_frame.setFrameStyle(QFrame.Box | QFrame.Raised)
        header_frame.setStyleSheet("background-color: white; padding: 10px;")
        
        header_layout = QVBoxLayout()
        header_frame.setLayout(header_layout)
        
        # Title
        title = QLabel("FieldNeuroToolbox")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 20, QFont.Bold))
        title.setStyleSheet("color: #007acc;")
        header_layout.addWidget(title)
        
        # Subtitle
        subtitle = QLabel("Preprocessing and analysis toolbox for neurobehavioral data")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setFont(QFont("Arial", 11))
        subtitle.setStyleSheet("color: #666666; font-style: italic;")
        header_layout.addWidget(subtitle)
        
        # Version
        version = QLabel("Version 0.1 | Professional PyQt Interface")
        version.setAlignment(Qt.AlignCenter)
        version.setFont(QFont("Arial", 9))
        version.setStyleSheet("color: #999999;")
        header_layout.addWidget(version)
        
        layout.addWidget(header_frame)
    
    def create_video_tab(self):
        """Create the video processing tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Description
        desc = QLabel("Tools for video file processing and manipulation")
        desc.setFont(QFont("Arial", 10, QFont.Bold))
        desc.setStyleSheet("color: #666666; margin: 10px;")
        layout.addWidget(desc)
        
        # Video processing group
        group = QGroupBox("Video Processing Tools")
        group_layout = QGridLayout()
        
        # Create buttons with descriptions
        buttons = [
            ("Video Trimming", "Interactively trim video files with preview", self.run_video_trim),
            ("Video Concatenation", "Join multiple video files together", self.run_video_concatenate),
            ("Video Downsampling", "Reduce video resolution and frame rate", self.run_video_downsample),
            ("Video Re-encoding", "Convert video formats and codecs", self.run_video_reencode),
        ]
        
        self.create_button_grid(group_layout, buttons)
        group.setLayout(group_layout)
        layout.addWidget(group)
        
        layout.addStretch()
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Video Processing")
    
    def create_sleap_tab(self):
        """Create the SLEAP processing tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Description
        desc = QLabel("SLEAP pose estimation pipeline tools")
        desc.setFont(QFont("Arial", 10, QFont.Bold))
        desc.setStyleSheet("color: #666666; margin: 10px;")
        layout.addWidget(desc)
        
        # SLEAP processing group
        group = QGroupBox("SLEAP Analysis Pipeline")
        group_layout = QGridLayout()
        
        buttons = [
            ("Video Inference + Tracking", "Run full SLEAP pipeline on videos", self.run_sleap_inference_track),
            ("Video Inference Only", "Run SLEAP inference without tracking", self.run_sleap_inference_only),
            ("Convert SLP to CSV/H5", "Convert SLEAP files to analysis formats", self.run_sleap_convert),
            ("Re-track SLP Files", "Re-run tracking on existing predictions", self.run_sleap_retrack),
        ]
        
        self.create_button_grid(group_layout, buttons)
        group.setLayout(group_layout)
        layout.addWidget(group)
        
        layout.addStretch()
        tab.setLayout(layout)
        self.tabs.addTab(tab, "SLEAP Analysis")
    
    def create_usv_tab(self):
        """Create the USV processing tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Description
        desc = QLabel("Ultrasonic vocalization analysis tools")
        desc.setFont(QFont("Arial", 10, QFont.Bold))
        desc.setStyleSheet("color: #666666; margin: 10px;")
        layout.addWidget(desc)
        
        # USV processing group
        group = QGroupBox("USV Analysis Tools")
        group_layout = QGridLayout()
        
        buttons = [
            ("USV Heterodyne Processing", "Process ultrasonic recordings", self.run_usv_heterodyne),
            ("Compress Audio Files", "Compress WAV files for storage", self.run_compress_wavs),
        ]
        
        self.create_button_grid(group_layout, buttons)
        group.setLayout(group_layout)
        layout.addWidget(group)
        
        layout.addStretch()
        tab.setLayout(layout)
        self.tabs.addTab(tab, "USV Analysis")
    
    def create_uwb_tab(self):
        """Create the UWB processing tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Description
        desc = QLabel("Ultra-wideband tracking and behavioral analysis")
        desc.setFont(QFont("Arial", 10, QFont.Bold))
        desc.setStyleSheet("color: #666666; margin: 10px;")
        layout.addWidget(desc)
        
        # UWB processing group
        group = QGroupBox("UWB Tracking & Analysis")
        group_layout = QGridLayout()
        
        buttons = [
            ("UWB Data Processing", "Preprocess and export UWB tracking data", self.run_uwb_preprocess),
            ("Animate UWB Paths", "Create animated videos of tracking paths", self.run_uwb_animate),
            ("Behavioral Analysis", "Analyze behavioral patterns from tracking", self.run_uwb_behavioral),
            ("Plot UWB Paths", "Generate static plots of tracking data", self.run_plot_uwb_path),
        ]
        
        self.create_button_grid(group_layout, buttons)
        group.setLayout(group_layout)
        layout.addWidget(group)
        
        layout.addStretch()
        tab.setLayout(layout)
        self.tabs.addTab(tab, "UWB Tracking")
    
    def create_utilities_tab(self):
        """Create the utilities tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Description
        desc = QLabel("General utilities and information")
        desc.setFont(QFont("Arial", 10, QFont.Bold))
        desc.setStyleSheet("color: #666666; margin: 10px;")
        layout.addWidget(desc)
        
        # Utilities group
        group = QGroupBox("Utilities & Information")
        group_layout = QGridLayout()
        
        buttons = [
            ("About FNT", "About FieldNeuroToolbox", self.show_about),
            ("Check Dependencies", "Verify required software is installed", self.check_dependencies),
            ("Open Documentation", "Open FNT documentation", self.open_documentation),
            ("Report Issue", "Report a bug or request a feature", self.report_issue),
        ]
        
        self.create_button_grid(group_layout, buttons)
        group.setLayout(group_layout)
        layout.addWidget(group)
        
        layout.addStretch()
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Utilities")
    
    def create_button_grid(self, layout, buttons):
        """Create a grid of buttons with descriptions"""
        row, col = 0, 0
        
        for title, description, callback in buttons:
            # Create button container
            button_widget = QWidget()
            button_layout = QVBoxLayout()
            button_widget.setLayout(button_layout)
            
            # Main button
            btn = QPushButton(title)
            btn.clicked.connect(callback)
            btn.setMinimumHeight(40)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button_layout.addWidget(btn)
            
            # Description label
            desc_label = QLabel(description)
            desc_label.setFont(QFont("Arial", 9))
            desc_label.setStyleSheet("color: #666666; margin: 5px;")
            desc_label.setWordWrap(True)
            desc_label.setAlignment(Qt.AlignCenter)
            button_layout.addWidget(desc_label)
            
            # Add to grid
            layout.addWidget(button_widget, row, col)
            
            col += 1
            if col > 1:  # 2 columns
                col = 0
                row += 1
    
    def run_function_safely(self, func, func_name):
        """Run a function safely with error handling and status updates"""
        if self.current_worker and self.current_worker.isRunning():
            QMessageBox.information(self, "Please Wait", "Another operation is currently running. Please wait for it to complete.")
            return
        
        self.current_worker = WorkerThread(func, func_name)
        self.current_worker.status_update.connect(self.status_bar.showMessage)
        self.current_worker.finished.connect(self.on_function_finished)
        self.current_worker.start()
    
    def on_function_finished(self, success, message):
        """Handle function completion"""
        if success:
            self.status_bar.showMessage(message, 5000)  # Show for 5 seconds
        else:
            QMessageBox.critical(self, "Error", message)
            self.status_bar.showMessage("Ready")
        
        self.current_worker = None
    
    # Video Processing Methods - Call existing tkinter functions
    def run_video_trim(self):
        """Launch video trimming tool"""
        def func():
            from fnt.video_processing.video_trim import video_trim
            video_trim()
        self.run_function_safely(func, "Video Trimming")
    
    def run_video_concatenate(self):
        """Launch video concatenation tool"""
        def func():
            from fnt.video_processing.video_concatenate import video_concatenate
            video_concatenate()
        self.run_function_safely(func, "Video Concatenation")
    
    def run_video_downsample(self):
        """Launch video downsampling tool"""
        def func():
            from fnt.video_processing.video_downsample import video_downsample
            video_downsample()
        self.run_function_safely(func, "Video Downsampling")
    
    def run_video_reencode(self):
        """Launch video re-encoding tool"""
        def func():
            from fnt.video_processing.video_reencode import video_reencode
            video_reencode()
        self.run_function_safely(func, "Video Re-encoding")
    
    # SLEAP Processing Methods - Call existing tkinter functions
    def run_sleap_inference_track(self):
        """Launch SLEAP inference and tracking"""
        def func():
            from fnt.sleap_processing.batch_video_inference_and_track import main
            main()
        self.run_function_safely(func, "SLEAP Inference + Tracking")
    
    def run_sleap_inference_only(self):
        """Launch SLEAP inference only"""
        def func():
            from fnt.sleap_processing.batch_video_inference_only import main
            main()
        self.run_function_safely(func, "SLEAP Inference Only")
    
    def run_sleap_convert(self):
        """Launch SLEAP file conversion"""
        def func():
            from fnt.sleap_processing.batch_convert_slp_to_csv_h5 import main
            main()
        self.run_function_safely(func, "SLEAP File Conversion")
    
    def run_sleap_retrack(self):
        """Launch SLEAP re-tracking"""
        def func():
            from fnt.sleap_processing.batch_slp_retrack import main
            main()
        self.run_function_safely(func, "SLEAP Re-tracking")
    
    # USV Processing Methods - Call existing tkinter functions
    def run_usv_heterodyne(self):
        """Launch USV heterodyne processing"""
        def func():
            from fnt.usv.usv_heterodyne import usv_batch_heterodyne
            usv_batch_heterodyne()
        self.run_function_safely(func, "USV Heterodyne Processing")
    
    def run_compress_wavs(self):
        """Launch WAV compression"""
        def func():
            from fnt.usv.compress_wavs import compress_wavs_main
            compress_wavs_main()
        self.run_function_safely(func, "WAV Compression")
    
    # UWB Processing Methods - Call existing tkinter functions
    def run_uwb_preprocess(self):
        """Launch UWB data preprocessing"""
        def func():
            from fnt.uwb.uwb_preprocess_sql import uwb_smoothing
            uwb_smoothing()
        self.run_function_safely(func, "UWB Data Processing")
    
    def run_uwb_animate(self):
        """Launch UWB path animation"""
        def func():
            from fnt.uwb.uwb_animate import uwb_animate_paths
            uwb_animate_paths()
        self.run_function_safely(func, "UWB Path Animation")
    
    def run_uwb_behavioral(self):
        """Launch UWB behavioral analysis"""
        def func():
            from fnt.uwb.uwb_behavioral_analysis import uwb_behavioral_analysis
            uwb_behavioral_analysis()
        self.run_function_safely(func, "UWB Behavioral Analysis")
    
    def run_plot_uwb_path(self):
        """Launch UWB path plotting"""
        def func():
            from fnt.uwb.plot_uwb_path import plot_uwb_path
            plot_uwb_path()
        self.run_function_safely(func, "UWB Path Plotting")
    
    # Utility Methods
    def show_about(self):
        """Show about dialog"""
        about_text = """<h2>FieldNeuroToolbox (FNT) v0.1</h2>
        
        <p>A comprehensive preprocessing and analysis toolbox for neurobehavioral data.</p>
        
        <h3>Features:</h3>
        <ul>
        <li>Video processing and manipulation</li>
        <li>SLEAP pose estimation pipeline</li>
        <li>Ultrasonic vocalization analysis</li>
        <li>Ultra-wideband tracking analysis</li>
        <li>Behavioral classification tools</li>
        </ul>
        
        <p><b>Interface:</b> Professional PyQt5-based GUI calling existing tkinter tools</p>
        
        <p>Developed for the neuroscience research community.</p>
        
        <p>For more information, visit: <a href='https://github.com/calebvogt/fnt'>github.com/calebvogt/fnt</a></p>
        """
        
        msg = QMessageBox()
        msg.setWindowTitle("About FieldNeuroToolbox")
        msg.setTextFormat(Qt.RichText)
        msg.setText(about_text)
        msg.exec_()
    
    def check_dependencies(self):
        """Check if required dependencies are available"""
        deps = {
            "FFmpeg": self.check_ffmpeg(),
            "OpenCV": self.check_opencv(),
            "NumPy": self.check_numpy(),
            "Pandas": self.check_pandas(),
            "Matplotlib": self.check_matplotlib(),
            "PyQt5": True,  # We know this is available since we're running
        }
        
        missing = [name for name, available in deps.items() if not available]
        available = [name for name, available in deps.items() if available]
        
        if missing:
            msg = f"<h3>Dependency Check Results</h3>"
            msg += f"<p><b style='color: green;'>Available:</b> {', '.join(available)}</p>"
            msg += f"<p><b style='color: red;'>Missing:</b> {', '.join(missing)}</p>"
            msg += f"<p>Please install missing dependencies for full functionality.</p>"
        else:
            msg = "<h3>Dependency Check Results</h3><p style='color: green;'><b>All required dependencies are available!</b></p>"
        
        msgbox = QMessageBox()
        msgbox.setWindowTitle("Dependencies Check")
        msgbox.setTextFormat(Qt.RichText)
        msgbox.setText(msg)
        msgbox.exec_()
    
    def check_ffmpeg(self):
        """Check if FFmpeg is available"""
        try:
            import subprocess
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
        except:
            return False
    
    def check_opencv(self):
        """Check if OpenCV is available"""
        try:
            import cv2
            return True
        except ImportError:
            return False
    
    def check_numpy(self):
        """Check if NumPy is available"""
        try:
            import numpy
            return True
        except ImportError:
            return False
    
    def check_pandas(self):
        """Check if Pandas is available"""
        try:
            import pandas
            return True
        except ImportError:
            return False
    
    def check_matplotlib(self):
        """Check if Matplotlib is available"""
        try:
            import matplotlib
            return True
        except ImportError:
            return False
    
    def open_documentation(self):
        """Open documentation in web browser"""
        webbrowser.open("https://github.com/calebvogt/fnt")
    
    def report_issue(self):
        """Open GitHub issues page"""
        webbrowser.open("https://github.com/calebvogt/fnt/issues")
    
    def closeEvent(self, event):
        """Handle window close event"""
        if self.current_worker and self.current_worker.isRunning():
            reply = QMessageBox.question(
                self, 'Close Application',
                "An operation is currently running. Are you sure you want to quit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                if self.current_worker:
                    self.current_worker.terminate()
                    self.current_worker.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    """Main entry point for the FNT PyQt GUI application"""
    if not PYQT_AVAILABLE:
        print("Error: PyQt5 is not available. Please install it with: pip install PyQt5")
        return
    
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("FieldNeuroToolbox")
    app.setApplicationVersion("0.1")
    app.setOrganizationName("FNT")
    
    # Create and show main window
    window = FNTMainWindow()
    window.show()
    
    # Start the event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()