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
        QScrollArea, QSizePolicy, QFileDialog, QInputDialog
    )
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
    from PyQt5.QtGui import QFont, QIcon, QPalette, QColor, QPixmap
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
        
        # Set window icon (check multiple possible locations)
        icon_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'icons', 'fnt_icon.ico'),
            os.path.join(os.path.dirname(__file__), 'icons', 'fnt_icon.ico'),
        ]
        
        for icon_path in icon_paths:
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
                break
        
        # Set dark theme style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #cccccc;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #cccccc;
            }
            QTabWidget::pane {
                border: 1px solid #3f3f3f;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #1e1e1e;
                color: #cccccc;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: #2b2b2b;
                border-bottom: 2px solid #0078d4;
            }
            QTabBar::tab:hover:!selected {
                background-color: #3f3f3f;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #3f3f3f;
                color: #cccccc;
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
            QStatusBar {
                background-color: #1e1e1e;
                color: #cccccc;
                border-top: 1px solid #3f3f3f;
            }
            QFrame {
                background-color: #2b2b2b;
                border-color: #3f3f3f;
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
        
        # Create tabs - Original order
        self.create_video_tab()
        self.create_sleap_tab()
        self.create_video_tracking_tab()
        self.create_uwb_tab()
        self.create_usv_tab()
        self.create_github_tab()
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
        header_frame.setStyleSheet("background-color: #1e1e1e; padding: 10px; border: 1px solid #3f3f3f;")
        
        header_layout = QHBoxLayout()
        header_frame.setLayout(header_layout)
        
        # Logo/Icon on the left
        logo_label = QLabel()
        icon_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'icons', 'ChatGPT Image Oct 29, 2025, 12_21_19 PM.png'),
            os.path.join(os.path.dirname(__file__), 'icons', 'ChatGPT Image Oct 29, 2025, 12_21_19 PM.png'),
        ]
        
        for icon_path in icon_paths:
            if os.path.exists(icon_path):
                pixmap = QPixmap(icon_path)
                # Scale to reasonable size while maintaining aspect ratio
                scaled_pixmap = pixmap.scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                logo_label.setPixmap(scaled_pixmap)
                break
        
        logo_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        header_layout.addWidget(logo_label)
        
        # Title and subtitle in the center
        text_layout = QVBoxLayout()
        
        title = QLabel("FieldNeuroToolbox")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Arial", 20, QFont.Bold))
        title.setStyleSheet("color: #0078d4; background-color: transparent;")
        text_layout.addWidget(title)
        
        # Subtitle
        subtitle = QLabel("Preprocessing and analysis toolbox for neurobehavioral data")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setFont(QFont("Arial", 11))
        subtitle.setStyleSheet("color: #999999; font-style: italic; background-color: transparent;")
        text_layout.addWidget(subtitle)
        
        # Add text layout to header
        header_layout.addLayout(text_layout)
        header_layout.addStretch()
        
        layout.addWidget(header_frame)
    
    def create_video_tab(self):
        """Create the video processing tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Description
        desc = QLabel("Tools for video file processing and manipulation")
        desc.setFont(QFont("Arial", 10, QFont.Bold))
        desc.setStyleSheet("color: #cccccc; margin: 10px;")
        layout.addWidget(desc)
        
        # Video processing group
        group = QGroupBox("Video Processing Tools")
        group_layout = QGridLayout()
        
        # Create buttons with descriptions
        buttons = [
            ("Video PreProcessing", "Comprehensive preprocessing: downsampling, re-encoding, format conversion", self.run_video_processing),
            ("Video Trim and Crop", "Interactively trim video files with preview", self.run_video_trim),
            ("Video Concatenation", "Join multiple video files together", self.run_video_concatenate),
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
        desc.setStyleSheet("color: #cccccc; margin: 10px;")
        layout.addWidget(desc)
        
        # SLEAP processing group
        group = QGroupBox("SLEAP Analysis Pipeline")
        group_layout = QGridLayout()
        
        buttons = [
            ("Run Inference", "Run SLEAP inference with optional tracking", self.run_sleap_inference_only),
            ("Convert SLP to CSV/H5", "Convert SLEAP files to analysis formats", self.run_sleap_convert),
            ("Re-track SLP Files", "Re-run tracking on existing predictions", self.run_sleap_retrack),
            ("Create Tracked Videos", "Render tracked videos from existing .slp files", self.run_sleap_render_videos),
        ]
        
        self.create_button_grid(group_layout, buttons)
        group.setLayout(group_layout)
        layout.addWidget(group)
        
        # SLEAP post-processing group
        post_group = QGroupBox("SLEAP Post-Processing")
        post_layout = QGridLayout()
        
        post_buttons = [
            ("ROI Tool", "Define ROIs and analyze spatial occupancy", self.run_sleap_roi_tool),
        ]
        
        self.create_button_grid(post_layout, post_buttons)
        post_group.setLayout(post_layout)
        layout.addWidget(post_group)
        
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
        desc.setStyleSheet("color: #cccccc; margin: 10px;")
        layout.addWidget(desc)
        
        # USV processing group
        group = QGroupBox("USV Processing Tools")
        group_layout = QGridLayout()
        
        buttons = [
            ("Compress Audio Files", "Compress WAV files for storage", self.run_compress_wavs),
            ("USV Heterodyne Processing", "Process ultrasonic recordings", self.run_usv_heterodyne),
            ("Trim Audio File", "Trim audio with spectrogram visualization and frequency filtering", self.run_audio_trim),
        ]
        
        self.create_button_grid(group_layout, buttons)
        group.setLayout(group_layout)
        layout.addWidget(group)
        
        layout.addStretch()
        tab.setLayout(layout)
        self.tabs.addTab(tab, "USV Processing")
    
    def create_uwb_tab(self):
        """Create the UWB processing tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Description
        desc = QLabel("Ultra-wideband tracking and behavioral analysis")
        desc.setFont(QFont("Arial", 10, QFont.Bold))
        desc.setStyleSheet("color: #cccccc; margin: 10px;")
        layout.addWidget(desc)
        
        # UWB Quick Review group
        quick_group = QGroupBox("UWB Quick Review")
        quick_layout = QGridLayout()
        
        quick_buttons = [
            ("UWB Quick Visualization", "Interactive tracking visualization with time slider", self.run_uwb_quick_visualization),
            ("UWB Quick Plots", "Generate quick visualization plots from tracking data", self.run_uwb_quick_plots),
        ]
        
        self.create_button_grid(quick_layout, quick_buttons)
        quick_group.setLayout(quick_layout)
        layout.addWidget(quick_group)
        
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
        self.tabs.addTab(tab, "UWB Processing")
    
    def create_github_tab(self):
        """Create the GitHub preprocessing tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Description
        desc = QLabel("Tools for preparing data files for GitHub repositories")
        desc.setFont(QFont("Arial", 10, QFont.Bold))
        desc.setStyleSheet("color: #cccccc; margin: 10px;")
        layout.addWidget(desc)
        
        # GitHub preprocessing group
        group = QGroupBox("GitHub Data Preparation")
        group_layout = QGridLayout()
        
        buttons = [
            ("File Splitter", "Split large files to meet GitHub's 50MB limit", self.run_file_splitter),
        ]
        
        self.create_button_grid(group_layout, buttons)
        group.setLayout(group_layout)
        layout.addWidget(group)
        
        layout.addStretch()
        tab.setLayout(layout)
        self.tabs.addTab(tab, "GitHub Preprocessing")

    def create_video_tracking_tab(self):
        """Create the video tracking tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Description
        desc = QLabel("Interactive tracking using SAM (Segment Anything Model) for behavioral tests")
        desc.setFont(QFont("Arial", 10, QFont.Bold))
        desc.setStyleSheet("color: #cccccc; margin: 10px;")
        layout.addWidget(desc)
        
        # SAM-based tracking group
        group = QGroupBox("SAM-Based Tracking Tools")
        group_layout = QGridLayout()
        
        buttons = [
            ("Simple Tracker", "Fast CPU-only tracking using classical computer vision methods", self.run_simple_tracker),
            ("Mask Pose Tracker", "SAM on every frame - extract rich pose features for behavioral clustering (requires GPU)", self.run_mask_pose_tracker),
        ]
        
        self.create_button_grid(group_layout, buttons)
        group.setLayout(group_layout)
        layout.addWidget(group)
        
        # Info box
        info_label = QLabel(
            "<b>Note:</b> Video tracking requires SAM model checkpoint.<br>"
            "Download from: <a href='https://github.com/facebookresearch/segment-anything#model-checkpoints'>"
            "github.com/facebookresearch/segment-anything</a><br><br>"
            "<b>Workflow:</b> Select video → Click on animal → Draw ROI → Track automatically"
        )
        info_label.setTextFormat(Qt.RichText)
        info_label.setOpenExternalLinks(True)
        info_label.setStyleSheet("color: #cccccc; background-color: #1e1e1e; padding: 10px; border: 1px solid #3f3f3f; border-radius: 4px; margin: 10px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        layout.addStretch()
        tab.setLayout(layout)
        self.tabs.addTab(tab, "Video Tracking")

    def create_utilities_tab(self):
        """Create the utilities tab"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Description
        desc = QLabel("General utilities and information")
        desc.setFont(QFont("Arial", 10, QFont.Bold))
        desc.setStyleSheet("color: #cccccc; margin: 10px;")
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
            desc_label.setStyleSheet("color: #cccccc; margin: 5px;")
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
    
    # Video Processing Methods
    def run_video_trim(self):
        """Launch video trimming tool"""
        # PyQt dialogs must run in main thread, not worker thread
        try:
            from fnt.videoProcessing.video_trim_pyqt import video_trim
            video_trim()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Video trimming failed: {str(e)}")
    
    def run_video_concatenate(self):
        """Launch video concatenation tool with PyQt interface"""
        try:
            from fnt.videoProcessing.video_concatenate_pyqt import VideoConcatenationGUI
            
            # Create and show the video concatenation window
            self.video_concatenation_window = VideoConcatenationGUI()
            self.video_concatenation_window.show()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to launch video concatenation tool: {str(e)}")
    
    def run_video_processing(self):
        """Launch combined video processing tool with PyQt interface"""
        try:
            from fnt.videoProcessing.videoProcessing import VideoProcessingGUI
            
            # Create a new instance each time to allow multiple windows
            video_processing_window = VideoProcessingGUI()
            video_processing_window.show()
            
            # Store reference to prevent garbage collection (use list to allow multiple instances)
            if not hasattr(self, 'video_processing_windows'):
                self.video_processing_windows = []
            self.video_processing_windows.append(video_processing_window)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to launch video processing tool: {str(e)}")
    
    # SLEAP Processing Methods
    def run_sleap_inference_only(self):
        """Launch SLEAP inference only with PyQt interface"""
        try:
            from fnt.sleapProcessing.batch_video_inference_only_pyqt import VideoInferenceWindow
            
            # Create and show the video inference window
            self.video_inference_window = VideoInferenceWindow()
            self.video_inference_window.show()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to launch video inference tool: {str(e)}")
    
    def run_sleap_convert(self):
        """Launch SLEAP file conversion"""
        def func():
            from fnt.sleapProcessing.batch_convert_slp_to_csv_h5 import main
            main()
        self.run_function_safely(func, "SLEAP File Conversion")
    
    def run_sleap_retrack(self):
        """Launch SLEAP re-tracking"""
        def func():
            from fnt.sleapProcessing.batch_slp_retrack import main
            main()
        self.run_function_safely(func, "SLEAP Re-tracking")
    
    def run_sleap_render_videos(self):
        """Launch SLEAP video rendering from existing .slp files"""
        try:
            from fnt.sleapProcessing.batch_render_videos_pyqt import RenderVideosWindow
            
            # Create and show the render videos window
            self.render_videos_window = RenderVideosWindow()
            self.render_videos_window.show()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to launch video rendering tool: {str(e)}")
    
    def run_sleap_roi_tool(self):
        """Launch SLEAP ROI Analysis Tool"""
        try:
            from fnt.sleapProcessing.sleap_roi_tool_pyqt import ROIToolGUI
            
            # Create and show the ROI tool window
            self.roi_tool_window = ROIToolGUI()
            self.roi_tool_window.show()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to launch ROI tool: {str(e)}")
    
    # USV Processing Methods - Call existing tkinter functions
    def run_usv_heterodyne(self):
        """Launch USV heterodyne processing"""
        def func():
            from fnt.usv.usv_heterodyne import usv_batch_heterodyne
            usv_batch_heterodyne()
        self.run_function_safely(func, "USV Heterodyne Processing")
    
    def run_audio_trim(self):
        """Launch audio trimming with spectrogram visualization"""
        try:
            from fnt.usv.audio_trim_pyqt import AudioTrimWindow
            
            # Create and show the audio trim window
            self.audio_trim_window = AudioTrimWindow()
            self.audio_trim_window.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Audio trimming failed: {str(e)}")
    
    def run_compress_wavs(self):
        """Launch WAV compression"""
        try:
            from fnt.usv.compress_wavs_pyqt import CompressWavsWindow
            
            # Create and show the compress wavs window
            self.compress_wavs_window = CompressWavsWindow()
            self.compress_wavs_window.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"WAV compression failed: {str(e)}")
    
    # UWB Processing Methods
    def run_uwb_quick_plots(self):
        """Launch UWB Quick Plots tool"""
        try:
            from fnt.uwb.uwb_quick_plots_pyqt import UWBQuickPlotsWindow
            
            # Create and show the UWB quick plots window
            self.uwb_quick_plots_window = UWBQuickPlotsWindow()
            self.uwb_quick_plots_window.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"UWB Quick Plots failed: {str(e)}")
    
    def run_uwb_quick_visualization(self):
        """Launch UWB Quick Visualization tool"""
        try:
            from fnt.uwb.uwb_quick_visualization_pyqt import UWBQuickVisualizationWindow
            
            # Create and show the UWB quick visualization window
            self.uwb_quick_viz_window = UWBQuickVisualizationWindow()
            self.uwb_quick_viz_window.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"UWB Quick Visualization failed: {str(e)}")
    
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
    
    # Video Tracking Methods
    def run_mask_pose_tracker(self):
        """Launch Mask Pose Tracker with SAM"""
        try:
            from fnt.videoTracking.mask_pose_tracker_gui import MaskPoseTrackerGUI
            
            # Create and show the Mask Pose Tracker window
            self.mask_pose_tracker_window = MaskPoseTrackerGUI()
            self.mask_pose_tracker_window.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to launch Mask Pose Tracker: {str(e)}\n\nMake sure dependencies are installed:\npip install opencv-python torch segment-anything pandas numpy")
    
    def run_simple_tracker(self):
        """Launch Simple Tracker (CPU-only, no SAM)"""
        try:
            from fnt.videoTracking.simple_tracker_gui_v2 import SimpleTrackerGUI
            
            # Create and show the Simple Tracker window
            self.simple_tracker_window = SimpleTrackerGUI()
            self.simple_tracker_window.show()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to launch Simple Tracker: {str(e)}\n\nMake sure dependencies are installed:\npip install opencv-python pandas numpy scipy")
    
    # GitHub Processing Methods - Pure PyQt implementations
    def run_file_splitter(self):
        """Launch file splitter for GitHub preparation"""
        # Run file selection in main thread to avoid Qt threading issues
        try:
            self.split_large_files()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"File splitter failed: {str(e)}")
    

    def split_large_files(self):
        """Split large files into smaller chunks for GitHub"""
        self.status_bar.showMessage("Opening file selection dialog...")
        
        # Select files to split - improved dialog with more options
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select files to split for GitHub (Use Ctrl+Click for multiple files)",
            "",
            "All Files (*.*);;"
            "CSV Files (*.csv);;"
            "Excel Files (*.xlsx *.xls);;"
            "Data Files (*.dat *.txt);;"
            "Archive Files (*.zip *.rar *.7z);;"
            "Video Files (*.mp4 *.avi *.mov);;"
            "Image Files (*.jpg *.png *.tiff)"
        )
        
        if not files:
            QMessageBox.information(self, "No Files Selected", "No files were selected for splitting.")
            self.status_bar.showMessage("Ready")
            return
        
        # Get max file size from user
        max_size_mb, ok = QInputDialog.getInt(
            self,
            "Maximum File Size",
            "Enter maximum file size in MB:\n\n" +
            "• GitHub limit: 50MB\n" +
            "• Recommended: 45MB (for safety)\n" +
            "• Minimum: 1MB",
            value=45,
            min=1,
            max=100
        )
        
        if not ok:
            self.status_bar.showMessage("Ready")
            return
        
        # Process files in worker thread to avoid blocking GUI
        def process_files():
            self.process_file_splitting(files, max_size_mb)
        
        self.run_function_safely(process_files, "File Splitting")
    
    def process_file_splitting(self, files, max_size_mb):
        """Process the actual file splitting (runs in worker thread)"""
        
    def process_file_splitting(self, files, max_size_mb):
        """Process the actual file splitting (runs in worker thread)"""
        max_size_bytes = max_size_mb * 1024 * 1024
        
        # Process each file
        total_files = len(files)
        files_split = 0
        files_skipped = 0
        
        for i, file_path in enumerate(files, 1):
            print(f"Processing file {i}/{total_files}: {os.path.basename(file_path)}")
            
            try:
                file_size = os.path.getsize(file_path)
                
                if file_size <= max_size_bytes:
                    print(f"Skipping {file_path}: already under size limit ({file_size/1024/1024:.1f}MB)")
                    files_skipped += 1
                    continue
                
                # Split the file
                self.split_file(file_path, max_size_bytes)
                files_split += 1
                
            except Exception as e:
                print(f"Error processing {os.path.basename(file_path)}: {str(e)}")
        
        # Print summary to console (will be shown in status)
        print(f"\nFile splitting completed!")
        print(f"Files split: {files_split}")
        print(f"Files skipped (already small enough): {files_skipped}")
        print(f"Total files processed: {total_files}")
        
        return f"Split {files_split} files, skipped {files_skipped}"
    
    def split_file(self, file_path, max_size_bytes):
        """Split a single file into chunks - smart splitting for CSV files"""
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        base_name, ext = os.path.splitext(file_name)
        
        # Check if it's a CSV file for smart splitting
        if ext.lower() in ['.csv', '.tsv']:
            return self.split_csv_file(file_path, max_size_bytes)
        else:
            return self.split_binary_file(file_path, max_size_bytes)
    
    def split_csv_file(self, file_path, max_size_bytes):
        """Split CSV file by rows to preserve data structure"""
        import csv
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        base_name, ext = os.path.splitext(file_name)
        
        chunk_number = 1
        current_size = 0
        current_rows = []
        header_row = None
        
        print(f"Smart CSV splitting: {file_name}")
        
        # Detect CSV dialect and read file
        with open(file_path, 'r', newline='', encoding='utf-8') as input_file:
            # Read first few lines to detect dialect
            sample = input_file.read(8192)
            input_file.seek(0)
            
            try:
                dialect = csv.Sniffer().sniff(sample)
            except csv.Error:
                # Fallback to excel dialect
                dialect = csv.excel
            
            reader = csv.reader(input_file, dialect)
            
            # Read header
            try:
                header_row = next(reader)
                header_size = len(','.join(header_row).encode('utf-8'))
                print(f"  Header: {len(header_row)} columns")
            except StopIteration:
                print("  Warning: Empty file")
                return 0
            
            # Process data rows
            row_count = 0
            for row in reader:
                row_count += 1
                row_text = ','.join(row)
                row_size = len(row_text.encode('utf-8'))
                
                # Check if adding this row would exceed size limit
                if (current_size + row_size + header_size) > max_size_bytes and current_rows:
                    # Write current chunk
                    self.write_csv_chunk(file_dir, base_name, ext, chunk_number, header_row, current_rows)
                    chunk_number += 1
                    current_rows = []
                    current_size = header_size  # Reset with header size
                
                current_rows.append(row)
                current_size += row_size
                
                if row_count % 10000 == 0:
                    print(f"  Processed {row_count:,} rows...")
        
        # Write final chunk if there are remaining rows
        if current_rows:
            self.write_csv_chunk(file_dir, base_name, ext, chunk_number, header_row, current_rows)
        
        # Create info file
        self.create_csv_info_file(file_path, chunk_number, row_count)
        
        print(f"✅ CSV split complete: {chunk_number} parts, {row_count:,} total rows")
        return chunk_number
    
    def write_csv_chunk(self, file_dir, base_name, ext, chunk_number, header_row, data_rows):
        """Write a single CSV chunk with header"""
        import csv
        
        chunk_filename = f"{base_name}.part{chunk_number:03d}{ext}"
        chunk_path = os.path.join(file_dir, chunk_filename)
        
        with open(chunk_path, 'w', newline='', encoding='utf-8') as chunk_file:
            writer = csv.writer(chunk_file)
            
            # Write header
            writer.writerow(header_row)
            
            # Write data rows
            writer.writerows(data_rows)
        
        chunk_size = os.path.getsize(chunk_path)
        print(f"  Created: {chunk_filename} ({chunk_size/1024/1024:.1f}MB, {len(data_rows):,} rows)")
    
    def create_csv_info_file(self, file_path, total_chunks, total_rows):
        """Create info file for CSV splits"""
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        base_name, ext = os.path.splitext(file_name)
        
        info_filename = f"{base_name}.split_info.txt"
        info_path = os.path.join(file_dir, info_filename)
        
        original_size = os.path.getsize(file_path)
        
        with open(info_path, 'w') as info_file:
            info_file.write(f"File Type: CSV (Smart Split)\n")
            info_file.write(f"Original file: {file_name}\n")
            info_file.write(f"Original size: {original_size} bytes ({original_size/1024/1024:.1f}MB)\n")
            info_file.write(f"Total chunks: {total_chunks}\n")
            info_file.write(f"Total rows: {total_rows:,}\n")
            info_file.write(f"Split method: Row-based (preserves data structure)\n")
            info_file.write(f"Headers: Included in each chunk\n")
            info_file.write(f"Split date: {os.path.getctime(file_path)}\n")
            info_file.write(f"\nNote: Each chunk contains headers and can be processed independently. Use pandas.concat() in Python or rbind() in R to recombine if needed.\n")
            info_file.write(f"\nChunk files:\n")
            for i in range(1, total_chunks + 1):
                chunk_name = f"{base_name}.part{i:03d}{ext}"
                info_file.write(f"  {chunk_name}\n")
            info_file.write(f"\nTo rejoin CSV files:\n")
            info_file.write(f"1. Use FNT GUI: GitHub Preprocessing -> File Joiner\n")
            info_file.write(f"2. Python: pd.concat([pd.read_csv(f) for f in chunk_files])\n")
            info_file.write(f"3. R: rbind(read.csv('part001'), read.csv('part002'), ...)\n")
    
    def split_binary_file(self, file_path, max_size_bytes):
        """Split non-CSV files using byte-level splitting (original method)"""
    def split_binary_file(self, file_path, max_size_bytes):
        """Split non-CSV files using byte-level splitting (original method)"""
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        base_name, ext = os.path.splitext(file_name)
        
        chunk_number = 1
        
        print(f"Binary splitting: {file_name}")
        
        with open(file_path, 'rb') as input_file:
            while True:
                chunk_data = input_file.read(max_size_bytes)
                if not chunk_data:
                    break
                
                # Create chunk filename
                chunk_filename = f"{base_name}.part{chunk_number:03d}{ext}"
                chunk_path = os.path.join(file_dir, chunk_filename)
                
                # Write chunk
                with open(chunk_path, 'wb') as chunk_file:
                    chunk_file.write(chunk_data)
                
                print(f"  Created: {chunk_filename} ({len(chunk_data)/1024/1024:.1f}MB)")
                chunk_number += 1
        
        # Create info file for binary splits
        info_filename = f"{base_name}.split_info.txt"
        info_path = os.path.join(file_dir, info_filename)
        
        total_chunks = chunk_number - 1
        original_size = os.path.getsize(file_path)
        
        with open(info_path, 'w') as info_file:
            info_file.write(f"File Type: Binary\n")
            info_file.write(f"Original file: {file_name}\n")
            info_file.write(f"Original size: {original_size} bytes ({original_size/1024/1024:.1f}MB)\n")
            info_file.write(f"Total chunks: {total_chunks}\n")
            info_file.write(f"Chunk size: {max_size_bytes} bytes\n")
            info_file.write(f"Split date: {os.path.getctime(file_path)}\n")
            info_file.write(f"\nNote: These chunks can be processed individually or recombined using standard tools like pandas (for CSV) or file concatenation commands.\n")
        
        print(f"✅ Binary split complete: {total_chunks} chunks")
        return total_chunks
    

    
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
    
    # Windows taskbar icon fix - must be set before creating QApplication
    try:
        import ctypes
        myappid = 'fnt.fieldneurotoolbox.gui.01'  # arbitrary string
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except:
        pass
    
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("FieldNeuroToolbox")
    app.setApplicationVersion("0.1")
    app.setOrganizationName("FNT")
    
    # Set application icon (check multiple possible locations)
    icon_paths = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'icons', 'fnt_icon.ico'),
        os.path.join(os.path.dirname(__file__), 'icons', 'fnt_icon.ico'),
    ]
    
    for icon_path in icon_paths:
        if os.path.exists(icon_path):
            app.setWindowIcon(QIcon(icon_path))
            break
    
    # Create and show main window
    window = FNTMainWindow()
    window.show()
    
    # Start the event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
