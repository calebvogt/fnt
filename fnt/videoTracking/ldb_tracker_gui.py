#!/usr/bin/env python3
"""
Light-Dark Box (LDB) Tracker GUI

Interactive SAM-based tracking for light-dark box anxiety tests.
Handles occlusion when animal enters dark compartment.

User workflow:
1. Select video file(s)
2. Click on animal in first frame -> SAM segments automatically
3. Draw rectangular ROIs for light zone, dark zone, and entrance
4. Track with zone-based occlusion handling
5. Export trajectory with anxiety metrics

Features:
- Zone-based tracking (light/dark/entrance)
- Occlusion handling in dark compartment
- Entry/exit detection
- Anxiety metrics (time in light/dark, transitions, latency)
- CSV export with behavioral data

Author: FieldNeuroToolbox Contributors
"""

import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QMessageBox,
    QGroupBox, QLineEdit, QCheckBox, QSpinBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

try:
    from .sam_tracker_base import SAMTrackerBase, calculate_distance_traveled, calculate_time_in_zone, detect_zone_transitions
    SAM_TRACKER_AVAILABLE = True
except ImportError:
    SAM_TRACKER_AVAILABLE = False
    print("Warning: SAM tracker base not available")


class LDBTrackerGUI(QMainWindow):
    """Main GUI for Light-Dark Box tracking - PLACEHOLDER"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Light-Dark Box Tracker - FieldNeuroToolbox [Coming Soon]")
        self.setGeometry(100, 100, 800, 600)
        
        # Apply dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #cccccc;
            }
            QLabel {
                color: #cccccc;
                font-size: 14px;
                padding: 20px;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
        """)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize placeholder UI."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setAlignment(Qt.AlignCenter)
        
        # Title
        title = QLabel("<h1>🚧 Light-Dark Box Tracker</h1>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Status message
        message = QLabel(
            "<h2>Coming Soon!</h2>"
            "<p>The Light-Dark Box tracker is currently under development.</p>"
            "<p><b>Planned Features:</b></p>"
            "<ul>"
            "<li>Rectangular ROI definition for light zone, dark zone, and entrance</li>"
            "<li>Zone-based tracking with occlusion handling</li>"
            "<li>Entry/exit detection with timestamp precision</li>"
            "<li>Anxiety metrics: time in light/dark, number of transitions, latency to enter dark</li>"
            "<li>CSV export with zone occupancy data</li>"
            "</ul>"
            "<p><b>Current Status:</b> The Open Field Test tracker is fully functional.<br>"
            "Use that tool to test SAM-based tracking workflow.</p>"
        )
        message.setAlignment(Qt.AlignCenter)
        message.setWordWrap(True)
        message.setStyleSheet("color: #cccccc; padding: 40px; max-width: 600px;")
        layout.addWidget(message)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        close_btn.setMaximumWidth(200)
        layout.addWidget(close_btn, alignment=Qt.AlignCenter)


def main():
    """Run the LDB Tracker GUI."""
    app = QApplication(sys.argv)
    
    window = LDBTrackerGUI()
    window.show()
    
    # Show info dialog
    QMessageBox.information(
        window,
        "LDB Tracker - Coming Soon",
        "The Light-Dark Box tracker is under development.\n\n"
        "The Open Field Test tracker is fully functional and demonstrates\n"
        "the SAM-based tracking workflow that will be used for LDB.\n\n"
        "Stay tuned for updates!"
    )
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
