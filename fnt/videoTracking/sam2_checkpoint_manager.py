#!/usr/bin/env python3
"""
SAM 2 Checkpoint Manager

Handles downloading and managing SAM 2 model checkpoints.
Provides user-friendly dialogs for checkpoint management.

Author: FieldNeuroToolbox Contributors
"""

import os
from pathlib import Path
from typing import Optional, Dict

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QComboBox, QFileDialog, QMessageBox, QCheckBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal


# SAM 2.1 model checkpoints
SAM2_CHECKPOINTS = {
    "sam2.1_hiera_tiny.pt": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        "config": "sam2.1_hiera_t.yaml",
        "size_mb": 155,
        "speed_fps": 91.2,
        "description": "Tiny (155 MB) - Fastest, good for quick tests"
    },
    "sam2.1_hiera_small.pt": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "config": "sam2.1_hiera_s.yaml",
        "size_mb": 184,
        "speed_fps": 84.8,
        "description": "Small (184 MB) - Fast with good accuracy"
    },
    "sam2.1_hiera_base_plus.pt": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "config": "sam2.1_hiera_b+.yaml",
        "size_mb": 323,
        "speed_fps": 64.1,
        "description": "Base+ (323 MB) - Balanced speed and accuracy"
    },
    "sam2.1_hiera_large.pt": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "config": "sam2.1_hiera_l.yaml",
        "size_mb": 897,
        "speed_fps": 39.5,
        "description": "Large (897 MB) - Best accuracy, slower"
    }
}


class DownloadThread(QThread):
    """Thread for downloading checkpoint files."""
    
    progress_signal = pyqtSignal(int, int)  # (downloaded_mb, total_mb)
    finished_signal = pyqtSignal(str)  # (file_path)
    error_signal = pyqtSignal(str)  # (error_message)
    
    def __init__(self, url: str, output_path: Path):
        super().__init__()
        self.url = url
        self.output_path = output_path
        self.is_cancelled = False
        
    def run(self):
        """Download file with progress reporting."""
        try:
            if not REQUESTS_AVAILABLE:
                self.error_signal.emit("requests library not installed. Install with: pip install requests")
                return
                
            # Create directory if it doesn't exist
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download with streaming
            response = requests.get(self.url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            total_mb = total_size / (1024 * 1024)
            downloaded = 0
            
            with open(self.output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if self.is_cancelled:
                        f.close()
                        if self.output_path.exists():
                            self.output_path.unlink()
                        return
                        
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        downloaded_mb = downloaded / (1024 * 1024)
                        self.progress_signal.emit(int(downloaded_mb), int(total_mb))
            
            self.finished_signal.emit(str(self.output_path))
            
        except Exception as e:
            self.error_signal.emit(f"Download failed: {str(e)}")
            
    def cancel(self):
        """Cancel download."""
        self.is_cancelled = True


class SAM2CheckpointDialog(QDialog):
    """Dialog for downloading SAM 2 checkpoints."""
    
    def __init__(self, default_dir: Optional[Path] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SAM 2 Model Checkpoints")
        self.setMinimumWidth(600)
        
        # Default to SAM_models in FNT repo
        if default_dir is None:
            # Assume we're in fnt/videoTracking, go up two levels
            default_dir = Path(__file__).parent.parent.parent / "SAM_models"
        
        self.checkpoint_dir = default_dir
        self.selected_checkpoint = None
        self.download_thread = None
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI components."""
        layout = QVBoxLayout()
        
        # Title and explanation
        title = QLabel("<h2>SAM 2 Model Setup</h2>")
        layout.addWidget(title)
        
        explanation = QLabel(
            "The Mask Tracker requires a SAM 2 model checkpoint to run.\n"
            "Select a model size based on your GPU and speed requirements.\n"
            "Models will be saved for future use."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        
        # Model selection
        model_group = QLabel("<b>Select Model:</b>")
        layout.addWidget(model_group)
        
        self.model_combo = QComboBox()
        for checkpoint_name, info in SAM2_CHECKPOINTS.items():
            display_text = f"{checkpoint_name.replace('.pt', '')} - {info['description']}"
            self.model_combo.addItem(display_text, checkpoint_name)
        
        # Default to base_plus (good balance)
        self.model_combo.setCurrentIndex(2)
        layout.addWidget(self.model_combo)
        
        # Download location
        location_layout = QHBoxLayout()
        location_label = QLabel("<b>Save Location:</b>")
        layout.addWidget(location_label)
        
        self.location_display = QLabel(str(self.checkpoint_dir))
        self.location_display.setStyleSheet("padding: 5px; background-color: #2b2b2b; border: 1px solid #3f3f3f;")
        location_layout.addWidget(self.location_display)
        
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_location)
        location_layout.addWidget(self.browse_btn)
        
        layout.addLayout(location_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.download_btn = QPushButton("Download")
        self.download_btn.clicked.connect(self.start_download)
        button_layout.addWidget(self.download_btn)
        
        self.skip_btn = QPushButton("Skip (Use Existing)")
        self.skip_btn.clicked.connect(self.skip_download)
        button_layout.addWidget(self.skip_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
        # Check for existing checkpoints
        self.check_existing_checkpoints()
        
    def check_existing_checkpoints(self):
        """Check if any checkpoints already exist."""
        if not self.checkpoint_dir.exists():
            return
            
        existing = []
        for checkpoint_name in SAM2_CHECKPOINTS.keys():
            checkpoint_path = self.checkpoint_dir / checkpoint_name
            if checkpoint_path.exists():
                existing.append(checkpoint_name)
        
        if existing:
            self.status_label.setText(
                f"✓ Found existing checkpoints: {', '.join([c.replace('.pt', '') for c in existing])}"
            )
            self.status_label.setStyleSheet("color: #4CAF50;")
            
    def browse_location(self):
        """Browse for save location."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Checkpoint Save Location",
            str(self.checkpoint_dir)
        )
        
        if directory:
            self.checkpoint_dir = Path(directory)
            self.location_display.setText(str(self.checkpoint_dir))
        if not REQUESTS_AVAILABLE:
            QMessageBox.critical(
                self,
                "Missing Dependency",
                "The 'requests' library is required for downloading.\n\n"
                "Please install it with:\n\n"
                "pip install requests\n\n"
                "Then restart the application."
            )
            return
            
            self.check_existing_checkpoints()
            
    def start_download(self):
        """Start downloading selected checkpoint."""
        checkpoint_name = self.model_combo.currentData()
        checkpoint_info = SAM2_CHECKPOINTS[checkpoint_name]
        
        output_path = self.checkpoint_dir / checkpoint_name
        
        # Check if already exists
        if output_path.exists():
            reply = QMessageBox.question(
                self,
                "File Exists",
                f"{checkpoint_name} already exists. Re-download?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                self.selected_checkpoint = output_path
                self.accept()
                return
        
        # Start download
        self.download_btn.setEnabled(False)
        self.skip_btn.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.model_combo.setEnabled(False)
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(checkpoint_info['size_mb'])
        
        self.status_label.setText(f"Downloading {checkpoint_name}...")
        self.status_label.setStyleSheet("color: #2196F3;")
        
        # Create download thread
        self.download_thread = DownloadThread(checkpoint_info['url'], output_path)
        self.download_thread.progress_signal.connect(self.update_progress)
        self.download_thread.finished_signal.connect(self.download_finished)
        self.download_thread.error_signal.connect(self.download_error)
        self.download_thread.start()
        
    def update_progress(self, downloaded_mb: int, total_mb: int):
        """Update progress bar."""
        self.progress_bar.setValue(downloaded_mb)
        self.status_label.setText(f"Downloading: {downloaded_mb} / {total_mb} MB")
        
    def download_finished(self, file_path: str):
        """Handle download completion."""
        self.selected_checkpoint = Path(file_path)
        self.status_label.setText("✓ Download complete!")
        self.status_label.setStyleSheet("color: #4CAF50;")
        
        QMessageBox.information(
            self,
            "Download Complete",
            f"Checkpoint saved to:\n{file_path}\n\nYou can now use the Mask Tracker!"
        )
        
        self.accept()
        
    def download_error(self, error_message: str):
        """Handle download error."""
        self.status_label.setText(f"✗ {error_message}")
        self.status_label.setStyleSheet("color: #f44336;")
        
        self.download_btn.setEnabled(True)
        self.skip_btn.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        QMessageBox.critical(self, "Download Error", error_message)
        
    def skip_download(self):
        """Skip download and select existing checkpoint."""
        # Look for any existing checkpoint
        if not self.checkpoint_dir.exists():
            QMessageBox.warning(
                self,
                "No Checkpoints",
                "No checkpoint directory found. Please download a model first."
            )
            return
            
        existing_checkpoints = [
            self.checkpoint_dir / name
            for name in SAM2_CHECKPOINTS.keys()
            if (self.checkpoint_dir / name).exists()
        ]
        
        if not existing_checkpoints:
            QMessageBox.warning(
                self,
                "No Checkpoints",
                "No checkpoints found in the selected directory. Please download one first."
            )
            return
        
        # Use first available checkpoint
        self.selected_checkpoint = existing_checkpoints[0]
        self.accept()
        
    def get_checkpoint_path(self) -> Optional[Path]:
        """Get selected checkpoint path."""
        return self.selected_checkpoint
        
    def get_config_name(self) -> Optional[str]:
        """Get config file name for selected checkpoint."""
        if self.selected_checkpoint:
            checkpoint_name = self.selected_checkpoint.name
            return SAM2_CHECKPOINTS.get(checkpoint_name, {}).get("config")
        return None


def get_sam2_checkpoint(parent=None) -> tuple[Optional[Path], Optional[str]]:
    """
    Show dialog to get SAM 2 checkpoint.
    
    Returns:
        (checkpoint_path, config_name) or (None, None) if cancelled
    """
    dialog = SAM2CheckpointDialog(parent=parent)
    
    if dialog.exec_() == QDialog.Accepted:
        return dialog.get_checkpoint_path(), dialog.get_config_name()
    
    return None, None
