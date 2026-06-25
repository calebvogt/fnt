#!/usr/bin/env python3
"""
Model Checkpoint Manager

Handles downloading and managing pretrained model checkpoints (SAM2, YOLO).
Provides user-friendly dialogs for checkpoint management.
All models are stored in ``LocalModels/`` at the FNT repo root
(auto-added to .gitignore).

Author: FieldNeuroethologyToolbox Contributors
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
    QProgressBar, QComboBox, QFileDialog, QMessageBox, QRadioButton,
    QButtonGroup, QGroupBox,
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

# YOLO pretrained model checkpoints
YOLO_CHECKPOINTS = {
    "yolo11n-seg.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt",
        "size_mb": 6,
        "description": "YOLOv11-nano-seg (6 MB) - Fastest, recommended for field use",
    },
    "yolo11s-seg.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt",
        "size_mb": 12,
        "description": "YOLOv11-small-seg (12 MB) - Higher accuracy, slightly slower",
    },
    "yolo11m-seg.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-seg.pt",
        "size_mb": 44,
        "description": "YOLOv11-medium-seg (44 MB) - More capacity, better accuracy",
    },
    "yolo11l-seg.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-seg.pt",
        "size_mb": 54,
        "description": "YOLOv11-large-seg (54 MB) - High accuracy, needs more data",
    },
}

# FNT repo root (go up from fnt/videoTracking/ to repo root)
_FNT_REPO_ROOT = Path(__file__).parent.parent.parent

# Canonical local model directory (auto-added to .gitignore)
LOCAL_MODELS_DIR = _FNT_REPO_ROOT / "LocalModels"

# Legacy directories (checked for backward compatibility)
_LEGACY_DIRS = [
    _FNT_REPO_ROOT / "sam_models_local",
    _FNT_REPO_ROOT / "SAM_models",
]


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
        try:
            if not REQUESTS_AVAILABLE:
                self.error_signal.emit("requests library not installed. Install with: pip install requests")
                return

            self.output_path.parent.mkdir(parents=True, exist_ok=True)

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
        self.is_cancelled = True


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


def _find_existing_checkpoints(*search_dirs: Path,
                               checkpoint_dict: Optional[Dict] = None) -> Dict[str, Path]:
    """Search multiple directories for existing checkpoints.

    Args:
        search_dirs: Directories to search (searched in order, first hit wins).
        checkpoint_dict: Dict of checkpoint names to look for.
            Defaults to SAM2_CHECKPOINTS if not given.
    """
    if checkpoint_dict is None:
        checkpoint_dict = SAM2_CHECKPOINTS
    found = {}
    for d in search_dirs:
        if not d.exists():
            continue
        for name in checkpoint_dict:
            p = d / name
            if p.exists() and p.stat().st_size > 0:
                if name not in found:
                    found[name] = p
    return found


def find_yolo_checkpoint(model_name: str) -> Optional[Path]:
    """Find a YOLO pretrained checkpoint in LocalModels or Ultralytics cache.

    Returns the path if found, None otherwise.
    """
    local = LOCAL_MODELS_DIR / model_name
    if local.exists() and local.stat().st_size > 0:
        return local

    for legacy in _LEGACY_DIRS:
        p = legacy / model_name
        if p.exists() and p.stat().st_size > 0:
            return p

    try:
        from ultralytics.utils import SETTINGS as ul_settings
        cache_dir = Path(ul_settings.get("weights_dir", ""))
        if cache_dir.exists():
            p = cache_dir / model_name
            if p.exists() and p.stat().st_size > 0:
                return p
    except Exception:
        pass

    home_ul = Path.home() / ".config" / "Ultralytics"
    for candidate in [home_ul, Path.home() / ".ultralytics"]:
        if candidate.exists():
            p = candidate / model_name
            if p.exists() and p.stat().st_size > 0:
                return p

    return None


def ensure_yolo_checkpoint(model_name: str) -> Path:
    """Return path to a YOLO checkpoint, downloading if necessary.

    Raises FileNotFoundError with a helpful message if offline and
    the checkpoint is not cached.
    """
    existing = find_yolo_checkpoint(model_name)
    if existing is not None:
        return existing

    LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_gitignore_entry(_FNT_REPO_ROOT, "LocalModels/")
    dest = LOCAL_MODELS_DIR / model_name

    info = YOLO_CHECKPOINTS.get(model_name, {})
    url = info.get("url")
    if not url:
        url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{model_name}"

    print(f"[FNT] Downloading {model_name} to {dest} ...")

    try:
        if REQUESTS_AVAILABLE:
            resp = requests.get(url, stream=True, timeout=30)
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"[FNT] Download complete: {dest}")
            return dest
        else:
            import urllib.request
            urllib.request.urlretrieve(url, str(dest))
            print(f"[FNT] Download complete: {dest}")
            return dest
    except Exception as e:
        if dest.exists():
            dest.unlink()
        raise FileNotFoundError(
            f"Cannot find or download YOLO pretrained weights '{model_name}'.\n\n"
            f"This is needed for transfer learning (COCO pre-trained weights).\n"
            f"You appear to be offline or the download failed:\n  {e}\n\n"
            f"To fix: connect to the internet and try again, or manually\n"
            f"download from:\n  {url}\n"
            f"and place it in:\n  {LOCAL_MODELS_DIR}/{model_name}"
        ) from e


class SAM2CheckpointDialog(QDialog):
    """Dialog for selecting / downloading SAM 2 checkpoints."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SAM 2 Model Setup")
        self.setMinimumWidth(620)

        self.selected_checkpoint: Optional[str] = None
        self.download_thread: Optional[DownloadThread] = None

        self._default_local_dir = LOCAL_MODELS_DIR
        self._custom_dir: Optional[Path] = None

        self._existing = _find_existing_checkpoints(
            self._default_local_dir,
            *_LEGACY_DIRS,
        )

        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Title
        title = QLabel("<h2>SAM 2 Model Setup</h2>")
        layout.addWidget(title)

        explanation = QLabel(
            "Select a SAM2 model to download. The Tiny model is recommended "
            "for annotation (fast, lower accuracy). The Large model provides "
            "the best segmentation quality but is slower and larger."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        # Model selection
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout()

        self.model_combo = QComboBox()
        self.model_combo.addItem(
            "SAM2.1 Tiny (155 MB) - Fast, recommended for annotation",
            "sam2.1_hiera_tiny.pt",
        )
        self.model_combo.addItem(
            "SAM2.1 Large (897 MB) - Best accuracy, slower",
            "sam2.1_hiera_large.pt",
        )
        self.model_combo.setCurrentIndex(0)
        model_layout.addWidget(self.model_combo)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Save location
        loc_group = QGroupBox("Save Location")
        loc_layout = QVBoxLayout()

        self._radio_group = QButtonGroup(self)
        self.radio_local = QRadioButton(
            "FNT repo: LocalModels/  (auto-added to .gitignore)"
        )
        self.radio_local.setChecked(True)
        self._radio_group.addButton(self.radio_local)
        loc_layout.addWidget(self.radio_local)

        browse_row = QHBoxLayout()
        self.radio_custom = QRadioButton("Custom location:")
        self._radio_group.addButton(self.radio_custom)
        browse_row.addWidget(self.radio_custom)

        self.lbl_custom_path = QLabel("(not set)")
        self.lbl_custom_path.setStyleSheet("color: #999999;")
        browse_row.addWidget(self.lbl_custom_path, 1)

        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.setMaximumWidth(100)
        self.btn_browse.clicked.connect(self._browse_custom)
        browse_row.addWidget(self.btn_browse)
        loc_layout.addLayout(browse_row)

        loc_group.setLayout(loc_layout)
        layout.addWidget(loc_group)

        # Existing checkpoints
        if self._existing:
            names = [n.replace(".pt", "") for n in self._existing]
            self.lbl_existing = QLabel(f"Found existing: {', '.join(names)}")
            self.lbl_existing.setStyleSheet("color: #4CAF50;")
        else:
            self.lbl_existing = QLabel("No existing checkpoints found")
            self.lbl_existing.setStyleSheet("color: #999999;")
        layout.addWidget(self.lbl_existing)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        # Buttons
        btn_layout = QHBoxLayout()
        self.download_btn = QPushButton("Download")
        self.download_btn.clicked.connect(self._start_download)
        btn_layout.addWidget(self.download_btn)

        self.use_existing_btn = QPushButton("Use Existing")
        self.use_existing_btn.clicked.connect(self._use_existing)
        self.use_existing_btn.setEnabled(bool(self._existing))
        btn_layout.addWidget(self.use_existing_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)

        layout.addLayout(btn_layout)

    def _get_save_dir(self) -> Path:
        if self.radio_custom.isChecked() and self._custom_dir:
            return self._custom_dir
        return self._default_local_dir

    def _browse_custom(self):
        d = QFileDialog.getExistingDirectory(self, "Select Checkpoint Save Location")
        if d:
            self._custom_dir = Path(d)
            self.lbl_custom_path.setText(d)
            self.lbl_custom_path.setStyleSheet("color: #cccccc;")
            self.radio_custom.setChecked(True)

            new_existing = _find_existing_checkpoints(self._custom_dir)
            if new_existing:
                self._existing.update(new_existing)
                names = [n.replace(".pt", "") for n in self._existing]
                self.lbl_existing.setText(f"Found existing: {', '.join(names)}")
                self.lbl_existing.setStyleSheet("color: #4CAF50;")
                self.use_existing_btn.setEnabled(True)

    def _start_download(self):
        if not REQUESTS_AVAILABLE:
            QMessageBox.critical(
                self, "Missing Dependency",
                "The 'requests' library is required for downloading.\n\n"
                "Install with: pip install requests"
            )
            return

        checkpoint_name = self.model_combo.currentData()
        checkpoint_info = SAM2_CHECKPOINTS[checkpoint_name]
        save_dir = self._get_save_dir()
        output_path = save_dir / checkpoint_name

        # If saving to repo-local dir, ensure gitignore
        if save_dir == self._default_local_dir:
            _ensure_gitignore_entry(_FNT_REPO_ROOT, "LocalModels/")

        if output_path.exists():
            reply = QMessageBox.question(
                self, "File Exists",
                f"{checkpoint_name} already exists at:\n{output_path}\n\nRe-download?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.No:
                self.selected_checkpoint = str(output_path)
                self.accept()
                return

        self.download_btn.setEnabled(False)
        self.use_existing_btn.setEnabled(False)
        self.btn_browse.setEnabled(False)
        self.model_combo.setEnabled(False)

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(checkpoint_info["size_mb"])

        self.status_label.setText(f"Downloading {checkpoint_name} ({checkpoint_info['size_mb']} MB)...")
        self.status_label.setStyleSheet("color: #2196F3;")

        self.download_thread = DownloadThread(checkpoint_info["url"], output_path)
        self.download_thread.progress_signal.connect(self._on_progress)
        self.download_thread.finished_signal.connect(self._on_download_finished)
        self.download_thread.error_signal.connect(self._on_download_error)
        self.download_thread.start()

    def _on_progress(self, downloaded_mb: int, total_mb: int):
        self.progress_bar.setValue(downloaded_mb)
        self.status_label.setText(f"Downloading: {downloaded_mb} / {total_mb} MB")

    def _on_download_finished(self, file_path: str):
        self.selected_checkpoint = file_path
        self.status_label.setText("Download complete!")
        self.status_label.setStyleSheet("color: #4CAF50;")
        QMessageBox.information(self, "Download Complete", f"Checkpoint saved to:\n{file_path}")
        self.accept()

    def _on_download_error(self, error_message: str):
        self.status_label.setText(f"Error: {error_message}")
        self.status_label.setStyleSheet("color: #f44336;")
        self.download_btn.setEnabled(True)
        self.use_existing_btn.setEnabled(bool(self._existing))
        self.btn_browse.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Download Error", error_message)

    def _use_existing(self):
        if not self._existing:
            QMessageBox.warning(self, "No Checkpoints", "No existing checkpoints found.")
            return

        # Prefer the model currently selected in the dropdown
        selected_name = self.model_combo.currentData()
        if selected_name in self._existing:
            self.selected_checkpoint = str(self._existing[selected_name])
            self.accept()
            return

        # Otherwise use the first available
        first_name = next(iter(self._existing))
        self.selected_checkpoint = str(self._existing[first_name])
        self.accept()

    def get_checkpoint_path(self) -> Optional[Path]:
        if self.selected_checkpoint:
            return Path(self.selected_checkpoint)
        return None

    def get_config_name(self) -> Optional[str]:
        if self.selected_checkpoint:
            checkpoint_name = Path(self.selected_checkpoint).name
            return SAM2_CHECKPOINTS.get(checkpoint_name, {}).get("config")
        return None


def get_sam2_checkpoint(parent=None) -> tuple:
    """Show dialog to get SAM 2 checkpoint.

    Returns:
        (checkpoint_path, config_name) or (None, None) if cancelled
    """
    dialog = SAM2CheckpointDialog(parent=parent)

    if dialog.exec_() == QDialog.Accepted:
        return dialog.get_checkpoint_path(), dialog.get_config_name()

    return None, None
