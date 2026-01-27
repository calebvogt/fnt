# CLAUDE.md - Instructions for Claude Code Sessions

This file provides context and patterns for Claude sessions working on the FieldNeuroToolbox (FNT) codebase.

## Project Overview

FNT (FieldNeuroToolbox) is a Python toolbox for preprocessing and analyzing neurobehavioral data. It provides PyQt5-based GUIs for various data processing tasks including:

- Video processing and SLEAP pose estimation
- Ultra-wideband (UWB) tracking analysis
- Ultrasonic vocalization (USV) detection and analysis
- RFID data processing
- Fiber photometry (Doric WiFP) processing

## Project Structure

```
fnt/
├── __init__.py           # Main package init (auto-imports all modules)
├── gui_pyqt.py          # Main GUI application with tabbed interface
├── CLAUDE.md            # This file
├── usv/                 # USV processing module
│   ├── usv_detector_pyqt.py      # USV detection GUI
│   ├── usv_inspector_pyqt.py     # USV ground-truthing GUI
│   ├── usv_detector/             # Core detection library
│   │   ├── config.py             # USVDetectorConfig dataclass
│   │   ├── dsp_detector.py       # DSP-based detection
│   │   ├── spectrogram.py        # Audio loading & spectrogram
│   │   ├── io.py                 # CSV/export utilities
│   │   └── batch.py              # Batch processing
│   ├── audio_trim_pyqt.py        # Audio trimming tool
│   └── compress_wavs_pyqt.py     # WAV compression tool
├── videoProcessing/     # Video tools
├── sleapProcessing/     # SLEAP pipeline tools
├── uwb/                 # UWB tracking tools
├── rfid/                # RFID processing tools
└── DoricFP/             # Fiber photometry tools
```

## PyQt5 Styling Patterns

### Dark Theme Stylesheet

All FNT GUIs use a consistent dark theme. Apply this base stylesheet:

```python
self.setStyleSheet("""
    QWidget {
        background-color: #2b2b2b;
        color: #cccccc;
        font-family: Arial;
    }
    QLabel {
        color: #cccccc;
        background-color: transparent;
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
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
        padding: 5px;
        border: 1px solid #3f3f3f;
        border-radius: 3px;
        background-color: #1e1e1e;
        color: #cccccc;
    }
    QProgressBar {
        border: 1px solid #3f3f3f;
        border-radius: 4px;
        text-align: center;
        background-color: #1e1e1e;
    }
    QProgressBar::chunk {
        background-color: #0078d4;
        border-radius: 3px;
    }
""")
```

### Title and Description Pattern

```python
# Title
title = QLabel("Tool Name")
title.setFont(QFont("Arial", 16, QFont.Bold))
title.setStyleSheet("color: #0078d4; background-color: transparent;")
layout.addWidget(title)

# Description
desc = QLabel("Brief description of what this tool does")
desc.setFont(QFont("Arial", 10))
desc.setStyleSheet("color: #999999; font-style: italic; margin-bottom: 10px;")
layout.addWidget(desc)
```

## File/Folder Selection Pattern

Use dual buttons for flexibility - "Add Folder" and "Add Files":

```python
# Button row
btn_layout = QHBoxLayout()

self.btn_add_folder = QPushButton("Add Folder")
self.btn_add_folder.clicked.connect(self.add_folder)
btn_layout.addWidget(self.btn_add_folder)

self.btn_add_files = QPushButton("Add Files")
self.btn_add_files.clicked.connect(self.add_files)
btn_layout.addWidget(self.btn_add_files)

self.btn_clear = QPushButton("Clear")
self.btn_clear.clicked.connect(self.clear_files)
self.btn_clear.setStyleSheet("background-color: #5c5c5c;")
btn_layout.addWidget(self.btn_clear)

btn_layout.addStretch()

# Implementation
def add_folder(self):
    folder = QFileDialog.getExistingDirectory(self, "Select Folder")
    if folder:
        files = list(Path(folder).glob("*.wav"))
        self.input_files.extend([str(f) for f in files])
        self.update_file_list()

def add_files(self):
    files, _ = QFileDialog.getOpenFileNames(
        self, "Select Files", "", "WAV Files (*.wav);;All Files (*.*)"
    )
    if files:
        self.input_files.extend(files)
        self.update_file_list()
```

## Progress Bar Best Practice

**CRITICAL**: Update progress bar AFTER file processing completes, not before.

```python
class Worker(QThread):
    progress = pyqtSignal(str, int, int)      # Before processing (for status text)
    file_complete = pyqtSignal(str, int, int)  # After completion (for progress bar)

    def run(self):
        for i, filepath in enumerate(self.files):
            filename = os.path.basename(filepath)

            # Emit BEFORE processing (for status update only)
            self.progress.emit(filename, i, total)

            # Do the actual processing
            result = process_file(filepath)

            # Emit AFTER completion (for progress bar)
            self.file_complete.emit(filename, i + 1, total)

# In main window:
self.worker.progress.connect(self.on_progress)        # Updates status label
self.worker.file_complete.connect(self.on_complete)   # Updates progress bar

def on_progress(self, filename, current, total):
    self.status_label.setText(f"Processing: {filename}")

def on_complete(self, filename, current, total):
    self.progress_bar.setValue(current)
```

## ADPCM Audio File Handling

Many ultrasonic recording systems (e.g., Avisoft) use ADPCM IMA WAV format. scipy.io.wavfile cannot read this. Use ffmpeg fallback:

```python
def load_audio(filepath):
    # Try soundfile first (handles most formats)
    try:
        import soundfile as sf
        audio, sr = sf.read(filepath, dtype='float32')
        return audio, sr
    except:
        pass

    # Fallback to ffmpeg for ADPCM
    return _load_with_ffmpeg(filepath)

def _load_with_ffmpeg(filepath):
    import subprocess
    import tempfile

    # Get sample rate
    probe_cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'stream=sample_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        filepath
    ]
    result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
    sr = int(result.stdout.strip())

    # Convert to raw PCM
    with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as tmp:
        temp_path = tmp.name

    try:
        convert_cmd = [
            'ffmpeg', '-i', filepath,
            '-f', 'f32le', '-acodec', 'pcm_f32le',
            '-ac', '1', '-y', temp_path
        ]
        subprocess.run(convert_cmd, capture_output=True, check=True)
        audio = np.fromfile(temp_path, dtype=np.float32)
    finally:
        os.unlink(temp_path)

    return audio, sr
```

## Configuration Dataclasses

Use dataclasses for configuration with JSON serialization:

```python
from dataclasses import dataclass, field, asdict
import json

@dataclass
class MyConfig:
    param1: float = 10.0
    param2: int = 100
    param3: str = "default"

    def to_json(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, filepath: str) -> 'MyConfig':
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)
```

## Common Issues and Solutions

### Import Errors with Main Package

The main `fnt/__init__.py` auto-imports all submodules. This can cause import errors if a dependency is missing (e.g., h5py). When testing specific modules, import them directly:

```python
# Instead of: from fnt.usv import usv_detector
# Use: from fnt.usv.usv_detector import DSPDetector
```

### QThread Blocking GUI

Always use QThread for long-running operations. Never do heavy processing in the main thread.

### Window References

Store references to child windows to prevent garbage collection:

```python
def open_tool(self):
    self.tool_window = ToolWindow()  # Store as instance variable
    self.tool_window.show()
```

## Adding New Tools to Main GUI

1. Create the tool in appropriate subdirectory (e.g., `fnt/usv/my_tool_pyqt.py`)
2. Add import and launch method in `gui_pyqt.py`:

```python
def run_my_tool(self):
    try:
        from fnt.usv.my_tool_pyqt import MyToolWindow
        self.my_tool_window = MyToolWindow()
        self.my_tool_window.show()
    except Exception as e:
        QMessageBox.critical(self, "Error", f"Failed: {str(e)}")
```

3. Add button to appropriate tab's button list:

```python
buttons = [
    ("My Tool", "Description of what it does", self.run_my_tool),
    # ... other buttons
]
```

## Testing Tools Standalone

Each PyQt tool should have a `main()` function for standalone testing:

```python
def main():
    app = QApplication(sys.argv)
    window = MyToolWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
```

Run with: `python -m fnt.usv.my_tool_pyqt`
