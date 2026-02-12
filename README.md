
# FieldNeuroToolbox (fnt)

Preprocessing and analysis toolbox for neurobehavioral data.

Authored by Caleb C. Vogt, PhD in collaboration with Claude.

## Pre-requisites
- Git
- Anaconda
- ffmpeg (and the bin file saved to your system PATH)
- GitHub Desktop (suggested)

## Windows Installation

Clone the fnt GitHub repo and install the package in editable mode:

```bash
conda create --name fnt python
conda activate fnt
cd path\to\fnt
pip install -e .
```

### GPU Support for Video Tracking (Recommended)

For SAM-based video tracking, GPU acceleration provides ~50x speedup. Install PyTorch with CUDA support:

```bash
conda activate fnt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Note**: This requires an NVIDIA GPU with CUDA support. The tracker will work on CPU but will be significantly slower (~10s per SAM frame vs ~0.1s on GPU).

To verify GPU detection:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

You should see `CUDA available: True` and your GPU name.

## Mac Installation

Open terminal:

```bash
conda create --name fnt python
conda activate fnt
cd path/to/fnt
pip install -e .
```

## Launch the GUI

After installation, launch the FieldNeuroToolbox GUI:

```bash
conda activate fnt
fnt-gui
```

## Updating

Because the package is installed in editable mode, updating is simple:

```bash
cd path/to/fnt
git pull
```

That's it — the updated code is immediately available the next time you run `fnt-gui`.

**Note:** If the update includes changes to `pyproject.toml` (e.g., new dependencies), you will need to re-run the install:

```bash
conda activate fnt
pip install -e .
```

## Notes
- Package dependencies are listed in pyproject.toml
