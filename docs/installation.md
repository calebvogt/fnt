# Installation

There are two ways to run FNT: the pre-built **standalone executable** (recommended for most users) or a **development installation** (if you want to modify the code).

## Standalone Executable (Recommended)

You do **not** need to install Python or Anaconda for this version.

### Requirements

- **ffmpeg** — must be installed and on your system PATH for audio/video features. On Windows, see [Installing ffmpeg on Windows](#installing-ffmpeg-on-windows) below.

### Steps

1. Download the latest release `.zip` or `.tar.gz` for your OS from the [Releases page](https://github.com/calebvogt/fnt/releases).
2. Extract the archive.
3. Run the extracted `fnt` executable to launch the GUI.

!!! warning "macOS security permission"
    Since the executable is unsigned, macOS Gatekeeper will block it. Open Terminal and run `xattr -cr path/to/fnt` on the extracted file (drag it from Finder into Terminal to paste the path). This removes the quarantine attribute from the executable and its bundled dependencies.

!!! warning "Windows SmartScreen"
    You may see a "Windows protected your PC" warning. Click **More info** → **Run anyway**.

## Development Installation

Use this method if you intend to modify the code or prefer managing your own Python environment.

### Prerequisites

- **Git** on the command line (on your PATH). Installing GitHub Desktop alone is **not** sufficient — it bundles a git that isn't exposed to the terminal. The steps below install git into the conda environment.
- **Anaconda or Miniconda**
- **Python 3.12 or newer** (3.13 recommended; the codebase uses 3.12+ syntax)
- **ffmpeg** — required for audio/video processing. Not a pip dependency, so `pip install -e .` alone does not provide it.

### Installing ffmpeg on Windows

!!! danger "Do not use `conda install ffmpeg` on Windows"
    The conda-forge ffmpeg package on Windows has a known DLL conflict (`libintl` / `gdk-pixbuf` / `fontconfig`) that causes it to crash during install and at runtime. Install ffmpeg system-wide instead.

1. Download the **"release essentials"** zip from [gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds/).
2. Extract it to a permanent location, e.g. `C:\ffmpeg`.
3. Add the `bin` folder (e.g. `C:\ffmpeg\bin`) to your system PATH:
   **Settings → System → About → Advanced system settings → Environment Variables → Path → New**.
4. Open a **new** terminal and verify: `ffmpeg -version`.

### Windows

```bat
git clone https://github.com/calebvogt/fnt.git
conda create --name fnt python=3.13
conda activate fnt
conda install git -y
cd path\to\fnt
pip install -e .
```

(ffmpeg is installed system-wide per the section above — do **not** `conda install ffmpeg`.)

### macOS / Linux

```bash
git clone https://github.com/calebvogt/fnt.git
conda create --name fnt python=3.13
conda activate fnt
conda install git -y
conda install -c conda-forge ffmpeg -y
cd path/to/fnt
pip install -e .
```

On macOS/Linux the conda-forge ffmpeg package works fine.

### Launch the GUI

```bash
conda activate fnt
fnt
```

## GPU Support (Optional)

For SAM-based video tracking and MAD model training/inference, GPU acceleration provides a large speedup. Install PyTorch with CUDA support:

```bash
conda activate fnt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Verify GPU detection:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

This requires an NVIDIA GPU with CUDA support. Tools run on CPU otherwise, but significantly slower.

## Updating

Because the package is installed in editable mode:

```bash
cd path/to/fnt
git pull
```

If the update changed `pyproject.toml` (new dependencies), re-run the install:

```bash
conda activate fnt
pip install -e .
```
