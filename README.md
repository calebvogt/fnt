[![build](https://github.com/calebvogt/fnt/actions/workflows/build.yml/badge.svg)](https://github.com/calebvogt/fnt/actions/workflows/build.yml)

# Field Neuroethology Toolbox (fnt)

Preprocessing and analysis toolbox for field and laboratory neurobehavioral data. This software is fully open source and made available freely to the research/hobbyist community. The software is released "as-is", and will continue to be updated as our own research needs evolve. Please open an issue for bugs or for feature requests! Thanks for checking us out! 

## Tutorials
I will be adding video tutorials to the following YouTube playlist: https://youtube.com/playlist?list=PLY8yLegR_viXsARZr460L3ZHRp3HZsOOC&si=FW3vPa8RUeZO-vz9

Subscribe for updates!

## Getting Started

### Standalone Executables (Recommended)

The easiest way to use FieldNeuroethologyToolbox is by downloading the pre-built standalone executable for your operating system. **You do not need to install Python or Anaconda to use this version.**

**Requirements for the Executable:**

- **ffmpeg**: Must be installed and added to your system PATH for video processing features to work. For windows, see here: https://www.youtube.com/watch?v=6sim9aF3g2c

1. Navigate to the [Releases](https://github.com/calebvogt/fnt/releases) page on the GitHub repository.
2. Download the latest release `.zip` or `.tar.gz` for your operating system (Windows, macOS, or Linux).
3. Extract the downloaded archive.
4. Run the extracted `fnt` executable to launch the GUI.
   * **macOS Security Permission:** Since the executable is unsigned, macOS Gatekeeper will block it. To fix this, open Terminal and run `xattr -cr path/to/fnt` on the extracted `fnt` file (you can drag it from Finder into Terminal to paste the path). This removes the quarantine attribute from the executable and all of its bundled dependencies.
   * **Windows SmartScreen:** When launching, you may see a "Windows protected your PC" warning. Click **"More info"** and then select **"Run anyway"** to allow the application to launch.


## Development Installation

If you intend to modify the code or prefer managing your own Python environment, use this method.

### Prerequisites

- **Git** — must be available on the command line (on your system PATH). Installing **GitHub Desktop alone is not sufficient**, because it bundles its own git that is not exposed to the terminal. The easiest fix is to install git into the conda environment with `conda install git` (included in the steps below).
- Anaconda or Miniconda
- ffmpeg — required for audio/video processing (used by USV and video tools). On **Windows**, install ffmpeg system-wide (see Windows instructions below). On **macOS/Linux**, `conda install -c conda-forge ffmpeg` works and is included in the steps below. ffmpeg is **not** a pip dependency, so `pip install -e .` alone does not provide it.
- Package dependencies are listed in `pyproject.toml`

### Windows Installation

Clone the fnt GitHub repo and install the package in editable mode. **Python 3.12 or newer is required** (3.13 recommended; the codebase uses Python 3.12+ syntax).

#### 1. Install ffmpeg system-wide

**Do not use `conda install ffmpeg`** — the conda-forge ffmpeg package on Windows has a known DLL conflict (`libintl` / `gdk-pixbuf`) that causes it to crash on install and at runtime.

Instead, install ffmpeg system-wide:

1. Download the **"release essentials"** zip from https://www.gyan.dev/ffmpeg/builds/
2. Extract it to a permanent location (e.g. `C:\ffmpeg`)
3. Add the `bin` folder (e.g. `C:\ffmpeg\bin`) to your system PATH:
   **Settings → System → About → Advanced system settings → Environment Variables → Path → New**
4. Open a **new** terminal and verify: `ffmpeg -version`

#### 2. Create the conda environment and install

```bash
git clone https://github.com/calebvogt/fnt.git
conda create --name fnt python=3.13
conda activate fnt
conda install git -y
cd path\to\fnt
pip install -e .
```

> **Note:** `conda install git` ensures a command-line git is available inside the environment. This is required because one dependency (SAM2) is installed directly from GitHub and pip needs `git` on the PATH to clone it. If git is missing you will see `ERROR: Cannot find command 'git'` during install.

#### GPU Support for Video Tracking (Recommended)

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

### Mac Installation

Open terminal:

```bash
git clone https://github.com/calebvogt/fnt.git
conda create --name fnt python=3.13
conda activate fnt
conda install git -y
conda install -c conda-forge ffmpeg -y
cd path/to/fnt
pip install -e .
```

### Launch the GUI

After installation, launch the FieldNeuroethologyToolbox GUI:

```bash
conda activate fnt
fnt
```

### Updating

Because the package is installed in editable mode, updating is simple:

```bash
cd path/to/fnt
git pull
```

That's it — the updated code is immediately available the next time you run `fnt`.

**Note:** If the update includes changes to `pyproject.toml` (e.g., new dependencies), you will need to re-run the install:

```bash
conda activate fnt
pip install -e .
```

## License

FieldNeuroethologyToolbox is released under the **GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)**. See the [LICENSE](LICENSE) file for the full text.

In short:

- You are free to use, modify, and redistribute this software, including for commercial purposes.
- If you distribute a modified version, you must release your source code under the same license.
- If you run a modified version as a network service, you must offer its source code to users of that service.

If you use FNT in published research, please cite this repository: [github.com/calebvogt/fnt](https://github.com/calebvogt/fnt)
