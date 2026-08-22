"""
SLEAP Inference Tool - PyQt5 Implementation

Run SLEAP inference with optional tracking on video files.
Supports both top-down (centroid + centered instance) and bottom-up models.
Automatically converts output to CSV format.
"""

import os
import re
import platform
import shutil
import subprocess
import glob
import tempfile
import threading
import time
from datetime import datetime

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QMessageBox, QGroupBox, QTextEdit, QCheckBox,
    QListWidget, QGridLayout, QFrame, QApplication, QComboBox, QDoubleSpinBox, QSpinBox,
    QAbstractItemView, QScrollArea, QListView, QTreeView, QDialog
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QFont, QTextCursor, QColor

# ---------------------------------------------------------------------------
# Console-output parsing
#
# SLEAP renders its progress with `rich`, which assumes a terminal: it emits
# ANSI colour/cursor codes, repaints the whole bar many times per second, and
# splits a single repaint across several \r-separated fragments.  Piped into a
# QTextEdit that turns into unreadable noise, so everything below exists to
# reduce that stream back down to one clean, self-formatted status line.
# ---------------------------------------------------------------------------

# ANSI escape sequences (colours, cursor movement, OSC titles).
_ANSI_RE = re.compile(r'\x1b[\[\(][0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07')

# Fields inside a progress repaint, e.g.
#   Predicting... ━━━━━━ 39% 7084/18000 ETA: 0:07:44 Elapsed: 0:06:51 23.8 FPS
_FRACTION_RE = re.compile(r'(\d[\d,]*)\s*/\s*(\d[\d,]*)')
_PERCENT_RE = re.compile(r'(\d{1,3})\s*%')
_ETA_RE = re.compile(r'ETA:?\s*(\d+:\d{2}(?::\d{2})?)')
_ELAPSED_RE = re.compile(r'Elapsed:?\s*(\d+:\d{2}(?::\d{2})?)')
_FPS_RE = re.compile(r'([\d.]+)\s*FPS')
_PHASE_RE = re.compile(r'(Predicting|Tracking|Inferring|Loading)', re.IGNORECASE)

# A fragment made up only of bar glyphs / repeated phase labels carries no
# information — it is a partial repaint and is dropped entirely.
_CHROME_RE = re.compile(
    r'(Predicting\.\.\.|Tracking\.\.\.|Inferring\.\.\.|[\s─━╸╺▁▔█░▏▎▍▌▋▊▉|.·\-–—])+')

PROGRESS_BAR_WIDTH = 24     # fixed width, so window resizing never reflows it
VIDEO_LABEL_WIDTH = 30      # video name column width in the status line


def detect_gpu():
    """Detect a GPU usable for inference. Returns (kind, name, detail).

    kind is 'cuda', 'metal' or None.  Detection is deliberately driver-level
    (nvidia-smi) rather than framework-level so it works regardless of which
    conda environment the GUI itself is running in.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
            if lines:
                parts = [p.strip() for p in lines[0].split(',')]
                name = parts[0]
                detail = f"{name} ({parts[1]})" if len(parts) > 1 else name
                if len(lines) > 1:
                    detail += f" (+{len(lines) - 1} more)"
                return 'cuda', name, detail
    except Exception:
        pass

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return 'metal', "Apple Silicon GPU", "Apple Silicon GPU (Metal)"

    return None, None, ""


# ---------------------------------------------------------------------------
# Network-drive detection
#
# Reading video frames over a network share is a common, quiet cause of slow
# inference: the GPU sits idle waiting on the network. We flag it so the user
# can choose to stage the videos on a local SSD first.
# ---------------------------------------------------------------------------

# Filesystem types that mean "not a local disk", for the POSIX code path.
_NETWORK_FSTYPES = {
    'nfs', 'nfs4', 'cifs', 'smb', 'smbfs', 'afpfs', 'ftpfs', 'webdav',
    'fuse.sshfs', 'fuse.davfs', 'fuse.rclone', 'fuse.gvfsd-fuse', '9p',
}


def _posix_path_is_network(path):
    """Best-effort network check on POSIX by matching the path to its mount."""
    try:
        real = os.path.realpath(path)
        entries = []  # (mountpoint, fstype)
        if os.path.exists('/proc/mounts'):
            with open('/proc/mounts', 'r', errors='ignore') as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 3:
                        entries.append((parts[1], parts[2]))
        else:
            # macOS / BSD: parse `mount` output, e.g.
            #   //user@host/share on /Volumes/share (smbfs, nodev, ...)
            out = subprocess.run(['mount'], capture_output=True, text=True, timeout=5)
            for line in out.stdout.splitlines():
                m = re.search(r' on (.+?) \(([^,) ]+)', line)
                if m:
                    entries.append((m.group(1), m.group(2)))

        # The deepest mountpoint that is a prefix of the path owns it.
        best = None
        for mountpoint, fstype in entries:
            mp = mountpoint.rstrip('/') or '/'
            if real == mp or real.startswith(mp + '/'):
                if best is None or len(mp) > len(best[0]):
                    best = (mp, fstype)
        if best:
            return best[1].lower() in _NETWORK_FSTYPES
    except Exception:
        pass
    return False


def is_network_path(path):
    """Best-effort: True if *path* lives on a network/remote drive.

    Conservative by design — returns True only when reasonably sure, so local
    disks never trigger a false warning. Handles UNC paths, Windows mapped
    network drives (via GetDriveType), and common POSIX network mounts.
    """
    if not path:
        return False
    p = str(path)

    # UNC paths (\\server\share or //server/share) are always remote.
    if p.startswith('\\\\') or p.startswith('//'):
        return True

    if os.name == 'nt':
        import ctypes
        drive = os.path.splitdrive(os.path.abspath(p))[0]
        if drive and drive.endswith(':'):
            DRIVE_REMOTE = 4
            try:
                return ctypes.windll.kernel32.GetDriveTypeW(drive + '\\') == DRIVE_REMOTE
            except Exception:
                return False
        return False

    return _posix_path_is_network(p)


# ---------------------------------------------------------------------------
# SLEAP installation discovery
#
# SLEAP can live in a conda environment (the traditional install) or be
# installed as a uv tool, which puts launcher shims on PATH.  Both are
# supported; an "install" is described by a dict:
#
#   kind    'conda' | 'uv' | 'path'
#   name    conda env name, or the tool name for uv
#   label   text shown in the dropdown
#   version SLEAP version if cheaply known (uv reports it, conda does not)
#   bin_dir directory holding sleap-track/sleap-convert (None => resolve on PATH
#           inside `conda run`)
#   python  interpreter for that install, used by the GPU probe
# ---------------------------------------------------------------------------

_EXE_SUFFIX = '.exe' if os.name == 'nt' else ''


def _uv_tool_paths(tool_dir):
    """Return (bin_dir, python) for a uv tool environment directory."""
    if os.name == 'nt':
        scripts = os.path.join(tool_dir, 'Scripts')
        return scripts, os.path.join(scripts, 'python.exe')
    bin_dir = os.path.join(tool_dir, 'bin')
    return bin_dir, os.path.join(bin_dir, 'python')


def _site_packages_dirs(env_path):
    """Candidate site-packages directories inside an environment."""
    if os.name == 'nt':
        return [os.path.join(env_path, 'Lib', 'site-packages')]
    return sorted(glob.glob(os.path.join(env_path, 'lib', 'python*', 'site-packages')))


def _torch_cuda_version(site_packages):
    """Read torch's built-in CUDA version from disk without importing torch.

    torch/version.py contains a line like ``cuda: Optional[str] = '12.8'`` for
    a CUDA build, or ``= None`` for a CPU-only build (the common reason a real
    GPU goes unused). Returns the version string, or None for a CPU-only build
    / when it can't be determined.
    """
    version_py = os.path.join(site_packages, 'torch', 'version.py')
    try:
        with open(version_py, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    except Exception:
        return None
    m = re.search(r'^\s*cuda\s*(?::[^=]*)?=\s*(None|["\']([^"\']+)["\'])',
                  text, re.MULTILINE)
    if m:
        return m.group(2)  # None when the build is CPU-only
    return None


def inspect_env(env_path):
    """Return (sleap_version, backend, gpu_cuda) for an environment, from disk.

    Reading package metadata off disk is instant, whereas importing SLEAP to
    ask it costs tens of seconds per environment — far too slow to do for
    every conda env just to populate a dropdown.  ``backend`` is 'torch'
    (SLEAP >= 1.5, via sleap-nn), 'tensorflow' (older SLEAP) or None.
    ``gpu_cuda`` is the CUDA version string of a CUDA-enabled torch build, or
    None (CPU-only torch, a TensorFlow backend, or no backend).
    """
    version = None
    backend = None
    gpu_cuda = None

    # conda-installed packages have no dist-info, but do leave conda-meta JSON
    for meta in glob.glob(os.path.join(env_path, 'conda-meta', 'sleap-*.json')):
        m = re.match(r'sleap-([0-9][^-]*)-', os.path.basename(meta))
        if m:
            version = m.group(1)
            break

    for site_packages in _site_packages_dirs(env_path):
        if not os.path.isdir(site_packages):
            continue
        if version is None:
            for dist in glob.glob(os.path.join(site_packages, 'sleap-*.dist-info')):
                m = re.match(r'sleap-([^-]+)\.dist-info$', os.path.basename(dist))
                if m:
                    version = m.group(1)
                    break
            if version is None and os.path.isdir(os.path.join(site_packages, 'sleap')):
                version = 'unknown'
        if backend is None:
            if os.path.isdir(os.path.join(site_packages, 'torch')):
                backend = 'torch'
                gpu_cuda = _torch_cuda_version(site_packages)
            elif os.path.isdir(os.path.join(site_packages, 'tensorflow')):
                backend = 'tensorflow'

    return version, backend, gpu_cuda


def _install_label(prefix, name, version, backend, gpu_cuda=None):
    """Dropdown text: what it is, which SLEAP, and whether it can run on GPU.

    SLEAP >= 1.6 ships the deep-learning backend (sleap-nn + PyTorch) as an
    optional extra; a plain ``uv tool install sleap`` is what SLEAP itself calls
    a "GUI-only installation" — it can label and view but cannot run inference.
    """
    bits = []
    if version:
        bits.append(f"SLEAP {version}" if version != 'unknown' else "SLEAP")
    if backend == 'torch':
        bits.append(f"PyTorch CUDA {gpu_cuda}" if gpu_cuda else "PyTorch CPU-only")
    elif backend == 'tensorflow':
        bits.append("TensorFlow")
    else:
        bits.append("GUI-only · no sleap-nn")
    return f"{prefix} — {name}  ·  {' · '.join(bits)}"


def detect_uv_installs():
    """Find SLEAP installed as a uv tool.

    Parses `uv tool list --show-paths`, whose output looks like:
        sleap v1.6.4 (C:\\Users\\me\\AppData\\Roaming\\uv\\tools\\sleap)
        - sleap-track (C:\\Users\\me\\.local\\bin\\sleap-track.exe)
    """
    installs = []
    try:
        result = subprocess.run(
            ["uv", "tool", "list", "--show-paths"],
            capture_output=True, text=True, shell=True, timeout=30,
        )
        if result.returncode != 0:
            return installs
    except Exception:
        return installs

    header = re.compile(r'^(\S+)\s+v(\S+)\s+\((.+)\)\s*$')
    for line in result.stdout.splitlines():
        line = line.rstrip()
        if not line or line.startswith('-'):
            continue
        m = header.match(line.strip())
        if not m:
            continue
        tool_name, version, tool_dir = m.group(1), m.group(2), m.group(3)
        if 'sleap' not in tool_name.lower():
            continue

        bin_dir, python = _uv_tool_paths(tool_dir)
        # Prefer the tool env's own scripts dir; fall back to the shims that uv
        # puts on PATH if the layout is unexpected.
        if not os.path.exists(os.path.join(bin_dir, 'sleap-track' + _EXE_SUFFIX)):
            shim = shutil.which('sleap-track')
            bin_dir = os.path.dirname(shim) if shim else bin_dir

        _, backend, gpu_cuda = inspect_env(tool_dir)
        installs.append({
            'kind': 'uv',
            'name': tool_name,
            'label': _install_label('⚡ uv', tool_name, version, backend, gpu_cuda),
            'version': version,
            'backend': backend,
            'gpu_cuda': gpu_cuda,
            'has_sleap': True,
            'bin_dir': bin_dir,
            'python': python if os.path.exists(python) else None,
        })
    return installs


_CONDA_EXE = None


def conda_executable():
    """Locate the conda launcher.

    PATH is checked first, then $CONDA_EXE, then the usual install locations —
    a GUI launched from a desktop shortcut often has no conda on its PATH even
    though conda is installed.
    """
    global _CONDA_EXE
    if _CONDA_EXE is not None:
        return _CONDA_EXE

    candidates = [shutil.which('conda'), os.environ.get('CONDA_EXE')]
    home = os.path.expanduser('~')
    for distro in ('miniconda3', 'anaconda3', 'miniforge3', 'mambaforge'):
        base = os.path.join(home, distro)
        if os.name == 'nt':
            candidates += [os.path.join(base, 'condabin', 'conda.bat'),
                           os.path.join(base, 'Scripts', 'conda.exe')]
        else:
            candidates.append(os.path.join(base, 'bin', 'conda'))

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            _CONDA_EXE = candidate
            return _CONDA_EXE

    _CONDA_EXE = 'conda'    # last resort: let the shell resolve it
    return _CONDA_EXE


def detect_conda_installs():
    """Return one install dict per conda environment."""
    installs = []
    try:
        result = subprocess.run(
            [conda_executable(), 'env', 'list'],
            capture_output=True, text=True, shell=True, timeout=30,
        )
        if result.returncode != 0:
            return installs
    except Exception:
        return installs

    seen = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        # "name              *  C:\path\to\env"  — * marks active, + frozen
        parts = [p for p in line.split() if p not in ('*', '+')]
        if not parts:
            continue
        env_name = parts[0]
        env_path = parts[-1] if len(parts) > 1 else ''

        # conda can list the same environment twice if it is registered in
        # more than one place; show it once.
        key = os.path.normcase(env_path) if env_path else env_name
        if key in seen:
            continue
        seen.add(key)

        version, backend, gpu_cuda = (
            inspect_env(env_path) if env_path else (None, None, None))
        label = (_install_label('🐍 conda', env_name, version, backend, gpu_cuda)
                 if version else f"🐍 conda — {env_name}  ·  no SLEAP")
        installs.append({
            'kind': 'conda',
            'name': env_name,
            'label': label,
            'version': version,
            'backend': backend,
            'gpu_cuda': gpu_cuda,
            'has_sleap': version is not None,
            'path': env_path,
            'bin_dir': None,       # resolved by `conda run` inside the env
            'python': None,        # ditto
        })
    return installs


def detect_sleap_installs():
    """Detect all SLEAP installations, best candidates first.

    Ordering drives which install the dropdown opens on:
      0. SLEAP + a GPU-capable backend (CUDA torch) — the fast path
      1. SLEAP + a backend, but CPU-only (e.g. a torch+cpu build)
      2. SLEAP present but no backend (GUI-only)
      3. no SLEAP at all
    so the install that will actually run fast on the GPU is auto-selected.
    """
    installs = detect_uv_installs() + detect_conda_installs()

    def rank(install):
        if not install.get('has_sleap'):
            return 3
        if not install.get('backend'):
            return 2
        # torch built with CUDA (or a TensorFlow backend we can't cheaply probe)
        if install.get('gpu_cuda') or install.get('backend') == 'tensorflow':
            return 0
        return 1  # backend present but CPU-only

    installs.sort(key=lambda i: (rank(i), i['kind'] != 'uv', i['name'].lower()))
    return installs


def install_bin(install, tool):
    """Resolve the executable for *tool* ('sleap-track', 'sleap-convert', ...).

    For conda installs the bare name is returned — it is resolved inside the
    environment by `conda run`.  Returning the bare name for other kinds too
    would be wrong: PATH lookup could silently pick a *different* SLEAP.
    """
    bin_dir = (install or {}).get('bin_dir')
    if bin_dir:
        candidate = os.path.join(bin_dir, tool + _EXE_SUFFIX)
        if os.path.exists(candidate):
            return candidate
    return tool


def install_cmd(install, tool, args):
    """Build the full command line to run *tool* under *install*."""
    if install and install.get('kind') == 'conda':
        # --no-capture-output keeps conda from buffering the child's output,
        # so progress lines reach our pipe as they are produced.
        return ([conda_executable(), "run", "--no-capture-output", "-n",
                 install['name'], tool] + list(args))
    return [install_bin(install, tool)] + list(args)


def describe_install(install):
    """Short human-readable description of an install, for logs."""
    if not install:
        return "unknown"
    version = f" (SLEAP {install['version']})" if install.get('version') else ""
    if install['kind'] == 'conda':
        return f"conda env '{install['name']}'{version}"
    return f"uv tool '{install['name']}'{version}"


class GpuProbeWorker(QThread):
    """Detect the GPU and ask the selected SLEAP install whether it can use it.

    The second half is what matters: a machine can have a perfectly good NVIDIA
    card while the SLEAP environment holds a CPU-only backend build, in which
    case inference silently falls back to the CPU at a fraction of the speed.
    SLEAP >= 1.5 runs on PyTorch (via sleap-nn); older versions use TensorFlow,
    so both backends are probed.  Importing either takes tens of seconds, hence
    the background thread.
    """

    result = pyqtSignal(dict)

    _PROBE_SCRIPT = (
        "import os\n"
        "os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')\n"
        "found = False\n"
        "try:\n"
        "    import torch\n"
        "    found = True\n"
        "    print('FNT_BACKEND=torch')\n"
        "    print('FNT_BACKEND_VERSION=' + torch.__version__)\n"
        "    print('FNT_CUDA_BUILD=' + str(torch.version.cuda is not None))\n"
        "    avail = torch.cuda.is_available()\n"
        "    print('FNT_GPUS=' + str(torch.cuda.device_count() if avail else 0))\n"
        "    if avail:\n"
        "        print('FNT_DEVICE_NAME=' + torch.cuda.get_device_name(0))\n"
        "except ImportError:\n"
        "    pass\n"
        "except Exception as e:\n"
        "    print('FNT_ERROR=' + str(e))\n"
        "    found = True\n"
        "if not found:\n"
        "    try:\n"
        "        import tensorflow as tf\n"
        "        found = True\n"
        "        gpus = tf.config.list_physical_devices('GPU')\n"
        "        print('FNT_BACKEND=tensorflow')\n"
        "        print('FNT_BACKEND_VERSION=' + tf.__version__)\n"
        "        print('FNT_CUDA_BUILD=' + str(tf.test.is_built_with_cuda()))\n"
        "        print('FNT_GPUS=' + str(len(gpus)))\n"
        "    except ImportError:\n"
        "        pass\n"
        "    except Exception as e:\n"
        "        print('FNT_ERROR=' + str(e))\n"
        "        found = True\n"
        "if not found:\n"
        "    print('FNT_ERROR=No PyTorch or TensorFlow backend found')\n"
    )

    def __init__(self, install=None):
        super().__init__()
        self.install = install

    def run(self):
        kind, name, detail = detect_gpu()
        info = {
            'kind': kind,
            'name': name,
            'detail': detail,
            'install': self.install,
            'backend': None,            # 'torch' | 'tensorflow' | None
            'backend_version': None,
            'backend_gpus': None,       # None => not probed / probe failed
            'cuda_build': None,
            'device_name': None,
            'error': None,
        }

        if self.install:
            info.update(self._probe_backend())

        self.result.emit(info)

    def _probe_cmd(self, script_path):
        install = self.install
        if install.get('kind') == 'conda':
            return [conda_executable(), "run", "--no-capture-output", "-n",
                    install['name'], "python", script_path]
        python = install.get('python')
        if not python:
            return None
        return [python, script_path]

    def _probe_backend(self):
        out = {}
        script_path = None
        try:
            fd, script_path = tempfile.mkstemp(suffix="_fnt_gpu_probe.py", text=True)
            with os.fdopen(fd, 'w') as f:
                f.write(self._PROBE_SCRIPT)

            cmd = self._probe_cmd(script_path)
            if cmd is None:
                return {'error': "Could not locate a Python interpreter for this install"}

            result = subprocess.run(
                cmd, capture_output=True, text=True, shell=True, timeout=300,
            )
            combined = (result.stdout or '') + (result.stderr or '')
            fields = {
                'FNT_BACKEND=': ('backend', str),
                'FNT_BACKEND_VERSION=': ('backend_version', str),
                'FNT_GPUS=': ('backend_gpus', int),
                'FNT_CUDA_BUILD=': ('cuda_build', lambda v: v == 'True'),
                'FNT_DEVICE_NAME=': ('device_name', str),
                'FNT_ERROR=': ('error', str),
            }
            for line in combined.splitlines():
                line = line.strip()
                for prefix, (key, cast) in fields.items():
                    if line.startswith(prefix):
                        try:
                            out[key] = cast(line[len(prefix):])
                        except Exception:
                            pass
                        break
            if not out:
                out['error'] = "No response from the GPU probe"
        except subprocess.TimeoutExpired:
            out['error'] = "GPU probe timed out"
        except Exception as e:
            out['error'] = str(e)
        finally:
            if script_path:
                try:
                    os.remove(script_path)
                except Exception:
                    pass
        return out


class InferenceWorker(QThread):
    """Worker thread for running SLEAP inference"""
    progress = pyqtSignal(str)
    progress_update = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, video_folders, individual_videos, model_paths, overwrite_existing, create_csv=True, add_tracking=False,
                 tracker_method="simple", similarity_method="instance", match_method="greedy",
                 max_tracks=0, track_window=10, robust_quantile=0.95, post_connect_breaks=False, install=None,
                 use_gpu=True, batch_size=4):
        super().__init__()
        self.video_folders = video_folders
        self.individual_videos = individual_videos
        self.model_paths = model_paths
        self.overwrite_existing = overwrite_existing
        self.create_csv = create_csv
        self.add_tracking = add_tracking
        self.tracker_method = tracker_method
        self.similarity_method = similarity_method
        self.match_method = match_method
        self.max_tracks = max_tracks
        self.track_window = track_window
        self.robust_quantile = robust_quantile
        self.post_connect_breaks = post_connect_breaks
        self.install = install
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self._stop_requested = False
        self._current_process = None
        self._current_video_label = ""   # shown at the start of the status line
        self._run_output_lines = []      # recent real output, scanned for errors

    # Output signatures that mean the run failed even if the process exits 0.
    _ERROR_SIGNATURES = (
        "sleap-nn is not installed",
        "modulenotfounderror",
        "traceback (most recent call last)",
        "importerror",
    )

    @staticmethod
    def _version_tuple(version):
        """Parse 'X.Y.Z' into (X, Y), or None if unknown/unparseable."""
        if not version or version == 'unknown':
            return None
        m = re.match(r'(\d+)\.(\d+)', str(version))
        return (int(m.group(1)), int(m.group(2))) if m else None

    def _uses_sleap_nn_cli(self):
        """True for SLEAP >= 1.6, whose working command is `sleap-nn track`.

        In 1.6 the legacy `sleap-track` shim references a sleap-nn module that
        was moved, so it errors out (and misreports it as "sleap-nn is not
        installed"); the real entry point is the `sleap-nn` CLI. SLEAP 1.5.x
        keeps using the old `sleap-track` with its `--gpu`/`--tracking.*` flags.
        """
        vt = self._version_tuple((self.install or {}).get('version'))
        return vt is not None and vt >= (1, 6)

    def _scan_for_errors(self, lines):
        """Return the first line matching a known failure signature, else None."""
        for line in lines:
            low = line.lower()
            for sig in self._ERROR_SIGNATURES:
                if sig in low:
                    return line.strip()
        return None

    def request_stop(self):
        """Request the worker to stop processing"""
        self._stop_requested = True
        if self._current_process:
            self._current_process.terminate()
        self.progress.emit("\n⚠️ Stop requested by user...")

    @staticmethod
    def _strip_ansi(text):
        """Remove ANSI escape codes (colors, cursor moves) from text."""
        return _ANSI_RE.sub('', text)

    @staticmethod
    def _parse_progress(text):
        """Extract progress fields from a rich repaint fragment, or None.

        A repaint may arrive truncated, so any subset of the fields may be
        present; the caller merges successive fragments into one state.
        """
        has_timing = ('ETA' in text) or ('Elapsed' in text) or ('FPS' in text)
        fraction = _FRACTION_RE.search(text)
        phase = _PHASE_RE.search(text)

        # Require real progress-bar evidence, so ordinary log lines that happen
        # to contain a path (with '/') are never mistaken for progress.
        if not (has_timing or (phase and fraction)):
            return None

        info = {}
        if phase:
            info['phase'] = phase.group(1).capitalize()
        if fraction:
            info['current'] = int(fraction.group(1).replace(',', ''))
            info['total'] = int(fraction.group(2).replace(',', ''))
        m = _PERCENT_RE.search(text)
        if m:
            info['percent'] = int(m.group(1))
        m = _ETA_RE.search(text)
        if m:
            info['eta'] = m.group(1)
        m = _ELAPSED_RE.search(text)
        if m:
            info['elapsed'] = m.group(1)
        m = _FPS_RE.search(text)
        if m:
            info['fps'] = m.group(1)
        return info or None

    @staticmethod
    def _is_chrome(text):
        """True if *text* is only progress-bar decoration (no information)."""
        return _CHROME_RE.fullmatch(text) is not None

    def _format_progress(self, state):
        """Render merged progress *state* as one fixed-width status line.

        The layout uses a fixed-width bar and fixed column widths so the line
        stays aligned and readable no matter how the window is resized.
        """
        percent = state.get('percent')
        total = state.get('total')
        current = state.get('current')
        if percent is None and total:
            percent = int(100 * current / total)
        percent = max(0, min(100, percent or 0))

        filled = int(round(PROGRESS_BAR_WIDTH * percent / 100.0))
        bar = '█' * filled + '░' * (PROGRESS_BAR_WIDTH - filled)

        label = self._current_video_label or state.get('phase', 'Working')
        if len(label) > VIDEO_LABEL_WIDTH:
            label = label[:VIDEO_LABEL_WIDTH - 1] + '…'
        parts = [f"  {label:<{VIDEO_LABEL_WIDTH}}", f"[{bar}]", f"{percent:3d}%"]

        if total:
            width = len(str(total))
            parts.append(f"{current:>{width}}/{total}")
        if state.get('fps'):
            parts.append(f"{state['fps']:>6} FPS")
        if state.get('eta'):
            parts.append(f"ETA {state['eta']}")
        if state.get('elapsed'):
            parts.append(f"elapsed {state['elapsed']}")

        return '  '.join(parts)

    def _read_stream(self, stream):
        """Read *stream*, collapsing SLEAP's progress repaints into one line.

        Each read may carry many \\r-separated repaint fragments at once.  All
        of them are merged into a single progress state and only the newest
        state is pushed to the GUI (via ``progress_update``, which rewrites the
        status line in place), at most a few times a second.  Genuine log
        messages go to ``progress`` and append as normal; pure bar decoration
        is discarded.
        """
        UPDATE_INTERVAL = 0.25      # seconds between status-line refreshes
        pending = None              # merged progress state not yet shown
        last_emit = 0.0

        try:
            for raw_line in iter(stream.readline, ''):
                if self._stop_requested:
                    break

                for segment in re.split(r'[\r\n]', raw_line):
                    clean = self._strip_ansi(segment).strip()
                    if not clean:
                        continue

                    info = self._parse_progress(clean)
                    if info is not None:
                        if pending is None:
                            pending = {}
                        pending.update(info)
                        continue

                    if self._is_chrome(clean):
                        continue

                    # A real message: show the latest progress first so the
                    # status line is left holding accurate numbers, then
                    # append the message beneath it.
                    if pending:
                        self.progress_update.emit(self._format_progress(pending))
                        pending = None
                        last_emit = time.time()
                    self.progress.emit(clean)
                    # Keep a bounded copy of real output lines so the run can be
                    # inspected for error signatures afterwards (some SLEAP CLIs
                    # log an error and still exit 0).
                    self._run_output_lines.append(clean)
                    if len(self._run_output_lines) > 400:
                        del self._run_output_lines[:200]

                now = time.time()
                if pending and now - last_emit >= UPDATE_INTERVAL:
                    self.progress_update.emit(self._format_progress(pending))
                    last_emit = now
        except Exception:
            pass
        finally:
            # Never leave the status line showing a stale intermediate value.
            if pending:
                try:
                    self.progress_update.emit(self._format_progress(pending))
                except Exception:
                    pass

    def run(self):
        try:
            total_processed = 0
            total_skipped = 0
            
            # Process folders
            for folder in self.video_folders:
                if self._stop_requested:
                    break
                    
                video_files = [f for f in os.listdir(folder)
                              if f.lower().endswith((".mp4", ".avi", ".mov"))
                              and not f.endswith("_roiTracked.mp4")]  # Ignore ROI tracked videos
                
                if not video_files:
                    self.progress.emit(f"⚠️ No video files found in: {folder}\n")
                    continue
                
                self.progress.emit(f"\n📁 Processing folder: {folder}")
                self.progress.emit(f"Found {len(video_files)} video file(s)\n")
                
                for video_file in video_files:
                    if self._stop_requested:
                        break
                        
                    full_path = os.path.join(folder, video_file)
                    
                    # Check for existing prediction files with any timestamp
                    existing_files = self.find_existing_predictions(full_path)
                    
                    if existing_files and not self.overwrite_existing:
                        self.progress.emit(f"⏭️ Skipping {video_file} (existing predictions detected)")
                        total_skipped += 1
                        continue
                    
                    # Delete existing files if overwrite is enabled
                    if existing_files and self.overwrite_existing:
                        self.progress.emit(f"🗑️ Deleting {len(existing_files)} existing prediction file(s) for {video_file}")
                        for existing_file in existing_files:
                            try:
                                os.remove(existing_file)
                                self.progress.emit(f"   Deleted: {os.path.basename(existing_file)}")
                            except Exception as e:
                                self.progress.emit(f"   ⚠️ Failed to delete {os.path.basename(existing_file)}: {str(e)}")
                    
                    # Run inference (CSV conversion happens inside, against the
                    # actual output path — regenerating it here would produce a
                    # different timestamp and point at a file that never existed)
                    if self.run_inference_on_video(full_path):
                        total_processed += 1
                    else:
                        self.progress.emit(f"❌ Failed to process {video_file}")
            
            # Process individual videos
            if self.individual_videos:
                self.progress.emit(f"\n📹 Processing {len(self.individual_videos)} individual video(s)\n")
            
            for video_path in self.individual_videos:
                if self._stop_requested:
                    break
                
                video_file = os.path.basename(video_path)
                
                # Check for existing prediction files with any timestamp
                existing_files = self.find_existing_predictions(video_path)
                
                if existing_files and not self.overwrite_existing:
                    self.progress.emit(f"⏭️ Skipping {video_file} (existing predictions detected)")
                    total_skipped += 1
                    continue
                
                # Delete existing files if overwrite is enabled
                if existing_files and self.overwrite_existing:
                    self.progress.emit(f"🗑️ Deleting {len(existing_files)} existing prediction file(s) for {video_file}")
                    for existing_file in existing_files:
                        try:
                            os.remove(existing_file)
                            self.progress.emit(f"   Deleted: {os.path.basename(existing_file)}")
                        except Exception as e:
                            self.progress.emit(f"   ⚠️ Failed to delete {os.path.basename(existing_file)}: {str(e)}")
                
                # Run inference (CSV conversion happens inside — see above)
                if self.run_inference_on_video(video_path):
                    total_processed += 1
                else:
                    self.progress.emit(f"❌ Failed to process {video_file}")
            
            summary = f"\n{'='*60}\n"
            if self._stop_requested:
                summary += f"⚠️ Inference stopped by user!\n"
            else:
                summary += f"✅ Inference complete!\n"
            summary += f"Videos processed: {total_processed}\n"
            summary += f"Videos skipped: {total_skipped}\n"
            summary += f"Total videos: {total_processed + total_skipped}\n"
            
            self.progress.emit(summary)
            
            if self._stop_requested:
                self.finished.emit(False, "Inference stopped by user")
            else:
                self.finished.emit(True, "Inference completed successfully!")
            
        except Exception as e:
            self.finished.emit(False, f"Error during inference: {str(e)}")
    
    def find_existing_predictions(self, video_path):
        """Find all existing prediction files for a video (with any timestamp)"""
        base = os.path.basename(video_path)
        parent = os.path.dirname(video_path)
        
        # Search patterns for .slp, .csv, and .mp4 prediction files
        patterns = [
            os.path.join(parent, f"{base}.*.predictions.slp"),
            os.path.join(parent, f"{base}.*.predictions.analysis.csv"),
            os.path.join(parent, f"{base}.*.predictions.mp4")
        ]
        
        existing_files = []
        for pattern in patterns:
            existing_files.extend(glob.glob(pattern))
        
        return existing_files
    
    def get_output_path(self, video_path):
        """Generate output file path with timestamp"""
        base = os.path.basename(video_path)
        parent = os.path.dirname(video_path)
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        filename = f"{base}.{timestamp}.predictions.slp"
        return os.path.join(parent, filename)
    
    def _build_legacy_track_args(self, video_file, output_file):
        """Arguments for SLEAP 1.5.x `sleap-track` (--gpu / --tracking.* syntax)."""
        args = [video_file]
        for model_path in self.model_paths:
            args += ["-m", model_path]

        # Device selection: "--gpu auto" picks the GPU with the most free
        # memory, while "--cpu" is a separate flag that forces CPU-only.
        args += [
            "-o", output_file,
            "--batch_size", str(self.batch_size),
            "--peak_threshold", "0.2",
        ]
        args += ["--gpu", "auto"] if self.use_gpu else ["--cpu"]

        if self.add_tracking:
            # max_instances first (required for post_connect_single_breaks)
            if self.max_tracks > 0:
                args += ["--max_instances", str(self.max_tracks)]
            args += ["--tracking.tracker", self.tracker_method]
            args += ["--tracking.similarity", self.similarity_method]
            args += ["--tracking.match", self.match_method]
            if self.max_tracks > 0:
                args += ["--tracking.max_tracks", str(self.max_tracks)]
            if self.track_window != 5:
                args += ["--tracking.track_window", str(self.track_window)]
            if self.robust_quantile < 1.0:
                args += ["--tracking.robust", str(self.robust_quantile)]
            if self.post_connect_breaks:
                args += ["--tracking.post_connect_single_breaks", "1"]
        return args

    def _build_nn_track_args(self, video_file, output_file):
        """Arguments for SLEAP >= 1.6 `sleap-nn track` (the new CLI).

        The flag names changed substantially from 1.5.x: `--device` replaces
        `--gpu`/`--cpu`, tracking options use `--tracking_*` / dedicated names,
        and similarity maps onto `--features`/`--scoring_method` the same way
        SLEAP's own legacy adaptor does it.
        """
        args = ["-i", video_file]
        for model_path in self.model_paths:
            args += ["-m", model_path]
        args += [
            "-o", output_file,
            "--batch_size", str(self.batch_size),
            "--peak_threshold", "0.2",
            "--device", "auto" if self.use_gpu else "cpu",
        ]

        if self.add_tracking:
            args += ["--tracking"]
            if self.max_tracks > 0:
                args += ["--max_instances", str(self.max_tracks)]

            tracker = self.tracker_method or ""
            if "flow" in tracker:
                args += ["--use_flow"]
            if "maxtracks" in tracker and self.max_tracks > 0:
                args += ["--candidates_method", "local_queues"]

            # similarity -> features/scoring (mirrors SLEAP's legacy adaptor);
            # "instance"/"normalized_instance" fall through to the defaults.
            sim = self.similarity_method
            if sim == "object_keypoint":
                args += ["--features", "keypoints", "--scoring_method", "oks"]
            elif sim == "centroid":
                args += ["--features", "centroids", "--scoring_method", "euclidean_dist"]
            elif sim == "iou":
                args += ["--features", "bboxes", "--scoring_method", "iou"]

            if self.match_method:
                args += ["--track_matching_method", self.match_method]
            if self.track_window:
                args += ["--tracking_window_size", str(self.track_window)]
            if self.robust_quantile < 1.0:
                args += ["--robust_best_instance", str(self.robust_quantile)]

            # post_connect_single_breaks REQUIRES a known instance count in
            # sleap-nn; skip it (with a warning) rather than hard-failing when
            # Max Tracks is left at "No limit".
            if self.post_connect_breaks:
                if self.max_tracks > 0:
                    args += ["--post_connect_single_breaks"]
                else:
                    self.progress.emit(
                        "   ⚠️ 'Connect single track breaks' needs Max Tracks set "
                        "(SLEAP 1.6+ requires it) — skipping it for this run.")
        return args

    def run_inference_on_video(self, video_file):
        """Run SLEAP inference on a single video (CLI depends on SLEAP version)."""
        output_file = self.get_output_path(video_file)
        self._run_output_lines = []

        use_nn_cli = self._uses_sleap_nn_cli()
        if use_nn_cli:
            track_args = self._build_nn_track_args(video_file, output_file)
            full_cmd = install_cmd(self.install, "sleap-nn", ["track"] + track_args)
        else:
            track_args = self._build_legacy_track_args(video_file, output_file)
            full_cmd = install_cmd(self.install, "sleap-track", track_args)

        # Label used by the live status line while this video is processing
        self._current_video_label = os.path.basename(video_file)

        self.progress.emit(f"\n🔁 Running inference on: {os.path.basename(video_file)}")
        self.progress.emit(
            f"Device: {'GPU (auto)' if self.use_gpu else 'CPU only'}   "
            f"Batch size: {self.batch_size}")

        # Build the full command for whichever install was selected (conda env
        # or uv tool); see install_cmd().
        if self.install:
            self.progress.emit(
                f"SLEAP install: {describe_install(self.install)} "
                f"[{'sleap-nn track' if use_nn_cli else 'sleap-track'}]")
        else:
            self.progress.emit("Warning: No SLEAP installation specified")

        self.progress.emit(f"Command: {' '.join(full_cmd)}\n")

        # PYTHONUNBUFFERED — prevents Python from buffering SLEAP's output.
        # FORCE_COLOR      — makes rich render its progress bar (it would
        #                    suppress it entirely on a non-TTY pipe).
        # COLUMNS          — tells rich the "terminal" width for formatting.
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["FORCE_COLOR"] = "1"
        env["COLUMNS"] = "120"

        try:
            self.progress.emit("🔄 Running SLEAP inference...\n")

            # stdout and stderr are read in separate threads so neither
            # pipe blocks the other.
            process = subprocess.Popen(
                full_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True,
                bufsize=1,
                env=env,
            )
            self._current_process = process

            # stderr thread — rich progress bar + INFO log lines go here
            stderr_thread = threading.Thread(
                target=self._read_stream,
                args=(process.stderr,),
                daemon=True,
            )
            stderr_thread.start()

            # stdout — normal print() output, if any
            self._read_stream(process.stdout)

            stderr_thread.join(timeout=30)
            process.wait()
            self._current_process = None

            # A clean exit code is not enough: some SLEAP CLIs log an error and
            # still exit 0 (the 1.6 legacy shim swallows an ImportError this
            # way). Treat a known error signature, or a missing/empty output
            # file, as a failure so we never report a false success.
            error_line = self._scan_for_errors(self._run_output_lines)
            produced = os.path.exists(output_file) and os.path.getsize(output_file) > 0

            if error_line:
                self.progress.emit(f"\n❌ Inference failed — {error_line}")
                if "not installed" in error_line.lower() and self._uses_sleap_nn_cli():
                    self.progress.emit(
                        "   (Reinstall SLEAP with the 'nn' extra so sleap-nn can run, "
                        "e.g. `uv tool install \"sleap[nn]\" --torch-backend auto`.)")
                return False

            if process.returncode != 0:
                self.progress.emit(
                    f"\n❌ Inference failed with return code {process.returncode}")
                return False

            if not produced:
                self.progress.emit(
                    f"\n❌ Inference exited cleanly but produced no predictions file "
                    f"({os.path.basename(output_file)}). Treating as a failure — "
                    f"see the log above for the underlying error.")
                return False

            self.progress.emit(f"\n✅ Inference completed: {os.path.basename(output_file)}")
            if self.create_csv:
                self.convert_to_csv(output_file)
            return True

        except Exception as e:
            self._current_process = None
            self.progress.emit(f"❌ Error running inference: {str(e)}")
            return False
    
    def convert_to_csv(self, slp_file):
        """Convert an .slp predictions file to analysis CSV (sleap-convert)."""
        csv_file = slp_file.replace(".predictions.slp", ".predictions.analysis.csv")

        if os.path.exists(csv_file) and not self.overwrite_existing:
            self.progress.emit(f"⏭️ Skipping CSV conversion: {os.path.basename(csv_file)} already exists")
            return

        # Don't try to convert a predictions file that isn't really there.
        if not (os.path.exists(slp_file) and os.path.getsize(slp_file) > 0):
            self.progress.emit(
                f"⚠️ Skipping CSV conversion: {os.path.basename(slp_file)} is missing or empty")
            return

        self.progress.emit(f"📄 Converting to CSV: {os.path.basename(slp_file)}")

        try:
            full_cmd = install_cmd(
                self.install, "sleap-convert",
                ["--format", "analysis.csv", "-o", csv_file, slp_file])

            result = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                shell=True
            )

            if result.returncode == 0 and os.path.exists(csv_file):
                self.progress.emit(f"✅ CSV created: {os.path.basename(csv_file)}")
            else:
                # Surface the full error so problems are diagnosable from the log,
                # rather than a 200-char truncation that cuts off the real cause.
                self.progress.emit(
                    f"⚠️ CSV conversion failed (exit code {result.returncode}):")
                detail = (result.stderr or result.stdout or "").strip()
                for line in (detail.splitlines() or ["Unknown error (no output)"]):
                    self.progress.emit(f"   {line}")

        except Exception as e:
            self.progress.emit(f"⚠️ CSV conversion error: {str(e)}")
    


class VideoInferenceWindow(QWidget):
    """PyQt5 window for SLEAP video inference configuration and execution"""
    
    def __init__(self):
        super().__init__()
        self.video_folders = []
        self.individual_videos = []  # Track individual video files separately
        self.folder_video_counts = {}  # folder -> cached video count (None if unreadable)
        self.model_paths = []
        self.is_top_down = False
        self.worker = None
        self.gpu_probe = None
        self.gpu_info = {}
        self._status_line_active = False   # is the last log block a live status line?
        self._network_warning_acknowledged = False  # network-drive warning shown this run?
        self.init_ui()

        # Auto-detect conda environments after UI is ready
        QTimer.singleShot(500, self.detect_sleap_installations)
        
    def init_ui(self):
        """Initialize the user interface"""
        self.setWindowTitle("SLEAP Video Inference Only")
        self.setGeometry(100, 100, 1000, 950)
        # Keep the minimum modest: the whole UI lives inside a scroll area
        # (see end of this method), so on a short screen it scrolls instead of
        # compressing the option rows into each other.
        self.setMinimumSize(820, 500)
        
        # Apply dark theme styling. This is applied to the scroll *content*
        # widget (see the end of this method), not to `self`: with the whole UI
        # nested inside a QScrollArea, styling only `self` no longer reliably
        # cascades the QPushButton rule down to the buttons, leaving them
        # looking like flat text. Owning the stylesheet on the widget that
        # directly parents the buttons keeps them blue.
        self._app_stylesheet = """
            QWidget {
                background-color: #2b2b2b;
                color: #cccccc;
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
            QLabel {
                color: #cccccc;
                background-color: transparent;
            }
            QListWidget {
                background-color: #1e1e1e;
                alternate-background-color: #232323;
                border: 1px solid #3f3f3f;
                border-radius: 4px;
                color: #cccccc;
                padding: 2px;
                outline: none;
            }
            QListWidget::item {
                padding: 5px 8px;
                border-radius: 3px;
            }
            QListWidget::item:hover {
                background-color: #2f2f2f;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
                color: #ffffff;
            }
            QTextEdit {
                background-color: #1e1e1e;
                border: 1px solid #3f3f3f;
                color: #cccccc;
                font-family: Consolas, Courier New, monospace;
            }
            QCheckBox {
                color: #cccccc;
                spacing: 8px;
                font-family: Consolas, Courier New, monospace;
            }
        """
        # A minimal sheet on the window itself so the frame/background is dark
        # even in the sliver the scroll area doesn't cover.
        self.setStyleSheet("QWidget { background-color: #2b2b2b; color: #cccccc; }")

        layout = QVBoxLayout()
        
        # Title
        title = QLabel("SLEAP Video Inference")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setStyleSheet("color: #0078d4; background-color: transparent;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Run SLEAP inference with optional tracking\nAutomatically converts output to CSV format")
        desc.setFont(QFont("Arial", 10))
        desc.setStyleSheet("color: #999999; font-style: italic; background-color: transparent; margin-bottom: 10px;")
        desc.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc)
        
        # Video Folders Group
        video_group = QGroupBox("1. Select Video Folders")
        video_layout = QVBoxLayout()
        
        btn_layout = QHBoxLayout()
        btn_add_folder = QPushButton("Add Folder(s)")
        btn_add_folder.setToolTip(
            "Add one or more folders of videos.\n"
            "Shift- or Ctrl-click to select several folders at once."
        )
        btn_add_folder.clicked.connect(self.add_video_folder)
        btn_layout.addWidget(btn_add_folder)
        
        btn_add_videos = QPushButton("Add Video(s)")
        btn_add_videos.clicked.connect(self.add_individual_videos)
        btn_layout.addWidget(btn_add_videos)
        
        btn_clear_folders = QPushButton("Clear All")
        btn_clear_folders.clicked.connect(self.clear_video_folders)
        btn_layout.addWidget(btn_clear_folders)
        btn_layout.addStretch()
        video_layout.addLayout(btn_layout)
        
        self.folder_list = QListWidget()
        self.folder_list.setMinimumHeight(90)
        self.folder_list.setMaximumHeight(120)
        self._style_list(self.folder_list)
        video_layout.addWidget(self.folder_list)

        self.lbl_video_summary = QLabel("No videos selected")
        self.lbl_video_summary.setStyleSheet("color: #999999; font-size: 11px;")
        video_layout.addWidget(self.lbl_video_summary)

        video_group.setLayout(video_layout)
        layout.addWidget(video_group)
        
        # Model Selection Group
        model_group = QGroupBox("2. Select SLEAP Model(s)")
        model_layout = QVBoxLayout()
        
        # Model type selection
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Model Type:"))
        
        btn_topdown = QPushButton("Top-Down (Centroid + Centered)")
        btn_topdown.clicked.connect(self.select_topdown_models)
        type_layout.addWidget(btn_topdown)
        
        btn_bottomup = QPushButton("Bottom-Up")
        btn_bottomup.clicked.connect(self.select_bottomup_models)
        type_layout.addWidget(btn_bottomup)
        
        type_layout.addStretch()
        model_layout.addLayout(type_layout)
        
        # Model paths display
        self.model_list = QListWidget()
        self.model_list.setMinimumHeight(70)
        self.model_list.setMaximumHeight(90)
        self._style_list(self.model_list)
        model_layout.addWidget(self.model_list)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # SLEAP Installation Group (conda environments and uv tools)
        env_group = QGroupBox("3. Select SLEAP Installation")
        env_layout = QVBoxLayout()

        env_select_layout = QHBoxLayout()
        env_select_layout.addWidget(QLabel("SLEAP Install:"))

        self.combo_install = QComboBox()
        self.combo_install.setMinimumWidth(320)
        self.combo_install.setStyleSheet("""
            QComboBox {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #3f3f3f;
                padding: 5px;
                border-radius: 3px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #1e1e1e;
                color: #cccccc;
                selection-background-color: #0078d4;
            }
        """)
        self.combo_install.setToolTip(
            "SLEAP installed as a uv tool (⚡) or inside a conda environment (🐍)."
        )
        self.combo_install.activated.connect(self.on_install_changed)
        env_select_layout.addWidget(self.combo_install)

        btn_refresh_envs = QPushButton("Refresh")
        btn_refresh_envs.clicked.connect(self.detect_sleap_installations)
        btn_refresh_envs.setMaximumWidth(80)
        env_select_layout.addWidget(btn_refresh_envs)

        btn_test_sleap = QPushButton("Test SLEAP")
        btn_test_sleap.clicked.connect(self.test_sleap_install)
        btn_test_sleap.setMaximumWidth(100)
        env_select_layout.addWidget(btn_test_sleap)

        env_select_layout.addStretch()
        env_layout.addLayout(env_select_layout)

        # Installation status
        self.lbl_env_status = QLabel("Click 'Refresh' to detect SLEAP installations")
        self.lbl_env_status.setStyleSheet("color: #999999; font-style: italic;")
        env_layout.addWidget(self.lbl_env_status)

        env_group.setLayout(env_layout)
        layout.addWidget(env_group)

        # Options Group
        options_group = QGroupBox("4. Processing Options")
        options_layout = QVBoxLayout()
        
        # --- Compute device -------------------------------------------------
        gpu_row = QHBoxLayout()

        self.chk_use_gpu = QCheckBox("Use GPU for inference")
        # On by default: the GPU probe takes ~30s, and until it reports back
        # the safe assumption is the previous behaviour (let SLEAP pick a GPU).
        # The probe unchecks and disables this if no GPU exists.
        self.chk_use_gpu.setChecked(True)
        self.chk_use_gpu.setToolTip(
            "Passes --gpu auto to sleap-track. Unchecked passes --cpu, forcing "
            "CPU-only inference."
        )
        gpu_row.addWidget(self.chk_use_gpu)

        self.lbl_gpu_status = QLabel("Detecting GPU...")
        self.lbl_gpu_status.setStyleSheet("color: #999999; font-style: italic;")
        gpu_row.addWidget(self.lbl_gpu_status)

        gpu_row.addStretch()

        self.btn_check_gpu = QPushButton("Re-check GPU")
        self.btn_check_gpu.setMaximumWidth(120)
        self.btn_check_gpu.setToolTip(
            "Re-detect the GPU and verify that the selected SLEAP install's\n"
            "deep-learning backend can actually see it (takes ~30s)."
        )
        self.btn_check_gpu.clicked.connect(lambda: self.probe_gpu(verbose=True))
        self.btn_check_gpu.setStyleSheet("""
            QPushButton {
                background-color: #3f3f3f;
                color: #cccccc;
                padding: 4px 10px;
                font-size: 9pt;
            }
            QPushButton:hover {
                background-color: #4f4f4f;
            }
        """)
        gpu_row.addWidget(self.btn_check_gpu)

        options_layout.addLayout(gpu_row)

        # CSV Creation Option (first and checked by default)
        self.chk_create_csv = QCheckBox("Create CSV prediction file (in addition to .slp)")
        self.chk_create_csv.setChecked(True)
        options_layout.addWidget(self.chk_create_csv)
        
        self.chk_overwrite = QCheckBox("Overwrite existing prediction files")
        self.chk_overwrite.setChecked(True)
        options_layout.addWidget(self.chk_overwrite)
        
        # Tracking Options
        self.chk_tracking = QCheckBox("Add Tracking (assign identities across frames)")
        self.chk_tracking.setChecked(True)  # Changed to True by default
        self.chk_tracking.stateChanged.connect(self.toggle_tracking_options)
        options_layout.addWidget(self.chk_tracking)
        
        # Tracking Parameters Container
        self.tracking_widget = QWidget()
        tracking_layout = QVBoxLayout()
        tracking_layout.setContentsMargins(20, 5, 0, 5)  # Indent tracking options
        
        # Create a grid layout for better organization
        grid_layout = QGridLayout()
        
        # Row 1: Tracker Method and Similarity Method
        grid_layout.addWidget(QLabel("Tracker Method:"), 0, 0)
        self.combo_tracker = QComboBox()
        self.combo_tracker.addItems(["simple", "flow", "simplemaxtracks", "flowmaxtracks", "None"])
        self.combo_tracker.setCurrentText("simple")
        self.combo_tracker.setStyleSheet("""
            QComboBox {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #3f3f3f;
                padding: 5px;
                border-radius: 3px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #1e1e1e;
                color: #cccccc;
                selection-background-color: #0078d4;
            }
        """)
        grid_layout.addWidget(self.combo_tracker, 0, 1)
        
        grid_layout.addWidget(QLabel("Similarity Method:"), 0, 2)
        self.combo_similarity = QComboBox()
        self.combo_similarity.addItems(["instance", "normalized_instance", "object_keypoint", "centroid", "iou"])
        self.combo_similarity.setCurrentText("instance")
        self.combo_similarity.setStyleSheet("""
            QComboBox {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #3f3f3f;
                padding: 5px;
                border-radius: 3px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #1e1e1e;
                color: #cccccc;
                selection-background-color: #0078d4;
            }
        """)
        grid_layout.addWidget(self.combo_similarity, 0, 3)
        
        # Row 2: Match Method and Max Tracks
        grid_layout.addWidget(QLabel("Match Method:"), 1, 0)
        self.combo_match = QComboBox()
        self.combo_match.addItems(["greedy", "hungarian"])
        self.combo_match.setCurrentText("greedy")
        self.combo_match.setStyleSheet("""
            QComboBox {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #3f3f3f;
                padding: 5px;
                border-radius: 3px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #1e1e1e;
                color: #cccccc;
                selection-background-color: #0078d4;
            }
        """)
        grid_layout.addWidget(self.combo_match, 1, 1)
        
        grid_layout.addWidget(QLabel("Max Tracks:"), 1, 2)
        self.spin_max_tracks = QSpinBox()
        self.spin_max_tracks.setMinimum(0)
        self.spin_max_tracks.setMaximum(50)
        self.spin_max_tracks.setValue(0)  # 0 = no limit
        self.spin_max_tracks.setSpecialValueText("No limit")
        self.spin_max_tracks.setStyleSheet("""
            QSpinBox {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #3f3f3f;
                padding: 5px;
                border-radius: 3px;
            }
        """)
        grid_layout.addWidget(self.spin_max_tracks, 1, 3)
        
        # Row 3: Track Window and Robust Quantile
        grid_layout.addWidget(QLabel("Track Window:"), 2, 0)
        self.spin_track_window = QSpinBox()
        self.spin_track_window.setMinimum(1)
        self.spin_track_window.setMaximum(20)
        self.spin_track_window.setValue(10)  # Default is 10 in SLEAP 1.5.2
        self.spin_track_window.setToolTip("How many frames back to look for matches")
        self.spin_track_window.setStyleSheet("""
            QSpinBox {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #3f3f3f;
                padding: 5px;
                border-radius: 3px;
            }
        """)
        grid_layout.addWidget(self.spin_track_window, 2, 1)
        
        grid_layout.addWidget(QLabel("Robust Quantile:"), 2, 2)
        self.spin_robust = QDoubleSpinBox()
        self.spin_robust.setMinimum(0.0)
        self.spin_robust.setMaximum(1.0)
        self.spin_robust.setSingleStep(0.05)
        self.spin_robust.setValue(0.95)  # Default is 0.95 (robust) in SLEAP 1.5.2
        self.spin_robust.setToolTip("Robust quantile of similarity score (1.0 = use max, non-robust)")
        self.spin_robust.setStyleSheet("""
            QDoubleSpinBox {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #3f3f3f;
                padding: 5px;
                border-radius: 3px;
            }
        """)
        grid_layout.addWidget(self.spin_robust, 2, 3)

        tracking_layout.addLayout(grid_layout)

        # Post-processing checkbox
        self.chk_post_connect = QCheckBox("Connect single track breaks (post-processing)")
        self.chk_post_connect.setChecked(False)
        self.chk_post_connect.setToolTip("Fill in single-frame gaps in tracks after tracking")
        tracking_layout.addWidget(self.chk_post_connect)

        self.tracking_widget.setLayout(tracking_layout)
        self.tracking_widget.setVisible(True)  # Visible by default since tracking is checked
        options_layout.addWidget(self.tracking_widget)

        # Batch Size — an inference setting, not a tracking one, so it lives in
        # its own always-visible row (grouped with the settings above but never
        # hidden when tracking is turned off). Indented and column-aligned so it
        # reads as a continuation of the settings grid, directly under Track
        # Window.
        batch_row = QGridLayout()
        batch_row.setContentsMargins(20, 0, 0, 5)
        batch_row.addWidget(QLabel("Batch Size:"), 0, 0)
        self.spin_batch_size = QSpinBox()
        self.spin_batch_size.setMinimum(1)
        self.spin_batch_size.setMaximum(128)
        self.spin_batch_size.setValue(4)
        self.spin_batch_size.setToolTip(
            "Frames sent to the model at once. On a GPU, larger batches are\n"
            "substantially faster but use more VRAM — try 16-32 on a 12 GB+\n"
            "card and lower it if inference runs out of memory."
        )
        self.spin_batch_size.setStyleSheet("""
            QSpinBox {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #3f3f3f;
                padding: 5px;
                border-radius: 3px;
            }
        """)
        batch_row.addWidget(self.spin_batch_size, 0, 1)
        # Match the settings grid's four-column proportions so "Batch Size:"
        # lines up under "Track Window:" and the spinner under its field.
        batch_row.setColumnStretch(0, 0)
        batch_row.setColumnStretch(1, 1)
        batch_row.setColumnStretch(2, 0)
        batch_row.setColumnStretch(3, 1)
        options_layout.addLayout(batch_row)
        
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Run and Stop Buttons
        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton("▶️ Run Inference")
        self.btn_run.clicked.connect(self.run_inference)
        self.btn_run.setEnabled(False)
        self.btn_run.setStyleSheet("""
            QPushButton {
                background-color: #16825d;
                padding: 10px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #1a9667;
            }
            QPushButton:disabled {
                background-color: #3f3f3f;
            }
        """)
        btn_layout.addWidget(self.btn_run)
        
        self.btn_stop = QPushButton("⏹️ Stop Processing")
        self.btn_stop.clicked.connect(self.stop_inference)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #c42b1c;
                padding: 10px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #d83b2c;
            }
            QPushButton:disabled {
                background-color: #3f3f3f;
            }
        """)
        btn_layout.addWidget(self.btn_stop)
        
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # Output Log Group
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout()
        
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFont(QFont("Consolas", 9))
        # No wrapping: a long command or status line scrolls horizontally
        # instead of reflowing into several lines when the window is resized.
        self.log_output.setLineWrapMode(QTextEdit.NoWrap)
        # A floor so the log stays usable, and a ceiling so its expanding policy
        # can't starve the option groups of height inside the scroll area.
        self.log_output.setMinimumHeight(150)
        self.log_output.setMaximumHeight(400)
        log_layout.addWidget(self.log_output)

        log_btn_layout = QHBoxLayout()
        secondary_btn_style = """
            QPushButton {
                background-color: #3f3f3f;
                color: #cccccc;
                padding: 5px 12px;
                font-size: 9pt;
            }
            QPushButton:hover {
                background-color: #4f4f4f;
            }
        """

        self.btn_copy_log = QPushButton("Copy Logs to Clipboard")
        self.btn_copy_log.setToolTip("Copy the full processing log to the clipboard")
        self.btn_copy_log.clicked.connect(self.copy_logs_to_clipboard)
        self.btn_copy_log.setStyleSheet(secondary_btn_style)
        log_btn_layout.addWidget(self.btn_copy_log)

        btn_clear_log = QPushButton("Clear Log")
        btn_clear_log.clicked.connect(self.clear_logs)
        btn_clear_log.setStyleSheet(secondary_btn_style)
        log_btn_layout.addWidget(btn_clear_log)

        log_btn_layout.addStretch()
        log_layout.addLayout(log_btn_layout)

        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        # Put the entire UI inside a vertical scroll area so nothing gets
        # squashed when the window is shorter than the content needs — the
        # option rows keep their natural height and the window scrolls instead.
        content = QWidget()
        content.setLayout(layout)
        # Own the full theme here (see note where _app_stylesheet is defined).
        content.setStyleSheet(self._app_stylesheet)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        # Keep the viewport dark without introducing a competing QPushButton-less
        # stylesheet in the cascade (which is what flattened the buttons before).
        scroll.viewport().setAutoFillBackground(True)
        pal = scroll.viewport().palette()
        pal.setColor(scroll.viewport().backgroundRole(), QColor("#2b2b2b"))
        scroll.viewport().setPalette(pal)

        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self.setLayout(outer)

        self.log("Ready to configure SLEAP inference...")
        self.log("1. Add video folder(s)")
        self.log("2. Select model type and model folder(s)")
        self.log("3. Select the SLEAP installation (uv tool or conda env)")
        self.log("4. Click 'Run Inference' to start\n")
    
    @staticmethod
    def _style_list(list_widget):
        """Apply the shared look for the folder/model lists.

        Long paths are elided in the middle so the informative tail (the
        folder name) stays visible at any window width; the full path is
        always available as a tooltip.
        """
        list_widget.setAlternatingRowColors(True)
        list_widget.setUniformItemSizes(True)
        list_widget.setTextElideMode(Qt.ElideMiddle)
        list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        list_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        list_widget.setFont(QFont("Consolas", 9))

    @staticmethod
    def _add_path_item(list_widget, text, full_path):
        """Add a list row showing *text* with *full_path* as its tooltip."""
        list_widget.addItem(text)
        list_widget.item(list_widget.count() - 1).setToolTip(full_path)

    @staticmethod
    def _count_videos_in_folder(folder):
        """Count inference-eligible videos in *folder* (same filter as the run).

        Returns None if the folder can't be listed (e.g. a dropped network
        drive), so the summary can say "?" rather than a misleading 0.
        """
        try:
            return sum(
                1 for f in os.listdir(folder)
                if f.lower().endswith((".mp4", ".avi", ".mov"))
                and not f.endswith("_roiTracked.mp4")
            )
        except Exception:
            return None

    def _pluralize(self, n, word):
        return f"{n} {word}{'' if n == 1 else 's'}"

    def update_video_summary(self):
        """Refresh the one-line summary under the video list.

        Shows folder count *and* how many videos those folders contain, since
        a folder count alone doesn't tell you how much work a run entails.
        Per-folder counts are cached (see add_video_folder) so this stays cheap
        even when the videos live on a slow network drive.
        """
        n_folders = len(self.video_folders)
        n_indiv = len(self.individual_videos)
        if not n_folders and not n_indiv:
            self.lbl_video_summary.setText("No videos selected")
            return

        parts = []
        if n_folders:
            counts = [self.folder_video_counts.get(f) for f in self.video_folders]
            known = [c for c in counts if c is not None]
            total = sum(known)
            # If any folder failed to enumerate, mark the total approximate.
            total_txt = f"{total}+" if len(known) < n_folders else str(total)
            parts.append(f"{self._pluralize(n_folders, 'folder')} "
                         f"containing {total_txt} "
                         f"video{'' if total == 1 and len(known) == n_folders else 's'}")
        if n_indiv:
            parts.append(self._pluralize(n_indiv, 'individual video'))
        self.lbl_video_summary.setText("Selected: " + ", ".join(parts))

    def log(self, message):
        """Add message to log output"""
        self.log_output.append(message)
        self._status_line_active = False
        self.log_output.verticalScrollBar().setValue(
            self.log_output.verticalScrollBar().maximum()
        )

    def log_update(self, message):
        """Rewrite the live status line in place.

        The status line is whatever was last written by this method; anything
        written by ``log`` ends it, so the next progress update starts a fresh
        line instead of overwriting a real message.
        """
        if not self._status_line_active:
            self.log_output.append(message)
            self._status_line_active = True
        else:
            cursor = self.log_output.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.movePosition(QTextCursor.StartOfBlock, QTextCursor.KeepAnchor)
            cursor.removeSelectedText()
            cursor.insertText(message)
        self.log_output.verticalScrollBar().setValue(
            self.log_output.verticalScrollBar().maximum()
        )

    def copy_logs_to_clipboard(self):
        """Copy the full processing log to the system clipboard"""
        text = self.log_output.toPlainText()
        QApplication.clipboard().setText(text)
        line_count = len(text.splitlines())
        self.btn_copy_log.setText(f"✅ Copied {line_count} lines")
        QTimer.singleShot(2000, lambda: self.btn_copy_log.setText("Copy Logs to Clipboard"))

    def clear_logs(self):
        """Clear the processing log"""
        self.log_output.clear()
        self._status_line_active = False


    def _make_dialog(self, caption, file_mode, allow_multi_dirs=False):
        """Build a non-native, dark-themed QFileDialog.

        Every file/folder picker in this tool uses the non-native Qt dialog so
        they look consistent (dark, matching the window) and so folder pickers
        can offer multi-selection — the native OS dialog can't select several
        directories at once.
        """
        dialog = QFileDialog(self, caption)
        dialog.setFileMode(file_mode)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        if file_mode == QFileDialog.Directory:
            dialog.setOption(QFileDialog.ShowDirsOnly, True)
        dialog.setStyleSheet(self._app_stylesheet)  # keep it dark like the tool
        if allow_multi_dirs:
            # Directory mode is single-select by default; flip the internal
            # list/tree views to extended selection so several folders can be
            # shift/ctrl-clicked at once.
            for view in dialog.findChildren((QListView, QTreeView)):
                view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        return dialog

    def _pick_folders(self):
        """Choose one or more folders (multi-select) — used for video folders."""
        dialog = self._make_dialog(
            "Select folder(s) containing video files for inference",
            QFileDialog.Directory, allow_multi_dirs=True)
        if dialog.exec_() == QDialog.Accepted:
            # Only keep real directories (selectedFiles can echo the line-edit text).
            return [f for f in dialog.selectedFiles() if os.path.isdir(f)]
        return []

    def _pick_folder(self, caption):
        """Choose a single folder — used for model directories."""
        dialog = self._make_dialog(caption, QFileDialog.Directory)
        if dialog.exec_() == QDialog.Accepted:
            dirs = [f for f in dialog.selectedFiles() if os.path.isdir(f)]
            return dirs[0] if dirs else ""
        return ""

    def _pick_files(self, caption, name_filter):
        """Choose one or more existing files — used for individual videos."""
        dialog = self._make_dialog(caption, QFileDialog.ExistingFiles)
        dialog.setNameFilter(name_filter)
        if dialog.exec_() == QDialog.Accepted:
            return [f for f in dialog.selectedFiles() if os.path.isfile(f)]
        return []

    def _add_folder(self, folder):
        """Add one folder to the processing list. Returns 'added'|'duplicate'."""
        if folder in self.video_folders:
            return 'duplicate'
        self.video_folders.append(folder)
        # Count videos once, now, and cache it (network drives are slow).
        count = self._count_videos_in_folder(folder)
        self.folder_video_counts[folder] = count
        count_txt = "?" if count is None else str(count)
        net = is_network_path(folder)
        marker = "🌐  " if net else "📁  "
        self._add_path_item(
            self.folder_list, f"{marker}{folder}   ({count_txt} videos)", folder)
        self.log(f"✅ Added video folder ({count_txt} videos): {folder}")
        if net:
            self.log("   🌐 On a network drive — inference may be slower "
                     "(see warning when you run).")
            self._network_warning_acknowledged = False
        return 'added'

    def add_video_folder(self):
        """Add one or more video folders to the processing list"""
        folders = self._pick_folders()
        if not folders:
            return

        added = duplicates = 0
        for folder in folders:
            if self._add_folder(folder) == 'added':
                added += 1
            else:
                duplicates += 1

        if added:
            self.update_video_summary()
            self.update_run_button()
        if added and len(folders) > 1:
            self.log(f"➕ Added {added} folder(s)"
                     + (f", skipped {duplicates} already in the list" if duplicates else ""))
        elif duplicates and not added:
            QMessageBox.information(
                self, "Already Added",
                "The selected folder(s) are already in the list.")

    def add_individual_videos(self):
        """Add individual video files to the processing list"""
        video_files = self._pick_files(
            "Select Video File(s) for Inference",
            "Video Files (*.mp4 *.avi *.mov);;All Files (*)"
        )

        if video_files:
            added = 0
            for video_path in video_files:
                # Add video if not already in list
                if video_path not in self.individual_videos:
                    self.individual_videos.append(video_path)
                    # Display just the filename; the full path is the tooltip
                    video_name = os.path.basename(video_path)
                    marker = "🌐  " if is_network_path(video_path) else "📹  "
                    self._add_path_item(self.folder_list, f"{marker}{video_name}", video_path)
                    added += 1

            if added > 0:
                self.log(f"✅ Added {added} individual video(s)")
                if any(is_network_path(v) for v in self.individual_videos):
                    self.log("   🌐 Some videos are on a network drive — inference "
                             "may be slower (see warning when you run).")
                    self._network_warning_acknowledged = False
                self.update_video_summary()
                self.update_run_button()
            else:
                self.log("ℹ️ All selected videos were already in the list")
    
    def clear_video_folders(self):
        """Clear all video folders and individual videos"""
        self.video_folders.clear()
        self.individual_videos.clear()
        self.folder_video_counts.clear()
        self.folder_list.clear()
        self._network_warning_acknowledged = False
        self.log("🗑️ Cleared all video folders and individual videos")
        self.update_video_summary()
        self.update_run_button()
    
    def toggle_tracking_options(self, state):
        """Show/hide tracking options based on checkbox state"""
        self.tracking_widget.setVisible(state == Qt.Checked)
    
    def on_install_changed(self, _index):
        """User picked a different SLEAP install — re-check GPU support."""
        self.update_run_button()
        self.probe_gpu(verbose=True)

    def probe_gpu(self, verbose=False):
        """Detect the GPU and check that the selected install can use it.

        Runs in the background because importing the deep-learning backend
        (PyTorch for SLEAP >= 1.5, TensorFlow before that) takes tens of
        seconds.
        """
        if self.gpu_probe and self.gpu_probe.isRunning():
            return

        install = self.combo_install.currentData()
        self.btn_check_gpu.setEnabled(False)
        self.lbl_gpu_status.setText("Detecting GPU...")
        self.lbl_gpu_status.setStyleSheet("color: #0078d4; font-style: italic;")
        if verbose:
            self.log(f"🔍 Checking GPU availability"
                     f"{f' for {describe_install(install)}' if install else ''}...")

        self._gpu_probe_verbose = verbose
        self.gpu_probe = GpuProbeWorker(install)
        self.gpu_probe.result.connect(self.on_gpu_probe_finished)
        self.gpu_probe.start()

    def on_gpu_probe_finished(self, info):
        """Update the GPU checkbox/label from the probe result."""
        self.gpu_info = info
        self.btn_check_gpu.setEnabled(True)
        verbose = getattr(self, '_gpu_probe_verbose', False)

        name = info.get('name')
        detail = info.get('detail') or name
        backend = info.get('backend')
        backend_name = {'torch': 'PyTorch', 'tensorflow': 'TensorFlow'}.get(backend, backend)
        backend_version = info.get('backend_version') or ''
        backend_gpus = info.get('backend_gpus')
        install = info.get('install')
        where = describe_install(install) if install else 'this install'
        error = info.get('error')

        self.lbl_gpu_status.setToolTip("")

        if not name:
            # Nothing usable found at the driver level.
            self.chk_use_gpu.setChecked(False)
            self.chk_use_gpu.setEnabled(False)
            self.lbl_gpu_status.setText("No GPU detected — inference will run on CPU")
            self.lbl_gpu_status.setStyleSheet("color: #ff9f43; font-style: italic;")
            if verbose:
                self.log("⚠️ No GPU detected (nvidia-smi found no NVIDIA device)")
            return

        self.chk_use_gpu.setEnabled(True)

        if backend_gpus is None:
            # A GPU exists, but the install's backend could not be questioned
            # (no install selected, probe failed, or no backend installed).
            self.chk_use_gpu.setChecked(True)
            self.lbl_gpu_status.setText(f"✅ {detail}")
            self.lbl_gpu_status.setStyleSheet("color: #4caf50; font-style: normal;")
            if verbose:
                self.log(f"✅ GPU detected: {detail}")
                if error:
                    self.log(f"   ⚠️ Could not verify GPU access for {where}: {error}")
        elif backend_gpus > 0:
            self.chk_use_gpu.setChecked(True)
            self.lbl_gpu_status.setText(
                f"✅ {detail} — {backend_name} sees {backend_gpus} GPU(s)")
            self.lbl_gpu_status.setStyleSheet("color: #4caf50; font-style: normal;")
            if verbose:
                self.log(f"✅ GPU detected and usable: {detail}")
                self.log(f"   {backend_name} {backend_version} in {where} "
                         f"sees {backend_gpus} GPU(s)")
        else:
            # The important case: a real GPU that SLEAP cannot actually use.
            # Leave the box on (harmless — SLEAP falls back to CPU) but say so.
            self.chk_use_gpu.setChecked(True)
            self.lbl_gpu_status.setText(
                f"⚠️ {name} present, but {backend_name} in {where} sees no GPU")
            self.lbl_gpu_status.setStyleSheet("color: #ff9f43; font-style: normal;")
            self.lbl_gpu_status.setToolTip(
                "SLEAP will fall back to CPU, which is many times slower.\n"
                "Usually means this install has a CPU-only build of its deep-learning\n"
                "backend, or its CUDA libraries do not match the installed driver."
            )
            self.log(f"⚠️ {name} is present, but {backend_name} {backend_version} in "
                     f"{where} reports 0 GPUs — inference will run on the CPU.")
            if info.get('cuda_build') is False:
                self.log(f"   {backend_name} in this install is a CPU-only build. "
                         f"Reinstall it with CUDA support to use the GPU.")
            else:
                self.log(f"   {backend_name} was built with CUDA but cannot see the "
                         f"device — check the NVIDIA driver and the install's CUDA "
                         f"library versions.")

    def detect_sleap_installations(self):
        """Detect SLEAP installations (uv tools and conda envs) into the dropdown"""
        self.combo_install.clear()
        self.combo_install.addItem("Detecting SLEAP installations...", None)
        self.lbl_env_status.setText("Detecting SLEAP installations...")
        self.lbl_env_status.setStyleSheet("color: #0078d4;")
        QApplication.processEvents()

        try:
            installs = detect_sleap_installs()
        except Exception as e:
            self.combo_install.clear()
            self.combo_install.addItem("Error", None)
            self.lbl_env_status.setText(f"Error: {str(e)}")
            self.lbl_env_status.setStyleSheet("color: #ff6b6b;")
            self.log(f"❌ Error detecting SLEAP installations: {str(e)}")
            return

        self.combo_install.clear()

        if not installs:
            self.combo_install.addItem("No SLEAP installations found", None)
            self.lbl_env_status.setText(
                "No SLEAP installation found (looked for uv tools and conda environments)")
            self.lbl_env_status.setStyleSheet("color: #ff6b6b;")
            self.log("❌ No SLEAP installation found — is conda or uv on your PATH?")
            self.update_run_button()
            return

        for install in installs:
            self.combo_install.addItem(install['label'], install)

        with_sleap = [i for i in installs if i.get('has_sleap')]
        n_uv = sum(1 for i in with_sleap if i['kind'] == 'uv')
        n_conda = len(with_sleap) - n_uv

        if with_sleap:
            found = []
            if n_uv:
                found.append(f"{n_uv} uv tool install(s)")
            if n_conda:
                found.append(f"{n_conda} conda environment(s)")
            summary = "Found SLEAP in " + " and ".join(found)
            self.lbl_env_status.setStyleSheet("color: #cccccc;")
        else:
            summary = ("No SLEAP found in any uv tool or conda environment "
                       "— pick one manually if you know where it lives")
            self.lbl_env_status.setStyleSheet("color: #ff9f43;")
        self.lbl_env_status.setText(summary)
        self.log(f"{'✅' if with_sleap else '⚠️'} {summary}")

        for install in with_sleap:
            self.log(f"   {install['label']}")

        # detect_sleap_installs() sorts runnable installs (SLEAP + a
        # deep-learning backend) first, so index 0 is the best candidate.
        self.combo_install.setCurrentIndex(0)
        selected = installs[0]
        self.log(f"🎯 Auto-selected: {describe_install(selected)}")
        if selected.get('has_sleap') and not selected.get('backend'):
            self._log_gui_only_hint(selected)

        self.update_run_button()
        # Now that we know which install SLEAP will run from, find out whether
        # a GPU exists and whether that install can actually use it.
        self.probe_gpu(verbose=True)

    def _log_gui_only_hint(self, install):
        """Explain that a backend-less SLEAP install can't run inference, and how to fix."""
        self.log("   ⚠️ This is a GUI-only SLEAP install: sleap-nn (and its PyTorch "
                 "backend) is not installed, so it can label/view but 'sleap track' "
                 "will error out and it cannot run inference.")
        if install.get('kind') == 'uv':
            self.log(f"      Fix: reinstall with the GPU extra, e.g. "
                     f"`uv tool install \"sleap[nn-cuda128]\"` (matches CUDA 12.8).")
        else:
            self.log("      Fix: install sleap-nn into this environment, e.g. "
                     "`pip install \"sleap-nn[torch]\"`.")

    def test_sleap_install(self):
        """Test that SLEAP actually runs in the selected installation"""
        install = self.combo_install.currentData()
        if not install:
            QMessageBox.warning(self, "No Installation",
                                "Please select a SLEAP installation first.")
            return

        where = describe_install(install)
        self.lbl_env_status.setText(f"Testing SLEAP in {where}...")
        self.lbl_env_status.setStyleSheet("color: #0078d4;")
        self.log(f"🔍 Testing SLEAP in {where}")
        QApplication.processEvents()

        try:
            result = subprocess.run(
                install_cmd(install, "sleap-track", ["--help"]),
                capture_output=True,
                text=True,
                shell=True,
                timeout=120
            )

            # sleap-track's click-based CLI exits non-zero for --help in some
            # versions, so treat "printed a usage block" as success too.
            help_text = (result.stdout or '') + (result.stderr or '')
            worked = result.returncode == 0 or 'Usage: sleap-track' in help_text

            if worked:
                version = install.get('version') or self._query_sleap_version(install)

                self.lbl_env_status.setText(f"✅ SLEAP v{version} available via {where}")
                self.lbl_env_status.setStyleSheet("color: #4caf50;")
                self.log(f"✅ SLEAP v{version} is working in {where}")
                self.log(f"   sleap-track: {install_bin(install, 'sleap-track')}")
                self.update_run_button()
                QMessageBox.information(
                    self,
                    "SLEAP Test Successful",
                    f"SLEAP v{version} is installed and working in {where}"
                )
            else:
                self.lbl_env_status.setText(f"❌ SLEAP not found in {where}")
                self.lbl_env_status.setStyleSheet("color: #ff6b6b;")
                self.log(f"❌ SLEAP not found in {where}")
                QMessageBox.warning(
                    self,
                    "SLEAP Not Found",
                    f"SLEAP is not installed or not working in {where}\n\n"
                    f"Error: {help_text[:300] if help_text else 'Command failed'}"
                )

        except subprocess.TimeoutExpired:
            self.lbl_env_status.setText(f"❌ Timeout testing {where}")
            self.lbl_env_status.setStyleSheet("color: #ff6b6b;")
            self.log(f"❌ Timeout testing SLEAP in {where}")
            QMessageBox.warning(self, "Timeout", f"Testing SLEAP in {where} timed out.")

        except Exception as e:
            self.lbl_env_status.setText(f"❌ Error testing {where}")
            self.lbl_env_status.setStyleSheet("color: #ff6b6b;")
            self.log(f"❌ Error testing SLEAP: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error testing SLEAP: {str(e)}")

    @staticmethod
    def _query_sleap_version(install):
        """Ask the install's Python for sleap.__version__ (conda envs)."""
        try:
            if install.get('kind') == 'conda':
                cmd = [conda_executable(), "run", "-n", install['name'], "python",
                       "-c", "import sleap; print(sleap.__version__)"]
            elif install.get('python'):
                cmd = [install['python'], "-c", "import sleap; print(sleap.__version__)"]
            else:
                return "unknown"
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    shell=True, timeout=120)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().splitlines()[-1].strip()
        except Exception:
            pass
        return "unknown"
    
    def find_centered_instance_model(self, centroid_folder):
        """Automatically find the matching centered instance model based on centroid folder.
        
        Looks for a sibling folder with the same date and n=X pattern but with 'centered_instance' in the name.
        Example: 
            Centroid: .../251110_231303.centroid.n=250
            Centered: .../251110_231345.centered_instance.n=250
        """
        import re
        
        parent_dir = os.path.dirname(centroid_folder)
        centroid_basename = os.path.basename(centroid_folder)
        
        # Extract date pattern (YYMMDD_HHMMSS) and n=X from centroid folder
        date_match = re.search(r'(\d{6}_\d{6})', centroid_basename)
        n_match = re.search(r'n=(\d+)', centroid_basename)
        
        if not date_match or not n_match:
            return None
        
        date_prefix = date_match.group(1)[:6]  # Just the YYMMDD part
        n_value = n_match.group(1)
        
        # Look for matching centered_instance folder
        if not os.path.exists(parent_dir):
            return None
        
        for folder_name in os.listdir(parent_dir):
            folder_path = os.path.join(parent_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            
            # Check if it matches the pattern: same date prefix, contains 'centered_instance', and same n=X
            if (folder_name.startswith(date_prefix) and 
                'centered_instance' in folder_name.lower() and 
                f'n={n_value}' in folder_name):
                return folder_path
        
        return None
    
    def select_topdown_models(self):
        """Select top-down models (centroid + centered instance)"""
        self.is_top_down = True
        self.model_paths.clear()
        self.model_list.clear()
        
        # Select centroid model
        centroid_folder = self._pick_folder("Select CENTROID model folder")
        if not centroid_folder:
            return
        
        # Try to automatically find centered instance model
        centered_folder = self.find_centered_instance_model(centroid_folder)
        
        if centered_folder:
            self.log(f"🔍 Auto-detected centered instance model: {os.path.basename(centered_folder)}")
        else:
            # Fallback to manual selection if auto-detection fails
            self.log("⚠️ Could not auto-detect centered instance model. Please select manually.")
            centered_folder = self._pick_folder("Select CENTERED INSTANCE model folder")
            if not centered_folder:
                return
        
        self.model_paths = [centroid_folder, centered_folder]
        self._add_path_item(self.model_list, f"◉  Centroid    {centroid_folder}", centroid_folder)
        self._add_path_item(self.model_list, f"◎  Centered    {centered_folder}", centered_folder)

        self.log(f"✅ Selected TOP-DOWN models:")
        self.log(f"   Centroid: {centroid_folder}")
        self.log(f"   Centered Instance: {centered_folder}")
        
        self.update_run_button()
    
    def select_bottomup_models(self):
        """Select bottom-up model"""
        self.is_top_down = False
        self.model_paths.clear()
        self.model_list.clear()
        
        # Select bottom-up model
        model_folder = self._pick_folder("Select BOTTOM-UP model folder")
        if not model_folder:
            return
        
        self.model_paths = [model_folder]
        self._add_path_item(self.model_list, f"◉  Bottom-Up   {model_folder}", model_folder)

        self.log(f"✅ Selected BOTTOM-UP model: {model_folder}")
        
        self.update_run_button()
    
    def update_run_button(self):
        """Enable/disable run button based on configuration"""
        has_videos = len(self.video_folders) > 0 or len(self.individual_videos) > 0
        has_models = len(self.model_paths) > 0
        has_environment = self.combo_install.currentData() is not None
        
        can_run = has_videos and has_models and has_environment
        self.btn_run.setEnabled(can_run)
    
    def _network_paths_in_selection(self):
        """Return the selected folders/videos that live on a network drive."""
        candidates = list(self.video_folders) + list(self.individual_videos)
        return [p for p in candidates if is_network_path(p)]

    def _maybe_warn_network_drive(self):
        """Warn once if inference will read from a network drive.

        Returns True to proceed, False if the user chose to cancel. After the
        user acknowledges (or there is nothing on a network drive), it stays
        quiet until the selection changes to include a new network path.
        """
        if self._network_warning_acknowledged:
            return True

        net_paths = self._network_paths_in_selection()
        if not net_paths:
            return True

        shown = "\n".join(f"   • {p}" for p in net_paths[:6])
        if len(net_paths) > 6:
            shown += f"\n   • …and {len(net_paths) - 6} more"

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Videos on a network drive")
        box.setText("Some of your videos are on a network / mapped drive:")
        box.setInformativeText(
            f"{shown}\n\n"
            "Reading video frames over the network is often the slowest part of "
            "inference — the GPU ends up waiting on the network instead of "
            "running at full speed.\n\n"
            "For faster inference, copy the videos to a local SSD first and run "
            "from there.\n\n"
            "Continue anyway?"
        )
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        box.button(QMessageBox.Yes).setText("Continue anyway")
        box.button(QMessageBox.Cancel).setText("Cancel")
        box.setDefaultButton(QMessageBox.Cancel)

        if box.exec_() != QMessageBox.Yes:
            self.log("⏹️ Run cancelled — videos are on a network drive. "
                     "Copy them to a local SSD for faster inference.")
            return False

        # Don't nag again until the selection changes.
        self._network_warning_acknowledged = True
        self.log("🌐 Proceeding with videos on a network drive "
                 "(inference may be slower than from a local SSD).")
        return True

    def run_inference(self):
        """Start the inference process"""
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(
                self,
                "Already Running",
                "Inference is already in progress. Please wait for it to complete."
            )
            return

        # Warn (once) if the videos live on a network drive — reading frames
        # over the network commonly starves the GPU and slows inference.
        if not self._maybe_warn_network_drive():
            return

        # Confirm with user
        video_count = 0
        for folder in self.video_folders:
            video_files = [f for f in os.listdir(folder)
                          if f.lower().endswith((".mp4", ".avi", ".mov"))
                          and not f.endswith("_roiTracked.mp4")]  # Ignore ROI tracked videos
            video_count += len(video_files)
        
        # Add individual videos to count
        video_count += len(self.individual_videos)
        
        model_type = "Top-Down (Centroid + Centered)" if self.is_top_down else "Bottom-Up"
        
        msg = f"Ready to run inference:\n\n"
        msg += f"Video folders: {len(self.video_folders)}\n"
        msg += f"Individual videos: {len(self.individual_videos)}\n"
        msg += f"Total videos: {video_count}\n"
        msg += f"Model type: {model_type}\n"
        msg += f"SLEAP install: {describe_install(self.combo_install.currentData())}\n"
        msg += f"Device: {'GPU (auto)' if self.chk_use_gpu.isChecked() else 'CPU only'}\n"
        msg += f"Batch size: {self.spin_batch_size.value()}\n"
        msg += f"Overwrite existing: {'Yes' if self.chk_overwrite.isChecked() else 'No'}\n"
        msg += f"Create CSV files: {'Yes' if self.chk_create_csv.isChecked() else 'No'}\n"
        msg += f"Tracking enabled: {'Yes' if self.chk_tracking.isChecked() else 'No'}\n"
        if self.chk_tracking.isChecked():
            msg += f"  - Tracker: {self.combo_tracker.currentText()}\n"
            msg += f"  - Similarity: {self.combo_similarity.currentText()}\n"
            msg += f"  - Match: {self.combo_match.currentText()}\n"
            msg += f"  - Max Tracks: {self.spin_max_tracks.value() if self.spin_max_tracks.value() > 0 else 'No limit'}\n"
            msg += f"  - Post-connect breaks: {'Yes' if self.chk_post_connect.isChecked() else 'No'}\n"
        msg += "\nContinue?"
        
        reply = QMessageBox.question(
            self,
            "Confirm Inference",
            msg,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Disable run button, enable stop button during processing
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.log("\n" + "="*60)
        self.log("🚀 Starting SLEAP inference...")
        self.log(f"Model type: {model_type}")
        self.log(f"Video folders: {len(self.video_folders)}")
        self.log(f"Individual videos: {len(self.individual_videos)}")
        self.log(f"Total videos: {video_count}")
        self.log(f"SLEAP install: {describe_install(self.combo_install.currentData())}")
        if self.chk_use_gpu.isChecked():
            gpu_name = self.gpu_info.get('name') or 'auto-selected GPU'
            self.log(f"Device: GPU — {gpu_name}")
        else:
            self.log("Device: CPU only")
        self.log(f"Batch size: {self.spin_batch_size.value()}")
        self.log("="*60 + "\n")
        
        # Start worker thread
        self.worker = InferenceWorker(
            self.video_folders,
            self.individual_videos,
            self.model_paths,
            self.chk_overwrite.isChecked(),
            self.chk_create_csv.isChecked(),
            self.chk_tracking.isChecked(),
            self.combo_tracker.currentText(),
            self.combo_similarity.currentText(),
            self.combo_match.currentText(),
            self.spin_max_tracks.value() if hasattr(self, 'spin_max_tracks') else 0,
            self.spin_track_window.value() if hasattr(self, 'spin_track_window') else 10,
            self.spin_robust.value() if hasattr(self, 'spin_robust') else 0.95,
            self.chk_post_connect.isChecked() if hasattr(self, 'chk_post_connect') else False,
            self.combo_install.currentData(),
            self.chk_use_gpu.isChecked(),
            self.spin_batch_size.value()
        )
        self.worker.progress.connect(self.log)
        self.worker.progress_update.connect(self.log_update)
        self.worker.finished.connect(self.on_inference_finished)
        self.worker.start()
    
    def stop_inference(self):
        """Stop the running inference process"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Stop Processing",
                "Are you sure you want to stop the inference process?\n\nThe current video will finish processing.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.worker.request_stop()
                self.btn_stop.setEnabled(False)
    
    def on_inference_finished(self, success, message):
        """Handle inference completion"""
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        
        if success:
            QMessageBox.information(
                self,
                "Inference Complete",
                message
            )
        else:
            QMessageBox.critical(
                self,
                "Inference Failed",
                message
            )


def main():
    """Main entry point for standalone execution"""
    import sys
    app = QApplication(sys.argv)
    window = VideoInferenceWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
