
# FieldNeuroToolbox (fnt)

Preprocessing and analysis toolbox for neurobehavioral data. 

Authored by Caleb C. Vogt, PhD in collaboration with Claude.

Overview
========


## Pre-requisites
- Github desktop
- Anaconda
- ffmpeg (and the bin file saved to windows environmental path variable, look up youtube tutorial)
- VS Code
- R and R studio (converting code to python slowly but surely)


## Windows Installation

Clone the fnt github repo. 

Install the package in editable mode. Any changes made to the code within the github repo will be availbale within the conda environment. 

```bash
conda create --name fnt python
conda activate fnt 
cd C:\GitHub\fnt 
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

## Launch the GUI

After installation, you can launch the FieldNeuroToolbox GUI in several ways:

**Option 3: Console script (if properly registered)**
```bash
conda activate fnt
fnt-gui
```



## Mac Installation
Open terminal

```bash
conda create --name fnt python
conda activate fnt 
cd ~/Documents/GitHub/fnt # relative to home directory
pip install -e . 
```


Notes: 
- if things aren't runnning in the jupyter notebooks, try restarting the kernel/environment. 
- package dependencies listed in the pyproject.toml

## Organization and general notes
- Core functions are stored in fnt folder
- The Notebooks folder contains notebooks and analyses that are either in progress and will eventually be made into standalone functions or are notebooks for processing data. 
- Analyses for projects are located in the notebooks folder, and organized by projects
- I attempt to use csv files as much as possible


## General contribution guidelines 
- Functions should always be written in lower case with underscores. 



## QuickScripts
- Here are python scripts that can be directly dragged and dropped within a folder of interest and ran via cmd line (or opened with cmd line) in order to a quick operation. Most of these QuickScripts are (or will be) included as standalone functions within the main FNT repo, but sometimes it is just faster and easier to drag and drop a file and delete it when it is done. 
- I try to make most scripts compatible with both MacOS and windows. 
