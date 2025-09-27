
# FieldNeuroToolbox (fnt)

Preprocessing and analysis toolbox for neurobehavioral data. 

Overview
========


## Pre-requisites
- Github desktop
- Anaconda
- ffmpeg (and the bin file saved to windows environmental path variable, look up youtube tutorial)
- VS Code
- R and R studio (converting code to python slowly but surely)


## Windows Installation

1. Clone the fieldneuro github repo. 

2. Add anaconda to system path so you can run fnt conda environment in standard cmd window
- open anaconda prompt:

```bash
conda info --base
```

- Open Windows Settings → Search for "Environment Variables" → Click Edit the system environment variables.
- In the System Properties window, click Environment Variables.
- Under System variables, find Path, select it, and click Edit.
- Click New and add the following paths:
C:\Users\YourUsername\anaconda3\Scripts
C:\Users\YourUsername\anaconda3
C:\Users\YourUsername\anaconda3\Library\bin


3. Create a new conda environment using a new cmd window (or the integrated terminal in VS code). 
- Note that conda environments can be stored in a user profile and across users (C:\anaconda3\envs). This can matter. I like to store things in the C: drive typically. 
- install the package in editable mode. Any changes made to the code within the github repo will be availbale within the conda environment. 

```bash
conda create --name fieldneuro
conda activate fieldneuro 
cd C:\GitHub\fieldneuro 
pip install -e . 
```



## Mac Installation

```bash
conda create --name fieldneuro
conda activate fieldneuro 
cd ~/Documents/GitHub/fieldneuro # relative to home directory
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
