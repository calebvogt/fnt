## batch_sleap_inference_cleaning.py
## conda activate sleap_env_v1.1.3 in working directory folder
## see here for details: https://sleap.ai/guides/proofreading.html#tracking-method-details

import os
import subprocess

## make sure to set -c for number of animals
subprocess.call('(for %i in (*.slp) do python -m sleap.info.trackcleaner ''%i'' -c 2)', shell=True)   


