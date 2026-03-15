import sys
import os
import random
import numpy as np

# 1. Setup Path to access 'src' from main python folder
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "../../python")
sys.path.append(src_path)

from src import simulation
from src import config

# 2. Force Deterministic Constants
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 3. Override Output Directory
# We want the data to land in ../data relative to this script
debug_data_dir = os.path.abspath(os.path.join(current_dir, "../data"))
config.SIM_DIR = debug_data_dir

# 4. Generate
run_name = "debug_run"
print(f"Generating deterministic world '{run_name}' in {debug_data_dir}...")

output_path = simulation.generate_scene(
    run_name=run_name,
    dataset='mixed',
    reverb=False,       # Anechoic for easier debugging initially? Or True? User said "same logic", usually default is True.
    n_interferers=1,
    snr_target=5
)

# Move the mixture to a standard name for easier access? 
# simulation.py creates a folder `debug_run`. 
# We'll leave it there: ../data/debug_run/mixture.wav

print(f"Generation Complete. Path: {output_path}")
print("Config used: Seed=42, SNR=5dB, Mixed Dataset")
