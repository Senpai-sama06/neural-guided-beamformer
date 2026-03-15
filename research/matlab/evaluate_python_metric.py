import sys
import os
import soundfile as sf
import numpy as np

# 1. Setup Environment to import from sibling 'python' directory
# Get current directory (.../matlab)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Path to python directory (.../python)
python_dir = os.path.join(current_dir, "../python")
sys.path.append(python_dir)

# Now we can import from the python project
try:
    from src import metrics
    from src import config
except ImportError as e:
    print(f"Error importing python modules: {e}")
    print(f"Added path: {python_dir}")
    sys.exit(1)

# 2. Configuration
# MATLAB Output path
est_path = os.path.join(current_dir, "output/output_raw_mask.wav")

# Ground Truth path (hardcoded to the batch run we are using)
# In a real scenario, this might need to be dynamic or passed as an arg
ground_truth_dir = os.path.join(python_dir, "data/simulated/batch_test_000")

def evaluate():
    print("="*60)
    print("CROSS-VERIFICATION: Evaluating MATLAB Output with Python Metrics")
    print("="*60)
    
    # 1. Validation
    if not os.path.exists(est_path):
        print(f"[ERROR] MATLAB output not found at: {est_path}")
        return
    if not os.path.exists(ground_truth_dir):
        print(f"[ERROR] Ground truth directory not found at: {ground_truth_dir}")
        return

    print(f"Estimate:     {os.path.basename(est_path)}")
    print(f"Ground Truth: {ground_truth_dir}")

    # 2. Load Audio
    # Estimate
    est, fs = sf.read(est_path, dtype='float32')
    if est.ndim > 1: est = est[:, 0] # Mono
    
    # Ground Truth
    s_tgt, s_int, s_mix = metrics.load_ground_truth(ground_truth_dir)
    
    if s_tgt is None:
        print("[ERROR] Failed to load ground truth files.")
        return

    # 3. Compute Metrics
    # Note check FS match
    if fs != config.FS:
        print(f"[WARNING] Sample rate mismatch: File={fs}, Config={config.FS}")

    m = metrics.calculate_metrics(est, s_tgt, s_int, config.FS)

    # 4. Report
    header = f"{'Method':<12} | {'SIR':<6} | {'SINR':<6} | {'P-NB':<6} | {'P-WB':<6} | {'STOI':<6}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    print(f"{'MATLAB_RAW':<12} | {m['sir']:.2f}   | {m['sinr']:.2f}   | {m['pesq_nb']:.2f}   | {m['pesq_wb']:.2f}   | {m['stoi']:.2f}")
    print("-" * len(header))

if __name__ == "__main__":
    evaluate()
