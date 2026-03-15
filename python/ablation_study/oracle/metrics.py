#!/usr/bin/env python3
"""
Simplified metrics calculation for Oracle TFLC ablation study.
Only evaluates the oracle output (no gauss/gan variants).
"""

import numpy as np
import soundfile as sf
import os
import sys
import datetime
import csv

# Add parent directory to path to import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from src import config

# Handle optional dependencies gracefully
try:
    from pystoi import stoi
    from pesq import pesq
    DEPENDENCIES_OK = True
except ImportError:
    print("Warning: 'pystoi' or 'pesq' not installed. Metrics will be skipped.")
    DEPENDENCIES_OK = False


class PESQEvaluator:
    def __init__(self, fs):
        self.fs = fs

    def evaluate(self, ref, deg):
        nb_score, wb_score = 0.0, 0.0
        if not DEPENDENCIES_OK:
            return 0.0, 0.0

        try:
            # PESQ requires 8k or 16k
            if self.fs in [8000, 16000]:
                nb_score = pesq(self.fs, ref, deg, 'nb')
            if self.fs == 16000:
                wb_score = pesq(self.fs, ref, deg, 'wb')
        except Exception:
            # PESQ can fail on silent/short clips
            pass
        return nb_score, wb_score


def load_ground_truth(sim_dir):
    """Load ground truth audio files from simulation directory."""
    try:
        s_tgt, _ = sf.read(os.path.join(sim_dir, "target.wav"), dtype='float32')
        s_int, _ = sf.read(os.path.join(sim_dir, "interference.wav"), dtype='float32')
        s_mix, _ = sf.read(os.path.join(sim_dir, "mixture.wav"), dtype='float32')
        
        if s_mix.ndim > 1:
            s_mix = s_mix[:, 0]
        if s_tgt.ndim > 1:
            s_tgt = s_tgt[:, 0]
        if s_int.ndim > 1:
            s_int = s_int[:, 0]
        
        return s_tgt, s_int, s_mix
    except FileNotFoundError:
        return None, None, None


def calculate_metrics(est, tgt, interf, fs):
    """Computes SIR, SINR, STOI, PESQ (NB & WB)."""
    # 1. Align Lengths
    min_len = min(len(est), len(tgt), len(interf))
    est = est[:min_len]
    tgt = tgt[:min_len]
    interf = interf[:min_len]
    
    # 2. Physics Metrics (OSINR)
    eps = 1e-10
    tgt_n = tgt / (np.linalg.norm(tgt) + eps)
    int_n = interf / (np.linalg.norm(interf) + eps)

    alpha = np.dot(est, tgt_n)
    beta = np.dot(est, int_n)
    
    e_target = alpha * tgt_n
    e_interf = beta * int_n
    e_noise = est - e_target - e_interf

    P_t = np.sum(e_target**2)
    P_i = np.sum(e_interf**2)
    P_n = np.sum(e_noise**2)

    sinr = 10 * np.log10(P_t / (P_i + P_n + eps))
    sir = 10 * np.log10(P_t / (P_i + eps))
    
    # 3. Perceptual Metrics
    stoi_val = 0.0
    pesq_nb, pesq_wb = 0.0, 0.0
    
    if DEPENDENCIES_OK:
        try:
            stoi_val = stoi(tgt, est, fs, extended=False)
            pesq_eval = PESQEvaluator(fs)
            pesq_nb, pesq_wb = pesq_eval.evaluate(tgt, est)
        except:
            pass

    return {
        "sir": sir,
        "sinr": sinr,
        "stoi": stoi_val,
        "pesq_nb": pesq_nb,
        "pesq_wb": pesq_wb
    }


def append_to_csv(run_name, mode, baseline_metrics, oracle_metrics):
    """Appends metrics to oracle_metrics.csv with one row per sample."""
    csv_path = os.path.join(os.path.dirname(__file__), "oracle_metrics.csv")
    
    # Define headers - one row per sample with reverb and no_reverb columns
    headers = [
        "Run_ID",
        # Reverb Baseline
        "Reverb_Base_SIR", "Reverb_Base_SINR", "Reverb_Base_PESQ_NB", "Reverb_Base_PESQ_WB", "Reverb_Base_STOI",
        # Reverb Oracle
        "Reverb_Oracle_SIR", "Reverb_Oracle_SINR", "Reverb_Oracle_PESQ_NB", "Reverb_Oracle_PESQ_WB", "Reverb_Oracle_STOI",
        # No_Reverb Baseline
        "NoReverb_Base_SIR", "NoReverb_Base_SINR", "NoReverb_Base_PESQ_NB", "NoReverb_Base_PESQ_WB", "NoReverb_Base_STOI",
        # No_Reverb Oracle
        "NoReverb_Oracle_SIR", "NoReverb_Oracle_SINR", "NoReverb_Oracle_PESQ_NB", "NoReverb_Oracle_PESQ_WB", "NoReverb_Oracle_STOI"
    ]
    
    file_exists = os.path.isfile(csv_path)
    
    # Read existing data if file exists
    existing_data = {}
    if file_exists:
        with open(csv_path, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_data[row['Run_ID']] = row
    
    # Update or create row for this run_name
    if run_name not in existing_data:
        existing_data[run_name] = {"Run_ID": run_name}
        # Initialize all fields with N/A
        for h in headers[1:]:
            existing_data[run_name][h] = "N/A"
    
    # Update the appropriate columns based on mode
    prefix = "Reverb" if mode == "reverb" else "NoReverb"
    existing_data[run_name][f"{prefix}_Base_SIR"] = f"{baseline_metrics['sir']:.2f}"
    existing_data[run_name][f"{prefix}_Base_SINR"] = f"{baseline_metrics['sinr']:.2f}"
    existing_data[run_name][f"{prefix}_Base_PESQ_NB"] = f"{baseline_metrics['pesq_nb']:.3f}"
    existing_data[run_name][f"{prefix}_Base_PESQ_WB"] = f"{baseline_metrics['pesq_wb']:.3f}"
    existing_data[run_name][f"{prefix}_Base_STOI"] = f"{baseline_metrics['stoi']:.3f}"
    existing_data[run_name][f"{prefix}_Oracle_SIR"] = f"{oracle_metrics['sir']:.2f}"
    existing_data[run_name][f"{prefix}_Oracle_SINR"] = f"{oracle_metrics['sinr']:.2f}"
    existing_data[run_name][f"{prefix}_Oracle_PESQ_NB"] = f"{oracle_metrics['pesq_nb']:.3f}"
    existing_data[run_name][f"{prefix}_Oracle_PESQ_WB"] = f"{oracle_metrics['pesq_wb']:.3f}"
    existing_data[run_name][f"{prefix}_Oracle_STOI"] = f"{oracle_metrics['stoi']:.3f}"
    
    # Write back the entire CSV
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        # Sort by Run_ID to maintain order
        for run_id in sorted(existing_data.keys(), key=lambda x: int(x.split('_')[-1])):
            writer.writerow(existing_data[run_id])


def evaluate_oracle(sample_name, mode='reverb'):
    """
    Evaluate oracle TFLC output for a single sample.
    
    Args:
        sample_name (str): Sample identifier (e.g., 'batch_test_000')
        mode (str): 'reverb' or 'anechoic'
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Setup paths
    sim_dir = os.path.join(config.SIM_DIR, sample_name, mode)
    output_dir = os.path.join(os.path.dirname(__file__), 'batch_test', mode)
    oracle_path = os.path.join(output_dir, f"{sample_name}_oracle.wav")
    
    # Check if oracle output exists
    if not os.path.exists(oracle_path):
        print(f"Oracle output not found: {oracle_path}")
        return False
    
    # Load ground truth
    s_tgt, s_int, s_mix = load_ground_truth(sim_dir)
    if s_tgt is None:
        print(f"Ground truth files missing for {sample_name} ({mode})")
        return False
    
    # Calculate baseline metrics
    baseline_metrics = calculate_metrics(s_mix, s_tgt, s_int, config.FS)
    
    # Load oracle output
    s_oracle, _ = sf.read(oracle_path, dtype='float32')
    if s_oracle.ndim > 1:
        s_oracle = s_oracle[:, 0]
    
    # Calculate oracle metrics
    oracle_metrics = calculate_metrics(s_oracle, s_tgt, s_int, config.FS)
    
    # Append to CSV
    append_to_csv(sample_name, mode, baseline_metrics, oracle_metrics)
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Oracle TFLC Metrics Evaluation')
    parser.add_argument('--sample', type=str, required=True, help='Sample name (e.g., batch_test_000)')
    parser.add_argument('--mode', type=str, default='reverb', choices=['reverb', 'no_reverb'], help='Dataset mode')
    
    args = parser.parse_args()
    
    success = evaluate_oracle(args.sample, args.mode)
    if success:
        print(f"Metrics calculated for {args.sample} ({args.mode})")
    else:
        print("Evaluation failed.")
