import numpy as np
import soundfile as sf
import os
import datetime 
import csv
from neural_beamformer import config

# Handle optional dependencies gracefully
try:
    from pystoi import stoi
    from pesq import pesq
    DEPENDENCIES_OK = True
except ImportError:
    print("Warning: 'pystoi' or 'pesq' not installed. Metrics will be skipped.")
    DEPENDENCIES_OK = False

def append_to_csv(run_name, metrics_dict):
    """Appends metrics for all methods to a central CSV file."""
    csv_path = os.path.join(config.RESULTS_DIR, "batch_comparison_full.csv")
    
    # Define headers
    headers = ["Run_ID"]
    # Metrics to track per method
    metric_keys = ["SIR", "SINR", "PESQ_NB", "PESQ_WB", "STOI"]
    
    # Generate headers: Base_SIR, Raw_SIR, etc.
    methods = ["Base", "Raw", "Gauss", "GAN"]
    for m in methods:
        for k in metric_keys:
            headers.append(f"{m}_{k}")
    
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        
        row = {"Run_ID": run_name}
        
        # Map internal keys to CSV headers
        # Internal keys: 'baseline', 'raw', 'gauss', 'gan'
        # CSV prefixes: 'Base', 'Raw', 'Gauss', 'GAN'
        mapping = {
            'baseline': 'Base',
            'raw': 'Raw',
            'gauss': 'Gauss',
            'gan': 'GAN'
        }
        
        for internal_k, prefix in mapping.items():
            data = metrics_dict.get(internal_k)
            if data:
                row[f"{prefix}_SIR"]     = f"{data['sir']:.2f}"
                row[f"{prefix}_SINR"]    = f"{data['sinr']:.2f}"
                row[f"{prefix}_PESQ_NB"] = f"{data['pesq_nb']:.3f}"
                row[f"{prefix}_PESQ_WB"] = f"{data['pesq_wb']:.3f}"
                row[f"{prefix}_STOI"]    = f"{data['stoi']:.3f}"
            else:
                for k in metric_keys:
                    row[f"{prefix}_{k}"] = "N/A"
                    
        writer.writerow(row)

# --- HELPER CLASSES ---

class PESQEvaluator:
    def __init__(self, fs):
        self.fs = fs

    def evaluate(self, ref, deg):
        nb_score, wb_score = 0.0, 0.0
        if not DEPENDENCIES_OK: return 0.0, 0.0

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

# --- CORE LOGIC ---

def load_ground_truth(sim_dir):
    try:
        s_tgt, _ = sf.read(os.path.join(sim_dir, "target.wav"), dtype='float32')
        s_int, _ = sf.read(os.path.join(sim_dir, "interference.wav"), dtype='float32')
        s_mix, _ = sf.read(os.path.join(sim_dir, "mixture.wav"), dtype='float32')
        
        if s_mix.ndim > 1: s_mix = s_mix[:, 0]
        if s_tgt.ndim > 1: s_tgt = s_tgt[:, 0]
        if s_int.ndim > 1: s_int = s_int[:, 0]
        
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
        except: pass

    return {
        "sir": sir, 
        "sinr": sinr, 
        "stoi": stoi_val, 
        "pesq_nb": pesq_nb,
        "pesq_wb": pesq_wb
    }

def evaluate_run(run_name):
    sim_dir = os.path.join(config.SIM_DIR, run_name)
    res_dir = os.path.join(config.RESULTS_DIR, f"{run_name}_results")
    report_path = os.path.join(res_dir, "report_full.txt")

    print(f"[EVAL] Evaluating Run: {run_name}...")

    # 1. Load Ground Truth
    s_tgt, s_int, s_mix = load_ground_truth(sim_dir)
    if s_tgt is None:
        print("[EVAL] Error: Ground truth files missing.")
        return

    # 2. Calculate Baseline Metrics
    base_metrics = calculate_metrics(s_mix, s_tgt, s_int, config.FS)
    
    # 3. Evaluate Each Method
    methods = {
        'raw': f"{run_name}_raw_mask.wav",
        'gauss': f"{run_name}_gauss_mask.wav",
        'gan': f"{run_name}_gan_filter.wav"
    }
    
    results = {'baseline': base_metrics}
    
    # Header format
    # Method | SIR | SINR | P-NB | P-WB | STOI
    header = f"{'Method':<12} | {'SIR':<6} | {'SINR':<6} | {'P-NB':<6} | {'P-WB':<6} | {'STOI':<6}"
    dash_line = "-" * len(header)
    
    print(dash_line)
    print(header)
    print(dash_line)
    
    # Print Baseline
    m = base_metrics
    print(f"{'Baseline':<12} | {m['sir']:.2f}   | {m['sinr']:.2f}   | {m['pesq_nb']:.2f}   | {m['pesq_wb']:.2f}   | {m['stoi']:.2f}")

    for key, filename in methods.items():
        path = os.path.join(res_dir, filename)
        if os.path.exists(path):
            est, _ = sf.read(path, dtype='float32')
            if est.ndim > 1: est = est[:, 0]
            
            m_metrics = calculate_metrics(est, s_tgt, s_int, config.FS)
            results[key] = m_metrics
            
            print(f"{key.upper():<12} | {m_metrics['sir']:.2f}   | {m_metrics['sinr']:.2f}   | {m_metrics['pesq_nb']:.2f}   | {m_metrics['pesq_wb']:.2f}   | {m_metrics['stoi']:.2f}")
        else:
            print(f"{key.upper():<12} | [FILE NOT FOUND]")
    print(dash_line)

    # 4. Save to CSV
    append_to_csv(run_name, results)
    
    # 5. Save Report Text
    with open(report_path, "w") as f:
        f.write(f"Evaluation Report: {run_name}\n")
        f.write(f"Date: {datetime.datetime.now()}\n")
        f.write(dash_line + "\n")
        f.write(header + "\n")
        f.write(dash_line + "\n")
        
        # Write Baseline
        m = base_metrics
        f.write(f"{'Baseline':<12} | {m['sir']:.2f}   | {m['sinr']:.2f}   | {m['pesq_nb']:.2f}   | {m['pesq_wb']:.2f}   | {m['stoi']:.2f}\n")
        
        # Write Methods
        for key in ['raw', 'gauss', 'gan']:
            if key in results:
                m = results[key]
                f.write(f"{key.upper():<12} | {m['sir']:.2f}   | {m['sinr']:.2f}   | {m['pesq_nb']:.2f}   | {m['pesq_wb']:.2f}   | {m['stoi']:.2f}\n")
            else:
                f.write(f"{key.upper():<12} | N/A\n")
    
    print(f"[EVAL] Metrics saved to CSV and {report_path}")