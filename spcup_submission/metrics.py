import numpy as np
import soundfile as sf
import os
import sys

# Optional Dependencies
try:
    from pystoi import stoi
    from pesq import pesq
    DEPENDENCIES_OK = True
except ImportError:
    print("Warning: 'pystoi' or 'pesq' missing. Install text: pip install pystoi pesq")
    DEPENDENCIES_OK = False

FS = 16000

class PESQEvaluator:
    def __init__(self, fs):
        self.fs = fs

    def evaluate(self, ref, deg):
        nb_score, wb_score = 0.0, 0.0
        if not DEPENDENCIES_OK: return 0.0, 0.0
        try:
            if self.fs in [8000, 16000]:
                nb_score = pesq(self.fs, ref, deg, 'nb')
            if self.fs == 16000:
                wb_score = pesq(self.fs, ref, deg, 'wb')
        except Exception as e:
            pass 
        return nb_score, wb_score

def calculate_metrics(est, tgt, interf, fs):
    """Computes SIR, SINR, STOI, PESQ."""
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
        except Exception as e:
            print(f"Metric Error: {e}")

    return {
        "sir": sir, 
        "sinr": sinr, 
        "stoi": stoi_val, 
        "pesq_nb": pesq_nb,
        "pesq_wb": pesq_wb
    }

def verify_task(task_folder):
    print(f"\n--- Verifying {os.path.basename(task_folder)} ---")
    
    p_est = os.path.join(task_folder, "processed_signal.wav")
    p_tgt = os.path.join(task_folder, "target_signal.wav")
    p_int = os.path.join(task_folder, "interference_signal1.wav")
    
    if not os.path.exists(p_est):
        print(f"[FAIL] processed_signal.wav missing in {task_folder}")
        return

    # Load
    est, fs_e = sf.read(p_est, dtype='float32')
    tgt, fs_t = sf.read(p_tgt, dtype='float32')
    int_sig, fs_i = sf.read(p_int, dtype='float32')
    
    # Mono check
    if est.ndim > 1: est = est[:, 0]
    if tgt.ndim > 1: tgt = tgt[:, 0]
    if int_sig.ndim > 1: int_sig = int_sig[:, 0]
    
    if fs_e != FS: print(f"Warning: FS mismatch {fs_e} vs {FS}")
    
    m = calculate_metrics(est, tgt, int_sig, FS)
    
    print(f"{'Metric':<10} | {'Value':<10}")
    print("-" * 25)
    print(f"{'SIR':<10} | {m['sir']:.2f}")
    print(f"{'SINR':<10} | {m['sinr']:.2f}")
    print(f"{'STOI':<10} | {m['stoi']:.4f}")
    print(f"{'PESQ_NB':<10} | {m['pesq_nb']:.3f}")
    print(f"{'PESQ_WB':<10} | {m['pesq_wb']:.3f}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    verify_task(os.path.join(base_dir, "Task1_Anechoic"))
    verify_task(os.path.join(base_dir, "Task2_Reverberant"))