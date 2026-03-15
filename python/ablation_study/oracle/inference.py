#!/usr/bin/env python3
"""
Oracle TFLC Beamforming with Oracle IRM
Uses ground truth target and interference to construct oracle mask.
Applies TFLC beamforming with iterative mask refinement.
"""

import os
import sys
import numpy as np
import soundfile as sf
import scipy.signal

# Add parent directory to path to import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from src import config

# ------------------------
# Config / Constants
# ------------------------
FS = config.FS
N_FFT = config.N_FFT
N_HOP = N_FFT - config.HOP_LEN
D = config.MIC_DIST
C = config.C_SPEED
ANGLE_TARGET = 90.0
SIGMA = 1e-3        # diagonal loading for MVDR

# ------------------------
# TFLC Beamforming
# ------------------------
def tflc_beamforming_broadside(Y_in, mask, n_beamformers=2, iterations=5):
    """
    Applied on the FULL sequence Y_in (2, F, T_full)
    No chunking means the covariance matrix is estimated globally.
    """
    X = Y_in.transpose(1, 2, 0)
    F_dim, T_dim, M = X.shape
    
    a_vec = np.ones((F_dim, M), dtype=np.complex64)
    noise_mask = (1.0 - mask)[:, :, None]
    
    # Estimate Global Noise Covariance
    Phi_noise_total = np.einsum('ftm,ftn,ft->fmn', X, X.conj(), noise_mask[:, :, 0])
    
    W_k = np.zeros((F_dim, M, n_beamformers), dtype=np.complex64)
    for k in range(n_beamformers):
        perturbation = (np.random.normal(0, 0.01, (F_dim, M, M)) + 1j * np.random.normal(0, 0.01, (F_dim, M, M)))
        Phi_init = Phi_noise_total + perturbation
        for f in range(F_dim):
            try:
                Phi_inv = np.linalg.inv(Phi_init[f] + 1e-2 * np.eye(M))
                a = a_vec[f][:, None]
                num = Phi_inv @ a
                den = a.conj().T @ Phi_inv @ a
                w = num / (den + 1e-10)
                W_k[f, :, k] = w.squeeze()
            except np.linalg.LinAlgError:
                W_k[f, :, k] = a.squeeze() / M 

    c_k = None
    for i in range(iterations):
        Y_k = np.einsum('fmk,ftm->ftk', W_k.conj(), X)
        y1 = Y_k[:, :, 0]; y2 = Y_k[:, :, 1]
        y21 = y1 - y2
        numerator = -np.real(y2 * y21.conj())
        denominator = np.abs(y21)**2 + 1e-10
        c1 = np.clip(numerator / denominator, 0.0, 1.0)
        c2 = 1.0 - c1
        c_k = np.stack([c1, c2], axis=-1) 
        
        for k in range(n_beamformers):
            mask_k = c_k[:, :, k]
            Phi_k = np.einsum('ftm,ftn,ft->fmn', X, X.conj(), mask_k) + 1e-2* np.eye(M)
            for f in range(F_dim):
                try:
                    Phi_inv = np.linalg.inv(Phi_k[f])
                    a = a_vec[f][:, None]
                    num = Phi_inv @ a
                    den = a.conj().T @ Phi_inv @ a
                    w = num / (den + 1e-10)
                    W_k[f, :, k] = w.squeeze()
                except: pass 

    Y_k_final = np.einsum('fmk,ftm->ftk', W_k.conj(), X)
    Y_final = np.sum(c_k * Y_k_final, axis=-1) 
    return Y_final

# ------------------------
# Main Oracle TFLC
# ------------------------
def oracle_mvdr_inference(sample_name, mode='reverb'):
    """
    Run Oracle TFLC inference on a single sample.
    
    Args:
        sample_name (str): Sample identifier (e.g., 'batch_test_000')
        mode (str): 'reverb' or 'no_reverb'
    
    Returns:
        str: Path to output file, or None if failed
    """
    # Setup paths
    sim_dir = os.path.join(config.SIM_DIR, sample_name, mode)
    output_dir = os.path.join(os.path.dirname(__file__), 'batch_test', mode)
    os.makedirs(output_dir, exist_ok=True)
    
    mix_path = os.path.join(sim_dir, "mixture.wav")
    tgt_path = os.path.join(sim_dir, "target.wav")
    int_path = os.path.join(sim_dir, "interference.wav")
    
    # Check if files exist
    for p in (mix_path, tgt_path, int_path):
        if not os.path.exists(p):
            print(f"Missing file: {p}")
            return None
    
    # Load stereo mixture (shape: (samples, 2))
    y_mix, sr = sf.read(mix_path, dtype="float32")
    if sr != FS:
        print(f"Warning: mixture SR {sr} != FS {FS}")
    
    if y_mix.ndim == 1:
        print("Error: mixture.wav is mono. Expecting stereo 2-channel.")
        return None
    
    Y_input = y_mix.T  # (2, samples)
    
    # Load target / interference as stereo -> mono (take mic-0)
    s_t_st, _ = sf.read(tgt_path, dtype="float32")
    s_i_st, _ = sf.read(int_path, dtype="float32")
    
    s_t = s_t_st[:, 0] if s_t_st.ndim > 1 else s_t_st
    s_i = s_i_st[:, 0] if s_i_st.ndim > 1 else s_i_st
    
    # STFTs
    f_bins, t_bins, Y_mix = scipy.signal.stft(Y_input, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
    _, _, S_t = scipy.signal.stft(s_t, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
    _, _, S_i = scipy.signal.stft(s_i, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
    
    # Magnitudes
    mag_t = np.abs(S_t)
    mag_i = np.abs(S_i)
    
    # Oracle IRM (power domain) - without noise component
    eps = 1e-10
    mask_t = (mag_t ** 2) / (mag_t ** 2 + mag_i ** 2 + eps)
    
    # TFLC beamforming with oracle mask
    S_out = tflc_beamforming_broadside(Y_mix, mask_t, n_beamformers=2, iterations=20)
    
    # Apply oracle spectral post-filter (mask_t)
    S_final = S_out * mask_t
    
    # ISTFT
    _, s_out = scipy.signal.istft(S_final, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
    
    # Normalize
    s_out = s_out / (np.max(np.abs(s_out)) + 1e-10)
    
    # Save output
    out_path = os.path.join(output_dir, f"{sample_name}_oracle.wav")
    sf.write(out_path, s_out.astype(np.float32), FS)
    
    return out_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Oracle TFLC Inference')
    parser.add_argument('--sample', type=str, required=True, help='Sample name (e.g., batch_test_000)')
    parser.add_argument('--mode', type=str, default='reverb', choices=['reverb', 'no_reverb'], help='Dataset mode')
    
    args = parser.parse_args()
    
    result = oracle_mvdr_inference(args.sample, args.mode)
    if result:
        print(f"Saved: {result}")
    else:
        print("Inference failed.")
