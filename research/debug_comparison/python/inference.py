import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import scipy.signal
import scipy.io
import os
import sys

# Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "../../python")
sys.path.append(src_path)

from src import config
from src.inference import EGE_Audio_UNet

# Constants
DEVICE = torch.device("cpu") # Force CPU for deterministic float behavior

# Inlined Deterministic TFLC Beamformer
def tflc_beamforming_debug(Y_in, mask, n_beamformers=2, iterations=20):
    X = Y_in.transpose(1, 2, 0) # (F, T, M)
    F_dim, T_dim, M = X.shape
    
    a_vec = np.ones((F_dim, M), dtype=np.complex64)
    noise_mask = (1.0 - mask)[:, :, None]
    
    # Estimate Global Noise Covariance
    Phi_noise_total = np.einsum('ftm,ftn,ft->fmn', X, X.conj(), noise_mask[:, :, 0])
    
    W_k = np.zeros((F_dim, M, n_beamformers), dtype=np.complex64)
    for k in range(n_beamformers):
        # DETERMINISTIC PERTURBATION
        # perturbation = (np.random.normal(0, 0.01, (F_dim, M, M)) + 1j * np.random.normal(0, 0.01, (F_dim, M, M)))
        perturbation = np.ones((F_dim, M, M), dtype=np.complex64) * 0.01 * (1 + 1j)
        
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
                
    W_init_debug = W_k.copy()
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
    Y_final = np.sum(c_k * Y_k_final, axis=-1) 
    return Y_final, W_k, c_k, Phi_noise_total, W_init_debug

def inference_debug(input_path, model_path):
    print(f"[DEBUG] Processing {input_path}")

    # 1. Load Audio
    y, sr = sf.read(input_path, dtype='float32')
    # y shape: (Samples, Channels) -> Transpose to (Channels, Samples)
    y = y.T 

    # 2. STFT
    f_bins, t_bins, Y = scipy.signal.stft(
        y, fs=config.FS, nperseg=config.N_FFT, noverlap=config.N_FFT-config.HOP_LEN
    )
    # Y shape: (Channels, F, T)
    
    # 3. Feature Extraction
    mag = np.abs(Y)
    log_mag = np.log(mag[0] + 1e-7)
    mean_val = log_mag.mean()
    std_val = log_mag.std()
    log_mag = (log_mag - mean_val) / (std_val + 1e-7)

    ipd = np.angle(Y[0]) - np.angle(Y[1])
    # Fix scalar f_bins
    fmap = np.tile(np.linspace(0, 1, len(f_bins))[:, np.newaxis], (1, log_mag.shape[1]))
    
    cross_spec = Y[0] * np.conj(Y[1])
    msc = np.abs(cross_spec) / (np.sqrt(np.abs(Y[0])**2 * np.abs(Y[1])**2) + 1e-9)

    feat = np.stack([log_mag, np.sin(ipd), np.cos(ipd), fmap, msc], axis=0) 
    # feat shape: (5, F, T)

    # 4. Model Inference
    # Pad
    original_T = feat.shape[2]
    pad_len = 0
    if original_T % 16 != 0:
        pad_len = 16 - (original_T % 16)
    
    X = torch.from_numpy(feat).float().unsqueeze(0).to(DEVICE)
    if pad_len > 0:
        X = F.pad(X, (0, pad_len))
        
    ege_model = EGE_Audio_UNet().to(DEVICE)
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    ege_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    ege_model.eval()

    with torch.no_grad():
        Mask_tensor = ege_model(X)
        if pad_len > 0:
            Mask_tensor = Mask_tensor[..., :-pad_len]
        Mask_np = Mask_tensor.squeeze(0).squeeze(0).cpu().numpy()

    # 5. Beamforming
    min_f = min(Mask_np.shape[0], Y.shape[1])
    min_t = min(Mask_np.shape[1], Y.shape[2])
    Mask_np = Mask_np[:min_f, :min_t]
    Y = Y[:, :min_f, :min_t]
    
    # Use Deterministic Debug Beamformer
    S_out, W_final, c_k, Phi_noise_total, W_init = tflc_beamforming_debug(Y, Mask_np, n_beamformers=2, iterations=20)
    
    # 6. Post Processing (Raw)
    S_raw = S_out * np.maximum(Mask_np, 0.05)
    
    # 7. ISTFT
    _, wav_out = scipy.signal.istft(S_raw, fs=config.FS, nperseg=config.N_FFT, noverlap=config.N_FFT-config.HOP_LEN)
    # Normalize
    wav_raw = wav_out.copy()
    wav_out = wav_out / (np.max(np.abs(wav_out)) + 1e-9)

    # --- SAVE DEBUG VARIABLES ---
    debug_file = os.path.join(current_dir, "../data/debug_python.mat")
    scipy.io.savemat(debug_file, {
        'py_audio_in': y,               # (2, T)
        'py_stft_in': Y,                # (2, F, T) complex
        'py_features': feat,            # (5, F, T)
        'py_mask_pred': Mask_np,        # (F, T)
        'py_stft_out': S_out,           # (F, T) complex
        'py_stft_final': S_raw,         # (F, T) complex (masked)
        'py_audio_out': wav_out,        # (T,)
        'py_audio_raw': wav_raw,        # (T,)
        'py_W_final': W_final,          # (F, M, K)
        'py_c_k': c_k,                  # (F, T, K)
        'py_Phi_init': Phi_noise_total, # (F, M, M)
        'py_W_init': W_init             # (F, M, K)
    })
    print(f"[DEBUG] Saved intermediate variables to {debug_file}")

    # Save Audio
    out_wav_path = os.path.join(current_dir, "../data/py_out.wav")
    sf.write(out_wav_path, wav_out, config.FS)
    print(f"[DEBUG] Saved output audio to {out_wav_path}")

if __name__ == "__main__":
    # Assuming gen_world.py has run
    mix_path = os.path.join(current_dir, "../data/debug_run/mixture.wav")
    
    # Correct relative path to model
    model_path = os.path.join(src_path, "models/best_model_mixed_soft.pth")
    
    inference_debug(mix_path, model_path)
