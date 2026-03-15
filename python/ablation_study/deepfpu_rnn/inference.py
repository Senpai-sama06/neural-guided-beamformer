#!/usr/bin/env python3
"""
DeepFPU-RNN + SMVB Beamforming
Uses learned mask from DeepFPU-RNN model with Steered Minimum Variance Beamformer.
"""

import os
import sys
import numpy as np
import soundfile as sf
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent directory to path to import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from src import config

# ------------------------
# Config / Constants
# ------------------------
FS = config.FS
N_FFT = config.N_FFT
N_HOP = N_FFT - config.HOP_LEN
FREQ_BINS = (N_FFT // 2) + 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SMVB Constants
ANGLE_TARGET = 90.0
MIC_DIST = 0.08  # 8cm microphone spacing
C_SPEED = 343.0
N_MICS = 2
LAMBDA_MIN = 1e-6
LAMBDA_MAX = 0.1
RANK_1_THRESH = 3.0

# ------------------------
# Model Architecture
# ------------------------
class ResBlock(nn.Module):
    def __init__(self, channels, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), 
            nn.BatchNorm2d(channels), 
            nn.ReLU(), 
            nn.Dropout2d(p=dropout),
            nn.Conv2d(channels, channels, 3, padding=1), 
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x): 
        return self.relu(x + self.conv(x))

class DeepFPU_CRNN(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=(1, 2))
        
        self.enc1 = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), 
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), 
            ResBlock(64, dropout)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), 
            ResBlock(128, dropout)
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), 
            ResBlock(256, dropout)
        )
        
        self.lstm_hidden = 256
        self.lstm_layers = 2
        
        self.proj_in = nn.Linear(256 * FREQ_BINS, 512)
        self.lstm = nn.LSTM(input_size=512, hidden_size=self.lstm_hidden, 
                            num_layers=self.lstm_layers, batch_first=True, bidirectional=True)
        self.proj_out = nn.Linear(self.lstm_hidden * 2, 256 * FREQ_BINS)
        
        self.up4 = nn.ConvTranspose2d(256, 256, (1, 2), stride=(1, 2))
        self.dec4 = nn.Sequential(
            nn.Conv2d(256+256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), 
            ResBlock(256, dropout)
        )
        
        self.up3 = nn.ConvTranspose2d(256, 128, (1, 2), stride=(1, 2))
        self.dec3 = nn.Sequential(
            nn.Conv2d(128+128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), 
            ResBlock(128, dropout)
        )
        
        self.up2 = nn.ConvTranspose2d(128, 64, (1, 2), stride=(1, 2))
        self.dec2 = nn.Sequential(
            nn.Conv2d(64+64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), 
            ResBlock(64, dropout)
        )
        
        self.up1 = nn.ConvTranspose2d(64, 32, (1, 2), stride=(1, 2))
        self.dec1 = nn.Sequential(
            nn.Conv2d(32+32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), 
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )
        
        self.out = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())

    def _match(self, x, target):
        if x.shape[3] != target.shape[3]: 
            x = F.interpolate(x, size=target.shape[2:], mode='nearest')
        return x

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        B, C, F, T = e4.shape
        rnn_in = e4.permute(0, 3, 1, 2).reshape(B, T, -1)
        
        rnn_out, _ = self.lstm(self.proj_in(rnn_in))
        
        b = self.proj_out(rnn_out).reshape(B, T, C, F).permute(0, 2, 3, 1)
        
        u4 = self._match(self.up4(b), e4)
        d4 = self.dec4(torch.cat([u4, e4], dim=1))
        
        u3 = self._match(self.up3(d4), e3)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))
        
        u2 = self._match(self.up2(d3), e2)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        
        u1 = self._match(self.up1(d2), e1)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        
        return self.out(d1)

# ------------------------
# SMVB Beamforming
# ------------------------
def get_steering_vector_single(f, angle_deg, d, c):
    """Compute normalized steering vector for given frequency and angle."""
    theta = np.deg2rad(angle_deg)
    tau1 = (d / 2) * np.cos(theta) / c
    tau2 = (d / 2) * np.cos(theta - np.pi) / c
    omega = 2 * np.pi * f
    v = np.array([[np.exp(-1j * omega * tau1)], [np.exp(-1j * omega * tau2)]])
    v = v / (v[0] + 1e-10)
    return v

def advanced_hybrid_bf(Y, mask, f_bins):
    """
    Advanced Hybrid Steered Minimum Variance Beamformer (SMVB).
    Adaptively switches between LCMV and MVDR based on interference rank.
    
    Args:
        Y: Complex STFT (2, F, T)
        mask: Target mask (F, T)
        f_bins: Frequency bins in Hz
    
    Returns:
        S_out: Beamformed output (F, T)
    """
    n_freq = Y.shape[1]
    n_frames = Y.shape[2]
    S_out = np.zeros((n_freq, n_frames), dtype=complex)
   
    desired_response = np.array([[1], [0]], dtype=np.complex64)
    mask_int = 1.0 - mask
   
    for i in range(n_freq):
        f_hz = f_bins[i]
       
        # Low-frequency bypass
        if f_hz < 200:
            S_out[i, :] = Y[0, i, :]
            continue
           
        Y_vec = Y[:, i, :]
        m_int_vec = mask_int[i, :]
       
        # Estimate interference covariance using mask
        denom_int = np.sum(m_int_vec) + 1e-6
        Y_weighted = Y_vec * np.sqrt(m_int_vec)
        Phi_int = (Y_weighted @ Y_weighted.conj().T) / denom_int
       
        # Total covariance
        R_yy = (Y_vec @ Y_vec.conj().T) / n_frames
       
        # Analyze interference rank via eigendecomposition
        try:
            eigvals, eigvecs = np.linalg.eigh(Phi_int)
            lambda_1 = eigvals[1]
            lambda_2 = eigvals[0] + 1e-10
            ratio = lambda_1 / lambda_2
            v_int = eigvecs[:, 1].reshape(2, 1)
        except:
            ratio = 0
            v_int = np.zeros((2, 1))

        # Get target steering vector
        v_tgt = get_steering_vector_single(f_hz, ANGLE_TARGET, MIC_DIST, C_SPEED)
       
        # Adaptive beamformer selection
        if ratio > RANK_1_THRESH:
            # High-rank interference: Use LCMV with null steering
            C_mat = np.column_stack((v_tgt, v_int))
            if np.linalg.cond(C_mat) > 10:
                w = v_tgt / N_MICS
            else:
                try:
                    w = np.linalg.solve(C_mat.conj().T, desired_response)
                except np.linalg.LinAlgError:
                    w = v_tgt / N_MICS
        else:
            # Low-rank interference: Use MVDR with diagonal loading
            trace_R = np.trace(R_yy).real
            diag_load = max(LAMBDA_MIN, min(LAMBDA_MAX, 0.01 * trace_R))
            R_loaded = Phi_int + diag_load * np.eye(N_MICS)
            try:
                w_unnorm = np.linalg.solve(R_loaded, v_tgt)
                norm_factor = (v_tgt.conj().T @ w_unnorm).item()
                w = w_unnorm / (norm_factor + 1e-10)
            except:
                w = v_tgt / N_MICS
               
        S_out[i, :] = (w.conj().T @ Y_vec).squeeze()
       
    return S_out

# ------------------------
# Main DeepFPU-RNN Inference
# ------------------------
def deepfpu_rnn_inference(sample_name, mode='reverb', model_path=None):
    """
    Run DeepFPU-RNN + SMVB inference on a single sample.
    
    Args:
        sample_name (str): Sample identifier (e.g., 'batch_test_000')
        mode (str): 'reverb' or 'no_reverb'
        model_path (str): Path to model .pth file
    
    Returns:
        str: Path to output file, or None if failed
    """
    # Setup paths
    sim_dir = os.path.join(config.SIM_DIR, sample_name, mode)
    output_dir = os.path.join(os.path.dirname(__file__), 'batch_test', mode)
    os.makedirs(output_dir, exist_ok=True)
    
    mix_path = os.path.join(sim_dir, "mixture.wav")
    
    # Check if file exists
    if not os.path.exists(mix_path):
        print(f"Missing file: {mix_path}")
        return None
    
    # Load stereo mixture
    y_mix, sr = sf.read(mix_path, dtype="float32")
    if sr != FS:
        print(f"Warning: mixture SR {sr} != FS {FS}")
    
    if y_mix.ndim == 1:
        print("Error: mixture.wav is mono. Expecting stereo 2-channel.")
        return None
    
    # Load model
    if model_path is None or not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return None
    
    model = DeepFPU_CRNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # STFT
    Y_input = y_mix.T  # (2, samples)
    f_bins, t_bins, Y_mix = scipy.signal.stft(Y_input, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
    
    # Feature extraction
    mag = np.abs(Y_mix)
    log_mag = np.log(mag[0] + 1e-7)
    log_mag = (log_mag - log_mag.mean()) / (log_mag.std() + 1e-7)

    ipd = np.angle(Y_mix[0]) - np.angle(Y_mix[1])
    fmap = np.tile(np.linspace(0, 1, FREQ_BINS)[:, np.newaxis], (1, log_mag.shape[1]))

    feat = np.stack([log_mag, np.sin(ipd), np.cos(ipd), fmap], axis=0)
    
    # Model inference
    X = torch.from_numpy(feat).float().unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        Mask_tensor = model(X)
        Mask_np = Mask_tensor.squeeze(0).squeeze(0).cpu().numpy()
    
    # Ensure dimensions match
    min_f = min(Mask_np.shape[0], Y_mix.shape[1])
    min_t = min(Mask_np.shape[1], Y_mix.shape[2])
    
    Mask_np = Mask_np[:min_f, :min_t]
    Y_mix = Y_mix[:, :min_f, :min_t]
    
    # SMVB beamforming with learned mask
    S_out = advanced_hybrid_bf(Y_mix, Mask_np, f_bins)
    
    # Apply mask post-filter
    S_final = S_out * np.maximum(Mask_np, 0.05)
    
    # ISTFT
    _, s_out = scipy.signal.istft(S_final, fs=FS, nperseg=N_FFT, noverlap=N_HOP)
    
    # Normalize
    s_out = s_out / (np.max(np.abs(s_out)) + 1e-10)
    
    # Save output
    out_path = os.path.join(output_dir, f"{sample_name}_deepfpu_rnn.wav")
    sf.write(out_path, s_out.astype(np.float32), FS)
    
    return out_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DeepFPU-RNN + SMVB Inference')
    parser.add_argument('--sample', type=str, required=True, help='Sample name (e.g., batch_test_000)')
    parser.add_argument('--mode', type=str, default='reverb', choices=['reverb', 'no_reverb'], help='Dataset mode')
    parser.add_argument('--model', type=str, required=True, help='Path to model .pth file')
    
    args = parser.parse_args()
    
    result = deepfpu_rnn_inference(args.sample, args.mode, args.model)
    if result:
        print(f"Saved: {result}")
    else:
        print("Inference failed.")
