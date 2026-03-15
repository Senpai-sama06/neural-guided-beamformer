import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf
import scipy.signal
import scipy.ndimage
import os
import time
import warnings
from neural_beamformer import config
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
# Paths
EGE_MODEL_PATH = "/home/cse-sdpl/Downloads/main-code/real-time-audio-visual-zooming-main/Final_pipeline/models/best_model_mixed_soft.pth"
MCDF_MODEL_PATH = "mcdf_safe_ep50.pth"  

# Constants
FREQ_BINS = (config.N_FFT // 2) + 1
INPUT_CHANNELS = 5
ERB_BANDS = 64      
HIDDEN_DIM = 128    

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. MODEL ARCHITECTURES (Keep exactly as before)
# ==============================================================================
class GHPA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groups = out_channels // 4
        self.p_h = nn.Parameter(torch.randn(1, self.groups, 10, 1))
        self.p_w = nn.Parameter(torch.randn(1, self.groups, 1, 10))
        self.p_c = nn.Parameter(torch.randn(1, self.groups, 1, 1))  
        self.proj_out = nn.Conv2d(out_channels, out_channels, kernel_size=1)
    def forward(self, x):
        B, C, H, W = x.shape
        x_groups = torch.chunk(x, 4, dim=1)
        p_hw = F.interpolate(self.p_h * self.p_w, size=(H, W), mode='bilinear', align_corners=False)
        x0 = x_groups[0] * p_hw
        p_ch = F.interpolate(self.p_c * self.p_h, size=(H, W), mode='bilinear', align_corners=False)
        x1 = x_groups[1] * p_ch
        p_cw = F.interpolate(self.p_c * self.p_w, size=(H, W), mode='bilinear', align_corners=False)
        x2 = x_groups[2] * p_cw
        x3 = x_groups[3]
        out = torch.cat([x0, x1, x2, x3], dim=1)
        out = self.proj_out(out)
        return out

class GAB(nn.Module):
    def __init__(self, low_channels, high_channels, out_channels):
        super().__init__()
        self.conv_high = nn.Sequential(nn.Conv2d(high_channels, low_channels, 1), nn.BatchNorm2d(low_channels), nn.ReLU())
        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(low_channels, low_channels, 3, padding=1, dilation=1, groups=low_channels),
            nn.Conv2d(low_channels, low_channels, 3, padding=2, dilation=2, groups=low_channels),
            nn.Conv2d(low_channels, low_channels, 3, padding=5, dilation=5, groups=low_channels),
            nn.Conv2d(low_channels, low_channels, 3, padding=7, dilation=7, groups=low_channels)
        ])
        self.proj = nn.Conv2d(low_channels * 4, out_channels, 1)
    def forward(self, low_feat, high_feat):
        high_feat = F.interpolate(high_feat, size=low_feat.shape[2:], mode='bilinear', align_corners=False)
        high_feat = self.conv_high(high_feat)
        fused = low_feat + high_feat
        outs = [conv(fused) for conv in self.dilated_convs]
        out = torch.cat(outs, dim=1)
        out = self.proj(out)
        return out

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1); self.bn1 = nn.BatchNorm2d(out_c); self.relu1 = nn.ReLU(); self.drop = nn.Dropout2d(p=dropout)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1); self.bn2 = nn.BatchNorm2d(out_c); self.relu2 = nn.ReLU()
        self.skip = nn.Identity()
        if in_c != out_c: self.skip = nn.Conv2d(in_c, out_c, 1)
    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x); out = self.bn1(out); out = self.relu1(out); out = self.drop(out)
        out = self.conv2(out); out = self.bn2(out)
        out = out + identity
        return self.relu2(out)

class EGE_Audio_UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1_conv = nn.Conv2d(INPUT_CHANNELS, 32, 3, padding=1); self.enc1_bn = nn.BatchNorm2d(32); self.enc1_relu = nn.ReLU()
        self.enc2_conv = nn.Conv2d(32, 64, 3, padding=1); self.enc2_bn = nn.BatchNorm2d(64); self.enc2_relu = nn.ReLU(); self.enc2_res = ResBlock(64, 64)
        self.enc3_conv = nn.Conv2d(64, 128, 3, padding=1); self.enc3_bn = nn.BatchNorm2d(128); self.enc3_relu = nn.ReLU(); self.enc3_res = ResBlock(128, 128)
        self.enc4_conv = nn.Conv2d(128, 256, 3, padding=1); self.enc4_bn = nn.BatchNorm2d(256); self.enc4_relu = nn.ReLU(); self.enc4_res = ResBlock(256, 256)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2)) 
        self.bottleneck = GHPA(256, 256)
        self.up4 = nn.ConvTranspose2d(256, 256, (1, 2), stride=(1, 2)); self.dec4 = ResBlock(256, 256); self.gab4 = GAB(256, 256, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, (1, 2), stride=(1, 2)); self.dec3 = ResBlock(128, 128); self.gab3 = GAB(128, 256, 128) 
        self.up2 = nn.ConvTranspose2d(128, 64, (1, 2), stride=(1, 2)); self.dec2 = ResBlock(64, 64); self.gab2 = GAB(64, 128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, (1, 2), stride=(1, 2)); self.dec1 = ResBlock(32, 32); self.gab1 = GAB(32, 64, 32)
        self.out_conv = nn.Conv2d(32, 1, 1)
    def _match_size(self, upsampled, target):
        if upsampled.shape[2:] != target.shape[2:]: return F.interpolate(upsampled, size=target.shape[2:], mode='bilinear', align_corners=False)
        return upsampled
    def forward(self, x):
        e1 = self.enc1_relu(self.enc1_bn(self.enc1_conv(x))); p1 = self.pool(e1)
        e2 = self.enc2_res(self.enc2_relu(self.enc2_bn(self.enc2_conv(p1)))); p2 = self.pool(e2)
        e3 = self.enc3_res(self.enc3_relu(self.enc3_bn(self.enc3_conv(p2)))); p3 = self.pool(e3)
        e4 = self.enc4_res(self.enc4_relu(self.enc4_bn(self.enc4_conv(p3)))); p4 = self.pool(e4)
        b = self.bottleneck(p4)
        u4 = self.up4(b); skip4 = self.gab4(e4, b); u4 = self._match_size(u4, skip4); d4 = self.dec4(u4 + skip4)
        u3 = self.up3(d4); skip3 = self.gab3(e3, d4); u3 = self._match_size(u3, skip3); d3 = self.dec3(u3 + skip3)
        u2 = self.up2(d3); skip2 = self.gab2(e2, d3); u2 = self._match_size(u2, skip2); d2 = self.dec2(u2 + skip2)
        u1 = self.up1(d2); skip1 = self.gab1(e1, d2); u1 = self._match_size(u1, skip1); d1 = self.dec1(u1 + skip1)
        out = torch.sigmoid(self.out_conv(d1))
        return out

class MCDF_Mask_Filter(nn.Module):
    def __init__(self):
        super().__init__()
        self.erb_enc = nn.Linear(FREQ_BINS, ERB_BANDS)
        self.gru = nn.GRU(input_size=ERB_BANDS * 3, hidden_size=HIDDEN_DIM, num_layers=2, batch_first=True)
        self.erb_dec = nn.Linear(HIDDEN_DIM, ERB_BANDS) 
        self.freq_dec = nn.Linear(ERB_BANDS, FREQ_BINS)
    def forward(self, beam_c, mic_mag, mask):
        beam_mag = torch.abs(beam_c)
        log_beam = torch.log(beam_mag + 1e-7).permute(0, 2, 1)
        log_mic = torch.log(mic_mag + 1e-7).permute(0, 2, 1)
        mask_p = mask.permute(0, 2, 1)
        b_erb = F.relu(self.erb_enc(log_beam))
        m_erb = F.relu(self.erb_enc(log_mic))
        msk_erb = F.relu(self.erb_enc(mask_p))
        rnn_in = torch.cat([b_erb, m_erb, msk_erb], dim=-1)
        rnn_out, _ = self.gru(rnn_in)
        mask_erb = F.relu(self.erb_dec(rnn_out)) 
        mask_full = torch.sigmoid(self.freq_dec(mask_erb)).permute(0, 2, 1)
        out_real = beam_c.real * mask_full
        out_imag = beam_c.imag * mask_full
        return torch.complex(out_real, out_imag)

# ==============================================================================
# 2. GLOBAL BEAMFORMER
# ==============================================================================
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

# ==============================================================================
# 3. MAIN INFERENCE (FULL CONTEXT)
# ==============================================================================
def enhance_audio(run_name, input_path, model_path):
    # Setup Output
    result_dir = os.path.join(config.RESULTS_DIR, f"{run_name}_results")
    os.makedirs(result_dir, exist_ok=True)
    
    path_raw = os.path.join(result_dir, f"{run_name}_raw_mask.wav")
    path_gauss = os.path.join(result_dir, f"{run_name}_gauss_mask.wav")
    path_gan = os.path.join(result_dir, f"{run_name}_gan_filter.wav")

    print(f"[INF] Processing FULL Context: {input_path}")

    # Load Audio
    y, sr = sf.read(input_path, dtype='float32')
    if sr != config.FS: print(f"Warning: SR mismatch {sr} vs {config.FS}")
    if y.ndim == 1: 
        print("Error: Input is mono. Requires 2 channels.")
        return

    # Load Models
    if not os.path.exists(model_path):
        print("Error: Model paths invalid.")
        return

    print("[INF] Loading Models...")
    ege_model = EGE_Audio_UNet().to(DEVICE)
    ege_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    ege_model.eval()

    # mcdf_model = MCDF_Mask_Filter().to(DEVICE)
    # mcdf_model.load_state_dict(torch.load(MCDF_MODEL_PATH, map_location=DEVICE))
    # mcdf_model.eval()

    start_time = time.time()

    # --- 1. GLOBAL STFT ---
    # Shape: (2, F, T_full)
    f_bins, t_bins, Y = scipy.signal.stft(
        y.T, fs=config.FS, nperseg=config.N_FFT, noverlap=config.N_FFT-config.HOP_LEN
    )
    
    # --- 2. GLOBAL FEATURE EXTRACTION ---
    mag = np.abs(Y)
    log_mag = np.log(mag[0] + 1e-7)
    log_mag = (log_mag - log_mag.mean()) / (log_mag.std() + 1e-7)

    ipd = np.angle(Y[0]) - np.angle(Y[1])
    fmap = np.tile(np.linspace(0, 1, FREQ_BINS)[:, np.newaxis], (1, log_mag.shape[1]))
    
    cross_spec = Y[0] * np.conj(Y[1])
    msc = np.abs(cross_spec) / (np.sqrt(np.abs(Y[0])**2 * np.abs(Y[1])**2) + 1e-9)

    feat = np.stack([log_mag, np.sin(ipd), np.cos(ipd), fmap, msc], axis=0) 
    
    # --- 3. PADDING FOR UNET ---
    # UNet needs time dim divisible by 16 because of 4 pooling layers (2^4 = 16)
    original_T = feat.shape[2]
    pad_len = 0
    if original_T % 16 != 0:
        pad_len = 16 - (original_T % 16)
    
    # Convert to Tensor & Pad
    X = torch.from_numpy(feat).float().unsqueeze(0).to(DEVICE) # (1, 5, F, T)
    if pad_len > 0:
        X = F.pad(X, (0, pad_len)) # Pad time dimension
        
    # --- 4. INFERENCE (Single Pass) ---
    with torch.no_grad():
        Mask_tensor = ege_model(X)
        
        # Remove Padding from output
        if pad_len > 0:
            Mask_tensor = Mask_tensor[..., :-pad_len]
            
        Mask_np = Mask_tensor.squeeze(0).squeeze(0).cpu().numpy()

    # --- 5. GLOBAL BEAMFORMING ---
    # Ensure dimensions match (Min check just in case)
    min_f = min(Mask_np.shape[0], Y.shape[1])
    min_t = min(Mask_np.shape[1], Y.shape[2])
    
    Mask_np = Mask_np[:min_f, :min_t]
    Y = Y[:, :min_f, :min_t]
    
    # Run TFLC on full sequence
    print("[INF] Running Global Beamforming...")
    S_out = tflc_beamforming_broadside(Y, Mask_np, n_beamformers=2, iterations=20)

    # --- 6. POST-PROCESSING (3 Paths) ---
    
    # A. Raw Mask
    S_raw = S_out * np.maximum(Mask_np, 0.05)
    
    # B. Gauss Mask (Conditional)
    # Blur the mask to reduce musical noise
    Mask_blurred = scipy.ndimage.gaussian_filter(Mask_np, sigma=2.0)
    # Only use blurred version where confidence is low (background)
    Mask_final = np.where(Mask_np < 0.75, Mask_blurred, Mask_np)
    S_gauss = S_out * np.maximum(Mask_final, 0.05)
    
    # C. GAN Filter
    # print("[INF] Running MCDF GAN Filter...")
    # with torch.no_grad():
    #     S_draft_t = torch.from_numpy(S_out).type(torch.complex64).unsqueeze(0).to(DEVICE)
    #     Z0_trimmed = Y[0]
    #     Mic_mag_t = torch.from_numpy(np.abs(Z0_trimmed)).float().unsqueeze(0).to(DEVICE)
    #     Mask_in_t = torch.from_numpy(Mask_np).float().unsqueeze(0).to(DEVICE)
        
    #     S_gan_c = mcdf_model(S_draft_t, Mic_mag_t, Mask_in_t)
    #     S_gan = S_gan_c.squeeze(0).cpu().numpy()

    # --- 7. ISTFT & SAVE ---
    def save_wav(spec, path):
        _, wav = scipy.signal.istft(spec, fs=config.FS, nperseg=config.N_FFT, noverlap=config.N_FFT-config.HOP_LEN)
        # Normalize
        wav = wav / (np.max(np.abs(wav)) + 1e-9)
        sf.write(path, wav, config.FS)

    save_wav(S_raw, path_raw)
    save_wav(S_gauss, path_gauss)
    # save_wav(S_gan, path_gan)

    print(f"Total time: {time.time() - start_time:.3f}s")
    print(f"[INF] Saved:\n  -> {path_raw}\n  -> {path_gauss}\n  -> {path_gan}")

# if __name__ == "__main__":
#     INPUT_FILE = "path/to/test.wav"
#     if os.path.exists(INPUT_FILE):
#         enhance_audio("full_context_run", INPUT_FILE, EGE_MODEL_PATH)