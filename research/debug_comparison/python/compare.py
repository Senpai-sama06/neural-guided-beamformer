import numpy as np
import scipy.io
import os
import sys

# Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "../data")

# Load Files
py_path = os.path.join(data_dir, "debug_python.mat")
mat_feat_path = os.path.join(data_dir, "debug_matlab_features.mat")
mat_inf_path = os.path.join(data_dir, "debug_matlab_inference.mat")

import h5py

print(f"Loading Python: {py_path}")
py_data = scipy.io.loadmat(py_path)

print(f"Loading MATLAB Feat (H5): {mat_feat_path}")
# h5py reads transposed (MATLAB column-major -> Python row-major)
# We need to be careful with transposes.
mat_feat = h5py.File(mat_feat_path, 'r')

print(f"Loading MATLAB Inf (H5): {mat_inf_path}")
mat_inf = h5py.File(mat_inf_path, 'r')

def compare(name, tensor_py, tensor_mat, tolerance=1e-5):
    print(f"\n--- Comparing {name} ---")
    
    # Cast to numpy if H5 dataset
    if hasattr(tensor_mat, 'shape'):
        tensor_mat = tensor_mat[:]
    
    # 1. Type Check
    print(f"Type: PY={type(tensor_py)} | MAT={type(tensor_mat)}")
    
    # 2. Shape Check/Fix
    tensor_py = np.squeeze(tensor_py)
    tensor_mat = np.squeeze(tensor_mat)
    
    # HDF5 from MATLAB is usually transposed. 
    # Heuristic: Match the frequency dimension (513)
    if tensor_py.shape != tensor_mat.shape:
        # Check if transposing helps align the "513" dimension or simply swaps them
        if (tensor_mat.ndim == 2 and tensor_mat.shape[1] == 513 and tensor_py.shape[0] == 513):
             print("  [INFO] Transposing MATLAB tensor (F=513 alignment)...")
             tensor_mat = tensor_mat.T
        elif tensor_py.ndim == tensor_mat.T.shape:
             # Fallback geometry check
             tensor_mat = tensor_mat.T
    
    # Special Case: Compound Complex
    if tensor_mat.dtype.names is not None:
         if 'real' in tensor_mat.dtype.names and 'imag' in tensor_mat.dtype.names:
             print("  [INFO] Converting Compound H5 Complex to Numpy Complex...")
             tensor_mat = tensor_mat['real'] + 1j * tensor_mat['imag']
             tensor_mat = np.squeeze(tensor_mat)
             # Re-check transpose after conversion
             if (tensor_mat.ndim == 2 and tensor_mat.shape[1] == 513 and tensor_py.shape[0] == 513):
                 tensor_mat = tensor_mat.T

    print(f"Shape Pre-Crop: PY={tensor_py.shape} | MAT={tensor_mat.shape}")

    # 2b. Auto-Crop Time Dimension
    if tensor_py.ndim == tensor_mat.ndim and tensor_py.shape != tensor_mat.shape:
        # Assuming last dim is Time for 2D (F, T) or 1D (T)
        # Check if deviation is small (padding diff)
        
        # 1D Case (Audio)
        if tensor_py.ndim == 1:
            min_len = min(tensor_py.shape[0], tensor_mat.shape[0])
            tensor_py = tensor_py[:min_len]
            tensor_mat = tensor_mat[:min_len]
            print(f"  [INFO] Cropped 1D Audio to {min_len}")
            
        # 2D Case (F, T)
        elif tensor_py.ndim == 2:
            # Assume (F, T) alignment now
            if tensor_py.shape[0] == tensor_mat.shape[0]: # F matches
                 min_t = min(tensor_py.shape[1], tensor_mat.shape[1])
                 tensor_py = tensor_py[:, :min_t]
                 tensor_mat = tensor_mat[:, :min_t]
                 print(f"  [INFO] Cropped Time dim to {min_t}")

    
    # 3. NaNs
    if np.any(np.isnan(tensor_py)) or np.any(np.isnan(tensor_mat)):
        print(f"  [WARNING] NaNs detected! PY: {np.any(np.isnan(tensor_py))}, MAT: {np.any(np.isnan(tensor_mat))}")
        tensor_py = np.nan_to_num(tensor_py)
        tensor_mat = np.nan_to_num(tensor_mat)

    # 4. Values
    # Align shapes if possible (small broadcast errors?)
    if tensor_py.shape != tensor_mat.shape:
        print("  [ERROR] Shape Mismatch persisted. Comparison aborted.")
        return

    diff = np.abs(tensor_py - tensor_mat)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"  Max Diff: {max_diff:.8f}")
    print(f"  Mean Diff: {mean_diff:.8f}")
    if max_diff > tolerance:
        print(f"  [FAIL] > Tol {tolerance}")
    else:
        print(f"  [PASS] < Tol {tolerance}")

# =========================================================
# 1. Audio Input
# =========================================================
# Python: py_audio_in (2, T)
# MATLAB: audio_in (T, 2)
# H5 Read: (2, T) or (T, 2)? usually (T, 2) which compares against (2,T).
# compare() handles transpose now.
compare("Audio Input", py_data['py_audio_in'], mat_feat['audio_in'], 1e-4)

# =========================================================
# 2. STFT
# =========================================================
# Python: py_stft_in (2, F, T) complex
# MATLAB: S1 (T, F), S2 (T, F) in H5
# Note H5 transposes: S1 was (F,T) in MATLAB -> (T,F) in H5
py_stft = py_data['py_stft_in']
mat_s1 = mat_feat['S1'][:] 
mat_s2 = mat_feat['S2'][:]

# Handle complex conversion manually if needed, compare handles it?
# Let's stack first. 
# They are likely compound. compare() handles compound conversion.
# But we need to pass them individually or stack properly.
# Let's verify S1 first.
compare("STFT Ch 1", py_stft[0], mat_feat['S1'], 1e-4)
compare("STFT Ch 2", py_stft[1], mat_feat['S2'], 1e-4)

# =========================================================
# 3. Features
# =========================================================
# Python: py_features (5, F, T)
# MATLAB: feat_1 (T, F) in H5
py_f1 = py_data['py_features'][0] # (F, T)
compare("Feature 1 (LogMag)", py_f1, mat_feat['feat_1'], 1e-4)

# =========================================================
# 4. Model Output (Mask)
# =========================================================
compare("Model Mask (Raw)", py_data['py_mask_pred'], mat_inf['mask_raw'], 1e-3)
compare("Model Mask (Resized)", py_data['py_mask_pred'], mat_inf['mask'], 1e-3) 

# =========================================================
# 5. Beamformed Output
# =========================================================
# Check Internals
# W_final: Python (F, M, K) vs MATLAB (F, M, K) -> H5 (K, M, F)?
print("Loading W_final from H5...")
w_mat = mat_inf['dbg']['W_final'][:]
print(f"  Raw W_final H5 Shape: {w_mat.shape}")

# Try to infer transpose
if w_mat.ndim == 3:
    # Python Target: (513, 190, 2) ? No, W is (513, 2, 2)
    # Python W_final is (F, M, K).
    # If H5 is (K, M, F) (2, 2, 513) -> Transpose (2, 1, 0) -> (513, 2, 2)
    if w_mat.shape[2] == 513:
         w_mat = np.transpose(w_mat, (2, 1, 0))
    elif w_mat.shape[0] == 513:
         pass # No transpose needed? (unlikely for H5)

# Phi_init Check
print("Loading Phi_init from H5...")
phi_mat = mat_inf['dbg']['Phi_init'][:] # (M, M, F) as usual for H5? (3,3,513) likely (K, M, F) logic
print(f"  Raw Phi_init H5 Shape: {phi_mat.shape}")
# Python Phi_noise_total is (F, M, M) -> (513, 2, 2)
# H5 is likely (M, M, F) -> Transpose (2, 1, 0)
if phi_mat.shape[2] == 513:
    phi_mat = np.transpose(phi_mat, (2, 1, 0))

# We need to capture python Phi_init. It wasn't saved!
# We need to update inference.py to save it.
# BUT we can't update inference.py easily without re-running everything.
# Wait, let's just inspect it if we can? No, we need to compare against valid python.
# I will update inference.py first.

# Check Phi_init
compare("Noise Covariance (Phi_init)", py_data['py_Phi_init'], phi_mat, 1e-4)

compare("Beamformer Weights (W_final)", py_data['py_W_final'], w_mat, 1e-4)

# c_k: Python (F, T, K) vs MATLAB (F, T, K) -> H5 (K, T, F)?
print("Loading c_k from H5...")
c_mat = mat_inf['dbg']['c_k'][:]
print(f"  Raw c_k H5 Shape: {c_mat.shape}")

if c_mat.ndim == 3:
    # Python Target: (513, 190, 2)
    # If H5 is (K, T, F) (2, 190, 513) -> Transpose (2, 1, 0) -> (513, 190, 2)
    if c_mat.shape[2] == 513 and c_mat.shape[0] == 2:
         c_mat = np.transpose(c_mat, (2, 1, 0))

compare("Beamformer Mask (c_k)", py_data['py_c_k'], c_mat, 1e-4)

compare("Beamformer Output (S_out)", py_data['py_stft_out'], mat_inf['S_out'], 1e-4)
compare("Masked Output (S_raw)", py_data['py_stft_final'], mat_inf['S_raw'], 1e-4)

# =========================================================
# 6. Final Audio
# =========================================================
compare("Raw Audio (Pre-Norm)", py_data['py_audio_raw'], mat_inf['wav_raw'], 1e-3)
compare("Final Audio", py_data['py_audio_out'], mat_inf['wav_out'], 1e-3)

# Correlation Check (to rule out scaling differences)
py_wav = py_data['py_audio_out'].squeeze()
mat_wav = mat_inf['wav_out'][:].squeeze()
min_len = min(len(py_wav), len(mat_wav))

print(f"  PY Audio: Max={py_wav.max():.4f}, Min={py_wav.min():.4f}, Mean={py_wav.mean():.4f}, Std={py_wav.std():.4f}")
print(f"  MAT Audio: Max={mat_wav.max():.4f}, Min={mat_wav.min():.4f}, Mean={mat_wav.mean():.4f}, Std={mat_wav.std():.4f}")

# Cross-Correlation Lag Check
from scipy import signal
corr_full = signal.correlate(py_wav[:min_len], mat_wav[:min_len], mode='full', method='fft')
lags = signal.correlation_lags(min_len, min_len, mode='full')
lag_idx = np.argmax(np.abs(corr_full))
max_lag = lags[lag_idx]
max_val = corr_full[lag_idx]
# Normalize max_val approx
norm_val = max_val / np.sqrt(np.sum(py_wav[:min_len]**2) * np.sum(mat_wav[:min_len]**2))

print(f"  [INFO] Max Cross-Corr: {norm_val:.4f} at Lag: {max_lag}")

corr = np.corrcoef(py_wav[:min_len], mat_wav[:min_len])[0, 1]
print(f"  [INFO] Zero-Lag Correlation: {corr:.6f}")
if norm_val > 0.9:
    print(f"  [PASS] Signals match with Lag {max_lag}. Alignment issue.")
elif corr > 0.99:
    print("  [PASS] Signals are highly correlated at Lag 0.")
else:
    print("  [FAIL] Signals are fundamentally different even with lag.")

