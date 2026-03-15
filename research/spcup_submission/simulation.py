import numpy as np
import soundfile as sf
import os
import glob
import random
import librosa
import kagglehub
import pyroomacoustics as pra
from scipy.signal import fftconvolve
from scipy.io import savemat

# =========================
# CONFIGURATION (Embedded)
# =========================
class Config:
    FS = 16000
    ROOM_DIM = [4.9, 4.9, 4.9]
    RT60_TARGET = 0.5
    SIR_TARGET_DB = 0
    MIC_LOCS_SIM = [[2.41, 2.45, 1.5], [2.49, 2.45, 1.5]] # 2 Mics
    MIC_DIST = 0.08
    # Directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_DIR = os.path.join(BASE_DIR, "input")

# Ensure input directory exists
os.makedirs(Config.INPUT_DIR, exist_ok=True)

def get_audio_files(dataset_name, n_needed):
    """
    Fetches n_needed random audio files from Kaggle datasets.
    """
    print(f"--- Fetching {n_needed} files from: {dataset_name} ---")
    files = []
    try:
        if dataset_name == 'librispeech':
            path = kagglehub.dataset_download("pypiahmad/librispeech-asr-corpus")
            files = glob.glob(os.path.join(path, "**", "*.flac"), recursive=True)
        elif dataset_name == 'musan':
            path = kagglehub.dataset_download("dogrose/musan-dataset")
            files = glob.glob(os.path.join(path, "**", "*.wav"), recursive=True)
        else: 
            # Default: ljspeech
            path = kagglehub.dataset_download("mathurinache/the-lj-speech-dataset")
            wav_path = os.path.join(path, "LJSpeech-1.1", "wavs")
            files = glob.glob(os.path.join(wav_path, "*.wav"))
        
        if len(files) == 0: 
            raise ValueError(f"No files found for {dataset_name}")
            
        if len(files) < n_needed:
            print(f"Warning: Only {len(files)} files found in {dataset_name}. Duplicating.")
            while len(files) < n_needed:
                files += files
        
        files.sort() # Ensure deterministic order
        return random.sample(files, n_needed)
    except Exception as e:
        print(f"Error getting data from {dataset_name}: {e}")
        return []

def add_awgn(signal, snr_db):
    """
    Adds Gaussian white noise to a signal at a specific SNR.
    """
    sig_power = np.mean(signal ** 2)
    if sig_power == 0: return signal, np.zeros_like(signal)
    
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise, noise

def generate_scene(run_name, dataset='mixed', reverb=True, n_interferers=1, snr_target=5, seed=42):
    """
    Generates a simulated audio scene and saves it in the specific Phase 2 Submission Format.
    """
    # Set seed for reproducibility (same files for both tasks)
    random.seed(seed)
    np.random.seed(seed)

    # 1. Setup Output Directory
    if "Task1" in run_name:
        save_dir = os.path.join(Config.BASE_DIR, "Task1_Anechoic")
    elif "Task2" in run_name:
        save_dir = os.path.join(Config.BASE_DIR, "Task2_Reverberant")
    else:
        save_dir = os.path.join(Config.BASE_DIR, "input")
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"[SIM] Generating '{run_name}' in {save_dir}...")

    # 2. Get Files & Load Audio
    files = []
    
    if dataset == 'mixed':
        # Force 1 interferer for this mode
        n_interferers = 1
        
        f_target = get_audio_files('librispeech', 1)
        
        # 50/50 Split for Interferer Type
        if random.random() < 0.5:
            f_int = get_audio_files('ljspeech', 1)
            int_type = "LJSpeech"
        else:
            f_int = get_audio_files('musan', 1)
            int_type = "MUSAN"
            
        print(f"[SIM] Mixed Mode: Target=LibriSpeech | Interferer={int_type}")
        files = f_target + f_int
        
    else:
        # Homogeneous Mode
        total_sources = 1 + n_interferers
        files = get_audio_files(dataset, total_sources)
    
    if not files:
        print("[SIM] Error: Not enough audio files retrieved.")
        return None

    # Load audio and determine minimum common length
    sigs = []
    min_len = float('inf')
    
    for f in files:
        # Load at 16kHz Mono
        y, _ = librosa.load(f, sr=Config.FS, mono=True)
        sigs.append(y)
        if len(y) < min_len: min_len = len(y)
    
    # Ensure minimum length is sufficient (e.g. 4s)
    if min_len < 4 * Config.FS:
        print(f"[WARNING] Audio too short: {min_len/Config.FS:.2f}s")

    # Truncate all signals to the minimum length
    sigs = [s[:min_len] for s in sigs]
    
    target_sig = sigs[0]       
    interferer_sigs = sigs[1:] 

    # 3. Setup Room (Physics)
    if reverb:
        e_absorption, max_order = pra.inverse_sabine(Config.RT60_TARGET, Config.ROOM_DIM)
        materials = pra.Material(e_absorption)
        m_order = 15 
        rir_type = "Reverberant"
    else:
        materials = pra.Material(1.0) # Anechoic (No reflections)
        m_order = 0
        rir_type = "Anechoic"

    room = pra.ShoeBox(Config.ROOM_DIM, fs=Config.FS, materials=materials, max_order=m_order)
    
    # Add Microphone Array
    mic_locs = np.array(Config.MIC_LOCS_SIM)
    room.add_microphone_array(mic_locs.T)
    
    # --- CALCULATE GEOMETRY ---
    mic_center = np.mean(mic_locs, axis=0) 
    
    # A. Add Target Source (Fixed Position)
    pos_target = [2.45, 3.45, 1.5]
    room.add_source(pos_target, signal=target_sig)
    
    # B. Add Interferer (Fixed Angle: 40 degrees)
    dist = 2.0  # Distance from array center
    pos_interferers = []
    
    if n_interferers > 0:
        for i in range(n_interferers):
            angle_deg = 40.0
            angle_rad = np.radians(angle_deg)
            x = mic_center[0] + dist * np.cos(angle_rad)
            y = mic_center[1] + dist * np.sin(angle_rad)
            z = 1.5 
            pos = [x, y, z]
            pos_interferers.append(pos)
            
            print(f"[SIM] Placing Interferer at {angle_deg}° -> [{x:.2f}, {y:.2f}, {z:.2f}]")
            room.add_source(pos, signal=interferer_sigs[i])

    # 4. Compute RIRs
    print("[SIM] Computing RIRs...")
    room.compute_rir()
    
    # Extract RIRs for saving
    # room.rir is a list of lists: rir[mic_idx][src_idx]
    # We flatten this for the .mat file
    rir_data = [] # List of lists
    for m in range(len(room.rir)):
        mic_rirs = []
        for s in range(len(room.rir[m])):
            mic_rirs.append(room.rir[m][s])
        rir_data.append(mic_rirs)

    def get_convolved(sig, rir):
        return fftconvolve(sig, rir, mode='full')[:min_len]

    # --- 5. Convolution & Mixing ---
    
    # A. Process Target
    target_ch1 = get_convolved(target_sig, room.rir[0][0])
    target_ch2 = get_convolved(target_sig, room.rir[1][0])

    # B. Process Interferers
    interf_ch1_total = np.zeros(min_len)
    interf_ch2_total = np.zeros(min_len)
    
    # Keep track of individual interferer signals for saving
    individual_interferers = [] 

    if n_interferers > 0:
        for i, i_sig in enumerate(interferer_sigs):
            src_idx = i + 1
            i_ch1 = get_convolved(i_sig, room.rir[0][src_idx])
            i_ch2 = get_convolved(i_sig, room.rir[1][src_idx])
            
            interf_ch1_total += i_ch1
            interf_ch2_total += i_ch2
            
            # Store stereo interference for this source
            individual_interferers.append(np.stack([i_ch1, i_ch2], axis=1))

    # --- 6. SIR Control ---
    p_target = np.mean(target_ch1 ** 2) + 1e-9
    p_interf = np.mean(interf_ch1_total ** 2) + 1e-9
    
    gain_interf = 1.0
    if n_interferers > 0:
        desired_ratio = 10**(Config.SIR_TARGET_DB/10)
        gain_interf = np.sqrt(p_target / (p_interf * desired_ratio))
        print(f"[SIM] Applying Gain {gain_interf:.4f} to Interferers")
        interf_ch1_total *= gain_interf
        interf_ch2_total *= gain_interf
        
        # Apply gain to individual stored signals too
        for i in range(len(individual_interferers)):
            individual_interferers[i] *= gain_interf
    
    clean_mix_ch1 = target_ch1 + interf_ch1_total
    clean_mix_ch2 = target_ch2 + interf_ch2_total

    # --- 7. SNR Control ---
    print(f"[SIM] Adding AWGN at {snr_target}dB SNR")
    final_ch1, n1 = add_awgn(clean_mix_ch1, snr_target)
    final_ch2, n2 = add_awgn(clean_mix_ch2, snr_target)

    # --- 8. Normalization & Saving ---
    stereo_mix = np.stack([final_ch1, final_ch2], axis=1)
    stereo_target = np.stack([target_ch1, target_ch2], axis=1)
    stereo_interf_total = np.stack([interf_ch1_total, interf_ch2_total], axis=1)
    stereo_noise = np.stack([n1, n2], axis=1)
    
    peak = np.max(np.abs(stereo_mix)) + 1e-9
    stereo_mix /= peak
    stereo_target /= peak
    stereo_interf_total /= peak
    stereo_noise /= peak
    
    # Normalize individual files with same peak? No, usually relative. 
    # But for WAV output we need to avoid clipping.
    # We'll use the mixture peak for consistency.
    for i in range(len(individual_interferers)):
        individual_interferers[i] /= peak

    # --- SAVE .WAV FILES ---
    # 1. Target
    sf.write(os.path.join(save_dir, "target_signal.wav"), stereo_target, Config.FS)
    # 2. Interferers
    if n_interferers > 0:
        for i in range(n_interferers):
            # 1-indexed naming for compliance: interference_signal1.wav
            fname = f"interference_signal{i+1}.wav"
            sf.write(os.path.join(save_dir, fname), individual_interferers[i], Config.FS)
    else:
        # Fallback if no interferer, create silence? Or skip.
         sf.write(os.path.join(save_dir, "interference_signal1.wav"), np.zeros_like(stereo_target), Config.FS)

    # 3. Mixture (Optional, but good for debug)
    sf.write(os.path.join(save_dir, "mixture_signal.wav"), stereo_mix, Config.FS)
    
    # 4. Noise (Optional, requested by user prompt "save target, interference and noise")
    sf.write(os.path.join(save_dir, "noise_signal.wav"), stereo_noise, Config.FS)

    # --- SAVE .MAT FILE ---
    # Construct Params Dictionary
    params = {
        'fs': Config.FS,
        'room_dim': Config.ROOM_DIM,
        'mic_locs': Config.MIC_LOCS_SIM,
        'target_pos': pos_target,
        'interferer_pos': pos_interferers,
        'rt60': Config.RT60_TARGET if reverb else 0,
        'snr': snr_target,
        'sir': Config.SIR_TARGET_DB,
        'rir_type': rir_type,
        'peak_norm_factor': peak
    }

    # Construct Output Dictionary
    mat_out = {
        'target_signal': stereo_target,
        'interference_signal': stereo_interf_total, # Total interference
        'mixture_signal': stereo_mix,
        'noise_signal': stereo_noise,
        'rir_data': rir_data, # This might need careful handling in Matlab, but saves as cells
        'params': params
    }
    
    mat_filename = f"{run_name}.mat"
    mat_path = os.path.join(save_dir, mat_filename)
    
    savemat(mat_path, mat_out)
    
    print(f"[SIM] Simulation Complete.")
    print(f"[SIM] Wave Files and {mat_filename} Saved to: {save_dir}")
    return mat_path


# =========================
# MAIN EXECUTION
# =========================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['1', '2', 'all'], default='all', help="Run Task 1 (Anechoic), 2 (Reverb), or both")
    args = parser.parse_args()

    # TASK 1: Anechoic, 5dB SNR
    if args.task in ['1', 'all']:
        generate_scene(
            run_name="Task1_Anechoic_5dB",
            dataset="mixed",
            reverb=False, 
            n_interferers=1, # Mixed mode forces 1 anyway
            snr_target=5
        )

    # TASK 2: Reverberant, 5dB SNR
    if args.task in ['2', 'all']:
        generate_scene(
            run_name="Task2_Reverberant_5dB",
            dataset="mixed",
            reverb=True,
            n_interferers=1,
            snr_target=5
        )