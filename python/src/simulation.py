import numpy as np
import soundfile as sf
import os
import glob
import random
import librosa
import kagglehub
import pyroomacoustics as pra
from scipy.signal import fftconvolve
from src import config

def get_audio_files(dataset_name, n_needed):
    """
    Fetches n_needed random audio files from Kaggle datasets.
    """
    print(f"--- Fetching {n_needed} files from: {dataset_name} ---")
    files = []
    try:
        if dataset_name == 'librispeech':
            # Downloads LibriSpeech
            path = kagglehub.dataset_download("pypiahmad/librispeech-asr-corpus")
            files = glob.glob(os.path.join(path, "**", "*.flac"), recursive=True)
        elif dataset_name == 'musan':
            # Downloads MUSAN
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
        
        return random.sample(files, n_needed)
    except Exception as e:
        print(f"Error getting data from {dataset_name}: {e}")
        return []

def add_awgn(signal, snr_db):
    """
    Adds Gaussian white noise to a signal at a specific SNR.
    """
    sig_power = np.mean(signal ** 2)
    if sig_power == 0: return signal
    
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise

def generate_scene(run_name, dataset='mixed', reverb=True, n_interferers=1, snr_target=5):
    """
    Generates a simulated audio scene.
    
    Args:
        run_name (str): Identifier for the run.
        dataset (str): 'mixed' for specific Libri+(LJ/Musan) setup.
        reverb (bool): True = Room Sim, False = Anechoic.
        n_interferers (int): Overridden to 1 if dataset='mixed'.
        snr_target (int): Target SNR in dB.
    """
    # 1. Setup Output Directory
    save_dir = os.path.join(config.SIM_DIR, run_name)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"[SIM] Generating '{run_name}' in {save_dir}...")
    print(f"[SIM] Config: Dataset={dataset} | Reverb={reverb} | SNR={snr_target}dB")

    # 2. Get Files & Load Audio
    files = []
    
    if dataset == 'mixed':
        # --- MIXED MODE CONFIGURATION ---
        # Target: LibriSpeech
        # Interferer: Randomly choose LJSpeech OR MUSAN
        
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
        
        # Order: [Target, Interferer]
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
        y, _ = librosa.load(f, sr=config.FS, mono=True)
        sigs.append(y)
        if len(y) < min_len: min_len = len(y)
    
    # Truncate all signals to the minimum length
    sigs = [s[:min_len] for s in sigs]
    
    target_sig = sigs[0]       
    interferer_sigs = sigs[1:] 

    # 3. Setup Room (Physics)
    # This logic respects the 'reverb' input argument
    if reverb:
        e_absorption, max_order = pra.inverse_sabine(config.RT60_TARGET, config.ROOM_DIM)
        materials = pra.Material(e_absorption)
        m_order = 15 
    else:
        materials = pra.Material(1.0) # Anechoic (No reflections)
        m_order = 0

    room = pra.ShoeBox(config.ROOM_DIM, fs=config.FS, materials=materials, max_order=m_order)
    
    # Add Microphone Array
    mic_locs = np.array(config.MIC_LOCS_SIM)
    room.add_microphone_array(mic_locs.T)
    
    # --- CALCULATE GEOMETRY ---
    mic_center = np.mean(mic_locs, axis=0) 
    
    # A. Add Target Source (Fixed Position)
    # 
    pos_target = [2.45, 3.45, 1.5]
    room.add_source(pos_target, signal=target_sig)
    
    # B. Add Interferer (Fixed Angle: 40 degrees)
    dist = 2.0  # Distance from array center
    
    if n_interferers > 0:
        # We only have 1 interferer in mixed mode, but loop handles generalized case
        for i in range(n_interferers):
            # Fixed 40 degrees
            angle_deg = 40.0
            angle_rad = np.radians(angle_deg)
            
            x = mic_center[0] + dist * np.cos(angle_rad)
            y = mic_center[1] + dist * np.sin(angle_rad)
            z = 1.5 
            
            print(f"[SIM] Placing Interferer at {angle_deg}° -> [{x:.2f}, {y:.2f}, {z:.2f}]")
            room.add_source([x, y, z], signal=interferer_sigs[i])

    # 4. Compute RIRs
    print("[SIM] Computing RIRs...")
    room.compute_rir()
    
    def get_convolved(sig, rir):
        return fftconvolve(sig, rir, mode='full')[:min_len]

    # --- 5. Convolution & Mixing ---
    
    # A. Process Target
    target_ch1 = get_convolved(target_sig, room.rir[0][0])
    target_ch2 = get_convolved(target_sig, room.rir[1][0])

    # B. Process Interferers
    interf_ch1_total = np.zeros(min_len)
    interf_ch2_total = np.zeros(min_len)

    if n_interferers > 0:
        for i, i_sig in enumerate(interferer_sigs):
            src_idx = i + 1
            i_ch1 = get_convolved(i_sig, room.rir[0][src_idx])
            i_ch2 = get_convolved(i_sig, room.rir[1][src_idx])
            
            interf_ch1_total += i_ch1
            interf_ch2_total += i_ch2

    # --- 6. SIR Control ---
    p_target = np.mean(target_ch1 ** 2) + 1e-9
    p_interf = np.mean(interf_ch1_total ** 2) + 1e-9
    
    if n_interferers > 0:
        desired_ratio = 10**(config.SIR_TARGET_DB/10)
        gain = np.sqrt(p_target / (p_interf * desired_ratio))
        print(f"[SIM] Applying Gain {gain:.4f} to Interferers")
        interf_ch1_total *= gain
        interf_ch2_total *= gain
    
    clean_mix_ch1 = target_ch1 + interf_ch1_total
    clean_mix_ch2 = target_ch2 + interf_ch2_total

    # --- 7. SNR Control ---
    print(f"[SIM] Adding AWGN at {snr_target}dB SNR")
    final_ch1 = add_awgn(clean_mix_ch1, snr_target)
    final_ch2 = add_awgn(clean_mix_ch2, snr_target)

    # --- 8. Normalization & Saving ---
    stereo_mix = np.stack([final_ch1, final_ch2], axis=1)
    stereo_target = np.stack([target_ch1, target_ch2], axis=1)
    stereo_interf = np.stack([interf_ch1_total, interf_ch2_total], axis=1)
    
    peak = np.max(np.abs(stereo_mix)) + 1e-9
    stereo_mix /= peak
    stereo_target /= peak
    stereo_interf /= peak

    mix_path = os.path.join(save_dir, "mixture.wav")
    tgt_path = os.path.join(save_dir, "target.wav")
    int_path = os.path.join(save_dir, "interference.wav")
    
    sf.write(mix_path, stereo_mix, config.FS)
    sf.write(tgt_path, stereo_target, config.FS)
    sf.write(int_path, stereo_interf, config.FS)
    
    print(f"[SIM] Simulation Complete.")
    print(f"[SIM] Files Saved to: {save_dir}")
    return mix_path