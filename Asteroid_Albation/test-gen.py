import os
import pandas as pd
import soundfile as sf
import numpy as np
import glob
import random
import scipy.signal
import librosa
import pyroomacoustics as pra
from tqdm import tqdm
import kagglehub

# --- CONFIGURATION ---
DATA_ROOT = "data_asteroid_benchmark"
CONF = {
    "fs": 16000,
    "room_dim": [4.9, 4.9, 4.9],
    "rt60_target": 0.5,
    "mic_locs": [[2.41, 2.45, 1.5], [2.49, 2.45, 1.5]],
    "sir_target_db": 0,
    "snr_target_db": 5,  # <--- FIXED: 5dB SNR Target
    "sample_len": 64000
}

# Load file lists
try:
    print("Loading datasets...")
    libri_path = kagglehub.dataset_download("pypiahmad/librispeech-asr-corpus")
    LIBRI_FILES = glob.glob(os.path.join(libri_path, "**", "*.flac"), recursive=True)
    
    lj_path = kagglehub.dataset_download("mathurinache/the-lj-speech-dataset")
    LJ_FILES = glob.glob(os.path.join(lj_path, "**", "*.wav"), recursive=True)
    
    musan_path = kagglehub.dataset_download("dogrose/musan-dataset")
    MUSAN_FILES = glob.glob(os.path.join(musan_path, "**", "*.wav"), recursive=True)
    print(f"Loaded: {len(LIBRI_FILES)} Libri, {len(LJ_FILES)} LJ, {len(MUSAN_FILES)} MUSAN")
except Exception as e:
    print(f"Error loading files: {e}")
    exit()

def add_awgn(signal, snr_db):
    """
    Adds Gaussian white noise to a signal at a specific SNR.
    """
    sig_power = np.mean(signal ** 2)
    if sig_power == 0: return signal
    
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise

def generate_waveform_sample(seed, force_reverb=None):
    """
    force_reverb: True (Reverb), False (Anechoic), None (Random)
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))

    # File Selection
    target_file = random.choice(LIBRI_FILES)
    interf_files = [random.choice(LJ_FILES), random.choice(MUSAN_FILES)]

    # Environment Logic
    if force_reverb is None:
        use_reverb = random.random() < 0.5
    else:
        use_reverb = force_reverb

    if use_reverb:
        e_abs, _ = pra.inverse_sabine(CONF["rt60_target"], CONF["room_dim"])
        max_order = 3
        materials = pra.Material(e_abs)
    else:
        max_order = 0
        materials = pra.Material(1.0) # Anechoic

    room = pra.ShoeBox(CONF["room_dim"], fs=CONF["fs"], materials=materials, max_order=max_order)
    room.add_microphone_array(np.array(CONF["mic_locs"]).T)

    L = CONF["sample_len"]
    def load_audio(path):
        try:
            y, _ = librosa.load(path, sr=CONF["fs"], mono=True)
            if len(y) < L: y = np.tile(y, int(np.ceil(L/len(y))))
            start = random.randint(0, len(y) - L)
            return y[start:start+L]
        except: return np.zeros(L)

    sig_tgt = load_audio(target_file)
    sig_interfs = [load_audio(f) for f in interf_files]

    # Add Sources
    # Target is fixed (broadside)
    room.add_source([2.45, 3.45, 1.5], signal=sig_tgt)
    # Interferers are random
    for s in sig_interfs:
        rx = random.uniform(0.5, CONF["room_dim"][0]-0.5)
        ry = random.uniform(0.5, CONF["room_dim"][1]-0.5)
        # Simple check to avoid placing source exactly on mic or target (optional but good practice)
        room.add_source([rx, ry, 1.5], signal=s)

    room.compute_rir()
    
    tgt_mic = np.zeros((2, L))
    interf_mic = np.zeros((2, L))

    # Convolve Target
    for m in range(2):
        tgt_mic[m] = scipy.signal.fftconvolve(sig_tgt, room.rir[m][0])[:L]
        
    # Convolve Interferers
    for idx, s in enumerate(sig_interfs):
        for m in range(2):
            interf_mic[m] += scipy.signal.fftconvolve(s, room.rir[m][idx+1])[:L]

    # SIR Scaling (Target vs Interference)
    p_t = np.mean(tgt_mic[0]**2) + 1e-9
    p_i = np.mean(interf_mic[0]**2) + 1e-9
    scaler = np.sqrt(p_t / (p_i * (10**(CONF["sir_target_db"]/10))))
    interf_mic *= scaler

    # --- 1. Create Clean Mix ---
    clean_mix = tgt_mic + interf_mic

    # --- 2. Add AWGN (5dB) ---
    # We apply noise to the mix based on the mix power
    noisy_mix_ch1 = add_awgn(clean_mix[0], CONF["snr_target_db"])
    noisy_mix_ch2 = add_awgn(clean_mix[1], CONF["snr_target_db"])
    final_mix = np.stack([noisy_mix_ch1, noisy_mix_ch2], axis=0)

    # Normalize
    peak = np.max(np.abs(final_mix)) + 1e-9
    final_mix /= peak
    tgt_mic /= peak
    interf_mic /= peak

    return final_mix, tgt_mic, interf_mic

def create_split(split_name, n_samples, reverb_condition):
    print(f"Generating {split_name} (Reverb={reverb_condition})...")
    save_dir = os.path.join(DATA_ROOT, split_name)
    os.makedirs(save_dir, exist_ok=True)
    csv_data = []

    for i in tqdm(range(n_samples)):
        seed = hash(f"{split_name}_{i}")
        audio = generate_waveform_sample(seed, force_reverb=reverb_condition)
        
        if audio is None: continue
        
        mix, tgt, intf = audio
        
        fname = f"{split_name}_{i:04d}"
        paths = {
            'mix': os.path.join(save_dir, f"{fname}_mix.wav"),
            's1': os.path.join(save_dir, f"{fname}_s1.wav"), # Target
            's2': os.path.join(save_dir, f"{fname}_s2.wav")  # Interf
        }
        
        # Save transposed (T, C) for soundfile
        sf.write(paths['mix'], mix.T, CONF["fs"])
        sf.write(paths['s1'], tgt.T, CONF["fs"])
        sf.write(paths['s2'], intf.T, CONF["fs"])

        csv_data.append({
            'mix_path': os.path.abspath(paths['mix']),
            'source_1_path': os.path.abspath(paths['s1']),
            'source_2_path': os.path.abspath(paths['s2']),
            'duration': CONF["sample_len"]/CONF["fs"],
            'sample_rate': CONF["fs"]
        })
    
    pd.DataFrame(csv_data).to_csv(os.path.join(save_dir, "metadata.csv"), index=False)

if __name__ == "__main__":
    # Create 500 samples for each condition
    create_split("test_anechoic", 500, reverb_condition=False)
    create_split("test_reverb", 500, reverb_condition=True)