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

# --- CONFIGURATION ---
DATA_ROOT = "data_asteroid_benchmark"
CONF = {
    "fs": 16000,
    "room_dim": [4.9, 4.9, 4.9],
    "rt60_target": 0.5,
    "mic_locs": [[2.41, 2.45, 1.5], [2.49, 2.45, 1.5]],
    "sir_target_db": 0,
    "sample_len": 64000  # 4 seconds at 16kHz
}

# --- 1. FILE LOADING (Same as your code) ---
def get_datasets():
    import kagglehub
    print("Loading datasets...")
    # NOTE: You might want to hardcode paths if already downloaded to save time
    try:
        lj_path = kagglehub.dataset_download("mathurinache/the-lj-speech-dataset")
        lj = glob.glob(os.path.join(lj_path, "**", "*.wav"), recursive=True)
        
        libri_path = kagglehub.dataset_download("pypiahmad/librispeech-asr-corpus")
        libri = glob.glob(os.path.join(libri_path, "**", "*.flac"), recursive=True)
        
        musan_path = kagglehub.dataset_download("dogrose/musan-dataset")
        musan = glob.glob(os.path.join(musan_path, "**", "*.wav"), recursive=True)
        return libri, lj, musan
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return [], [], []

LIBRI_FILES, LJ_FILES, MUSAN_FILES = get_datasets()

# --- 2. WAVEFORM GENERATOR (Stripped of STFT) ---
def generate_waveform_sample(split_seed):
    """
    Generates raw time-domain audio (Mix, Target, Interferer)
    """
    if len(LIBRI_FILES) == 0: return None

    # Use split_seed to ensure randomness but consistency if needed
    random.seed(split_seed)
    np.random.seed(split_seed % (2**32))

    # File Selection
    target_file = random.choice(LIBRI_FILES)
    interf1_file = random.choice(LJ_FILES)
    interf2_file = random.choice(MUSAN_FILES)
    interf_files = [interf1_file, interf2_file]

    # Room Setup (50% Reverb)
    use_reverb = random.random() < 0.5
    if use_reverb:
        e_abs, _ = pra.inverse_sabine(CONF["rt60_target"], CONF["room_dim"])
        max_order = 3
        materials = pra.Material(e_abs)
    else:
        max_order = 0
        materials = pra.Material(1.0)

    room = pra.ShoeBox(CONF["room_dim"], fs=CONF["fs"], materials=materials, max_order=max_order)
    room.add_microphone_array(np.array(CONF["mic_locs"]).T)

    # Load Audio Helper
    L = CONF["sample_len"]
    def load_audio(path):
        try:
            y, _ = librosa.load(path, sr=CONF["fs"], mono=True)
            if len(y) < L:
                tile_n = int(np.ceil(L/len(y)))
                y = np.tile(y, tile_n)
            start = random.randint(0, len(y) - L)
            return y[start:start+L]
        except:
            return np.zeros(L)

    sig_tgt = load_audio(target_file)
    sig_interfs = [load_audio(f) for f in interf_files]

    # Add Sources
    room.add_source([2.45, 3.45, 1.5], signal=sig_tgt) # Fixed Target
    for s in sig_interfs:
        rx = random.uniform(0.5, CONF["room_dim"][0]-0.5)
        ry = random.uniform(0.5, CONF["room_dim"][1]-0.5)
        room.add_source([rx, ry, 1.5], signal=s) # Random Interf

    # Simulate
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

    # SIR Scaling
    p_t = np.mean(tgt_mic[0]**2) + 1e-9
    p_i = np.mean(interf_mic[0]**2) + 1e-9
    scaler = np.sqrt(p_t / (p_i * (10**(CONF["sir_target_db"]/10))))
    interf_mic *= scaler

    # Noise
    snr = 25
    p_mix = np.mean((tgt_mic[0] + interf_mic[0])**2)
    noise_sigma = np.sqrt(p_mix / (10**(snr/10)))
    sensor_noise = np.random.normal(0, noise_sigma, tgt_mic.shape)

    mix = tgt_mic + interf_mic + sensor_noise
    
    # Normalize
    peak = np.max(np.abs(mix)) + 1e-9
    mix /= peak
    tgt_mic /= peak
    interf_mic /= peak

    return mix, tgt_mic, interf_mic

# --- 3. DATASET GENERATION LOOP ---
def create_dataset_csv(split_name, n_samples):
    print(f"\n--- Generating {split_name.upper()} Dataset ({n_samples} samples) ---")
    
    save_dir = os.path.join(DATA_ROOT, split_name)
    os.makedirs(save_dir, exist_ok=True)
    
    csv_data = []
    
    for i in tqdm(range(n_samples)):
        # Generate Audio
        # We pass a seed to ensure if we re-run this logic, it behaves somewhat predictably
        seed = hash(f"{split_name}_{i}")
        audio_tuple = generate_waveform_sample(seed)
        
        if audio_tuple is None: continue
        
        mix, tgt, intf = audio_tuple
        
        # Define Paths
        fname = f"{split_name}_{i:05d}"
        path_mix = os.path.join(save_dir, f"{fname}_mix.wav")
        path_tgt = os.path.join(save_dir, f"{fname}_tgt.wav")
        path_int = os.path.join(save_dir, f"{fname}_int.wav")
        
        # Save Audio (Shape: L, C)
        sf.write(path_mix, mix.T, CONF["fs"])
        sf.write(path_tgt, tgt.T, CONF["fs"])
        sf.write(path_int, intf.T, CONF["fs"])
        
        # Append to CSV Data
        # Asteroid expects absolute paths usually
        entry = {
            'mix_path': os.path.abspath(path_mix),
            'source_1_path': os.path.abspath(path_tgt), # Target
            'source_2_path': os.path.abspath(path_int), # Interference
            'noise_path': os.path.abspath(path_int),    # Noise (same as int for separation)
            'duration': CONF["sample_len"] / CONF["fs"],
            'sample_rate': CONF["fs"],
            'id': fname
        }
        csv_data.append(entry)
        
    # Save CSV
    df = pd.DataFrame(csv_data)
    csv_path = os.path.join(save_dir, "metadata.csv")
    df.to_csv(csv_path, index=False)
    print(f"Manifest saved to: {csv_path}")

# --- 4. EXECUTION ---
if __name__ == "__main__":
    # Generate 3 separate sets
    # Adjust numbers as needed (Train should be large)
    create_dataset_csv("train", 5000) 
    create_dataset_csv("val", 500)
    create_dataset_csv("test", 500)
    
    print("\n[Done] All datasets and CSVs ready.")