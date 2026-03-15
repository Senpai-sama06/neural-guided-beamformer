import os
import torch
import pandas as pd
import numpy as np
import soundfile as sf
import pytorch_lightning as pl
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

# Asteroid Imports
from asteroid.models import ConvTasNet, FasNetTAC
from asteroid.engine.system import System
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr

# Metric Libraries
from pesq import pesq
from pystoi import stoi

# ==========================================
# --- CONFIGURATION ---
# ==========================================
MODEL_TYPE = "convtasnet"  # Options: "convtasnet", "fasnet"

# Paths
DATA_ROOT = "data_asteroid_benchmark"
TRAIN_CSV = os.path.join(DATA_ROOT, "train", "metadata.csv")
VAL_CSV = os.path.join(DATA_ROOT, "val", "metadata.csv")
TEST_SETS = {
    "Anechoic": os.path.join(DATA_ROOT, "test_anechoic", "metadata.csv"),
    "Reverb":   os.path.join(DATA_ROOT, "test_reverb", "metadata.csv")
}

BATCH_SIZE = 2
MAX_EPOCHS = 100
LR = 1e-3
FS = 16000
CHECKPOINT_ROOT = "trained_benchmarks"

# ==========================================
# --- 1. CUSTOM DATASET CLASS ---
# ==========================================
class CustomAsteroidDataset(Dataset):
    def __init__(self, csv_path, sample_rate=16000, force_mono=False):
        self.df = pd.read_csv(csv_path)
        self.sample_rate = sample_rate
        self.force_mono = force_mono

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        mix, _ = sf.read(row['mix_path'], dtype='float32')
        src1, _ = sf.read(row['source_1_path'], dtype='float32')
        src2, _ = sf.read(row['source_2_path'], dtype='float32')
        
        sources = np.stack([src1, src2], axis=0)
        
        if mix.ndim > 1: mix = mix.T
        else: mix = mix[None, :]

        if self.force_mono:
            mix = mix[0:1, :] 

        if sources.ndim > 2: sources = sources.transpose(0, 2, 1)

        # IMPORTANT: For FaSNet, sources are usually mono ground truth. 
        # Ensure sources are (Src, Time) if they are mono files, or (Src, Chan, Time) if stereo.
        # Standard Asteroid loss expects (Src, Time). 
        # If your ground truth 'source_1_path' is stereo, we must force it to mono here for the loss to work.
        if sources.ndim == 3: 
             # Take Channel 0 of source as ground truth reference
            sources = sources[:, 0, :] 

        return torch.from_numpy(mix), torch.from_numpy(sources)

# ==========================================
# --- 2. CUSTOM SYSTEM (THE FIX) ---
# ==========================================
class StereoAwareSystem(System):
    def common_step(self, batch, batch_nb, train=True):
        inputs, targets = batch
        est_targets = self(inputs)
        
        # --- FIX: Handle 4D Output from Spatial Models ---
        # FaSNet Output: (Batch, n_src, n_mics, Time) -> [B, 2, 2, T]
        # Loss Expects:  (Batch, n_src, Time)         -> [B, 2, T]
        
        if est_targets.ndim == 4:
            # Take Channel 0 (Reference Mic) for loss calculation
            est_targets = est_targets[:, :, 0, :]
            
        loss = self.loss_func(est_targets, targets)
        return loss

# ==========================================
# --- 3. METRIC HELPER ---
# ==========================================
def compute_metrics_single_file(est, tgt, mix):
    def get_sisdr(x, y):
        alpha = np.dot(x, y) / (np.linalg.norm(y)**2 + 1e-8)
        target_scaled = alpha * y
        noise = x - target_scaled
        return 10 * np.log10(np.linalg.norm(target_scaled)**2 / (np.linalg.norm(noise)**2 + 1e-8) + 1e-8)

    si_sdr_val = get_sisdr(est, tgt)
    si_sdr_imp = si_sdr_val - get_sisdr(mix, tgt)

    try: stoi_val = stoi(tgt, est, FS, extended=False)
    except: stoi_val = 0.0

    try: pesq_wb = pesq(FS, tgt, est, 'wb')
    except: pesq_wb = 0.0

    return {"SI_SDR": si_sdr_val, "SI_SDRi": si_sdr_imp, "STOI": stoi_val, "PESQ_WB": pesq_wb}

# ==========================================
# --- 4. MAIN PIPELINE ---
# ==========================================
def main():
    print(f"\n{'='*50}\n   FULL PIPELINE: {MODEL_TYPE.upper()}\n{'='*50}\n")

    USE_MONO = (MODEL_TYPE == "convtasnet")
    print(f"[Config] Mode: {'MONO (Blind)' if USE_MONO else 'STEREO (Spatial)'}")

    # --- PART A: DATA LOADING ---
    print(f"[Data] Loading Custom CSVs...")
    train_set = CustomAsteroidDataset(TRAIN_CSV, sample_rate=FS, force_mono=USE_MONO)
    val_set = CustomAsteroidDataset(VAL_CSV, sample_rate=FS, force_mono=USE_MONO)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=4, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=4, drop_last=True)

    # --- PART B: MODEL DEFINITION ---
    print(f"[Model] Initializing {MODEL_TYPE.upper()}...")
    
    if MODEL_TYPE == "convtasnet":
        model = ConvTasNet(n_src=2,  n_blocks=8, n_repeats=3, mask_act="relu")
    elif MODEL_TYPE == "fasnet":
        model = FasNetTAC(n_src=2, enc_dim=64, feature_dim=64, hidden_dim=128, chunk_size=50, hop_size=25)
    else:
        raise ValueError("Unknown Model Type")

    # --- PART C: SYSTEM SETUP ---
    optimizer = Adam(model.parameters(), lr=LR)
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx')

    # USE CUSTOM SYSTEM CLASS HERE
    system = StereoAwareSystem(
        model=model,
        optimizer=optimizer,
        loss_func=loss_func,
        train_loader=train_loader,
        val_loader=val_loader
    )

    # --- PART D: TRAINING ---
    save_dir = os.path.join(CHECKPOINT_ROOT, MODEL_TYPE)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=save_dir, monitor="val_loss", mode="min", save_top_k=1, filename="best_model"
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback],
        gradient_clip_val=5.0
    )

    print("[Train] Starting Training Loop...")
    trainer.fit(system)
    
    # --- PART E: EVALUATION ---
    print(f"\n{'='*50}\n   STARTING EVALUATION\n{'='*50}\n")
    
    best_ckpt = trainer.checkpoint_callback.best_model_path
    system = StereoAwareSystem.load_from_checkpoint(best_ckpt, model=model, train_loader=train_loader, val_loader=val_loader)
    system.model.eval()
    system.model.cuda()

    all_results = []

    for env_name, csv_path in TEST_SETS.items():
        print(f"\n[Eval] Environment: {env_name}")
        test_set = CustomAsteroidDataset(csv_path, sample_rate=FS, force_mono=USE_MONO)
        test_loader = DataLoader(test_set, batch_size=1, num_workers=2)

        for i, batch in enumerate(tqdm(test_loader)):
            mix, sources = batch
            mix, sources = mix.cuda(), sources.cuda()

            with torch.no_grad():
                est_sources = system.model(mix)
            
            # Fix Dimensions for Loss/Alignment
            if est_sources.ndim == 4:
                est_sources_loss = est_sources[:, :, 0, :] # Take Chan 0
            else:
                est_sources_loss = est_sources

            # PIT Alignment 
            _, reordered_est = loss_func(est_sources_loss, sources, return_est=True)

            # Metrics
            def get_cpu(tensor): return tensor[0].cpu().numpy()
            
            # Sources are already mono (B, Src, T) from dataset
            tgt_np = get_cpu(sources)[0] # Src 0
            est_np = get_cpu(reordered_est)[0] # Est matched to Src 0
            
            if mix.ndim == 3: mix_np = mix[0, 0].cpu().numpy()
            else: mix_np = mix[0].cpu().numpy()

            metrics = compute_metrics_single_file(est_np, tgt_np, mix_np)
            metrics['Environment'] = env_name
            metrics['Model'] = MODEL_TYPE
            metrics['ID'] = i
            all_results.append(metrics)

    out_csv = f"results_{MODEL_TYPE}.csv"
    df = pd.DataFrame(all_results)
    df.to_csv(out_csv, index=False)
    
    print(f"\n[Eval] Saved to: {out_csv}")
    print(df.groupby(['Environment']).mean()[['SI_SDR', 'SI_SDRi', 'PESQ_WB', 'STOI']])

if __name__ == "__main__":
    main()