import os
import logging
import warnings
import torch
import pandas as pd
import numpy as np
import soundfile as sf
import pytorch_lightning as pl
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

# --- SILENCE ALL VERBOSE OUTPUT ---
warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.ERROR)
logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.ERROR)

# Asteroid Imports
from asteroid.models import ConvTasNet, FasNetTAC, DPRNNTasNet
from asteroid.engine.system import System
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr

# Metric Libraries
from pesq import pesq
from pystoi import stoi

# ==========================================
# --- CONFIGURATION ---
# ==========================================
MODELS_TO_TRAIN = ["convtasnet", "fasnet", "dprnntasnet"] 

# Paths
DATA_ROOT = "data_asteroid_benchmark"
TRAIN_CSV = os.path.join(DATA_ROOT, "train", "metadata.csv")
VAL_CSV = os.path.join(DATA_ROOT, "val", "metadata.csv")
TEST_SETS = {
    "Anechoic": os.path.join(DATA_ROOT, "test_anechoic", "metadata.csv"),
    "Reverb":   os.path.join(DATA_ROOT, "test_reverb", "metadata.csv")
}

BATCH_SIZE = 2
MAX_EPOCHS = 1
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

        if sources.ndim == 3: 
            sources = sources[:, 0, :] 

        return torch.from_numpy(mix), torch.from_numpy(sources)

# ==========================================
# --- 2. CUSTOM SYSTEM ---
# ==========================================
class StereoAwareSystem(System):
    def common_step(self, batch, batch_nb, train=True):
        inputs, targets = batch
        est_targets = self(inputs)
        
        if est_targets.ndim == 4:
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
# --- 4. PIPELINE RUNNER FOR A SINGLE MODEL ---
# ==========================================
def run_model_pipeline(model_type, global_pbar):
    use_mono = (model_type in ["convtasnet", "dprnntasnet"])
    
    train_set = CustomAsteroidDataset(TRAIN_CSV, sample_rate=FS, force_mono=use_mono)
    val_set = CustomAsteroidDataset(VAL_CSV, sample_rate=FS, force_mono=use_mono)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=4, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, num_workers=4, drop_last=True)
    
    if model_type == "convtasnet":
        model = ConvTasNet(n_src=2, n_blocks=8, n_repeats=3, mask_act="relu")
    elif model_type == "fasnet":
        model = FasNetTAC(n_src=2, enc_dim=64, feature_dim=64, hidden_dim=128, chunk_size=50, hop_size=25)
    elif model_type == "dprnntasnet":
        model = DPRNNTasNet(n_src=2, n_repeats=2, bn_chan=64, hid_size=128, chunk_size=100)
    else:
        raise ValueError(f"Unknown Model Type: {model_type}")

    optimizer = Adam(model.parameters(), lr=LR)
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from='pw_mtx')

    system = StereoAwareSystem(
        model=model,
        optimizer=optimizer,
        loss_func=loss_func,
        train_loader=train_loader,
        val_loader=val_loader
    )

    save_dir = os.path.join(CHECKPOINT_ROOT, model_type)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=save_dir, monitor="val_loss", mode="min", save_top_k=1, filename="best_model"
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback],
        gradient_clip_val=5.0,
        enable_progress_bar=False, 
        enable_model_summary=False,
        # logger=False,                 # Turns off "LitLogger" tip and default logging
        precision="16-mixed",         
        accumulate_grad_batches=4     
    )

    global_pbar.set_description(f"Training {model_type.upper()}")
    trainer.fit(system)
    
    best_ckpt = trainer.checkpoint_callback.best_model_path
    system = StereoAwareSystem.load_from_checkpoint(best_ckpt, model=model, train_loader=train_loader, val_loader=val_loader)
    system.model.eval()
    system.model.cuda()

    all_results = []
    
    global_pbar.set_description(f"Evaluating {model_type.upper()}")
    
    for env_name, csv_path in TEST_SETS.items():
        test_set = CustomAsteroidDataset(csv_path, sample_rate=FS, force_mono=use_mono)
        test_loader = DataLoader(test_set, batch_size=1, num_workers=2)

        for i, batch in enumerate(test_loader):
            mix, sources = batch
            mix, sources = mix.cuda(), sources.cuda()

            with torch.no_grad():
                est_sources = system.model(mix)
            
            if est_sources.ndim == 4:
                est_sources_loss = est_sources[:, :, 0, :]
            else:
                est_sources_loss = est_sources

            _, reordered_est = loss_func(est_sources_loss, sources, return_est=True)

            def get_cpu(tensor): return tensor[0].cpu().numpy()
            
            tgt_np = get_cpu(sources)[0] 
            est_np = get_cpu(reordered_est)[0] 
            
            if mix.ndim == 3: mix_np = mix[0, 0].cpu().numpy()
            else: mix_np = mix[0].cpu().numpy()

            metrics = compute_metrics_single_file(est_np, tgt_np, mix_np)
            metrics['Environment'] = env_name
            metrics['Model'] = model_type
            metrics['ID'] = i
            all_results.append(metrics)

    out_csv = f"results_{model_type}.csv"
    df = pd.DataFrame(all_results)
    df.to_csv(out_csv, index=False)
    
# ==========================================
# --- 5. MAIN EXECUTION ---
# ==========================================
if __name__ == "__main__":
    # Single global progress bar, no other prints
    with tqdm(total=len(MODELS_TO_TRAIN), desc="Initializing...", position=0, leave=True) as global_pbar:
        for m_type in MODELS_TO_TRAIN:
            run_model_pipeline(m_type, global_pbar)
            global_pbar.update(1)
        global_pbar.set_description("All Complete!")