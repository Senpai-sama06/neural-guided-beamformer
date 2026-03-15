# Ablation Study Framework

This directory contains ablation studies for testing alternative architectures and approaches on the same dataset used for the main solution.

## Structure

Each ablation study has its own folder with the following structure:

```
ablation_study/
├── <study_name>/
│   ├── inference.py          # Inference implementation for this approach
│   ├── metrics.py             # Metrics calculation (simplified)
│   ├── run_batch.py           # Batch processing script
│   ├── batch_test/
│   │   ├── reverb/            # Results for reverb dataset
│   │   └── anechoic/          # Results for anechoic dataset
│   └── <study_name>_metrics.csv  # Metrics CSV
```

## Current Studies

### 1. Oracle TFLC (`oracle/`)

Tests the upper bound performance using ground truth target and interference signals to construct oracle masks.

**Approach:**
- Uses oracle IRM: `mask_t = |S_target|² / (|S_target|² + |S_interference|²)`
- Applies TFLC (Time-Frequency Linearly Constrained) beamforming with iterative mask refinement
- Post-filters with oracle mask

**Purpose:** Establishes performance ceiling for mask-based beamforming approaches.

### 2. DeepFPU-RNN + SMVB (`deepfpu_rnn/`)

Tests learned mask estimation using DeepFPU-RNN model with Steered Minimum Variance Beamformer.

**Approach:**
- Uses DeepFPU-RNN (CRNN architecture) to estimate masks from mixture
- Applies SMVB (Steered Minimum Variance Beamformer) with adaptive LCMV/MVDR switching
- Adaptively selects beamforming strategy based on interference rank
- Post-filters with learned mask

**Purpose:** Evaluates performance of learned mask estimation with physics-based steering compared to oracle upper bound.

## Adding New Studies

1. Create a new folder: `ablation_study/<study_name>/`
2. Implement `inference.py` with your approach
3. Copy and adapt `metrics.py` from an existing study
4. Create `run_batch.py` for batch processing
5. Update this README with study description

## Running Studies

Each study has its own `run_batch.py` script. Example:

```bash
cd ablation_study/oracle
python run_batch.py --start 0 --end 499
```

This will process samples 0-499 for both reverb and anechoic modes.
