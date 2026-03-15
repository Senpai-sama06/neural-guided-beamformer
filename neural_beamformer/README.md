# 🐍 Neural-Guided Beamformer (Python Package)

The `neural_beamformer` package serves as the core research and desktop inference pipeline for the Hybrid Audio Zooming system. It contains the logic to extract spatial features, run the SAEGE-UNet neural prior, and execute the Restricted TFLC optimization algorithm.

## Installation

Ensure you have Python 3.8+ installed. You can install the package in editable mode directly from the root repository:

```bash
# From the root of the neural-guided-beamformer repo:
pip install .
```

This will automatically install necessary dependencies such as `torch`, `soundfile`, `numpy`, and `scipy`. Note that **PyTorch model weights are required** and must be provided separately.

## Usage

You can use this package either directly through the Command Line Interface (CLI) or import it as a library into your own Python code.

### 1. Command Line Interface (CLI)

Once installed, a global `enhance-audio` command becomes available in your terminal environment.

```bash
enhance-audio --input path/to/noisy.wav --output path/to/clean.wav --model path/to/saege_unet.pth
```

**Arguments:**
- `--input`: Path to the input noisy audio file. Ideally, a 2-channel `.wav` file matching the specific 8cm dual-mic geometry the model was tuned for.
- `--output`: Path where the enhanced `.wav` file will be saved.
- `--model`: Path to the pre-trained `EGE-Unet` model weights `.pth` file.

### 2. Using as a Library

You can directly invoke the pipeline programmatically in your own scripts:

```python
from neural_beamformer import enhance_audio

INPUT_PATH = "data/noisy_sample.wav"
OUTPUT_PATH = "data/enhanced_output.wav"
MODEL_PATH = "models/best_model_weights.pth"

# Run the inference pipeline
enhance_audio(INPUT_PATH, OUTPUT_PATH, MODEL_PATH)
```

## Internal Architecture

The logic is split across several modules within `neural_beamformer/`:
- **`cli.py`**: Handles terminal argument parsing.
- **`inference.py`**: The core logic containing the STFT extraction, `EGE_Audio_UNet` class, and the `tflc_beamforming_broadside` algorithm.
- **`config.py`**: Centralizes physical constants (e.g., sample rate `16000`, FFT sizes).
- **`simulation.py` & `metrics.py`**: Tooling for generating RIR scenes and evaluating performance (SIR, PESQ, STOI).
