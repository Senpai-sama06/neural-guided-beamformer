<div align="center">

# 🎧 IRM_TFLC: Real-Time Audio Enhancement
**Neural Network Estimated Masks + Iterative TFLC Beamforming**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Android Min SDK](https://img.shields.io/badge/Android-Min%20SDK%2024-green.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📖 Overview

**IRM_TFLC** is an advanced, real-time audio enhancement and source separation pipeline. It is designed to significantly improve speech intelligibility in noisy and reverberant environments.

The system utilizes a two-stage approach:
1. **Mask Estimation**: An efficient, lightweight **EGE-Unet** neural network estimates Ideal Ratio Masks (IRM) from raw audio features.
2. **Spatial Filtering**: An **Iterative Time-Frequency Line Constrained (TFLC) Beamformer** utilizes these masks to spatially isolate the target speaker and suppress background noise.

This repository provides **both** a research-friendly Python pipeline and a highly optimized C++/Kotlin implementation for mobile Android devices.

---

## ✨ Key Features

- **Hybrid Architecture**: Combines the pattern recognition strengths of Deep Learning with the spatial precision of classic array signal processing (Beamforming).
- **Mobile Optimized**: Includes a complete Android Studio project featuring a C++ (JNI) audio engine for ultra-low latency on-device processing.
- **Pre-Trained Models**: Ready-to-use ONNX and PyTorch weights for immediate inference.
- **Cross-Language Parity**: The Python and Android implementations are mathematically aligned, ensuring identical audio quality across platforms.

---

## 📂 Repository Structure

The project has been organized to support both end-users and researchers:

| Directory | Description |
| :--- | :--- |
| [`python/`](python/) | **Core Python Implementation**: Clean inference scripts, metrics evaluation, and command-line interfaces. *Start here for desktop usage.* |
| [`AudioEnhancerCpp/`](AudioEnhancerCpp/) | **Android Application**: Complete Android Studio project (`app/`). Features a native C++ audio engine built via CMake and JNI. |
| [`models/`](models/) | **Weights**: Pre-trained `.pth` and `.onnx` models used by the inference pipelines. |
| [`research/`](research/) | **Development & Training**: Contains the original PyTorch Lightning (Asteroid) training code, MATLAB prototyping scripts, and SP-Cup competition submissions. |

---

## 🚀 Getting Started

### 🐍 Python (Desktop / Server)

1. Navigate to the Python directory and install dependencies:
   ```bash
   cd python
   pip install -r requirements.txt
   ```
2. Run the enhancement inference on a noisy audio file:
   ```bash
   python -m src.inference --input data/noisy_sample.wav --output data/clean_output.wav
   ```
   *(See the [Python README](python/README.md) for advanced usage, batch processing, and evaluation metrics).*

### 📱 Android (Mobile App)

The mobile application is built to run the enhancement entirely on-device using C++ for performance.

1. Open Android Studio and select **Open an Existing Project**.
2. Select the [`AudioEnhancerCpp/`](AudioEnhancerCpp/) folder.
3. Allow Gradle to sync and fetch the required NDK version (specified in `build.gradle.kts`).
4. Build and run on a physical Android device or emulator (API 24+).

*(See the [Android README](AudioEnhancerCpp/README.md) for detailed JNI/C++ compilation instructions).*

---

## 🔬 How It Works

1. **Feature Extraction**: Multi-channel STFTs are computed. The pipeline extracts Log-Magnitude, Inter-channel Phase Differences (IPD), and Magnitude Squared Coherence (MSC).
2. **Neural Network Inference**: The EGE-Unet model processes these spatial and spectral features to estimate a soft mask representing the probability of speech dominance in each time-frequency bin.
3. **Iterative Beamforming**: The estimated mask is used to construct spatial covariance matrices. The TFLC beamformer iteratively refines these matrices to cleanly extract the target signal without creating "musical noise" artifacts.

---

## 📄 License & Citation

This project is licensed under the [MIT License](LICENSE).

If you use this code in your research, please consider citing our work:

```bibtex
@misc{irm_tflc_2026,
  author = {Audio Signal Processing and Intelligence REsearch labs (ASPIRE labs) IIITDM Kurnool },
  title = {Neural Guided beamformer for two channel microphones for mobile edge devices},
  year = {2026},
  howpublished = {\url{https://github.com/your-username/IRM-TFLC}}
}
```
