<div align="center">

# Neural-Guided Beamformer
**Real-Time Speech Enhancement via Neural Priors and Restricted TFLC Optimization**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Android Min SDK](https://img.shields.io/badge/Android-Min%20SDK%2024-green.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📖 Overview

This is the official implementation of a hybrid audio enhancement system, for real-time operation on dual-microphone edge devices (with smartphone like form-factors). 

<!-- The system directly addresses the fragility of deep complex spectral mapping models (e.g., DCUNet, DCCRN) in reverberant, dual-mic geometries. By combining the powerful pattern recognition of deep learning with the strict distortionless constraints of classic array signal processing, we achieve aggressive interference suppression while mathematically preserving the target speech. -->

The pipeline comprises two main stages:
1. **Neural Priori**: A Speech-Adapted Efficient Group Enhanced UNet estimates a probabilistic Ideal Ratio Mask (IRM).
2. **Spatial Filtering**: A beamforming backend, which iteratively optimizes spatial filters using the neural priors.

---

## ✨ Key Features & Performance

- **Robust to Reverberation**: Outperforms state-of-the-art complex spectral mapping models. Benchmarks available.
- **Strictly Distortionless**: Unlike purely time-domain non-linear synthesizers (like ConvTasNet) that suffer from spectral degradation, the spatial filtering stage guarantees the target speech remains unaltered.
- **Low-Latency and Edge compatible**
<!-- - **Low Latency Edge Deployment**: Processing a 4-second audio segment takes just **0.3s for neural inference** and **0.1s for TFLC optimization** on a standard mobile NPU. -->
<!-- - **Cross-**: Includes a Python research pipeline and a fully-functional Android C++/Kotlin application for on-device testing. -->

---

<!-- ## 🔬 Methodology: How It Works

### 1. Feature Extraction & The SAEGE-UNet
At each time-frequency bin, we extract a composite tensor $\Psi(f, t) \in \mathbb{R}^5$ capturing spectral texture and spatial correlations:
- Log-magnitude of the reference mic
- Inter-channel Phase Difference (IPD) projected as $\sin(\Delta\phi)$ and $\cos(\Delta\phi)$
- Magnitude Squared Coherence (MSC)
- Fixed frequency map representation

The **SAEGE-UNet** takes this feature space and bounds it via a Sigmoid activation to estimate an Ideal Ratio Mask, $M(f, t) \in [0, 1]$. This mask acts strictly as an initialization prior for the next stage.

### 2. Iterative Restricted TFLC (RTFLC) Beamformer
The RTFLC beamformer enhances the target by forming a convex combination of $K=2$ component beamformers. 
Instead of blind initialization, we use the probability mask $M(f, t)$ to construct a noise-dominant spatial covariance matrix:
$$ \Phi_{\text{noise}}(f) = \frac{1}{T} \sum_{t=1}^{T} (1 - M(f, t)) x(f, t)x^H(f, t) $$

To avoid inter-beamformer coupling and prevent target cancellation, **Restricted TFLC** independently updates each spatial filter using mask-weighted statistics associated with its *own* beamformer selection mask. The filters are then resolved using a standard MVDR solution, strictly enforcing the distortionless constraint ($w_k^H(f)a(f) = 1$).

--- -->

## 📂 Repository Structure

| Directory | Description |
| :--- | :--- |
| [`python/`](python/) | **Core Python Implementation**: Clean inference scripts, metrics evaluation, and command-line interfaces. *Start here for desktop usage.* |
| [`AudioEnhancerCpp/`](AudioEnhancerCpp/) | **Android Application**: Complete Android Studio project (`app/`). Features a native C++ audio engine built via CMake and JNI. |
| [`models/`](models/) | **Weights**: Pre-trained `.pth` and `.onnx` models used by the inference pipelines. |
| [`research/`](research/) | **Development & Training**: Contains PyTorch Lightning training code, MATLAB prototyping scripts, and evaluation logic. |

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

### 📱 Android (Mobile App)

The mobile application is built to run the enhancement entirely on-device using C++ for low-latency NPU/CPU execution.

1. Open Android Studio and select **Open an Existing Project**.
2. Select the [`AudioEnhancerCpp/`](AudioEnhancerCpp/) folder.
3. Allow Gradle to sync and fetch the required NDK version.
4. Build and run on a physical Android device or emulator (API 24+).

---

## 📄 License & Citation

This project is licensed under the [MIT License](LICENSE).

If you use this code in your research, please cite our ICASSP paper:

```bibtex
@inproceedings{hybrid_neural_beamforming_2026,
  author = {Audio Signal Processing Intelligence and REsearch Labs (ASPIRE), Indian Institute of Information Technology Design and Manufacturing Kurnool},
  title = {Hybrid Neural Beamforming with SAEGE-UNet and Restricted TFLC for Edge Devices},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year = {2026}
}
```
