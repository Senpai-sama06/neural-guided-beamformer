#include "AudioEngine.h"
#ifndef AUDIO_DEBUG_LINUX
#include <android/log.h>
#endif
#define _USE_MATH_DEFINES
#ifndef AUDIO_DEBUG_LINUX
#include "nnapi_provider_factory.h"
#endif
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <thread>
#include <vector>

#define LOG_TAG "AudioEngine"
#define LOG_TAG "AudioEngine"

#ifdef AUDIO_DEBUG_LINUX
#include <stdio.h>
#include <stdarg.h>
#include <iostream>
#include <fstream>
void __android_log_print(int prio, const char* tag, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    printf("[%s] ", tag);
    vprintf(fmt, args);
    printf("\n");
    va_end(args);
}
#define ANDROID_LOG_DEBUG 3
#endif

#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

using namespace Eigen;

// --- Helper for Stats ---
void logStats(const std::string &name, const std::vector<float> &data) {
  if (data.empty())
    return;
  float sum = 0, min_val = data[0], max_val = data[0];
  for (float v : data) {
    sum += v;
    if (v < min_val)
      min_val = v;
    if (v > max_val)
      max_val = v;
  }
  float mean = sum / data.size();
  LOGD("[%s] Min: %.4f, Max: %.4f, Mean: %.4f", name.c_str(), min_val, max_val,
       mean);
  
  #ifdef AUDIO_DEBUG_LINUX
  std::ofstream file("debug_cpp_" + name + ".txt");
  for(float v : data) file << v << "\n";
  #endif
}

void logStatsMatrix(const std::string &name, const MatrixXf &mat) {
  float min_val = mat.minCoeff();
  float max_val = mat.maxCoeff();
  float mean = mat.mean();
  LOGD("[%s] Min: %.4f, Max: %.4f, Mean: %.4f", name.c_str(), min_val, max_val,
       mean);

  #ifdef AUDIO_DEBUG_LINUX
  std::ofstream file("debug_cpp_" + name + ".txt");
  file << mat;
  #endif
}

void dumpMaskToDisk(const MatrixXf &mask, int F, int T) {
    #ifdef AUDIO_DEBUG_LINUX
    std::ofstream file("debug_cpp_mask.txt");
    file << mask;
    #endif
}

void dumpMatrixXcd(const std::string &name, const MatrixXcd &mat) {
   #ifdef AUDIO_DEBUG_LINUX
   std::ofstream file("debug_cpp_" + name + ".txt");
   file << mat.real(); // Just dump real part for quick check or handle complex
   // Better dump custom CSV
   std::ofstream file2("debug_cpp_" + name + "_complex.csv");
   for(int i=0; i<mat.rows(); ++i) {
       for(int j=0; j<mat.cols(); ++j) {
           file2 << mat(i,j).real() << "+" << mat(i,j).imag() << "i,";
       }
       file2 << "\n";
   }
   #endif
}

// --- Helper for Complex Matrices in Eigen ---
// Eigen doesn't have a direct "Hermitian Transpose" symbol like ' in MATLAB for
// matrix multiplication in expr templates easily without .adjoint() We utilize
// .adjoint() which is the conjugate transpose.

AudioEngine::AudioEngine(const std::string &modelPath)
    : env(ORT_LOGGING_LEVEL_WARNING, "AudioEnhancer"), session(nullptr) {

  // 1. Initialize ONNX Session (Optimized for XNNPACK/CPU)
  Ort::SessionOptions sessionOptions;
  sessionOptions.SetIntraOpNumThreads(
      4); // Sweet spot for mobile CPU (Big Cores)
  sessionOptions.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);

  // Disable spinning to prevent thermal throttling (crucial for sustained audio
  // performance)
  sessionOptions.AddConfigEntry("session.intra_op.allow_spinning", "0");

  // Explicitly disabling NNAPI fallback overhead.
  // We rely on the default CPU provider which uses XNNPACK for ARM.

  // Log available providers to confirm setup
  std::vector<std::string> available_providers = Ort::GetAvailableProviders();
  LOGD("Available ONNX Execution Providers:");
  for (const auto &provider : available_providers) {
    LOGD(" - %s", provider.c_str());
  }

  session = Ort::Session(env, modelPath.c_str(), sessionOptions);

  // 1b. Resolving Model Inputs/Outputs dynamically
  Ort::AllocatorWithDefaultOptions allocator;

  size_t num_input_nodes = session.GetInputCount();
  input_node_names.resize(num_input_nodes);

  for (size_t i = 0; i < num_input_nodes; i++) {
    auto input_name = session.GetInputNameAllocated(i, allocator);
    input_node_names_allocated.push_back(
        input_name.get()); // Deep copy to std::string
    input_node_names[i] = input_node_names_allocated.back().c_str();
  }

  size_t num_output_nodes = session.GetOutputCount();
  output_node_names.resize(num_output_nodes);

  for (size_t i = 0; i < num_output_nodes; i++) {
    auto output_name = session.GetOutputNameAllocated(i, allocator);
    output_node_names_allocated.push_back(
        output_name.get()); // Deep copy to std::string
    output_node_names[i] = output_node_names_allocated.back().c_str();
  }

  // 2. Initialize FFT
  fft_cfg = kiss_fftr_alloc(N_FFT, 0, nullptr, nullptr);
  ifft_cfg = kiss_fftr_alloc(N_FFT, 1, nullptr, nullptr);

  // 3. Create Hanning Window
  initWindow();

  // 4. Optimize CPU Utilization for OpenMP
  // Set to 4 threads to match typical Big Core count and avoid context
  // switching overhead
  omp_set_num_threads(4);
  LOGD("OpenMP configured for 4 threads (Optimized)");
}

AudioEngine::~AudioEngine() {
  free(fft_cfg);
  free(ifft_cfg);
}

void AudioEngine::initWindow() {
  window.resize(WIN_LENGTH);
  window_sum = 0.0f;
  // Use PERIODIC Hanning window to match scipy.signal.get_window('hann', N)
  // Periodic: 0.5 * (1 - cos(2*pi*i / N)) for i=0..N-1
  // Symmetric: 0.5 * (1 - cos(2*pi*i / (N-1))) for i=0..N-1
  for (int i = 0; i < WIN_LENGTH; ++i) {
    window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / WIN_LENGTH));  // Changed from (WIN_LENGTH - 1)
    window_sum += window[i];
  }
  LOGD("Window sum: %.6f (using periodic formula to match scipy)", window_sum);
}

// =================================================================================
// TFLC BEAMFORMING (Iterative)
// =================================================================================
Eigen::MatrixXcd
AudioEngine::tflc_beamforming(const std::vector<MatrixXcd> &Y_in,
                              const MatrixXf &mask) {
  int F = Y_in[0].rows();
  int T = Y_in[0].cols();
  int M = 2; // Two microphones

  VectorXcd a_vec = VectorXcd::Ones(M);

  // W_k: Filter weights per frequency, per source (2 sources: Speech, Noise)
  std::vector<std::vector<VectorXcd>> W_k(
      F, std::vector<VectorXcd>(2, VectorXcd::Zero(M)));

// --- 1. Initialization ---
// Note: Reverting OpenMP usage to match verified "Good" state logic carefully
#pragma omp parallel for
  for (int f = 0; f < F; ++f) {
    MatrixXcd Phi_init = MatrixXcd::Zero(M, M);

    for (int t = 0; t < T; ++t) {
      VectorXcd x(M);
      x(0) = Y_in[0](f, t);
      x(1) = Y_in[1](f, t);

      float noise_prob = 1.0f - mask(f, t);
      if (noise_prob < 0)
        noise_prob = 0;

      Phi_init += (x * x.adjoint()) * noise_prob;
    }

    for (int k = 0; k < 2; ++k) {
      MatrixXcd Phi_reg = Phi_init + MatrixXcd::Identity(M, M) * 1e-2f;
      VectorXcd num = Phi_reg.ldlt().solve(a_vec);
      std::complex<double> den = a_vec.adjoint() * num;
      W_k[f][k] = num / (std::real(den) + 1e-10);
    }
    
    // DEBUG: Dump Phi_init at f=100 for comparison
    if (f == 100) {
      // First dump mask values at f=100
      float mask_sum = 0;
      LOGD("[DEBUG_TFLC] f=100: Mask first 10 values:");
      for (int t = 0; t < std::min(10, T); ++t) {
        LOGD("  mask[100,%d] = %.8f", t, mask(100, t));
        mask_sum += mask(100, t);
      }
      LOGD("[DEBUG_TFLC] f=100: Mask mean (first 10) = %.8f", mask_sum / std::min(10, T));
      
      // Dump full mask at f=100 to file for comparison
      std::ofstream mf("debug_cpp_mask_f100.txt");
      for(int t = 0; t < T; ++t) mf << mask(100, t) << "\n";
      mf.close();
      
      LOGD("[DEBUG_TFLC] f=%d: Phi_init[0,0]=(%.8f, %.8f), Phi_init[0,1]=(%.8f, %.8f)",
           f, Phi_init(0,0).real(), Phi_init(0,0).imag(),
           Phi_init(0,1).real(), Phi_init(0,1).imag());
      LOGD("[DEBUG_TFLC] f=%d: Phi_init[1,0]=(%.8f, %.8f), Phi_init[1,1]=(%.8f, %.8f)",
           f, Phi_init(1,0).real(), Phi_init(1,0).imag(),
           Phi_init(1,1).real(), Phi_init(1,1).imag());
      LOGD("[DEBUG_TFLC] f=%d: W_k[f][0]=(%.8f, %.8f), (%.8f, %.8f)",
           f, W_k[f][0](0).real(), W_k[f][0](0).imag(),
           W_k[f][0](1).real(), W_k[f][0](1).imag());
    }
  }

  // --- 2. Iterative Update ---
  int iterations = 20;

  std::vector<MatrixXcd> Y_sep(2, MatrixXcd(F, T));
  std::vector<MatrixXf> c_k(2, MatrixXf(F, T));

  for (int iter = 0; iter < iterations; ++iter) {

// A. Apply Beamformers
#pragma omp parallel for
    for (int f = 0; f < F; ++f) {
      for (int t = 0; t < T; ++t) {
        VectorXcd x(M);
        x(0) = Y_in[0](f, t);
        x(1) = Y_in[1](f, t);
        for (int k = 0; k < 2; ++k) {
          Y_sep[k](f, t) = W_k[f][k].dot(x);
        }
      }
    }

// B. Update weights c_k
#pragma omp parallel for
    for (int f = 0; f < F; ++f) {
      for (int t = 0; t < T; ++t) {
        std::complex<float> y1 = (std::complex<float>)Y_sep[0](f, t);
        std::complex<float> y2 = (std::complex<float>)Y_sep[1](f, t);
        std::complex<float> y21 = y1 - y2;

        float num = -std::real(y2 * std::conj(y21));
        float den = std::norm(y21) + 1e-10f;
        float val = std::clamp(num / den, 0.0f, 1.0f);

        c_k[0](f, t) = val;
        c_k[1](f, t) = 1.0f - val;
      }
    }

// C. Update Filters
#pragma omp parallel for
    for (int f = 0; f < F; ++f) {
      for (int k = 0; k < 2; ++k) {
        MatrixXcd Phi_k = MatrixXcd::Zero(M, M);
        for (int t = 0; t < T; ++t) {
          VectorXcd x(M);
          x(0) = Y_in[0](f, t);
          x(1) = Y_in[1](f, t);
          Phi_k += (x * x.adjoint()) * c_k[k](f, t);
        }

        Phi_k += MatrixXcd::Identity(M, M) * 1e-2f;
        VectorXcd num = Phi_k.ldlt().solve(a_vec);
        std::complex<double> den = a_vec.adjoint() * num;
        W_k[f][k] = num / (std::real(den) + 1e-10);
      }
    }
    }

  
  #ifdef AUDIO_DEBUG_LINUX
  dumpMatrixXcd("tflc_out_final_loop", Y_sep[0]); // Dump last sep
  #endif

  // --- 3. Final Reconstruction (RESTORED logic) ---
  MatrixXcd Y_final(F, T);
#pragma omp parallel for
  for (int f = 0; f < F; ++f) {
    for (int t = 0; t < T; ++t) {
      VectorXcd x(M);
      x(0) = Y_in[0](f, t);
      x(1) = Y_in[1](f, t);

      // Reconstruct the Mixture based on the model
      std::complex<double> sum_val = 0;
      for (int k = 0; k < 2; ++k) {
        std::complex<double> y_k = W_k[f][k].dot(x);
        sum_val += (double)c_k[k](f, t) * y_k;
      }

      // Apply the original U-Net Mask to the reconstructed signal
      // (Post-Filtering) This was the key quality step missed in the previous
      // iteration
      Y_final(f, t) = sum_val * (double)std::max((float)mask(f, t), 0.05f);
    }
  }
  
  // DEBUG: Dump Y_final stats for comparison with Python
  #ifdef AUDIO_DEBUG_LINUX
  double sum_mag = 0, max_mag = 0;
  for(int f=0; f<F; ++f) for(int t=0; t<T; ++t) {
    double m = std::abs(Y_final(f,t));
    sum_mag += m;
    max_mag = std::max(max_mag, m);
  }
  LOGD("[DEBUG_TFLC] Y_final: Mean Mag: %.8f, Max Mag: %.8f", sum_mag/(F*T), max_mag);
  LOGD("[DEBUG_TFLC] Y_final[100, 0:5] = (%.8f, %.8f), (%.8f, %.8f), (%.8f, %.8f), (%.8f, %.8f), (%.8f, %.8f)",
       Y_final(100,0).real(), Y_final(100,0).imag(),
       Y_final(100,1).real(), Y_final(100,1).imag(),
       Y_final(100,2).real(), Y_final(100,2).imag(),
       Y_final(100,3).real(), Y_final(100,3).imag(),
       Y_final(100,4).real(), Y_final(100,4).imag());
  #endif
  
  return Y_final;
}

// =================================================================================
// MAIN PIPELINE
// =================================================================================
// =================================================================================
// MAIN PIPELINE
// =================================================================================
// =================================================================================
// MAIN PIPELINE (Fixed Buffer Sizing)
// =================================================================================
std::vector<float> AudioEngine::processAudio(const std::vector<float> &inputAudio) {
    auto t_start = std::chrono::high_resolution_clock::now();

    // 1. INPUT NORMALIZATION
    float max_in = 0.0f;
    for(float v : inputAudio) max_in = std::max(max_in, std::abs(v));
    std::vector<float> normAudio = inputAudio;
    if (max_in > 1.0f) {
        for(float &v : normAudio) v /= max_in;
    }



    // 2. SCIPY PADDING (Start + End = N_FFT/2)
    // Matches scipy.signal.stft(padded=True)
    int padding = N_FFT / 2;
    std::vector<float> audio_L, audio_R;

    // Reserve space
    size_t est_size = normAudio.size() / 2 + 2 * padding;
    audio_L.reserve(est_size);
    audio_R.reserve(est_size);

    // START PADDING
    for(int i=0; i<padding; ++i) {
        audio_L.push_back(0.0f);
        audio_R.push_back(0.0f);
    }

    // DATA
    for (size_t i = 0; i < normAudio.size(); i += 2) {
        audio_L.push_back(normAudio[i]);
        if (i + 1 < normAudio.size()) audio_R.push_back(normAudio[i + 1]);
        else audio_R.push_back(normAudio[i]);
    }
    
    // END PADDING
    for(int i=0; i<padding; ++i) {
        audio_L.push_back(0.0f);
        audio_R.push_back(0.0f);
    }
    
    // 2.5 SCIPY FIT PADDING (Ensure coverage)
    // Calculate required frames using ceil to match Scipy's 'padded=True' which covers the whole signal
    size_t current_size = audio_L.size();
    int numFramesScipy = (int)std::ceil((float)(current_size - WIN_LENGTH) / HOP_LENGTH) + 1;
    size_t required_len = (numFramesScipy - 1) * HOP_LENGTH + WIN_LENGTH;
    
    if (required_len > current_size) {
        size_t pad_extra = required_len - current_size;
        for(size_t i=0; i<pad_extra; ++i) {
            audio_L.push_back(0.0f);
            audio_R.push_back(0.0f);
        }
    }

    // Interleave
    std::vector<float> stereo_interleaved(audio_L.size() * 2);
    for (size_t i = 0; i < audio_L.size(); ++i) {
        stereo_interleaved[2 * i] = audio_L[i];
        stereo_interleaved[2 * i + 1] = audio_R[i];
    }

    // 4. Processing
    auto t_stft_start = std::chrono::high_resolution_clock::now();
    auto Y_stft = stft_multichannel(stereo_interleaved, 2);
    auto t_stft_end = std::chrono::high_resolution_clock::now();

    // Features
    int numFreqs = Y_stft[0].rows();
    int numFrames = Y_stft[0].cols();
    
    // Calculate Feature Padding (Divisible by 16)
    int stride = 16;
    int pad_len = (stride - (numFrames % stride)) % stride;
    int numFramesPadded = numFrames + pad_len;

    std::vector<float> features = computeFeatures(Y_stft, pad_len);

    std::vector<int64_t> inputShape = {1, 5, numFreqs, numFramesPadded};
    
    // Debug
    LOGD("Frames: %d, Pad: %d, Total: %d", numFrames, pad_len, numFramesPadded);

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, features.data(), features.size(), inputShape.data(), inputShape.size());

    // Inference
    auto t_infer_start = std::chrono::high_resolution_clock::now();
    auto outputTensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(),
                                     &inputTensor, 1, output_node_names.data(), 1);
    float *maskData = outputTensors[0].GetTensorMutableData<float>();
    auto t_infer_end = std::chrono::high_resolution_clock::now();

    MatrixXf mask(numFreqs, numFrames);
    for (int f = 0; f < numFreqs; ++f) {
        for (int t = 0; t < numFrames; ++t) {
            // Read from Padded Output (Row Major: F x T_padded)
            mask(f, t) = maskData[f * numFramesPadded + t];
        }
    }
    
    #ifdef AUDIO_DEBUG_LINUX
    logStatsMatrix("ege_mask", mask);
    #endif

    // TFLC
    auto t_tflc_start = std::chrono::high_resolution_clock::now();
    MatrixXcd cleanSpec = tflc_beamforming(Y_stft, mask);
    auto t_tflc_end = std::chrono::high_resolution_clock::now();

    // ISTFT
    std::vector<float> enhancedAudio = istft(cleanSpec);

    // 5. Cleanup Padding (Crucial Step for Continuity)
    // First, remove the start padding we added
    if (enhancedAudio.size() > padding) {
        enhancedAudio.erase(enhancedAudio.begin(), enhancedAudio.begin() + padding);
    }

    // Now, resize strictly to the original input length to avoid silence gaps/clicks
    // Input was stereo, so Mono length is size/2
    size_t original_mono_len = inputAudio.size() / 2;

    if (enhancedAudio.size() > original_mono_len) {
        enhancedAudio.resize(original_mono_len);
    } else if (enhancedAudio.size() < original_mono_len) {
        // This handles the edge case where output is slightly shorter due to ISTFT loss
        enhancedAudio.resize(original_mono_len, 0.0f);
    }

    // Final Norm - REMOVED to match Python behavior (no peak normalization)
    // float max_out = 0.0f;
    // for (float v : enhancedAudio) max_out = std::max(max_out, std::abs(v));
    // float norm_factor = max_out + 1e-9f;
    // for (float &v : enhancedAudio) v /= norm_factor;

    auto t_end = std::chrono::high_resolution_clock::now();

    // --- TIMING LOGS ---
    double ms_stft = std::chrono::duration<double, std::milli>(t_stft_end - t_stft_start).count();
    double ms_infer = std::chrono::duration<double, std::milli>(t_infer_end - t_infer_start).count();
    double ms_tflc = std::chrono::duration<double, std::milli>(t_tflc_end - t_tflc_start).count();
    double ms_total = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    LOGD("=== TIMING REPORT ===");
    LOGD("Input Size     : %zu samples", inputAudio.size());
    LOGD("Output Size    : %zu samples", enhancedAudio.size());
    LOGD("STFT + Feats   : %.2f ms", ms_stft);
    LOGD("Model Inference: %.2f ms", ms_infer);
    LOGD("TFLC Beamform  : %.2f ms", ms_tflc);
    LOGD("Total Process  : %.2f ms", ms_total);
    LOGD("=====================");

    return enhancedAudio;
}
// =================================================================================
// FEATURE EXTRACTION (Fixed Debug Prints)
// =================================================================================
std::vector<float> AudioEngine::computeFeatures(const std::vector<MatrixXcd> &stft_channels, int pad_len) {
    const MatrixXcd &Y0 = stft_channels[0];
    const MatrixXcd &Y1 = stft_channels[1];
    int F = Y0.rows();
    int T = Y0.cols();

    std::vector<float> features;
    features.reserve(5 * F * (T + pad_len));

    // --- 1. Log Magnitude ---
    MatrixXf mag = Y0.cwiseAbs().cast<float>();
    MatrixXf log_mag(F, T);

    float sum = 0.0f;
    float sum_sq = 0.0f;
    int count = F * T;

    for (int idx = 0; idx < count; ++idx) {
        float val = std::log(mag(idx) + 1e-7f);
        log_mag(idx) = val;
        sum += val;
        sum_sq += val * val;
    }

    float mean = sum / count;
    float variance = (sum_sq / count) - (mean * mean);
    float std_val = std::sqrt(std::max(variance, 1e-9f)) + 1e-7f;

    MatrixXf feat_logmag = (log_mag.array() - mean) / std_val;

    // --- 2. IPD ---
    MatrixXf feat_sin(F, T);
    MatrixXf feat_cos(F, T);

    for (int t = 0; t < T; ++t) {
        for (int f = 0; f < F; ++f) {
            float angle0 = std::arg(Y0(f, t));
            float angle1 = std::arg(Y1(f, t));
            float diff = angle0 - angle1;
            feat_sin(f, t) = std::sin(diff);
            feat_cos(f, t) = std::cos(diff);
        }
    }

    // --- 3. FMap ---
    MatrixXf feat_fmap(F, T);
    for (int f = 0; f < F; ++f) {
        float val = (float)f / (F - 1);
        for (int t = 0; t < T; ++t) {
            feat_fmap(f, t) = val;
        }
    }

    // --- 4. MSC ---
    MatrixXf feat_msc(F, T);
    for (int t = 0; t < T; ++t) {
        for (int f = 0; f < F; ++f) {
            std::complex<float> y0 = (std::complex<float>)Y0(f, t);
            std::complex<float> y1 = (std::complex<float>)Y1(f, t);
            std::complex<float> cross = y0 * std::conj(y1);
            float cross_mag = std::abs(cross);
            float denom = std::sqrt(std::norm(y0) * std::norm(y1)) + 1e-9f;
            feat_msc(f, t) = cross_mag / denom;
        }
    }

    // Flattening (Channels, Freq, Time)
    auto push_channel = [&](const MatrixXf &M) {
        for (int f = 0; f < F; ++f) {
            for (int t = 0; t < T; ++t) {
                features.push_back(M(f, t));
            }
            // Logic: Pad Time Dimension
            for(int p=0; p < pad_len; ++p) {
                features.push_back(0.0f);
            }
        }
    };

    push_channel(feat_logmag);
    push_channel(feat_sin);
    push_channel(feat_cos);
    push_channel(feat_fmap);
    push_channel(feat_msc);

    #ifdef AUDIO_DEBUG_LINUX
    std::ofstream fs("debug_cpp_features_flat.txt");
    for(float f : features) fs << f << "\n";
    fs.close();
    #endif



    return features;
}

// =================================================================================
// STFT / ISTFT Boilerplate
// =================================================================================

std::vector<MatrixXcd>
AudioEngine::stft_multichannel(const std::vector<float> &audio,
                               int num_channels) {
  // Audio is interleaved
  // Returns 2 Matrices
  int numFrames = (audio.size() / num_channels - WIN_LENGTH) / HOP_LENGTH + 1;

  std::vector<MatrixXcd> specs;
  for (int c = 0; c < num_channels; ++c) {
    specs.push_back(MatrixXcd(N_FFT / 2 + 1, numFrames));
  }

  // DEBUG: Print specific frame Y values
  int debug_f = 433;
  int debug_t = 12;

  std::vector<double> timeBuffer(N_FFT);
  std::vector<kiss_fft_cpx> freqBuffer(N_FFT / 2 + 1);

  for (int t = 0; t < numFrames; ++t) {
    int startSample = t * HOP_LENGTH * num_channels;

    for (int c = 0; c < num_channels; ++c) {
      // Extract window
      for (int i = 0; i < WIN_LENGTH; ++i) {
        int idx = startSample + i * num_channels + c;
        if (idx < audio.size()) {
          timeBuffer[i] = (double)audio[idx] * (double)window[i];
        } else {
          timeBuffer[i] = 0.0;
        }
      }
      
      // DEBUG: Print input samples for frame 12
      if (t == debug_t && c == 1) {
        LOGD("[DEBUG_INPUT_FFT] t=%d, c=%d: First 10 samples (after window):", t, c);
        LOGD("  %.8f %.8f %.8f %.8f %.8f", timeBuffer[0], timeBuffer[1], timeBuffer[2], timeBuffer[3], timeBuffer[4]);
        LOGD("  %.8f %.8f %.8f %.8f %.8f", timeBuffer[5], timeBuffer[6], timeBuffer[7], timeBuffer[8], timeBuffer[9]);
        double tb_sum = 0;
        for(int i=0; i<WIN_LENGTH; ++i) tb_sum += timeBuffer[i];
        LOGD("[DEBUG_INPUT_FFT] Sum of windowed input: %.10f", tb_sum);
        
        // Dump full windowed input to file for comparison
        std::ofstream dump("debug_cpp_windowed_input.bin", std::ios::binary);
        dump.write(reinterpret_cast<char*>(timeBuffer.data()), WIN_LENGTH * sizeof(double));
        dump.close();
        LOGD("[DEBUG_INPUT_FFT] Dumped windowed input to debug_cpp_windowed_input.bin");
      }

      // FFT
      kiss_fftr(fft_cfg, timeBuffer.data(), freqBuffer.data());
      
      // DEBUG: Print raw FFT output before scaling for frame 12 ch1
      if (t == debug_t && c == 1) {
        LOGD("[DEBUG_RAW_FFT] t=%d, c=%d: Raw FFT[%d] = (%.10f, %.10f)",
             t, c, debug_f, freqBuffer[debug_f].r, freqBuffer[debug_f].i);
      }

      // Store with scipy 'spectrum' scaling (divide by window sum)
      double scale = 1.0 / window_sum;
      for (int f = 0; f < N_FFT / 2 + 1; ++f) {
        specs[c](f, t) = std::complex<double>(freqBuffer[f].r * scale, freqBuffer[f].i * scale);
      }
      
      // DEBUG: Print Y values at specific frame
      if (t == debug_t) {
        LOGD("[DEBUG_STFT] t=%d, c=%d: Y[%d]=(%.8f, %.8f), Mag=%.8f, Angle=%.8f",
             t, c, debug_f, specs[c](debug_f, t).real(), specs[c](debug_f, t).imag(),
             std::abs(specs[c](debug_f, t)), std::arg(specs[c](debug_f, t)));
      }
    }
  }
  return specs;
}

std::vector<float> AudioEngine::istft(const MatrixXcd &spec) {
  int numFreqs = spec.rows();
  int numFrames = spec.cols();
  int outputLen = (numFrames - 1) * HOP_LENGTH + WIN_LENGTH;

  std::vector<float> output(outputLen, 0.0f);
  std::vector<kiss_fft_cpx> freqBuffer(numFreqs);
  std::vector<double> timeBuffer(N_FFT);

  for (int t = 0; t < numFrames; ++t) {
    for (int f = 0; f < numFreqs; ++f) {
      freqBuffer[f].r = spec(f, t).real();
      freqBuffer[f].i = spec(f, t).imag();
    }

    kiss_fftri(ifft_cfg, freqBuffer.data(), timeBuffer.data());

    int startSample = t * HOP_LENGTH;
    for (int i = 0; i < WIN_LENGTH; ++i) {
      // Multiply by window_sum to compensate for STFT spectrum scaling (/ window_sum)
      output[startSample + i] += ((float)timeBuffer[i] / N_FFT) * window_sum;
    }
  }
  return output;
}
