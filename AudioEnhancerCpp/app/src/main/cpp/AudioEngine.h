#ifndef AUDIO_ENGINE_H
#define AUDIO_ENGINE_H

#include "kiss_fftr.h"
#include "onnxruntime_cxx_api.h"
#include <Eigen/Dense>
#include <complex>
#include <string>
#include <vector>

// Constants - MUST match model training parameters

#define SAMPLE_RATE 16000
#define N_FFT 1024
#define HOP_LENGTH 512  // <-- MATCHES PYTHON config.py
#define WIN_LENGTH 1024

class AudioEngine {
public:
  AudioEngine(const std::string &modelPath);
  ~AudioEngine();

  // Main processing function
  std::vector<float> processAudio(const std::vector<float> &inputAudio);

private:
  // ONNX Runtime variables
  Ort::Env env;
  Ort::Session session;

  // --- NEW MEMBERS (Required for the new .cpp code) ---
  std::vector<std::string> input_node_names_allocated;
  std::vector<const char *> input_node_names;
  std::vector<std::string> output_node_names_allocated;
  std::vector<const char *> output_node_names;

  // FFT variables
  kiss_fftr_cfg fft_cfg;
  kiss_fftr_cfg ifft_cfg;

  std::vector<float> window;

  // --- THE MISSING VARIABLE CAUSING YOUR ERROR ---
  float window_sum;

  // Helper functions
  void initWindow();

  std::vector<float>
  computeFeatures(const std::vector<Eigen::MatrixXcd> &stft_channels, int pad_len);
  std::vector<Eigen::MatrixXcd>
  stft_multichannel(const std::vector<float> &audio, int num_channels);
  std::vector<float> istft(const Eigen::MatrixXcd &spec);
  Eigen::MatrixXcd tflc_beamforming(const std::vector<Eigen::MatrixXcd> &Y_in,
                                    const Eigen::MatrixXf &mask);
};

#endif // AUDIO_ENGINE_H