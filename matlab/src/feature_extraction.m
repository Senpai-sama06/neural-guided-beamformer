clear; clc;

% =========================================================================
% FEATURE EXTRACTION (PRODUCTION)
% =========================================================================
% This script incorporates the debugged STFT and feature extraction logic
% to ensure parity with the Python implementation.

% Input/Output Paths
mixture = '../../python/data/simulated/batch_test_000/mixture.wav'; 
output_file = '../input/model_input.mat';

if exist(mixture, 'file') ~= 2
    error('Input file not found: %s', mixture); 
end

[audio_in, fs_raw] = audioread(mixture);

% 1. Resampling
FS = 16000;
if fs_raw ~= FS
    audio_in = resample(audio_in, FS, fs_raw);
end

% 2. Ensure Stereo
if size(audio_in, 2) == 1
    audio_in = [audio_in, audio_in];
end

% --- STFT (Aligned to Scipy) ---
N_FFT = 1024;
HOP_LEN = 512;
OVERLAP = N_FFT - HOP_LEN;
WINDOW = hann(N_FFT, 'periodic');

% 3. Padding (Scipy parity)
pad_width = N_FFT / 2;
tail_pad = 512; 
audio_padded = [zeros(pad_width, size(audio_in, 2)); audio_in; zeros(pad_width + tail_pad, size(audio_in, 2))];

% 4. Spectrogram
[S1, f, t] = spectrogram(audio_padded(:,1), WINDOW, OVERLAP, N_FFT, FS);
[S2, ~, ~] = spectrogram(audio_padded(:,2), WINDOW, OVERLAP, N_FFT, FS);

% 5. Scaling (Scipy parity)
win_sum = sum(WINDOW);
S1 = S1 / win_sum;
S2 = S2 / win_sum;

% --- Feature Extraction ---
mag_S1 = abs(S1);
log_mag = log(mag_S1 + 1e-7);

% Global Normalization (Channel 1)
mu = mean(log_mag(:));
sigma = std(log_mag(:), 1); % ddof=0
feat_1 = (log_mag - mu) / (sigma + 1e-7);

% IPD Features (Channels 2 & 3)
phase1 = angle(S1);
phase2 = angle(S2);
ipd_raw = phase1 - phase2;
feat_2 = sin(ipd_raw);
feat_3 = cos(ipd_raw);

% Frequency Map (Channel 4)
[n_freq, n_time] = size(S1);
feat_4 = repmat(linspace(0, 1, n_freq)', 1, n_time);

% MSC (Channel 5)
cross_spec = S1 .* conj(S2);
auto_spec1 = S1 .* conj(S1);
auto_spec2 = S2 .* conj(S2);
feat_5 = abs(cross_spec) ./ (sqrt(abs(auto_spec1) .* abs(auto_spec2)) + 1e-9);

% Stack Features
features_3d = cat(3, feat_1, feat_2, feat_3, feat_4, feat_5);

% Permute for Model: (Freq, Time, Channels) -> (Time, Channels, Freq)
neural_input = permute(features_3d, [2, 3, 1]);

% Save
save(output_file, 'neural_input', '-v7.3');
fprintf('Saved features to %s\n', output_file);
fprintf('Input Shape: %d Time x %d Channels x %d Freq\n', size(neural_input,1), size(neural_input,2), size(neural_input,3));
