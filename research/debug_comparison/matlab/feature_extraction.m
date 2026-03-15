clear; clc;

% Input Path (Hardcoded to debug_run)
mixture = "../data/debug_run/mixture.wav";
output_file = "../data/debug_matlab_features.mat";

if ~isfile(mixture), error("Input file not found: %s", mixture); end

[audio_in, fs_raw] = audioread(mixture);
% Make sure we match python (16k)
FS = 16000;
if fs_raw ~= FS
    audio_in = resample(audio_in, FS, fs_raw);
end
% Ensure Stereo
if size(audio_in, 2) == 1, audio_in = [audio_in, audio_in]; end

% --- 1. STFT (Aligned to Scipy) ---
N_FFT = 1024;
HOP_LEN = 512;
OVERLAP = N_FFT - HOP_LEN;
WINDOW = hann(N_FFT, 'periodic');

% 1. Padding (Scipy boundary='zeros' pads N_FFT/2)
pad_width = N_FFT / 2;
% Scipy also pads the end if the signal doesn't fit integers frames. 
% We add extra buffer (N_FFT) to force MATLAB to keep that last frame.
tail_pad = 512; 
audio_padded = [zeros(pad_width, size(audio_in, 2)); audio_in; zeros(pad_width + tail_pad, size(audio_in, 2))];

% 2. Spectrogram
[S1, f, t] = spectrogram(audio_padded(:,1), WINDOW, OVERLAP, N_FFT, FS);
[S2, ~, ~] = spectrogram(audio_padded(:,2), WINDOW, OVERLAP, N_FFT, FS);

% 3. Scaling (Scipy scaling='spectrum' divides by sum(window))
win_sum = sum(WINDOW);
S1 = S1 / win_sum;
S2 = S2 / win_sum;


% --- 2. Feature Extraction ---
mag_S1 = abs(S1);
log_mag = log(mag_S1 + 1e-7);
mu = mean(log_mag(:));
% Match Numpy std (ddof=0 -> normalization by N)
sigma = std(log_mag(:), 1);
feat_1 = (log_mag - mu) / (sigma + 1e-7);

phase1 = angle(S1);
phase2 = angle(S2);
ipd_raw = phase1 - phase2;
feat_2 = sin(ipd_raw);
feat_3 = cos(ipd_raw);

[n_freq, n_time] = size(S1);
feat_4 = repmat(linspace(0, 1, n_freq)', 1, n_time);

cross_spec = S1 .* conj(S2);
auto_spec1 = S1 .* conj(S1);
auto_spec2 = S2 .* conj(S2);
feat_5 = abs(cross_spec) ./ (sqrt(abs(auto_spec1) .* abs(auto_spec2)) + 1e-9);

features_3d = cat(3, feat_1, feat_2, feat_3, feat_4, feat_5);
neural_input = permute(features_3d, [2, 3, 1]);

% --- SAVE DEBUG DATA ---
% Saving as 'model_input.mat' structure for main.m, but also raw vars for python compare
save(output_file, 'neural_input', 'audio_in', 'S1', 'S2', 'feat_1', 'feat_5', '-v7.3');
fprintf("Saved features to %s\n", output_file);
