% =========================================================================
% IEEE SIGNAL PROCESSING CUP 2026 - NEURAL MODEL SCRIPT
% =========================================================================
% This script performs STFT (Short Time Fourier Transform) and then
% calculates the five features required for the neural model - log
% spectrum, frequency map, MSC, sin(IPD) and cos(IPD).
%
% Generates:
%   - .mat files containing target, interference, mixture, and RIRs.
%   - .wav files for listening.
%   - 2D Room Geometry Plot.
% =========================================================================
clear; clc;

%% 1. CONFIGURATION & SETUP
mixture = "/home/communications-lab/ramakrishna/avz_testing/real-time-audio-visual-zooming/modal_solutions/matlab_base/data/Task1_Anechoic_Mixture.wav";
neural_input = "/home/communications-lab/ramakrishna/avz_testing/real-time-audio-visual-zooming/modal_solutions/IRM_TFLC/matlab/model_input.mat";
model_onnx = "/home/communications-lab/ramakrishna/avz_testing/real-time-audio-visual-zooming/modal_solutions/matlab_base/models/ege_unet_1024_fp32.onnx";

if isfile(neural_input)
    % Load the data. We assume the variable inside is named 'model_input'
    load(neural_input, 'model_input');
else
    error("model_input.mat not found!")
end

if ~isfile(model_onnx)
    error("model_onnx is not found!")
end

% Loading Neural model
% Note: If this line fails, ensure you have the Deep Learning Toolbox Converter for ONNX
lgraph = importNetworkFromONNX(model_onnx);
net = dlnetwork(lgraph);

%% 2. RUNNING THE NEURAL MODEL
disp("Running Neural Model...");

% --- FIX 1: PERMUTATION CHECK ---
% Ensure data is (Freq, Time, Channels, Batch)
% If features are (Chan, Freq, Time) [Python Style], we must flip it.
if size(model_input, 1) == 5
    disp('Detected Python format (Chan, Freq, Time). Permuting to MATLAB format...');
    model_input = permute(model_input, [2, 3, 1]); 
end

dl_in = dlarray(model_input, 'SSCB');

% --- Handle Padding (Divisible by 16 Rule) ---
% [h, w, c, b] = size(dl_in);
% pad_amount = 0;
% if mod(w, 16) ~= 0
%     pad_amount = 16 - mod(w, 16);
%     padding = zeros(h, pad_amount, c, b, 'single');
%     dl_in = cat(2, dl_in, dlarray(padding, 'SSCB'));
% end

dl_mask_pred = predict(net, dl_in);
mask_pred = extractdata(dl_mask_pred);
mask_pred = mask_pred(:, 1:w, :, :);
mask = double(squeeze(mask_pred)); 
mask = max(0, min(1, mask));
disp('Mask Generated.');

%% 3. Re-STFT
if isfile(mixture)
    [audio_raw, FS_raw] = audioread(mixture);
else
    error("Audio file not found!");
end

% STFT Constants
FS = 16000;             
N_FFT = 1024;
HOP_LEN = 512;
OVERLAP = N_FFT - HOP_LEN;
WINDOW = hann(N_FFT, 'periodic');

if FS_raw ~= FS
    disp("Resampling audio to 16KHz...");
    audio = resample(audio_raw, FS, FS_raw);
else
    audio = audio_raw;
end

% Recheck: To convert code to Double Mono-Stereo
if size(audio, 2) == 1
    audio = [audio, audio]; 
end

% --- Channel 1 (Mic 1) ---
[S1, ~, ~] = spectrogram(audio(:,1), WINDOW, OVERLAP, N_FFT, FS);
% --- Channel 2 (Mic 2) ---
[S2, ~, ~] = spectrogram(audio(:,2), WINDOW, OVERLAP, N_FFT, FS);

%% 4. BEAMFORMER LOGIC
[n_bins, n_frames] = size(S1);

% --- FIX 2: SYNC SIZES ---
% Ensure Mask and STFT have the exact same number of frames
% (The mask might be slightly shorter due to padding cuts)
min_len = min(n_frames, size(mask, 2));
S1 = S1(:, 1:min_len);
S2 = S2(:, 1:min_len);
mask = mask(:, 1:min_len);
n_frames = min_len; % Update frame count

Y_mvdr = zeros(n_bins, n_frames);
d = ones(2, 1); 

for f = 1:n_bins
    % 1. Estimate Noise Covariance Matrix (Rnn) for this frequency
    noise_weight = 1 - mask(f, :);
    
    % Stack mics: 2 x Time
    X_f = [S1(f, :); S2(f, :)]; 
    
    % Weighted Covariance calculation
    X_weighted = X_f .* sqrt(noise_weight);
    Rnn = (X_weighted * X_weighted') / (sum(noise_weight) + 1e-6);
    
    % Diagonal Loading (Stability)
    Rnn = Rnn + 1e-3 * eye(2);
    
    % 2. MVDR Weights Formula
    Rnn_inv = inv(Rnn);
    numerator = Rnn_inv * d;
    denominator = d' * Rnn_inv * d;
    w_mvdr = numerator / (denominator + 1e-10);
    
    % 3. Apply Weights
    Y_mvdr(f, :) = w_mvdr' * X_f;
end

%% 5. INVERSE STFT (BACK TO AUDIO)
% --- FIX 3: VARIABLE NAMING ---
final_spec = Y_mvdr; 
enhanced_audio = istft(final_spec, FS, ...
                       'Window', WINDOW, ...          % Changed WIN -> WINDOW
                       'OverlapLength', OVERLAP, ...  % Changed N_FFT-HOP -> OVERLAP
                       'FFTLength', N_FFT);

% Normalize volume to avoid clipping
enhanced_audio = enhanced_audio / max(abs(enhanced_audio));

% Save
output_filename = 'processed_output.wav';
audiowrite(output_filename, enhanced_audio, FS);

%% 6. VISUALIZATION OF RESULTS
figure('Name', 'Results', 'Color', 'w');
subplot(3,1,1);
imagesc(log(abs(S1)+eps)); axis xy; colormap inferno; title('Original Noisy Input');
subplot(3,1,2);
imagesc(mask); axis xy; colormap parula; title('Predicted Mask (1=Target)');
subplot(3,1,3);
imagesc(log(abs(Y_mvdr)+eps)); axis xy; colormap inferno; title('Enhanced MVDR Output');
fprintf('Processing Complete. Listen to %s\n', output_filename);