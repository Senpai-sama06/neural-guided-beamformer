clear; clc;

% =========================================================================
% INFERENCE SCRIPT (PRODUCTION)
% =========================================================================

% Add utils to path
addpath('utils');

% Paths
mixture = '../../python/data/simulated/batch_test_000/mixture.wav';
neural_input_path = '../input/model_input.mat';
model_onnx = '../models/ege_unet_1024_fp32.onnx';

% Constants
FS = 16000; 
N_FFT = 1024; 
HOP_LEN = 512; 
OVERLAP = N_FFT - HOP_LEN; 
WINDOW = hann(N_FFT, 'periodic');

% 1. Load Audio
if exist(mixture, 'file') ~= 2
    error('Input file not found: %s', mixture); 
end

[audio, fs_raw] = audioread(mixture);
if fs_raw ~= FS
    disp('Resampling...');
    audio = resample(audio, FS, fs_raw); 
end
if size(audio,2)==1, audio=[audio,audio]; end

% 2. STFT (Debugged/Parity with Scipy)
pad_width = N_FFT / 2;
tail_pad = 512;
audio_padded = [zeros(pad_width, size(audio, 2)); audio; zeros(pad_width + tail_pad, size(audio, 2))];

[S1, ~, ~] = spectrogram(audio_padded(:,1), WINDOW, N_FFT-HOP_LEN, N_FFT, FS);
[S2, ~, ~] = spectrogram(audio_padded(:,2), WINDOW, N_FFT-HOP_LEN, N_FFT, FS);

win_sum = sum(WINDOW);
S1 = S1 / win_sum;
S2 = S2 / win_sum;

[n_f, n_t] = size(S1);

% 3. Prepare Model Input
if exist(neural_input_path, 'file') ~= 2
    error('Features file (model_input.mat) not found. Run feature_extraction.m first.');
end
load(neural_input_path, 'neural_input');

% Permute back to (Freq, Time, Channels)
new_model_input = permute(neural_input, [3, 1, 2]);
final_model_input = new_model_input;
[n_f_in, n_t_in, n_ch_in] = size(final_model_input);

% Pad time dimension to multiple of 16 for UNet
pad_t = 0;
if mod(n_t_in, 16) ~= 0
    pad_t = 16 - mod(n_t_in, 16);
    padding = zeros(n_f_in, pad_t, n_ch_in);
    final_model_input = cat(2, final_model_input, padding);
end

n_t_padded = size(final_model_input, 2);

% 4. Load Model
if exist(model_onnx, 'file') ~= 2
    error('Model ONNX file not found: %s', model_onnx);
end

% Note: ImageInputSize set dynamically to match padded input length
lgraph = importONNXLayers(model_onnx, 'OutputLayerType', 'regression', 'ImportWeights', true, 'ImageInputSize', [513, n_t_padded, 5]);
try
    lgraph = removeLayers(lgraph, 'RegressionLayer_mask'); 
catch
    % Layer might not exist, proceed
end
net = dlnetwork(lgraph);

% 5. Inference
disp('Running Inference...');
input_4d = reshape(final_model_input, 513, n_t_padded, 5, 1);
dl_in = dlarray(input_4d, 'SSCB');
output_mask = predict(net, dl_in);
mask_pred = extractdata(output_mask);

% 6. Post-Process Mask
mask_raw = double(squeeze(mask_pred));

% Remove padding
if pad_t > 0
    mask_raw = mask_raw(:, 1:end-pad_t);
end

% Resize to original STFT dimensions
mask = imresize(mask_raw, [n_f, n_t], 'bilinear');
mask = max(0, min(1, mask));

% 7. Beamforming
disp('Running TFLC Beamformer...');
Y = zeros(2, n_f, n_t);
Y(1,:,:) = S1; 
Y(2,:,:) = S2;
[S_out, ~] = tflc_beamformer(Y, mask, 2, 20);

% Apply Raw Mask to Beamformed Output
S_raw = S_out .* max(mask, 0.05);

% 8. ISTFT / Reconstruction
% ISTFT Parity
spec = S_raw;
if size(spec, 1) == 513
    spec_pos = spec(2:end-1, :);
    spec_neg = conj(flipud(spec_pos));
    spec_2sided = [spec(1,:); spec_pos; spec(end,:); spec_neg];
else
    spec_2sided = spec;
end

wav_out = istft(spec_2sided, FS, 'Window', WINDOW, 'OverlapLength', N_FFT-HOP_LEN, 'FFTLength', N_FFT, 'Centered', false);
wav_out = real(wav_out);

% Trim Padding
start_idx = 512 + 1;
len_orig = size(audio, 1);
end_idx = start_idx + len_orig - 1;

if length(wav_out) >= end_idx
    wav_out = wav_out(start_idx:end_idx);
else
    if length(wav_out) > start_idx
        wav_out = wav_out(start_idx:end);
    end
end

% Calibration/Normalization
calibration_factor = 512.0; 
wav_out = wav_out * calibration_factor;
wav_out = wav_out / (max(abs(wav_out)) + 1e-9);

% Save
output_wav = '../output/output_raw_mask.wav';
audiowrite(output_wav, wav_out, FS);
fprintf('Saved final output to %s\n', output_wav);
