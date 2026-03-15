clear; clc;
mixture = '../data/debug_run/mixture.wav';
neural_input_path = '../data/debug_matlab_features.mat';
model_onnx = '../../matlab/models/ege_unet_1024_fp32.onnx';
FS = 16000; N_FFT = 1024; HOP_LEN = 512; WINDOW = hann(N_FFT, 'periodic');
[audio, fs_raw] = audioread(mixture);
if fs_raw ~= FS, audio = resample(audio, FS, fs_raw); end
if size(audio,2)==1, audio=[audio,audio]; end
pad_width = N_FFT / 2;
tail_pad = 512;
audio_padded = [zeros(pad_width, size(audio, 2)); audio; zeros(pad_width + tail_pad, size(audio, 2))];
[S1, ~, ~] = spectrogram(audio_padded(:,1), WINDOW, N_FFT-HOP_LEN, N_FFT, FS);
[S2, ~, ~] = spectrogram(audio_padded(:,2), WINDOW, N_FFT-HOP_LEN, N_FFT, FS);
win_sum = sum(WINDOW);
S1 = S1 / win_sum;
S2 = S2 / win_sum;
[n_f, n_t] = size(S1);
load(neural_input_path, 'neural_input');
new_model_input = permute(neural_input, [3, 1, 2]);
final_model_input = new_model_input;
[n_f_in, n_t_in, n_ch_in] = size(final_model_input);
pad_t = 0;
if mod(n_t_in, 16) ~= 0
    pad_t = 16 - mod(n_t_in, 16);
    padding = zeros(n_f_in, pad_t, n_ch_in);
    final_model_input = cat(2, final_model_input, padding);
end
n_t_padded = size(final_model_input, 2);
lgraph = importONNXLayers(model_onnx, 'OutputLayerType', 'regression', 'ImportWeights', true, 'ImageInputSize', [513, n_t_padded, 5]);
lgraph = removeLayers(lgraph, 'RegressionLayer_mask');
net = dlnetwork(lgraph);
input_4d = reshape(final_model_input, 513, n_t_padded, 5, 1);
dl_in = dlarray(input_4d, 'SSCB');
output_mask = predict(net, dl_in);
mask_pred = extractdata(output_mask);
mask_raw = double(squeeze(mask_pred));
if pad_t > 0
    mask_raw = mask_raw(:, 1:end-pad_t);
end
mask = imresize(mask_raw, [n_f, n_t], 'bilinear');
mask = max(0, min(1, mask));
Y = zeros(2, n_f, n_t);
Y(1,:,:) = S1; Y(2,:,:) = S2;
[S_out, ~] = tflc_beamformer(Y, mask, 2, 20);
S_raw = S_out .* max(mask, 0.05);
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
start_idx = 512 + 1;
end_idx = start_idx + 96524 - 1;
if length(wav_out) >= end_idx
    wav_out = wav_out(start_idx:end_idx);
end
calibration_factor = 512.0;
wav_out = wav_out * calibration_factor;
wav_out = wav_out / (max(abs(wav_out)) + 1e-9);
audiowrite('../data/matlab_out.wav', wav_out, FS);
fprintf('Saved final matlab output to ../data/matlab_out.wav\n');
