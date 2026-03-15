function wav_out = inference_func(neural_input, mixture_audio, fs_in, model_path)
% INFERENCE_FUNC Runs the enhancement pipeline (Mask Inference + Beamforming)
%   wav_out = inference_func(neural_input, mixture_audio, fs_in, model_path)
%
%   Input:
%       neural_input:  Features from feature_extraction_func (Time x Ch x Freq)
%       mixture_audio: Original mixture audio (for beamformer physics)
%       fs_in:         Sampling rate
%       model_path:    Path to ONNX model
%
%   Output:
%       wav_out:       Enhanced, single-channel audio

    % Constants
    FS = 16000;
    N_FFT = 1024;
    HOP_LEN = 512;
    WINDOW = hann(N_FFT, 'periodic');
    
    % Ensure correct FS for processing
    if fs_in ~= FS
        mixture_audio = resample(mixture_audio, FS, fs_in);
    end
    if size(mixture_audio,2)==1, mixture_audio=[mixture_audio,mixture_audio]; end

    % --- 1. Prepare Model Input ---
    % Permute back to (Freq, Time, Channels)
    final_model_input = permute(neural_input, [3, 1, 2]);
    [n_f_in, n_t_in, n_ch_in] = size(final_model_input);

    % Pad time dimension to multiple of 16 for UNet
    pad_t = 0;
    if mod(n_t_in, 16) ~= 0
        pad_t = 16 - mod(n_t_in, 16);
        padding = zeros(n_f_in, pad_t, n_ch_in);
        final_model_input = cat(2, final_model_input, padding);
    end
    n_t_padded = size(final_model_input, 2);

    % --- 2. Load Model ---
    if exist(model_path, 'file') ~= 2
        error('Model ONNX file not found: %s', model_path);
    end

    % Import ONNX Layers
    lgraph = importONNXLayers(model_path, 'OutputLayerType', 'regression', 'ImportWeights', true, 'ImageInputSize', [513, n_t_padded, 5]);
    try
        lgraph = removeLayers(lgraph, 'RegressionLayer_mask'); 
    catch
        % Layer might not exist, proceed
    end
    net = dlnetwork(lgraph);

    % --- 3. Inference ---
    input_4d = reshape(final_model_input, 513, n_t_padded, 5, 1);
    dl_in = dlarray(input_4d, 'SSCB');
    
    % Predict
    output_mask = predict(net, dl_in);
    mask_pred = extractdata(output_mask);

    % --- 4. Post-Process Mask ---
    mask_raw = double(squeeze(mask_pred));

    % Remove padding
    if pad_t > 0
        mask_raw = mask_raw(:, 1:end-pad_t);
    end

    % Recover original STFT dimensions for Beamforming
    % STFT of original audio to get dimensions
    pad_width = N_FFT / 2;
    tail_pad = 512;
    audio_padded = [zeros(pad_width, size(mixture_audio, 2)); mixture_audio; zeros(pad_width + tail_pad, size(mixture_audio, 2))];
    [S1, ~, ~] = spectrogram(audio_padded(:,1), WINDOW, N_FFT-HOP_LEN, N_FFT, FS);
    [n_f, n_t] = size(S1);
    
    win_sum = sum(WINDOW);
    S1 = S1 / win_sum;
    
    % Resize mask to match original features (in case of slight mismatch)
    mask = imresize(mask_raw, [n_f, n_t], 'bilinear');
    mask = max(0, min(1, mask));

    % --- 5. Beamforming ---
    % Compute full STFT for beamforming
    [S2, ~, ~] = spectrogram(audio_padded(:,2), WINDOW, N_FFT-HOP_LEN, N_FFT, FS);
    S2 = S2 / win_sum;

    Y = zeros(2, n_f, n_t);
    Y(1,:,:) = S1; 
    Y(2,:,:) = S2;
    
    % Run TFLC Beamformer
    [S_out, ~] = tflc_beamformer(Y, mask, 2, 20);

    % Apply Raw Mask to Beamformed Output (Post-filtering)
    S_raw = S_out .* max(mask, 0.05);

    % --- 6. ISTFT / Reconstruction ---
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
    len_orig = size(mixture_audio, 1);
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

end
