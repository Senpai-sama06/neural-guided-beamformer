% =========================================================================
% IEEE SIGNAL PROCESSING CUP 2026 - INFERENCE SCRIPT
% (PYTHON-MATCHING VERSION)
% =========================================================================
clear; clc;

%% 1. CONFIGURATION & SETUP
mixture = "/home/communications-lab/ramakrishna/avz_testing/real-time-audio-visual-zooming/modal_solutions/matlab_base/correction/mixture.wav";
neural_input = "/home/communications-lab/ramakrishna/avz_testing/real-time-audio-visual-zooming/modal_solutions/IRM_TFLC/matlab/model_input.mat";
model_onnx = "/home/communications-lab/ramakrishna/avz_testing/real-time-audio-visual-zooming/modal_solutions/matlab_base/models/ege_unet_1024_fp32.onnx";

if isfile(neural_input)
    model_input = load(neural_input);
    new_model_input = permute(model_input.neural_input,[3,1,2]);
    final_model_input = new_model_input(:,1:128,:);
else
    error("model_input.mat not found!")
end

if ~isfile(model_onnx)
    error("model_onnx is not found!")
end

% Load ONNX Model
dag_net = importONNXNetwork(model_onnx, 'TargetNetwork', 'dagnetwork','OutputLayerType','regression','ImageInputSize',[513,128,5]);
lgraph = layerGraph(dag_net);
lgraph = removeLayers(lgraph, 'RegressionLayer_mask');
net = dlnetwork(lgraph);

%% 2. RUNNING THE NEURAL MODEL
disp("Running Neural Model...");
input_4d = reshape(final_model_input, 513, 128, 5, 1);
dl_in = dlarray(input_4d, 'SSCB');
output_mask = predict(net, dl_in);
mask_pred = extractdata(output_mask);
mask_pred = mask_pred(:, 1:128, :, :);
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

if size(audio, 2) == 1
    audio = [audio, audio]; 
end

[S1, ~, ~] = spectrogram(audio(:,1), WINDOW, OVERLAP, N_FFT, FS);
[S2, ~, ~] = spectrogram(audio(:,2), WINDOW, OVERLAP, N_FFT, FS);

%% 4. BEAMFORMER LOGIC (PYTHON MATCHING)
% Matches 'tflc_beamforming_broadside' from Python script
disp('Running Global Beamforming (Python Match)...');

% Align Dimensions
[n_bins, n_frames] = size(S1);
min_len = min(n_frames, size(mask, 2));
S1 = S1(:, 1:min_len);
S2 = S2(:, 1:min_len);
mask = mask(:, 1:min_len);
n_frames = min_len;

% Create 3D Input X: [Freq x Time x Mics] -> [F, T, M]
X = cat(3, S1, S2); 
[F, T, M] = size(X);

n_beamformers = 2;
iterations = 20; % Matches Python 'iterations=20' call

% --- A. Estimate Global Noise Covariance ---
% Python: Phi_noise_total = einsum(..., noise_mask)
noise_mask = 1.0 - mask;
Phi_noise_total = zeros(F, M, M);

for f = 1:F
    X_f = squeeze(X(f, :, :)).'; % [M x T]
    w_f = sqrt(noise_mask(f, :));
    X_w = X_f .* w_f;
    % Sum over time (Equivalent to einsum summation)
    Phi_noise_total(f, :, :) = (X_w * X_w'); 
end

% --- B. Initialization (Random Perturbation) ---
% Python: Broadside vector (ones) + Random Noise Covariance
a_vec = ones(M, 1); 
W_k = zeros(F, M, n_beamformers);

for k = 1:n_beamformers
    % Generate Perturbation (Real + Imag) matching Python np.random.normal(0, 0.01)
    % Note: Python random and Matlab randn won't yield identical numbers, 
    % but the statistical behavior is now identical.
    perturbation = (randn(F, M, M) + 1i * randn(F, M, M)) * 0.01;
    
    Phi_init = Phi_noise_total + perturbation;
    
    for f = 1:F
        % Python: Phi_inv = np.linalg.inv(Phi_init[f] + 1e-2 * np.eye(M))
        R = squeeze(Phi_init(f,:,:)) + 1e-2 * eye(M); 
        
        try
            R_inv = inv(R);
            num = R_inv * a_vec;
            den = a_vec' * R_inv * a_vec;
            w = num / (den + 1e-10);
            W_k(f, :, k) = w.';
        catch
            W_k(f, :, k) = (a_vec / M).';
        end
    end
end

% --- C. TFLC Iterations ---
c_k = zeros(F, T, n_beamformers);

for iter = 1:iterations
    % 1. Apply Beamformers
    Y_k = zeros(F, T, n_beamformers);
    for k = 1:n_beamformers
        for f = 1:F
            w_f = squeeze(W_k(f, :, k)).';
            x_f = squeeze(X(f, :, :)).';
            Y_k(f, :, k) = w_f' * x_f;
        end
    end
    
    % 2. Update Weights (c_k) - Geometric Projection
    y1 = Y_k(:,:,1);
    y2 = Y_k(:,:,2);
    y21 = y1 - y2;
    
    numerator = -real(y2 .* conj(y21));
    denominator = abs(y21).^2 + 1e-10;
    
    c1 = max(0, min(1, numerator ./ denominator));
    c2 = 1.0 - c1;
    
    c_k(:,:,1) = c1;
    c_k(:,:,2) = c2;
    
    % 3. Update Filters (W_k)
    for k = 1:n_beamformers
        mask_k = c_k(:,:,k); % Python: mask_k = c_k (Not squared in broadside code)
        
        for f = 1:F
            X_f = squeeze(X(f, :, :)).';
            % Weighted Covariance
            w_f = sqrt(mask_k(f, :)); 
            X_w = X_f .* w_f;
            Phi_k = (X_w * X_w') + 1e-2 * eye(M); % Matching 1e-2 loading
            
            try
                R_inv = inv(Phi_k);
                num = R_inv * a_vec;
                den = a_vec' * R_inv * a_vec;
                w = num / (den + 1e-10);
                W_k(f, :, k) = w.';
            catch
                % Keep previous w if singular
            end
        end
    end
end

% --- D. Final Reconstruction ---
Y_k_final = zeros(F, T);
for k = 1:n_beamformers
    Y_k_out = zeros(F, T);
    for f = 1:F
        w_f = squeeze(W_k(f, :, k)).';
        x_f = squeeze(X(f, :, :)).';
        Y_k_out(f, :) = w_f' * x_f;
    end
    Y_k_final = Y_k_final + c_k(:,:,k) .* Y_k_out;
end

% --- E. Post-Processing (Raw Mask) ---
% Python: S_raw = S_out * np.maximum(Mask_np, 0.05)
Y_final_masked = Y_k_final .* max(mask, 0.05);

%% 5. INVERSE STFT (BACK TO AUDIO)
part_positive = Y_final_masked(2:end-1, :); 
part_negative = conj(flipud(part_positive));
Y_twosided = [Y_final_masked; part_negative];

% Use 'istft' to match scipy.signal.istft logic
enhanced_audio = istft(Y_twosided, FS, ...
                       'Window', WINDOW, ...          
                       'OverlapLength', OVERLAP, ...  
                       'FFTLength', N_FFT);
                   
enhanced_audio = real(enhanced_audio);

% Normalize (Matches Python: wav / (np.max(np.abs(wav)) + 1e-9))
% enhanced_audio = enhanced_audio / (max(abs(enhanced_audio)) + 1e-9);

% Save
output_filename = 'processed_output_1.wav';
audiowrite(output_filename, enhanced_audio, FS);

%% 6. VISUALIZATION
figure('Name', 'Results', 'Color', 'w');
subplot(1,3,1);
imagesc(log(abs(S1)+eps)); axis xy; colormap jet; title('Noisy Input');
subplot(1,3,2);
imagesc(mask); axis xy; colormap parula; title('Mask');
subplot(1,3,3);
imagesc(log(abs(Y_final_masked)+eps)); axis xy; colormap jet; title('Enhanced Output');
fprintf('Processing Complete. Saved to %s\n', output_filename);