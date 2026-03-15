% STFT Constants
FS = 16000;             % Sampling Rate [cite: 79]
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

%Recheck: To convert code to Double Mono-Stereo
if size(audio, 2) == 1
    audio = [audio, audio]; 
end

[n_bins, n_frames] = size(S1);
Y_mvdr = zeros(n_bins, n_frames);

% Steering Vector for 90 degrees (Broadside)
% At 90 deg, sound hits both mics at the same time (assuming centered array).
% If your target is 90, the phase diff is 0. Steering vector is [1; 1].
d = ones(2, 1); 

for f = 1:n_bins
    % 1. Estimate Noise Covariance Matrix (Rnn) for this frequency
    % We use (1 - mask) because low mask values = high noise
    noise_weight = 1 - mask(f, :);
    
    % Stack mics: 2 x Time
    X_f = [S1(f, :); S2(f, :)]; 
    
    % Weighted Covariance calculation
    % Rnn = (X * W * X') / sum(W)
    X_weighted = X_f .* sqrt(noise_weight);
    Rnn = (X_weighted * X_weighted') / (sum(noise_weight) + 1e-6);
    
    % Diagonal Loading (Stability)
    Rnn = Rnn + 1e-6 * eye(2);
    
    % 2. MVDR Weights Formula: w = (Rnn^-1 * d) / (d' * Rnn^-1 * d)
    Rnn_inv = inv(Rnn);
    numerator = Rnn_inv * d;
    denominator = d' * Rnn_inv * d;
    w_mvdr = numerator / (denominator + 1e-10);
    
    % 3. Apply Weights
    % Output = w^H * x
    Y_mvdr(f, :) = w_mvdr' * X_f;
end

%% 4. INVERSE STFT (BACK TO AUDIO)
% -------------------------------------------------------------------------
% Choose which result to save (Y_simple or Y_mvdr)
final_spec = Y_mvdr; 

enhanced_audio = istft(final_spec, FS, ...
                       'Window', WIN, 'OverlapLength', N_FFT-HOP, ...
                       'FFTLength', N_FFT);

% Normalize volume to avoid clipping
enhanced_audio = enhanced_audio / max(abs(enhanced_audio));

% Save
output_filename = 'processed_output.wav';
audiowrite(output_filename, enhanced_audio, FS);

%% 5. VISUALIZATION OF RESULTS
% -------------------------------------------------------------------------
figure('Name', 'Results', 'Color', 'w');

subplot(3,1,1);
imagesc(log(abs(S1)+eps)); axis xy; colormap inferno; title('Original Noisy Input');

subplot(3,1,2);
imagesc(mask); axis xy; colormap parula; title('Predicted Mask (1=Target)');

subplot(3,1,3);
imagesc(log(abs(Y_mvdr)+eps)); axis xy; colormap inferno; title('Enhanced MVDR Output');

fprintf('Processing Complete. Listen to %s\n', output_filename);