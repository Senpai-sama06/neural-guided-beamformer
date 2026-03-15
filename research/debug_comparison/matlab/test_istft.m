clear; clc;

FS = 16000;
N_FFT = 1024;
HOP_LEN = 512;
WINDOW = hann(N_FFT, 'periodic');

% 1. Synthesize signal
t = (0:FS-1).' / FS;
x = sin(2*pi*440*t); 

% 2. MATLAB Spectrogram (mimic main.m)
[S, f, t_stft] = spectrogram(x, WINDOW, N_FFT-HOP_LEN, N_FFT, FS);

% 3. Apply the scaling we used in main.m to match Python
win_sum = sum(WINDOW);
S_scaled = S / win_sum; 

% 4. Reconstruction Logic (One-sided to Two-sided)
spec = S_scaled;
if size(spec, 1) == 513
    spec_pos = spec(2:end-1, :);
    spec_neg = conj(flipud(spec_pos));
    spec_2sided = [spec(1,:); spec_pos; spec(end,:); spec_neg];
else
    spec_2sided = spec;
end

% 5. ISTFT
y_rec = istft(spec_2sided, FS, 'Window', WINDOW, 'OverlapLength', N_FFT-HOP_LEN, 'FFTLength', N_FFT, 'Centered', false);
y_rec = real(y_rec);

% 6. Align and Compare
L = min(length(x), length(y_rec));
x_trunc = x(1:L);
y_rec = y_rec(1:L);

% Calculate Ratio
ratio_amp = mean(abs(x_trunc)) / mean(abs(y_rec));
ratio_rms = rms(x_trunc) / rms(y_rec);

fprintf("Window Sum: %.4f\n", win_sum);
fprintf("Amplitude Ratio (Target/Actual): %.6f\n", ratio_amp);
fprintf("RMS Ratio     (Target/Actual): %.6f\n", ratio_rms);

% Verify with correction
y_corrected = y_rec * ratio_rms;
mse = mean((x_trunc - y_corrected).^2);
fprintf("MSE after correction: %.10f\n", mse);
