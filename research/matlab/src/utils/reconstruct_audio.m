function reconstructed_signal = reconstruct_audio(spec, fs, nfft, hop_len, window, target_len)
    % RECONSTRUCT_AUDIO - Inverse STFT with specific calibration for TFLC Pipeline
    %
    % Usage:
    %   audio = reconstruct_audio(spec, fs, nfft, hop_len, window, target_len)
    %
    % Inputs:
    %   spec       : [F, T] Matrix (Single Channel, One-Sided or Two-Sided)
    %   fs         : Sampling Rate (e.g. 16000)
    %   nfft       : FFT Size (e.g. 1024)
    %   hop_len    : Hop Size (e.g. 512)
    %   window     : Window Vector (e.g. hann(1024, 'periodic'))
    %   target_len : (Optional) Expected length of output signal for cropping
    %
    % Output:
    %   reconstructed_signal : [T, 1] Time-domain signal (Normalized)
    
    % 1. Ensure Two-Sided Spectrum
    if size(spec, 1) == (nfft/2 + 1)
        % One-sided spectrum - need to create two-sided
        % Negative frequencies are complex conjugate of positive (reversed)
        spec_positive = spec(2:end-1, :);
% Bins 2 to 512 spec_negative = conj(flipud(spec_positive));
% Reversed conjugate

    % Reconstruct : [DC; positive; Nyquist; negative] spec_twosided =
    [spec(1, :); spec_positive; spec(end, :); spec_negative];
else spec_twosided = spec;
end

        % 2. ISTFT(CRITICAL PARAMS MATCHING SCIPY) %
        -'Window' : Must match forward transform % -'OverlapLength' : N
    - Hop % -'Centered'
    : FALSE(Scipy default is True leading to padding, we want exact alignment)
          enhanced_audio = istft(spec_twosided, fs, ... 'Window', window,
                                 ... 'OverlapLength', nfft - hop_len,
                                 ... 'FFTLength', nfft, ... 'Centered', false);

enhanced_audio = real(enhanced_audio);

    % 3. Cropping Logic (Match Python Scipy defaults)
    % Scipy centered=True (default) pads input. We used centered=False.
    % We verified start_idx = 512 + 1 works for this pipeline alignment.
    start_idx = 512 + 1;

    if nargin
      >= 6 && ~isempty(target_len) end_idx = start_idx + target_len - 1;
    if length (enhanced_audio)
      >= end_idx enhanced_audio = enhanced_audio(start_idx : end_idx);
    end end

        % 4. Scaling Correction %
        The forward pipeline divided by sum(window)(512).%
        MATLAB's istft does NOT undo this scaling automatically in this config. calibration_factor =
        sum(window);
    % Dynamically calculate 512.0 enhanced_audio =
        enhanced_audio * calibration_factor;

    reconstructed_signal = enhanced_audio;
    end
