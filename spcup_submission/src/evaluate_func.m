function metrics = evaluate_func(y_est, y_tgt, y_int, fs)
% EVALUATE_FUNC Calculates SIR, SINR, STOI, PESQ, ViSQOL (NB/WB)
%   metrics = evaluate_func(y_est, y_tgt, y_int, fs)

    % Ensure Mono
    if size(y_est,2)>1, y_est=y_est(:,1); end
    if size(y_tgt,2)>1, y_tgt=y_tgt(:,1); end
    if size(y_int,2)>1, y_int=y_int(:,1); end

    % Alignment (Cross-Correlation)
    [c, lags] = xcorr(y_tgt, y_est);
    [~, I] = max(abs(c));
    delay = lags(I);

    if delay > 0
        y_est = y_est(delay+1:end);
        y_tgt = y_tgt(1:length(y_est));
        y_int = y_int(1:length(y_est));
    elseif delay < 0
        delay = -delay;
        y_tgt = y_tgt(delay+1:end);
        y_int = y_int(delay+1:end);
        y_est = y_est(1:length(y_tgt));
    end

    % Truncate to min length
    min_len = min([length(y_est), length(y_tgt), length(y_int)]);
    y_est = y_est(1:min_len);
    y_tgt = y_tgt(1:min_len);
    y_int = y_int(1:min_len);

    % --- Physics Metrics (OSINR/SIR) ---
    eps = 1e-10;
    tgt_n = y_tgt / (norm(y_tgt) + eps);
    int_n = y_int / (norm(y_int) + eps);

    alpha = dot(y_est, tgt_n);
    beta  = dot(y_est, int_n);

    e_target = alpha * tgt_n;
    e_interf = beta * int_n;
    e_noise  = y_est - e_target - e_interf;

    P_t = sum(e_target.^2);
    P_i = sum(e_interf.^2);
    P_n = sum(e_noise.^2);

    metrics.SIR  = 10 * log10(P_t / (P_i + eps));
    metrics.SINR = 10 * log10(P_t / (P_i + P_n + eps));

    % --- Perceptual Metrics ---
    % STOI
    metrics.STOI = NaN;
    if exist('stoi', 'file') == 2
        try
            metrics.STOI = stoi(y_tgt, y_est, fs);
        catch
        end
    end

    % PESQ
    metrics.PESQ = NaN;
    if exist('pesq', 'file') == 2
        try
            metrics.PESQ = pesq(y_tgt, y_est, fs);
        catch
        end
    end

    % ViSQOL Wideband (Default Speech Mode at current FS, usually 16k or 48k)
    metrics.ViSQOL_WB = NaN;
    if exist('visqol', 'file') == 2
        try
            metrics.ViSQOL_WB = visqol(y_est, y_tgt, fs, 'Mode', 'speech');
        catch
        end
    end
    
    % ViSQOL Narrowband (Resample to 8kHz)
    metrics.ViSQOL_NB = NaN;
    if exist('visqol', 'file') == 2
        try
            % Resample to 8000 Hz for Narrowband simulation
            target_fs = 8000;
            est_8k = resample(y_est, target_fs, fs);
            tgt_8k = resample(y_tgt, target_fs, fs);
            metrics.ViSQOL_NB = visqol(est_8k, tgt_8k, target_fs, 'Mode', 'speech');
        catch
        end
    end

    fprintf('SIR: %.2f | SINR: %.2f | STOI: %.4f | PESQ: %.4f | ViSQOL_WB: %.4f | ViSQOL_NB: %.4f\n', ...
        metrics.SIR, metrics.SINR, metrics.STOI, metrics.PESQ, metrics.ViSQOL_WB, metrics.ViSQOL_NB);

end
