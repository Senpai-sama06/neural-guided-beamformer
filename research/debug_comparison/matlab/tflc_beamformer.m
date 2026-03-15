function [Y_final, debug_data] = tflc_beamformer(Y_in, mask, n_beamformers, iterations)
    % tflc_beamformer.m - Manual Inversion 2x2 Single Precision
    
    if nargin < 3, n_beamformers = 2; end
    if nargin < 4, iterations = 20; end

    X = permute(Y_in, [2, 3, 1]);
    X = single(X);
    [F_dim, T_dim, M] = size(X);
    
    a_vec = ones(F_dim, M, 'single');
    noise_mask = single(1.0) - single(mask);
    
    % 1. Estimate Global Noise Covariance
    Phi_noise_total = zeros(F_dim, M, M, 'single');
    for f = 1:F_dim
        X_f = squeeze(X(f, :, :));
        m_f = noise_mask(f, :).';
        for m1 = 1:M
            for m2 = 1:M
                Phi_noise_total(f, m1, m2) = sum(m_f .* X_f(:, m1) .* conj(X_f(:, m2)));
            end
        end
    end
    debug_data.Phi_init = Phi_noise_total;
    
    % 2. Initialize
    W_k = zeros(F_dim, M, n_beamformers, 'single');
    
    for k = 1:n_beamformers
        perturbation = single(0.01) * (ones(F_dim, M, M, 'single') + 1i * ones(F_dim, M, M, 'single'));
        for f = 1:F_dim
            Phi_f = squeeze(Phi_noise_total(f, :, :)) + squeeze(perturbation(f, :, :)) + single(1e-2) * eye(M, 'single');
            
            % Manual 2x2 Inversion
            det_val = Phi_f(1,1)*Phi_f(2,2) - Phi_f(1,2)*Phi_f(2,1);
            inv_det = single(1.0) / det_val;
            
            if isinf(inv_det) || isnan(inv_det) || abs(det_val) < single(1e-12)
                W_k(f, :, k) = a_vec(f, :) / single(M);
            else
                Phi_inv = zeros(2, 2, 'single');
                Phi_inv(1,1) = Phi_f(2,2) * inv_det;
                Phi_inv(1,2) = -Phi_f(1,2) * inv_det;
                Phi_inv(2,1) = -Phi_f(2,1) * inv_det;
                Phi_inv(2,2) = Phi_f(1,1) * inv_det;
                
                a = a_vec(f, :).';
                num = Phi_inv * a;
                den = a' * Phi_inv * a;
                w = num / (den + single(1e-10));
                W_k(f, :, k) = w.';
            end
        end
    end
    debug_data.W_init = W_k;
    
    % 3. Iterative Optimization
    for iter = 1:iterations
        Y_k = zeros(F_dim, T_dim, n_beamformers, 'single');
        for f = 1:F_dim
            X_f = squeeze(X(f, :, :)).';
            for k = 1:n_beamformers
                w_f = squeeze(W_k(f, :, k)).';
                Y_k(f, :, k) = conj(w_f)' * X_f;
            end
        end
        
        y1 = Y_k(:, :, 1);
        y2 = Y_k(:, :, 2);
        y21 = y1 - y2;
        
        numerator = -real(y2 .* conj(y21));
        denominator = abs(y21).^2 + single(1e-10);
        c1 = max(single(0), min(single(1), numerator ./ denominator));
        c2 = single(1.0) - c1;
        c_k = cat(3, c1, c2);
        
        for k = 1:n_beamformers
            mask_k = c_k(:, :, k);
            Phi_k = zeros(F_dim, M, M, 'single');
            for f = 1:F_dim
                X_f = squeeze(X(f, :, :));
                m_f = mask_k(f, :).';
                for m1 = 1:M
                    for m2 = 1:M
                        Phi_k(f, m1, m2) = sum(m_f .* X_f(:, m1) .* conj(X_f(:, m2)));
                    end
                end
                Phi_k(f, :, :) = squeeze(Phi_k(f, :, :)) + single(1e-2) * eye(M, 'single');
            end
            
            for f = 1:F_dim
                Phi_f = squeeze(Phi_k(f, :, :));
                % Manual Inversion
                det_val = Phi_f(1,1)*Phi_f(2,2) - Phi_f(1,2)*Phi_f(2,1);
                inv_det = single(1.0) / det_val;
                
                if ~isinf(inv_det) && ~isnan(inv_det) && abs(det_val) > single(1e-12)
                    Phi_inv = zeros(2, 2, 'single');
                    Phi_inv(1,1) = Phi_f(2,2) * inv_det;
                    Phi_inv(1,2) = -Phi_f(1,2) * inv_det;
                    Phi_inv(2,1) = -Phi_f(2,1) * inv_det;
                    Phi_inv(2,2) = Phi_f(1,1) * inv_det;
                    
                    a = a_vec(f, :).';
                    w = (Phi_inv * a) / (a' * Phi_inv * a + single(1e-10));
                    W_k(f, :, k) = w.';
                end
            end
        end
    end
    
    debug_data.W_final = W_k;
    debug_data.c_k = c_k;
    
    Y_k_final = zeros(F_dim, T_dim, n_beamformers, 'single');
    for f = 1:F_dim
        X_f = squeeze(X(f, :, :)).';
        for k = 1:n_beamformers
            w_f = squeeze(W_k(f, :, k)).';
            Y_k_final(f, :, k) = conj(w_f)' * X_f;
        end
    end
    
    Y_final = sum(c_k .* Y_k_final, 3);
end
