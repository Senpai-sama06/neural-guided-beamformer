% PROCESS_TASK2 - Reverberant Room Simulation Phase 2
clear; clc;

% 1. Setup Paths
addpath('../src');
addpath('../src/utils');

% 2. Load Data
filename = 'Task2_Reverberant_5dB.mat';
if exist(filename, 'file') ~= 2
    error('Data file %s not found.', filename);
end
fprintf('Loading %s...\n', filename);
load(filename);

% 3. Config
if isfield(params, 'save_fs')
    fs = double(params.save_fs);
elseif isfield(params, 'fs')
    fs = double(params.fs);
else
    fs = 16000;
end
model_path = '../models/ege_unet_1024_fp32.onnx';

% 4. Feature Extraction
fprintf('Extracting Features...\n');
neural_input = feature_extraction_func(mixture_signal, fs);

% 5. Inference & Beamforming
fprintf('Running Inference...\n');
processed_signal = inference_func(neural_input, mixture_signal, fs, model_path);

% 6. Save Processed Audio Wav
audiowrite('processed_signal.wav', processed_signal, fs);

% 7. Evaluation
fprintf('Evaluating...\n');
tgt = target_signal;
int = interference_signal;

% Evaluate Function
metrics = evaluate_func(processed_signal, tgt, int, fs);

% 8. Append Results to MAT file
fprintf('Saving results to %s...\n', filename);
save(filename, 'processed_signal', 'metrics', '-append');

fprintf('Task 2 Complete.\n');
