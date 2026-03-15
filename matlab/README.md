# MATLAB Audio Enhancement Pipeline

## Directory Structure
- **`src/`**: Source code (`feature_extraction.m`, `main.m`).
- **`input/`**: Stores intermediate features (`model_input.mat`) and input audio.
- **`output/`**: Stores generated enhanced audio (`.wav`).
- **`models/`**: Contains the ONNX models.

## How to Run

### Step 1: Feature Extraction
This step converts your input audio (mixture) into the specific feature set required by the neural network.

1.  Open `src/feature_extraction.m`.
2.  **Update the Input Path**: Change the `mixture` variable (Line 23) to point to your dataset audio file.
    ```matlab
    mixture = "/path/to/your/audio.wav";
    ```
3.  **Run the script**.
    - It saves `model_input.mat` to the `../input/` directory.

### Step 2: Inference & Beamforming
This step runs the Neural Network to estimate a mask, then applies Iterative TFLC Beamforming.

1.  Open `src/main.m`.
2.  **Update the Input Path**: Change the `mixture` variable (Line 8) to point to the **SAME** audio file you used in Step 1.
    ```matlab
    mixture = "/path/to/your/audio.wav";
    ```
    *(Note: `main.m` needs the original audio to compute the beamforming covariance matrices)*.
3.  **Run the script**.
    - It reads features from `../input/model_input.mat`.
    - It saves the enhanced audio to `../output/output_raw_mask.wav` and `output_gauss_mask.wav`.

## Parameters
- **`n_beamformers`** (Default: 2): Set in `src/main.m`.
- **`iterations`** (Default: 20): Number of TFLC iterations.

### Step 3: Evaluation
This step calculates OSINR, SIR, and ViSQOL metrics by comparing the enhanced audio against the ground truth (`target.wav`).

1.  Open `src/evaluate.m`.
2.  **Verify Data Path**: Ensure `dataDir` (Line 9) points to the folder containing `target.wav`, `mixture.wav`, and `interference.wav`.
3.  **Run the script**.
    - It prints **SIR**, **SINR**, **STOI**, and **ViSQOL** scores to the command window.

