import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
import os

def generate_rankings(input_file):
    """
    1. Loads benchmarking data.
    2. Filters out outliers (Zero PESQ, Negative SIR/SINR).
    3. Calculates weighted improvement scores prioritizing PESQ/STOI.
    4. Generates ranked CSVs for Raw and Gauss pipelines.
    """
    
    # --- 1. Load Data ---
    try:
        df = pd.read_csv(input_file)
        print(f"Successfully loaded {input_file}")
        print(f"Total test cases loaded: {len(df)}")
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    # --- 2. Define High-Priority Weights ---
    # Total Quality Weight = 80%
    # Total Signal Weight = 20%
    weights = {
        'PESQ_NB': 0.25,  # High Priority
        'PESQ_WB': 0.25,  # High Priority
        'STOI':    0.30,  # Highest Priority (Intelligibility)
        'SINR':    0.12,  # Low Priority
        'SIR':     0.08   # Lowest Priority
    }

    metrics = ['PESQ_NB', 'PESQ_WB', 'STOI', 'SINR', 'SIR']
    pipelines = ['Raw', 'Gauss'] 

    print("-" * 40)

    for pipe in pipelines:
        print(f"Processing {pipe} pipeline...")
        
        # Work on a copy
        rank_df = df.copy()
        initial_count = len(rank_df)
        
        # --- 3. Outlier Removal / Cleaning ---
        # Condition 1: PESQ must be > 0.1 (removes 0 or near-0 failures)
        # Condition 2: SIR and SINR must be >= 0 (removes negative ratios)
        
        # We verify the absolute output of the pipeline (e.g., Raw_PESQ_NB), not the delta yet.
        try:
            rank_df = rank_df[
                (rank_df[f'{pipe}_PESQ_NB'] > 0.1) & 
                (rank_df[f'{pipe}_PESQ_WB'] > 0.1) & 
                (rank_df[f'{pipe}_SIR'] >= 0) & 
                (rank_df[f'{pipe}_SINR'] >= 0)
            ]
        except KeyError as e:
            print(f"Error: Missing columns in CSV for {pipe} pipeline: {e}")
            continue

        dropped_count = initial_count - len(rank_df)
        if dropped_count > 0:
            print(f"   -> Removed {dropped_count} outliers (0 PESQ or Negative SIR/SINR).")
            print(f"   -> Remaining cases: {len(rank_df)}")
        else:
            print("   -> No outliers found. All cases retained.")

        if len(rank_df) == 0:
            print("   -> Warning: No valid test cases remaining after filtering.")
            continue

        # --- 4. Calculate Deltas (Improvement over Base) ---
        delta_cols = []
        for m in metrics:
            base_col = f"Base_{m}"
            pipe_col = f"{pipe}_{m}"
            delta_col = f"Delta_{m}"
            
            rank_df[delta_col] = rank_df[pipe_col] - rank_df[base_col]
            delta_cols.append(delta_col)

        # --- 5. Normalize Deltas (0 to 1 Scaling) ---
        scaler = MinMaxScaler()
        # Create temp array for calculation only
        norm_data = scaler.fit_transform(rank_df[delta_cols])
        
        # --- 6. Calculate Weighted Score ---
        scores = np.zeros(len(rank_df))
        
        for idx, col_name in enumerate(delta_cols):
            metric_name = col_name.replace("Delta_", "")
            w = weights.get(metric_name, 0)
            scores += norm_data[:, idx] * w
        
        rank_df['Performance_Score'] = scores

        # --- 7. Formatting & Saving ---
        # We output: Run_ID, The calculated Score, and the Raw Metric Values (for reference)
        output_cols = ['Run_ID', 'Performance_Score'] + \
                      [f"{pipe}_{m}" for m in metrics]
        
        final_df = rank_df[output_cols].sort_values(by='Performance_Score', ascending=False)
        
        output_filename = f"ranklist_{pipe.lower()}_cleaned_anechoic.csv"
        final_df.to_csv(output_filename, index=False)
        print(f"   -> Saved ranked list to: {output_filename}")

    print("-" * 40)
    print("Processing complete.")

if __name__ == "__main__":
    # Check for command line argument or use default
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # REPLACE 'data.csv' with your actual filename if running directly
        file_path = 'benchmark_results.csv' 
        
    generate_rankings(file_path)