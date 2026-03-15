#!/usr/bin/env python3
"""
Batch processing script for DeepFPU-RNN + SMVB ablation study.
Runs inference and metrics calculation on samples 0-499 for both reverb and no_reverb modes.
"""

import os
import sys
import argparse
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Import deepfpu_rnn modules
from inference import deepfpu_rnn_inference
from metrics import evaluate_deepfpu_rnn


def run_batch(start_idx=0, end_idx=499, model_path=None):
    """
    Run DeepFPU-RNN + SMVB on batch samples.
    
    Args:
        start_idx (int): Starting sample index
        end_idx (int): Ending sample index (inclusive)
        model_path (str): Path to model .pth file
    """
    if model_path is None:
        print("Error: Model path is required")
        return
    
    modes = ['reverb', 'no_reverb']
    total_samples = (end_idx - start_idx + 1) * len(modes)
    
    print(f"Starting DeepFPU-RNN + SMVB batch processing...")
    print(f"Samples: {start_idx} to {end_idx}")
    print(f"Modes: {modes}")
    print(f"Model: {model_path}")
    print(f"Total runs: {total_samples}")
    print("-" * 60)
    
    success_count = 0
    fail_count = 0
    
    with tqdm(total=total_samples, desc="Processing") as pbar:
        for idx in range(start_idx, end_idx + 1):
            sample_name = f"batch_test_{idx:03d}"
            
            for mode in modes:
                try:
                    # Run inference
                    output_path = deepfpu_rnn_inference(sample_name, mode, model_path)
                    
                    if output_path is None:
                        fail_count += 1
                        pbar.set_postfix({"Success": success_count, "Failed": fail_count})
                        pbar.update(1)
                        continue
                    
                    # Calculate metrics
                    eval_success = evaluate_deepfpu_rnn(sample_name, mode)
                    
                    if eval_success:
                        success_count += 1
                    else:
                        fail_count += 1
                    
                except Exception as e:
                    print(f"\nError processing {sample_name} ({mode}): {e}")
                    fail_count += 1
                
                pbar.set_postfix({"Success": success_count, "Failed": fail_count})
                pbar.update(1)
    
    print("-" * 60)
    print(f"Batch processing complete!")
    print(f"Successful: {success_count}/{total_samples}")
    print(f"Failed: {fail_count}/{total_samples}")
    print(f"Results saved to: ablation_study/deepfpu_rnn/deepfpu_rnn_metrics.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepFPU-RNN + SMVB Batch Processing')
    parser.add_argument('--start', type=int, default=0, help='Starting sample index (default: 0)')
    parser.add_argument('--end', type=int, default=499, help='Ending sample index (default: 499)')
    parser.add_argument('--model', type=str, required=True, help='Path to model .pth file')
    
    args = parser.parse_args()
    
    run_batch(args.start, args.end, args.model)
