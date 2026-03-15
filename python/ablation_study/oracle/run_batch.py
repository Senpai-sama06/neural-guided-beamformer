#!/usr/bin/env python3
"""
Batch processing script for Oracle TFLC ablation study.
Runs inference and metrics calculation on samples 0-499 for both reverb and no_reverb modes.
"""

import os
import sys
import argparse
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Import oracle modules
from inference import oracle_mvdr_inference
from metrics import evaluate_oracle


def run_batch(start_idx=0, end_idx=499):
    """
    Run oracle TFLC on batch samples.
    
    Args:
        start_idx (int): Starting sample index
        end_idx (int): Ending sample index (inclusive)
    """
    modes = ['reverb', 'no_reverb']
    total_samples = (end_idx - start_idx + 1) * len(modes)
    
    print(f"Starting Oracle TFLC batch processing...")
    print(f"Samples: {start_idx} to {end_idx}")
    print(f"Modes: {modes}")
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
                    output_path = oracle_mvdr_inference(sample_name, mode)
                    
                    if output_path is None:
                        fail_count += 1
                        pbar.set_postfix({"Success": success_count, "Failed": fail_count})
                        pbar.update(1)
                        continue
                    
                    # Calculate metrics
                    eval_success = evaluate_oracle(sample_name, mode)
                    
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
    print(f"Results saved to: ablation_study/oracle/oracle_metrics.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Oracle TFLC Batch Processing')
    parser.add_argument('--start', type=int, default=0, help='Starting sample index (default: 0)')
    parser.add_argument('--end', type=int, default=499, help='Ending sample index (default: 499)')
    
    args = parser.parse_args()
    
    run_batch(args.start, args.end)
