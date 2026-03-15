'''
python batch_run.py --n 50 --interferers 2
'''

import argparse
import time
import os
import soundfile as sf # Required to check audio duration
from src import simulation, inference, metrics, config
import matplotlib.pyplot as plt

# Optional: Progress bar
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

def run_batch(n_runs, start_idx=0, n_interferers=2):
    model_path = os.path.join("models", "best_model_mixed_soft.pth")  #"/home/communications-lab/Downloads/final_pipeline_submission_grade/final_pipeline/models/best_model_mixed_soft.pth"
    
    print(f"=== STARTING BATCH RUN: {n_runs} Iterations ===")
    
    # Loop over the number of requested runs
    for i in tqdm(range(start_idx, start_idx + n_runs), desc="Processing"):
        run_name = f"batch_test_{i:03d}" # e.g., batch_test_001
        
        mix_path = None
        
        # --- 1. REGENERATION LOOP ---
        # Keep generating until we get a file >= 4 seconds
        while True:
            try:
                # Simulate Scene
                mix_path = simulation.generate_scene(
                    run_name=run_name,
                    dataset='mixed', # or 'musan'
                    reverb=False,
                    n_interferers=n_interferers,
                    snr_target=5
                )
                
                if not mix_path: 
                    # If simulation failed completely, try again
                    continue 

                # Check Duration
                audio_info = sf.info(mix_path)
                if audio_info.duration < 4.0:
                    print(f" [REGEN] {run_name} too short ({audio_info.duration:.2f}s). Regenerating...")
                    continue # Pass and regenerate
                
                # If we reached here, the audio is valid (>= 4s)
                break

            except Exception as e:
                print(f"\n[ERROR] Simulation failed on {run_name}: {e}. Retrying...")
                continue

        # --- 2. PIPELINE EXECUTION ---
        try:
            # Inference
            inference.enhance_audio(
                run_name=run_name,
                input_path=mix_path,
                model_path=model_path
            )

            # Evaluate (Auto-logs to CSV)
            metrics.evaluate_run(run_name)
            
            # Optional: Clean up huge wav files
            # import shutil
            # shutil.rmtree(os.path.join(config.SIM_DIR, run_name))

        except Exception as e:
            print(f"\n[ERROR] Pipeline failed on {run_name}: {e}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=10, help="Number of runs")
    parser.add_argument('--start', type=int, default=0, help="Start index for naming")
    parser.add_argument('--interferers', type=int, default=2, help="Number of interferers")
    
    args = parser.parse_args()
    
    run_batch(args.n, args.start, args.interferers)
