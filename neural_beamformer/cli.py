import argparse
import sys
import os
from neural_beamformer.inference import enhance_audio

def main():
    parser = argparse.ArgumentParser(
        description="Neural-Guided Beamformer: Real-Time Speech Enhancement via Neural Priors and RTFLC Optimization."
    )
    
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to the input noisy audio file (.wav format, ideally 2-channel)."
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        required=True, 
        help="Path where the enhanced audio file will be saved."
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="Path to the pre-trained EGE-Unet model weights (.pth)."
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)
        
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
        sys.exit(1)
        
    # Ensure output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        
    print(f"Starting enhancement process...")
    enhance_audio(args.input, args.output, args.model)
    print(f"Process complete. File saved to {args.output}")

if __name__ == "__main__":
    main()
