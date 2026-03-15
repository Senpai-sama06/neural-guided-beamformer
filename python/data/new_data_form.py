import pandas as pd
import shutil
import os
import sys

# ================= USER CONFIGURATION =================
# 1. Input CSV Paths
csv_anechoic_path = '/home/cse-sdpl/real-time-audio-visual-zooming/modal_solutions/IRM_TFLC/python/data/results/ranklist_raw_cleaned_anechoic.csv'
csv_reverb_path   = '/home/cse-sdpl/real-time-audio-visual-zooming/modal_solutions/IRM_TFLC/python/data/results/rank_csv/ranklist_raw_cleaned.csv'

# 2. Source Directories
src_results_dir   = 'results'     
src_simulated_dir = 'simulated'   

# 3. Output Directory
dest_root = 'final_data'
# ======================================================

def extract_top_1000():
    print("--- Starting Top 100 Extraction ---")

    # 1. Load the Rank Lists
    if not os.path.exists(csv_anechoic_path) or not os.path.exists(csv_reverb_path):
        print("Error: One or both CSV files not found.")
        return

    df_anec = pd.read_csv(csv_anechoic_path)
    df_rev = pd.read_csv(csv_reverb_path)

    # 2. Prepare Columns to Keep and Rename
    # We assume the columns in your source CSVs are named like 'Raw_PESQ_NB', etc.
    metrics = ['Raw_PESQ_NB', 'Raw_PESQ_WB', 'Raw_STOI', 'Raw_SINR', 'Raw_SIR']
    cols_to_extract = ['Run_ID', 'Performance_Score'] + metrics

    # Filter to ensure we only grab existing columns
    # (Just in case one file has slightly different headers)
    anec_cols = [c for c in cols_to_extract if c in df_anec.columns]
    rev_cols  = [c for c in cols_to_extract if c in df_rev.columns]
    
    df_anec = df_anec[anec_cols].copy()
    df_rev = df_rev[rev_cols].copy()

    # Create Renaming Maps to distinguish Anechoic vs Reverb in the final sheet
    # Format: "Raw_PESQ_NB" -> "Anechoic_PESQ_NB"
    anec_rename_map = {'Performance_Score': 'Score_Anechoic'}
    rev_rename_map  = {'Performance_Score': 'Score_Reverb'}

    for m in metrics:
        if m in df_anec.columns:
            # e.g., Raw_PESQ_NB -> Anechoic_PESQ_NB
            clean_name = m.replace('Raw_', '') 
            anec_rename_map[m] = f"Anechoic_{clean_name}"
            
        if m in df_rev.columns:
            # e.g., Raw_PESQ_NB -> Reverb_PESQ_NB
            clean_name = m.replace('Raw_', '')
            rev_rename_map[m] = f"Reverb_{clean_name}"

    df_anec.rename(columns=anec_rename_map, inplace=True)
    df_rev.rename(columns=rev_rename_map, inplace=True)

    # 3. Merge Dataframes (Intersection of Run_IDs)
    merged = pd.merge(df_anec, df_rev, on='Run_ID', how='inner')
    
    # 4. Calculate Combined Score & Rank
    merged['Total_Score'] = merged['Score_Anechoic'] + merged['Score_Reverb']
    top_1000 = merged.sort_values(by='Total_Score', ascending=False).head(1000)
    
    selected_ids = top_1000['Run_ID'].tolist()
    print(f"Selected Top 1000 IDs (Score range: {top_1000['Total_Score'].min():.4f} - {top_1000['Total_Score'].max():.4f})")

    # 5. Create Destination Structure
    dest_results = os.path.join(dest_root, 'results')
    dest_simulated = os.path.join(dest_root, 'simulated')
    
    if os.path.exists(dest_root):
        print(f"Note: '{dest_root}' already exists. Merging into it...")
    
    os.makedirs(dest_results, exist_ok=True)
    os.makedirs(dest_simulated, exist_ok=True)

    # 6. Copy Folders
    success_count = 0
    missing_paths = []

    for run_id in selected_ids:
        # Check if ID is string (e.g., 'batch_test_3587') or int (3587)
        folder_name = str(run_id)
        
        # --- Copy Results Folder ---
        src_res = os.path.join(src_results_dir, folder_name)
        dst_res = os.path.join(dest_results, folder_name)
        
        if os.path.exists(src_res):
            if os.path.exists(dst_res):
                shutil.rmtree(dst_res) 
            shutil.copytree(src_res, dst_res)
        else:
            missing_paths.append(src_res)

        # --- Copy Simulated Folder ---
        src_sim = os.path.join(src_simulated_dir, folder_name)
        dst_sim = os.path.join(dest_simulated, folder_name)
        
        if os.path.exists(src_sim):
            if os.path.exists(dst_sim):
                shutil.rmtree(dst_sim) 
            shutil.copytree(src_sim, dst_sim)
        else:
            missing_paths.append(src_sim)
            
        success_count += 1

    print("-" * 30)
    print(f"Extraction Complete.")
    print(f"Processed {success_count} IDs.")
    if missing_paths:
        # Only showing first missing example to avoid clutter
        print(f"WARNING: {len(missing_paths)} source folders were missing. Example: {missing_paths[0]}")
    print(f"Files stored in: {os.path.abspath(dest_root)}")

    # 7. Save the enriched Metadata CSV
    # Reordering columns for readability: ID, Total Score, Anechoic Metrics, Reverb Metrics
    
    # Identify the columns we created
    anec_metric_cols = [c for c in top_1000.columns if 'Anechoic_' in c]
    rev_metric_cols  = [c for c in top_1000.columns if 'Reverb_' in c]
    
    final_cols = ['Run_ID', 'Total_Score'] + anec_metric_cols + rev_metric_cols
    
    # Save
    top_1000[final_cols].to_csv(os.path.join(dest_root, 'top_1000_metadata.csv'), index=False)
    print("Saved extended metadata to 'top_1000_metadata.csv'")

if __name__ == "__main__":
    extract_top_1000()