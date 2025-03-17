#!/usr/bin/env python3
"""
Run all benchmark analysis scripts in one go.

This script runs all five benchmark analysis scripts sequentially, 
allowing for batch processing of all evaluation types. It also combines
all individual CSV files into one comprehensive CSV for easier analysis.
"""

import argparse
import os
import subprocess
import sys
import csv
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict

# List of analysis scripts and their corresponding evaluation names
ANALYSIS_SCRIPTS = [
    ("analyze_blog_summary.py", "blog_summary"),
    ("analyze_flappy_bird.py", "flappy_bird"),
    ("analyze_goose_wiki.py", "goose_wiki"),
    ("analyze_restaurant_research.py", "restaurant_research"),
    ("analyze_squirrel_census.py", "squirrel_census")
]

def run_script(script_path: str, eval_name: str, base_dir: str, output_dir: str) -> Tuple[str, str, int]:
    """Run a single analysis script with the given parameters."""
    script_name = os.path.basename(script_path)
    output_name = eval_name + "-analysis.csv"
    output_path = os.path.join(output_dir, output_name)
    
    cmd = [sys.executable, script_path, "--base-dir", base_dir, "--output", output_path]
    print(f"Running: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate()
    
    if process.returncode == 0:
        print(f"✅ {script_name} completed successfully")
    else:
        print(f"❌ {script_name} failed with code {process.returncode}")
        print(f"Error: {stderr}")
    
    return script_name, output_path, process.returncode

def combine_csv_files(csv_files: List[Tuple[str, str]], output_path: str) -> bool:
    """
    Combine individual CSV files into one comprehensive CSV.
    
    Args:
        csv_files: List of tuples containing (eval_name, file_path)
        output_path: Path to save the combined CSV
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not csv_files:
        print("No files provided to combine")
        return False
        
    try:
        # Use pandas to read and combine CSV files
        all_data = []
        processed_files = 0
        
        print(f"\nProcessing {len(csv_files)} CSV files for combination:")
        for eval_name, file_path in csv_files:
            print(f"  - Processing {eval_name}: {file_path}")
            
            if not os.path.exists(file_path):
                print(f"    Warning: CSV file not found")
                continue
                
            try:
                # Read with error handling
                df = pd.read_csv(file_path)
                print(f"    Success: Read {len(df)} rows")
                
                # Simple validation - ensure it has expected columns
                if len(df.columns) < 3:
                    print(f"    Warning: CSV has only {len(df.columns)} columns, may be invalid")
                
                # Add evaluation name column
                df['evaluation'] = eval_name
                
                all_data.append(df)
                processed_files += 1
            except Exception as e:
                print(f"    Error reading file: {str(e)}")
                continue
        
        if not all_data:
            print("\nNo valid CSV files found to combine")
            return False
            
        print(f"\nSuccessfully processed {processed_files}/{len(csv_files)} files")
        
        # Combine all dataframes
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Combined into dataframe with {len(combined_df)} rows and {len(combined_df.columns)} columns")
        
        # Save the combined data
        combined_df.to_csv(output_path, index=False)
        print(f"Combined CSV created successfully: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error combining CSV files: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_all_analyses(base_dir: str, output_dir: str, parallel: bool = False) -> None:
    """Run all analysis scripts, either sequentially or in parallel."""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get full paths to all scripts
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_info = [(os.path.join(script_dir, script), eval_name) for script, eval_name in ANALYSIS_SCRIPTS]
    
    # Check that all scripts exist
    missing_scripts = [script for script, _ in script_info if not os.path.isfile(script)]
    if missing_scripts:
        print(f"Error: The following scripts are missing: {', '.join(missing_scripts)}")
        return
    
    results = []
    
    if parallel:
        # Run scripts in parallel using ThreadPoolExecutor
        print(f"Running {len(script_info)} analysis scripts in parallel...")
        with ThreadPoolExecutor(max_workers=len(script_info)) as executor:
            futures = [
                executor.submit(run_script, script, eval_name, base_dir, output_dir)
                for script, eval_name in script_info
            ]
            results = [future.result() for future in futures]
    else:
        # Run scripts sequentially
        print(f"Running {len(script_info)} analysis scripts sequentially...")
        for script, eval_name in script_info:
            results.append(run_script(script, eval_name, base_dir, output_dir))
    
    # Report results
    success_count = sum(1 for _, _, code in results if code == 0)
    print("\nAnalysis Summary:")
    print(f"✅ {success_count}/{len(results)} scripts completed successfully")
    
    # Combine CSV files from successful runs
    successful_files = []
    for script_name, output_path, code in results:
        if code == 0:
            # Find the corresponding eval_name for this script
            for script, eval_name in ANALYSIS_SCRIPTS:
                if os.path.basename(script) == script_name:
                    successful_files.append((eval_name, output_path))
                    break
    
    print(f"DEBUG: Successful files to combine: {successful_files}")
    
    combined_csv_path = os.path.join(output_dir, "all-evaluations.csv")
    combine_successful = combine_csv_files(successful_files, combined_csv_path)
    
    if success_count == len(results):
        print(f"\nAll analyses completed successfully. Results saved to: {output_dir}")
    else:
        failed_scripts = [script_name for script_name, _, code in results if code != 0]
        print(f"\nThe following scripts failed: {', '.join(failed_scripts)}")
    
    if combine_successful:
        print(f"\nCombined analysis file created: {combined_csv_path}")

def main():
    parser = argparse.ArgumentParser(description='Run all benchmark analysis scripts')
    parser.add_argument('--base-dir', default='.', 
                        help='Base directory containing benchmark-provider-model directories')
    parser.add_argument('--output-dir', default='./analysis-results',
                        help='Directory to save all analysis CSV files')
    parser.add_argument('--parallel', action='store_true',
                        help='Run analysis scripts in parallel (default: sequential)')
    
    args = parser.parse_args()
    run_all_analyses(args.base_dir, args.output_dir, args.parallel)

if __name__ == "__main__":
    main()