#!/usr/bin/env python3
"""
Script to analyze goose_wiki evaluation results from benchmark directories.

This script finds all benchmark-[provider]-[model] directories, extracts metrics
from goose_wiki eval_result.json files, and outputs a CSV summary.
"""

import os
import json
import glob
import csv
import re
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional

def find_benchmark_directories(base_dir: str) -> List[str]:
    """Find all benchmark directories matching the pattern benchmark-provider-model."""
    if not os.path.isdir(base_dir):
        print(f"Error: Base directory {base_dir} does not exist")
        return []
    
    # Find all directories matching the pattern benchmark-{provider}-{model}
    benchmark_dirs = []
    for item in os.listdir(base_dir):
        if item.startswith('benchmark-') and os.path.isdir(os.path.join(base_dir, item)):
            benchmark_dirs.append(os.path.join(base_dir, item))
    
    return benchmark_dirs

def find_goose_wiki_results(benchmark_dir: str) -> List[str]:
    """Find all eval_result.json files for goose_wiki evaluation."""
    pattern = os.path.join(benchmark_dir, "**", "vibes", "goose_wiki", "eval_result.json")
    return glob.glob(pattern, recursive=True)

def parse_provider_model(dir_path: str) -> Tuple[str, str]:
    """Extract provider and model from the benchmark directory name."""
    dir_name = os.path.basename(dir_path)
    match = re.match(r'benchmark-([^-]+)-(.+)', dir_name)
    
    if match:
        provider = match.group(1)
        model = match.group(2)
        return provider, model
    else:
        # Fallback if the format doesn't match expected pattern
        parts = dir_name.split('-', 2)
        if len(parts) >= 3:
            return parts[1], parts[2]
        else:
            return "unknown", "unknown"

def extract_metric_value(metrics: List, metric_name: str) -> Any:
    """Extract a specific metric value from metrics list."""
    for metric in metrics:
        if isinstance(metric, list) and len(metric) >= 2 and metric[0] == metric_name:
            # Handle different metric formats
            value = metric[1]
            if isinstance(value, dict):
                # Check for different value types
                if "Boolean" in value:
                    return value["Boolean"]
                elif "Integer" in value:
                    return value["Integer"]
                elif "Float" in value:
                    return value["Float"]
                elif "String" in value:
                    return value["String"]
                else:
                    return next(iter(value.values()), None)
            else:
                return value
    return None

def load_and_analyze_results(file_path: str) -> Dict[str, Any]:
    """Load and analyze a single eval_result.json file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metrics = data.get("metrics", [])
        
        # Extract key metrics
        used_write_tool = extract_metric_value(metrics, "used_write_tool")
        valid_implementation = extract_metric_value(metrics, "valid_implementation")
        total_tokens = extract_metric_value(metrics, "total_tokens")
        total_tool_calls = extract_metric_value(metrics, "total_tool_calls")
        execution_time = extract_metric_value(metrics, "prompt_execution_time_seconds")
        
        # Determine correctness based on the rules:
        # 1. If used_write_tool is False, the result is immediately False
        # 2. If used_write_tool is True, check valid_implementation to also be True
        correct_results = False
        if isinstance(used_write_tool, bool) and used_write_tool:
            if isinstance(valid_implementation, bool) and valid_implementation:
                correct_results = True
        
        return {
            "correct_results": correct_results,
            "used_write_tool": used_write_tool,
            "valid_implementation": valid_implementation,
            "total_tokens": total_tokens,
            "total_tool_calls": total_tool_calls,
            "prompt_execution_time_seconds": execution_time,
        }
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return {}

def analyze_goose_wiki(base_dir: str, output_csv: str) -> None:
    """Analyze all goose_wiki results and write to CSV."""
    benchmark_dirs = find_benchmark_directories(base_dir)
    results_by_provider_model = defaultdict(list)
    
    print(f"Found {len(benchmark_dirs)} benchmark directories")
    
    for benchmark_dir in benchmark_dirs:
        provider, model = parse_provider_model(benchmark_dir)
        key = (provider, model)
        
        wiki_results = find_goose_wiki_results(benchmark_dir)
        if wiki_results:
            print(f"Found {len(wiki_results)} goose_wiki results for {provider}-{model}")
            
            for result_file in wiki_results:
                result_data = load_and_analyze_results(result_file)
                if result_data:
                    results_by_provider_model[key].append(result_data)
    
    # Generate CSV data
    csv_rows = []
    
    for (provider, model), runs in results_by_provider_model.items():
        # Count correct runs
        correct_runs = [run for run in runs if run.get("correct_results") is True]
        n_correct_runs = len(correct_runs)
        best_run = n_correct_runs > 0
        
        # Calculate averages for correct runs
        if correct_runs:
            avg_tokens_correct = sum(run.get("total_tokens", 0) for run in correct_runs) / n_correct_runs
            avg_tool_calls_correct = sum(run.get("total_tool_calls", 0) for run in correct_runs) / n_correct_runs
            avg_exec_time_correct = sum(run.get("prompt_execution_time_seconds", 0) for run in correct_runs) / n_correct_runs
        else:
            avg_tokens_correct = 0
            avg_tool_calls_correct = 0
            avg_exec_time_correct = 0
            
        # Calculate averages for all runs
        avg_tokens_all = sum(run.get("total_tokens", 0) for run in runs) / len(runs) if runs else 0
        avg_tool_calls_all = sum(run.get("total_tool_calls", 0) for run in runs) / len(runs) if runs else 0
        avg_exec_time_all = sum(run.get("prompt_execution_time_seconds", 0) for run in runs) / len(runs) if runs else 0
        
        csv_rows.append({
            "provider": provider,
            "model": model,
            "best_run": best_run,
            "n_correct_runs": n_correct_runs,
            "total_runs": len(runs),
            "success_rate": n_correct_runs / len(runs) if runs else 0,
            "avg_tokens_correct": avg_tokens_correct,
            "avg_tool_calls_correct": avg_tool_calls_correct,
            "avg_exec_time_correct": avg_exec_time_correct,
            "avg_tokens_all": avg_tokens_all,
            "avg_tool_calls_all": avg_tool_calls_all, 
            "avg_exec_time_all": avg_exec_time_all
        })
    
    # Sort by provider, then model
    csv_rows.sort(key=lambda x: (x["provider"], x["model"]))
    
    # Write to CSV
    if csv_rows:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = list(csv_rows[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        
        print(f"Goose Wiki analysis complete. Results saved to {output_csv}")
    else:
        print("No goose_wiki evaluation results found")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze goose_wiki benchmark results')
    parser.add_argument('--base-dir', default='.', 
                        help='Base directory containing benchmark-provider-model directories')
    parser.add_argument('--output', default='goose-wiki-analysis.csv',
                        help='Output CSV file for analysis results')
    
    args = parser.parse_args()
    analyze_goose_wiki(args.base_dir, args.output)

if __name__ == "__main__":
    main()