#!/usr/bin/env python3
"""
Script to analyze squirrel_census evaluation results from benchmark directories.

This script finds all benchmark-[provider]-[model] directories, extracts metrics
from squirrel_census eval_result.json files, and outputs a CSV summary.
"""

import json
from typing import Dict, Any, List
from benchmark_utils import (
    extract_metric_value,
    extract_standard_metrics,
    analyze_benchmark_results,
    create_argparser
)

def load_and_analyze_results(file_path: str) -> Dict[str, Any]:
    """Load and analyze a single eval_result.json file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metrics = data.get("metrics", [])
        
        # Extract key metrics
        wrote_script = extract_metric_value(metrics, "wrote_script")
        ran_script = extract_metric_value(metrics, "ran_script")
        correct_results = extract_metric_value(metrics, "correct_results")
        
        # Get standard metrics (tokens, tool calls, execution time)
        standard_metrics = extract_standard_metrics(metrics)
        
        # Determine overall success:
        # The result is correct if wrote_script, ran_script, and correct_results are all true
        is_successful = False
        if isinstance(wrote_script, bool) and wrote_script and \
           isinstance(ran_script, bool) and ran_script and \
           isinstance(correct_results, bool) and correct_results:
            is_successful = True
        
        return {
            "is_successful": is_successful,
            "wrote_script": wrote_script,
            "ran_script": ran_script,
            "correct_results": correct_results,
            **standard_metrics
        }
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return {}

def calculate_additional_stats(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate additional statistics specific to squirrel census."""
    if not runs:
        return {
            "wrote_script_rate": 0,
            "ran_script_rate": 0,
            "correct_results_rate": 0
        }
    
    # Count runs by individual metrics
    wrote_script_count = sum(1 for run in runs if run.get("wrote_script") is True)
    ran_script_count = sum(1 for run in runs if run.get("ran_script") is True)
    correct_results_count = sum(1 for run in runs if run.get("correct_results") is True)
    
    return {
        "wrote_script_rate": wrote_script_count / len(runs),
        "ran_script_rate": ran_script_count / len(runs),
        "correct_results_rate": correct_results_count / len(runs)
    }

def main():
    parser = create_argparser("squirrel_census", "squirrel-census-analysis.csv")
    args = parser.parse_args()
    
    analyze_benchmark_results(
        base_dir=args.base_dir,
        eval_name="squirrel_census",
        results_processor=load_and_analyze_results,
        output_csv=args.output,
        stats_preprocessor=calculate_additional_stats
    )

if __name__ == "__main__":
    main()