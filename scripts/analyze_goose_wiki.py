#!/usr/bin/env python3
"""
Script to analyze goose_wiki evaluation results from benchmark directories.

This script finds all benchmark-[provider]-[model] directories, extracts metrics
from goose_wiki eval_result.json files, and outputs a CSV summary.
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
        used_write_tool = extract_metric_value(metrics, "used_write_tool")
        valid_implementation = extract_metric_value(metrics, "valid_implementation")
        
        # Get standard metrics (tokens, tool calls, execution time)
        standard_metrics = extract_standard_metrics(metrics)
        
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
            **standard_metrics
        }
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return {}

def calculate_additional_stats(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate additional statistics specific to goose wiki."""
    if not runs:
        return {
            "implementation_success_rate": 0,
            "write_tool_success_rate": 0
        }
    
    # Count runs by individual metrics
    write_tool_count = sum(1 for run in runs if run.get("used_write_tool") is True)
    implementation_count = sum(1 for run in runs if run.get("valid_implementation") is True)
    
    return {
        "write_tool_success_rate": write_tool_count / len(runs),
        "implementation_success_rate": implementation_count / len(runs)
    }

def main():
    parser = create_argparser("goose_wiki", "goose-wiki-analysis.csv")
    args = parser.parse_args()
    
    analyze_benchmark_results(
        base_dir=args.base_dir,
        eval_name="goose_wiki",
        results_processor=load_and_analyze_results,
        output_csv=args.output,
        stats_preprocessor=calculate_additional_stats
    )

if __name__ == "__main__":
    main()