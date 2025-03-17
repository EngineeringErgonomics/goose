#!/usr/bin/env python3
"""
Script to analyze restaurant_research evaluation results from benchmark directories.

This script finds all benchmark-[provider]-[model] directories, extracts metrics
from restaurant_research eval_result.json files, and outputs a CSV summary.
"""

import os
import json
from typing import Dict, Any, List
from benchmark_utils import (
    extract_metric_value,
    extract_standard_metrics,
    analyze_benchmark_results,
    create_argparser,
    load_output_file,
    evaluate_with_openai
)

# System prompt for OpenAI evaluation of restaurant research
RESTAURANT_RESEARCH_PROMPT = """You are evaluating an AI assistant's response to a restaurant research task. The instructions were:

'Search the internet for and provide a current, detailed list of the best Sichuanese restaurants specifically 
in the East Village neighborhood of NYC. Format your response in Markdown using bullet points (either - or *) 
for each restaurant. For each restaurant include:
- Restaurant name and what they're known for
- Signature dishes
- Atmosphere/setting
- Any relevant details about reservations or dining experience
- What distinguishes them from others

Present the information in order of significance or quality. Focus specifically on Sichuanese establishments, 
not general Chinese restaurants.'

Does this response appropriately answer the query? It should list Sichuanese restaurants in NYC's East Village, 
including the details requested, and formatted with bullet points."""

def load_and_analyze_results(file_path: str) -> Dict[str, Any]:
    """Load and analyze a single eval_result.json file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metrics = data.get("metrics", [])
        
        # Extract key metrics from eval_result.json
        used_fetch_tool = extract_metric_value(metrics, "used_fetch_tool")
        valid_markdown_format = extract_metric_value(metrics, "valid_markdown_format")
        bullet_point_count = extract_metric_value(metrics, "bullet_point_count")
        
        # Get standard metrics (tokens, tool calls, execution time)
        standard_metrics = extract_standard_metrics(metrics)
        
        # Get the directory containing the eval_result.json file
        dir_path = os.path.dirname(file_path)
        
        # Evaluate with OpenAI if output file exists
        openai_evaluation = 0
        response_text = load_output_file(dir_path, "restaurant_research_output.txt")
        if response_text:
            print(f"Evaluating output from {dir_path}/restaurant_research_output.txt")
            openai_evaluation = evaluate_with_openai(response_text, RESTAURANT_RESEARCH_PROMPT)
        
        # Calculate correctness score (sum of two boolean metrics plus OpenAI score)
        correctness_score = (used_fetch_tool or False) + (valid_markdown_format or False) + openai_evaluation
        
        # Determine if run was successful (correctness_score of 4 means all three criteria were met)
        correct_results = correctness_score == 4
        
        return {
            "correct_results": correct_results,
            "correctness_score": correctness_score/4, # Normalize to a scale of 0-1
            "used_fetch_tool": used_fetch_tool or False,
            "valid_markdown_format": valid_markdown_format or False,
            "openai_evaluation": openai_evaluation,
            "bullet_point_count": bullet_point_count or 0,
            **standard_metrics
        }
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return {}

def calculate_additional_stats(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate additional statistics specific to restaurant research."""
    if not runs:
        return {
            "avg_correctness_score": 0,
            "fetch_success_rate": 0,
            "markdown_success_rate": 0,
            "openai_success_rate": 0,
            "avg_bullets_correct": 0,
            "avg_bullets_all": 0
        }
    
    # Get average correctness score
    avg_correctness_score = sum(run.get("correctness_score", 0) for run in runs) / len(runs)
    
    # Count runs that pass each individual criteria
    fetch_success = sum(1 for run in runs if run.get("used_fetch_tool") is True)
    markdown_success = sum(1 for run in runs if run.get("valid_markdown_format") is True)
    openai_success = sum(1 for run in runs if run.get("openai_evaluation", 0) > 0)
    
    # Calculate bullet point metrics
    successful_runs = [run for run in runs if run.get("correct_results") is True]
    n_successful_runs = len(successful_runs)
    
    avg_bullets_correct = 0
    if successful_runs:
        avg_bullets_correct = sum(run.get("bullet_point_count", 0) for run in successful_runs) / n_successful_runs
    
    avg_bullets_all = sum(run.get("bullet_point_count", 0) for run in runs) / len(runs)
    
    return {
        "avg_correctness_score": avg_correctness_score,
        "fetch_success_rate": fetch_success / len(runs),
        "markdown_success_rate": markdown_success / len(runs),
        "openai_success_rate": openai_success / len(runs),
        "avg_bullets_correct": avg_bullets_correct,
        "avg_bullets_all": avg_bullets_all
    }

def main():
    parser = create_argparser("restaurant_research", "restaurant-research-analysis.csv")
    args = parser.parse_args()
    
    analyze_benchmark_results(
        base_dir=args.base_dir,
        eval_name="restaurant_research",
        results_processor=load_and_analyze_results,
        output_csv=args.output,
        stats_preprocessor=calculate_additional_stats
    )

if __name__ == "__main__":
    main()