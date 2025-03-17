#!/usr/bin/env python3
"""
Script to analyze blog_summary evaluation results from benchmark directories.

This script finds all benchmark-[provider]-[model] directories, extracts metrics
from blog_summary eval_result.json files, and outputs a CSV summary.
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

# System prompt for OpenAI evaluation of blog summaries
BLOG_SUMMARY_PROMPT = """You are evaluating an AI assistant's response to a blog summary task. The instructions were:

'What are the top 5 most counterintuitive insights from this blog post? Format your response in Markdown with 5 numbered points (1. 2. 3. 4. 5.) https://huyenchip.com/2025/01/07/agents.html'

Does this response appropriately answer the query? It should list the top 5 most counterintuitive insights from the blog post in a Markdown numbered list format."""

def load_and_analyze_results(file_path: str) -> Dict[str, Any]:
    """Load and analyze a single eval_result.json file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metrics = data.get("metrics", [])
        
        # Extract key metrics from eval_result.json
        used_fetch_tool = extract_metric_value(metrics, "used_fetch_tool")
        valid_markdown_format = extract_metric_value(metrics, "valid_markdown_format")
        
        # Get standard metrics (tokens, tool calls, execution time)
        standard_metrics = extract_standard_metrics(metrics)
        
        # Get the directory containing the eval_result.json file
        dir_path = os.path.dirname(file_path)
        
        # Evaluate with OpenAI if output file exists
        openai_evaluation = 0
        response_text = load_output_file(dir_path, "blog_summary_output.txt")
        if response_text:
            print(f"Evaluating output from {dir_path}/blog_summary_output.txt")
            openai_evaluation = evaluate_with_openai(response_text, BLOG_SUMMARY_PROMPT)
        
        # Calculate correctness score (sum of two boolean metrics plus OpenAI score)
        correctness_score = (used_fetch_tool or False) + (valid_markdown_format or False) + openai_evaluation
        
        # Determine if run was successful (correctness_score of 4 means all criteria were met at the highest level)
        correct_results = correctness_score == 4
        
        return {
            "correct_results": correct_results,
            "correctness_score": correctness_score,
            "used_fetch_tool": used_fetch_tool or False,
            "valid_markdown_format": valid_markdown_format or False,
            "openai_evaluation": openai_evaluation,
            **standard_metrics
        }
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return {}

def calculate_additional_stats(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate additional statistics specific to blog summary."""
    if not runs:
        return {
            "avg_correctness_score": 0,
            "fetch_success_rate": 0,
            "markdown_success_rate": 0,
            "openai_success_rate": 0,
            "avg_openai_score": 0
        }
    
    # Get average correctness score
    avg_correctness_score = sum(run.get("correctness_score", 0) for run in runs) / len(runs)
    
    # Count runs that pass each individual criteria
    fetch_success = sum(1 for run in runs if run.get("used_fetch_tool") is True)
    markdown_success = sum(1 for run in runs if run.get("valid_markdown_format") is True)
    openai_success = sum(1 for run in runs if run.get("openai_evaluation", 0) > 0)
    
    # Calculate average openai evaluation score
    avg_openai_score = sum(run.get("openai_evaluation", 0) for run in runs) / len(runs)
    
    return {
        "avg_correctness_score": avg_correctness_score,
        "fetch_success_rate": fetch_success / len(runs),
        "markdown_success_rate": markdown_success / len(runs),
        "openai_success_rate": openai_success / len(runs),
        "avg_openai_score": avg_openai_score
    }

def main():
    parser = create_argparser("blog_summary", "blog-summary-analysis.csv")
    args = parser.parse_args()
    
    analyze_benchmark_results(
        base_dir=args.base_dir,
        eval_name="blog_summary",
        results_processor=load_and_analyze_results,
        output_csv=args.output,
        stats_preprocessor=calculate_additional_stats
    )

if __name__ == "__main__":
    main()