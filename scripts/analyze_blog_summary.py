#!/usr/bin/env python3
"""
Script to analyze blog_summary evaluation results from benchmark directories.

This script finds all benchmark-[provider]-[model] directories, extracts metrics
from blog_summary eval_result.json files, and outputs a CSV summary.
"""

import os
import json
import glob
import csv
import re
from openai import OpenAI
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

def find_blog_summary_results(benchmark_dir: str) -> List[str]:
    """Find all eval_result.json files for blog_summary evaluation."""
    pattern = os.path.join(benchmark_dir, "**", "vibes", "blog_summary", "eval_result.json")
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

def evaluate_with_openai(response_text: str) -> int:
    """
    Use OpenAI to evaluate the quality of the blog summary response.
    
    Args:
        response_text: The text content of the AI's response
        
    Returns:
        int: Score of 0, 1, or 2 (0 = totally wrong, 1 = partially correct, 2 = perfect answer)
    """
    try:
        # Get OpenAI API key from environment
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not found in environment variables")
            return 0
        
        # Create OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Create the prompt for evaluation
        prompt = f"""You are evaluating an AI assistant's response to a blog summary task. The instructions were:

'What are the top 5 most counterintuitive insights from this blog post? Format your response in Markdown with 5 numbered points (1. 2. 3. 4. 5.) https://huyenchip.com/2025/01/07/agents.html'

Here is the AI assistant's response:

{response_text}

Does this response appropriately answer the query? It should list the top 5 most counterintuitive insights from the blog post in a Markdown numbered list format.

Evaluate with a score of 0, 1, or 2 (0 = totally wrong, 1 = partially correct, 2 = perfect answer). Your response should be in json format with fields "reasoning" and "score". Output as a JSON parseable raw string "{{"reasoning": "", "score": xxx}}" do not output markdown."""
        
        # Make the API request using the OpenAI SDK
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.8
        )
        
        # Extract the model's response
        assistant_response = response.choices[0].message.content
        print(assistant_response)
        
        # Parse JSON response to extract score
        try:
            import json
            response_json = json.loads(assistant_response)
            score = int(response_json.get("score", 0))
            print(f"OpenAI evaluation score: {score}")
            return score
        except Exception as e:
            print(f"Error parsing OpenAI evaluation response: {e}")
            # Fallback to the old method if parsing fails
            is_correct = "1" in assistant_response and "0" not in assistant_response
            print(f"Fallback evaluation result: {'Passed' if is_correct else 'Failed'}")
            return 1 if is_correct else 0
    
    except Exception as e:
        print(f"Error during OpenAI evaluation: {e}")
        return 0

def load_and_analyze_results(file_path: str) -> Dict[str, Any]:
    """Load and analyze a single eval_result.json file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metrics = data.get("metrics", [])
        
        # Extract key metrics from eval_result.json
        used_fetch_tool = extract_metric_value(metrics, "used_fetch_tool")
        valid_markdown_format = extract_metric_value(metrics, "valid_markdown_format")
        total_tokens = extract_metric_value(metrics, "total_tokens")
        total_tool_calls = extract_metric_value(metrics, "total_tool_calls")
        execution_time = extract_metric_value(metrics, "prompt_execution_time_seconds")
        
        # Get the directory containing the eval_result.json file
        dir_path = os.path.dirname(file_path)
        
        # Look for blog_summary_output.txt file
        output_file = os.path.join(dir_path, "blog_summary_output.txt")
        
        # Evaluate with OpenAI if output file exists
        openai_evaluation = 0
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    response_text = f.read()
                
                print(f"Evaluating output from {output_file}")
                openai_evaluation = evaluate_with_openai(response_text)
            except Exception as e:
                print(f"Error reading or evaluating {output_file}: {str(e)}")
        else:
            print(f"Warning: blog_summary_output.txt not found in {dir_path}")
        
        # Calculate correctness score (sum of two boolean metrics plus OpenAI score)
        correctness_score = (used_fetch_tool or False) + (valid_markdown_format or False) + openai_evaluation
        
        # Determine if run was successful (correctness_score of 4 means all criteria were met at the highest level)
        correct_results = correctness_score >= 3 and openai_evaluation == 2
        
        return {
            "correct_results": correct_results,
            "correctness_score": correctness_score,
            "used_fetch_tool": used_fetch_tool or False,
            "valid_markdown_format": valid_markdown_format or False,
            "openai_evaluation": openai_evaluation,
            "total_tokens": total_tokens or 0,
            "total_tool_calls": total_tool_calls or 0,
            "prompt_execution_time_seconds": execution_time or 0,
        }
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return {}

def analyze_blog_summary(base_dir: str, output_csv: str) -> None:
    """Analyze all blog_summary results and write to CSV."""
    benchmark_dirs = find_benchmark_directories(base_dir)
    results_by_provider_model = defaultdict(list)
    
    print(f"Found {len(benchmark_dirs)} benchmark directories")
    
    for benchmark_dir in benchmark_dirs:
        provider, model = parse_provider_model(benchmark_dir)
        key = (provider, model)
        
        summary_results = find_blog_summary_results(benchmark_dir)
        if summary_results:
            print(f"Found {len(summary_results)} blog_summary results for {provider}-{model}")
            
            for result_file in summary_results:
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
        
        # Get average correctness score
        avg_correctness_score = sum(run.get("correctness_score", 0) for run in runs) / len(runs) if runs else 0
        
        # Count runs that pass each individual criteria
        fetch_success = sum(1 for run in runs if run.get("used_fetch_tool") is True)
        markdown_success = sum(1 for run in runs if run.get("valid_markdown_format") is True)
        openai_success = sum(1 for run in runs if run.get("openai_evaluation", 0) > 0)
        
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
        
        # Calculate average openai evaluation score
        avg_openai_score = sum(run.get("openai_evaluation", 0) for run in runs) / len(runs) if runs else 0
        
        csv_rows.append({
            "provider": provider,
            "model": model,
            "best_run": best_run,
            "n_correct_runs": n_correct_runs,
            "total_runs": len(runs),
            "success_rate": n_correct_runs / len(runs) if runs else 0,
            "avg_correctness_score": avg_correctness_score,
            "fetch_success_rate": fetch_success / len(runs) if runs else 0,
            "markdown_success_rate": markdown_success / len(runs) if runs else 0,
            "openai_success_rate": openai_success / len(runs) if runs else 0,
            "avg_openai_score": avg_openai_score,
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
        
        print(f"Blog Summary analysis complete. Results saved to {output_csv}")
    else:
        print("No blog_summary evaluation results found")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze blog_summary benchmark results')
    parser.add_argument('--base-dir', default='.', 
                        help='Base directory containing benchmark-provider-model directories')
    parser.add_argument('--output', default='blog-summary-analysis.csv',
                        help='Output CSV file for analysis results')
    
    args = parser.parse_args()
    analyze_blog_summary(args.base_dir, args.output)

if __name__ == "__main__":
    main()