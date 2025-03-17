#!/usr/bin/env python3
"""
Utility module for benchmark analysis scripts.

This module contains common functions used across various benchmark analysis scripts
to extract metrics, process results, and generate standardized reports.
"""

import os
import json
import glob
import csv
import re
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional, Callable

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

def find_eval_results(benchmark_dir: str, eval_name: str) -> List[str]:
    """Find all eval_result.json files for the specified evaluation."""
    
    # Get all subdirectories in the benchmark directory
    all_subdirs = [d for d in os.listdir(benchmark_dir) if os.path.isdir(os.path.join(benchmark_dir, d))]
    
    # Filter for timestamp-like directories (allow more flexible formats)
    timestamp_dirs = []
    for d in all_subdirs:
        # Check if directory name contains date-like elements
        if (d.startswith("20") and ("-" in d or ":" in d)) or re.match(r"\d{4}-\d{2}-\d{2}", d):
            timestamp_dirs.append(os.path.join(benchmark_dir, d))
    
    # Define specific category dirs to check under each timestamp
    category_dirs = ["vibes", "squirrel"]
    
    results = []
    
    # Check each timestamp directory for the evaluation in all category dirs
    for ts_dir in timestamp_dirs:
        for category in category_dirs:
            # Build the specific path for this timestamp/category/eval_name
            eval_path = os.path.join(ts_dir, category, eval_name, "eval_result.json")
            
            # Check if the file exists (no globbing needed)
            if os.path.exists(eval_path):
                results.append(eval_path)
    
    # Debug: Show what we found
    print(f"DEBUG: Found {len(results)} {eval_name} evaluation results in {len(timestamp_dirs)} timestamp directories")
    print(f"DEBUG: Timestamp directories checked: {timestamp_dirs[:3] if timestamp_dirs else 'none'}")
    
    # Print first few results to understand the pattern
    for i, result in enumerate(results[:3]):
        rel_path = os.path.relpath(result, benchmark_dir)
        print(f"  - Result {i+1}: {rel_path}")
    
    return results

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

def extract_standard_metrics(metrics: List) -> Dict[str, Any]:
    """Extract common metrics found in most eval results."""
    return {
        "total_tokens": extract_metric_value(metrics, "total_tokens") or 0,
        "total_tool_calls": extract_metric_value(metrics, "total_tool_calls") or 0,
        "prompt_execution_time_seconds": extract_metric_value(metrics, "prompt_execution_time_seconds") or 0
    }

def load_output_file(result_dir: str, filename: str) -> Optional[str]:
    """Load content from an output file if it exists."""
    output_file = os.path.join(result_dir, filename)
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading {output_file}: {str(e)}")
    else:
        print(f"Warning: {filename} not found in {result_dir}")
    return None

def calculate_run_statistics(runs: List[Dict[str, Any]], success_field: str = "correct_results") -> Dict[str, Any]:
    """Calculate common statistics for a set of runs."""
    if not runs:
        return {
            "best_run": False,
            "n_successful_runs": 0,
            "total_runs": 0,
            "success_rate": 0,
            "avg_tokens_all": 0,
            "avg_tool_calls_all": 0,
            "avg_exec_time_all": 0,
            "avg_tokens_successful": 0,
            "avg_tool_calls_successful": 0,
            "avg_exec_time_successful": 0
        }
    
    # Count successful runs
    successful_runs = [run for run in runs if run.get(success_field) is True]
    n_successful_runs = len(successful_runs)
    best_run = n_successful_runs > 0
    
    # Calculate averages for successful runs
    if successful_runs:
        avg_tokens_successful = sum(run.get("total_tokens", 0) for run in successful_runs) / n_successful_runs
        avg_tool_calls_successful = sum(run.get("total_tool_calls", 0) for run in successful_runs) / n_successful_runs
        avg_exec_time_successful = sum(run.get("prompt_execution_time_seconds", 0) for run in successful_runs) / n_successful_runs
    else:
        avg_tokens_successful = 0
        avg_tool_calls_successful = 0
        avg_exec_time_successful = 0
        
    # Calculate averages for all runs
    avg_tokens_all = sum(run.get("total_tokens", 0) for run in runs) / len(runs)
    avg_tool_calls_all = sum(run.get("total_tool_calls", 0) for run in runs) / len(runs)
    avg_exec_time_all = sum(run.get("prompt_execution_time_seconds", 0) for run in runs) / len(runs)
    
    return {
        "best_run": best_run,
        "n_successful_runs": n_successful_runs,
        "total_runs": len(runs),
        "success_rate": n_successful_runs / len(runs),
        "avg_tokens_successful": avg_tokens_successful,
        "avg_tool_calls_successful": avg_tool_calls_successful,
        "avg_exec_time_successful": avg_exec_time_successful,
        "avg_tokens_all": avg_tokens_all,
        "avg_tool_calls_all": avg_tool_calls_all,
        "avg_exec_time_all": avg_exec_time_all
    }

def analyze_benchmark_results(
    base_dir: str, 
    eval_name: str,
    results_processor: Callable[[str], Dict[str, Any]],
    output_csv: str,
    stats_preprocessor: Callable[[List[Dict[str, Any]]], Dict[str, Any]] = None
) -> None:
    """Generic function to analyze benchmark results for a specified evaluation type."""
    benchmark_dirs = find_benchmark_directories(base_dir)
    results_by_provider_model = defaultdict(list)
    
    print(f"\n=== Analyzing {eval_name} ===")
    print(f"Found {len(benchmark_dirs)} benchmark directories")
    
    # Count total result files found across all providers
    total_result_files = 0
    
    for benchmark_dir in benchmark_dirs:
        provider, model = parse_provider_model(benchmark_dir)
        key = (provider, model)
        
        print(f"\nChecking {provider}-{model}:")
        
        # Check the timestamp directories
        timestamp_dirs = [d for d in glob.glob(os.path.join(benchmark_dir, "*")) if os.path.isdir(d)]
        print(f"  Found {len(timestamp_dirs)} timestamp directories")
        for i, ts_dir in enumerate(timestamp_dirs[:3]):  # Show first 3
            print(f"    - {os.path.basename(ts_dir)}")
        if len(timestamp_dirs) > 3:
            print(f"    - ... and {len(timestamp_dirs) - 3} more")
            
        # Find results for this evaluation
        eval_results = find_eval_results(benchmark_dir, eval_name)
        total_result_files += len(eval_results)
        
        if eval_results:
            print(f"\n  Processing {len(eval_results)} {eval_name} results for {provider}-{model}")
            
            for result_file in eval_results:
                print(f"    Processing: {result_file}")
                result_data = results_processor(result_file)
                if result_data:
                    results_by_provider_model[key].append(result_data)
                    print(f"      ✅ Success")
                else:
                    print(f"      ❌ Failed to process result file")
    
    # Generate CSV data
    csv_rows = []
    
    print("\n=== Summary by Provider/Model ===")
    for (provider, model), runs in results_by_provider_model.items():
        print(f"{provider}-{model}: {len(runs)} valid runs processed")
        
        # Get standard statistics
        stats = calculate_run_statistics(runs)
        
        # Create row with provider and model
        row = {
            "provider": provider,
            "model": model,
            **stats
        }
        
        # Allow custom stats preprocessing if provided
        if stats_preprocessor:
            custom_stats = stats_preprocessor(runs)
            row.update(custom_stats)
            
        csv_rows.append(row)
    
    # Sort by provider, then model
    csv_rows.sort(key=lambda x: (x["provider"], x["model"]))
    
    # Write to CSV
    if csv_rows:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = list(csv_rows[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        
        print(f"\n{eval_name.replace('_', ' ').title()} analysis complete. Results saved to {output_csv}")
        print(f"Total result files found: {total_result_files}")
        print(f"Total valid runs processed: {sum(len(runs) for runs in results_by_provider_model.values())}")
    else:
        print(f"\nNo {eval_name} evaluation results found")

def create_argparser(eval_name: str, default_output: str) -> Any:
    """Create a standard argument parser for benchmark analysis scripts."""
    import argparse
    
    parser = argparse.ArgumentParser(description=f'Analyze {eval_name} benchmark results')
    parser.add_argument('--base-dir', default='.', 
                        help='Base directory containing benchmark-provider-model directories')
    parser.add_argument('--output', default=default_output,
                        help='Output CSV file for analysis results')
    
    return parser

# OpenAI evaluation utility (used by both restaurant_research and blog_summary)
def evaluate_with_openai(response_text: str, system_prompt: str) -> int:
    """
    Use OpenAI to evaluate the quality of a response.
    
    Args:
        response_text: The text content of the AI's response
        system_prompt: The prompt to send to OpenAI
        
    Returns:
        int: Score of 0, 1, or 2 (0 = totally wrong, 1 = partially correct, 2 = perfect answer)
    """
    try:
        # First, try to import OpenAI and check for dependencies
        try:
            from openai import OpenAI
        except ImportError:
            print("ERROR: OpenAI package not installed. Run: pip install openai")
            return 0
        
        # Get OpenAI API key from environment
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY not found in environment variables")
            print("Please set your OpenAI API key: export OPENAI_API_KEY='your-api-key'")
            return 0
        
        print(f"OpenAI API Key found (length: {len(api_key)})")
        
        # Create OpenAI client
        client = OpenAI(api_key=api_key)
        print("OpenAI client created")
        
        # Truncate response_text if it's too long
        max_length = 12000  # Safe limit for context window
        if len(response_text) > max_length:
            print(f"Warning: Truncating response text from {len(response_text)} to {max_length} characters")
            response_text = response_text[:max_length] + "\n[... truncated ...]"
        
        # Create the prompt for evaluation
        prompt = f"""{system_prompt}

Here is the AI assistant's response:

{response_text}

Evaluate with a score of 0, 1, or 2 (0 = totally wrong, 1 = partially correct, 2 = perfect answer). Your response should be in json format with fields "reasoning" and "score". Output as a JSON parseable raw string "{{"reasoning": "", "score": xxx}}" do not output markdown."""
        
        print(f"Sending evaluation request to OpenAI (model: gpt-4o)...")
        
        # Make the API request using the OpenAI SDK
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5  # Lower temperature for more consistent responses
            )
            print("OpenAI response received")
        except Exception as api_error:
            print(f"ERROR with OpenAI API call: {api_error}")
            # Try fallback model if the first one fails
            try:
                print("Trying fallback to gpt-3.5-turbo model...")
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5
                )
                print("Fallback model response received")
            except Exception as fallback_error:
                print(f"ERROR with fallback model: {fallback_error}")
                return 0
        
        # Extract the model's response
        assistant_response = response.choices[0].message.content
        print(f"OpenAI response content: {assistant_response[:100]}...")
        
        # Parse JSON response to extract score
        try:
            import json
            
            # Try to find JSON object in the response if there's surrounding text
            import re
            json_match = re.search(r'\{.*\}', assistant_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                print(f"Extracted JSON: {json_str[:100]}...")
                response_json = json.loads(json_str)
            else:
                response_json = json.loads(assistant_response)
            
            score = int(response_json.get("score", 0))
            print(f"OpenAI evaluation score: {score}")
            return score
        except Exception as e:
            print(f"Error parsing OpenAI evaluation response: {e}")
            print(f"Raw response: {assistant_response}")
            
            # Fallback pattern matching if JSON parsing fails
            if "score: 2" in assistant_response or "score\":2" in assistant_response or '"score": 2' in assistant_response:
                print("Pattern matching fallback: Score 2")
                return 2
            elif "score: 1" in assistant_response or "score\":1" in assistant_response or '"score": 1' in assistant_response:
                print("Pattern matching fallback: Score 1")
                return 1
            elif "score: 0" in assistant_response or "score\":0" in assistant_response or '"score": 0' in assistant_response:
                print("Pattern matching fallback: Score 0") 
                return 0
            
            # Last resort fallback based on keywords
            is_correct = "perfect" in assistant_response.lower() or "complete" in assistant_response.lower()
            is_partial = "partial" in assistant_response.lower() or "incomplete" in assistant_response.lower()
            
            if is_correct:
                print("Keyword fallback: Score 2 (perfect/complete)")
                return 2
            elif is_partial:
                print("Keyword fallback: Score 1 (partial/incomplete)")
                return 1
            else:
                print("Keyword fallback: Score 0 (default)")
                return 0
    
    except Exception as e:
        print(f"Critical error during OpenAI evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 0