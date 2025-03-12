#!/usr/bin/env python3
"""
Analyze benchmark results from Goose evaluations.

This script reads eval_result.json files from benchmark-provider-model directories,
aggregates the data, and outputs in CSV or markdown format.
"""

import argparse
import csv
import datetime
import glob
import json
import os
import re
import sys
from collections import defaultdict
from typing import Dict, List, Any, Set, Tuple, Optional


def find_benchmark_directories(base_dir: str) -> List[str]:
    """Find all benchmark directories in the given base directory."""
    if not os.path.isdir(base_dir):
        print(f"Error: Base directory {base_dir} does not exist")
        sys.exit(1)
    
    # Find all directories matching the pattern benchmark-{provider}-{model}
    benchmark_dirs = []
    for item in os.listdir(base_dir):
        if item.startswith('benchmark-') and os.path.isdir(os.path.join(base_dir, item)):
            benchmark_dirs.append(os.path.join(base_dir, item))
    
    return benchmark_dirs


def find_eval_result_files(benchmark_dir: str) -> List[str]:
    """Find all eval_result.json files in the given benchmark directory."""
    pattern = os.path.join(benchmark_dir, "**", "eval_result.json")
    return glob.glob(pattern, recursive=True)


def parse_provider_model(dir_path: str) -> Tuple[str, str]:
    """Extract provider and model from the benchmark directory name."""
    dir_name = os.path.basename(dir_path)
    match = re.match(r'benchmark-(\w+)-(.+)', dir_name)
    
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


def parse_timestamp_from_path(file_path: str) -> str:
    """Extract timestamp from the eval_result.json file path."""
    # Path format: .../benchmark-provider-model/YYYY-MM-DD-HH:MM:SS/suite/eval/eval_result.json
    parts = file_path.split(os.sep)
    for part in parts:
        if re.match(r'\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2}', part):
            return part
    return ""


def parse_suite_eval_from_path(file_path: str) -> Tuple[str, str]:
    """Extract suite and evaluation names from the eval_result.json file path."""
    # Path format: .../benchmark-provider-model/timestamp/suite/eval/eval_result.json
    parts = file_path.split(os.sep)
    
    # Look for the timestamp to find our position
    timestamp_index = -1
    for i, part in enumerate(parts):
        if re.match(r'\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2}', part):
            timestamp_index = i
            break
    
    if timestamp_index >= 0 and timestamp_index + 2 < len(parts):
        suite = parts[timestamp_index + 1]
        eval_name = parts[timestamp_index + 2]
        return suite, eval_name
    else:
        return "unknown", "unknown"


def load_eval_result_file(filename: str) -> Dict[str, Any]:
    """Load and parse an eval_result.json file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading {filename}: {str(e)}")
        return {}


def get_common_metrics(results: List[Dict[str, Any]]) -> List[str]:
    """Get a list of metrics that appear in all benchmark results."""
    # First collect all metrics by suite and eval name
    all_metrics: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    
    for result in results:
        suite_name = result.get("suite", "unknown")
        eval_name = result.get("evaluation", "unknown")
        
        for metric_entry in result.get("metrics", []):
            if len(metric_entry) >= 2:
                metric_name = metric_entry[0]
                all_metrics[suite_name][eval_name].add(metric_name)
    
    # Find common metrics across all evaluations
    common_metrics = set()
    first = True
    
    for suite_name, evals in all_metrics.items():
        for eval_name, metrics in evals.items():
            if first:
                common_metrics = metrics
                first = False
            else:
                common_metrics &= metrics
    
    # Add some standard metrics we always want
    always_include = {
        "total_tokens", 
        "prompt_execution_time_seconds", 
        "total_tool_calls"
    }
    
    return sorted(list(common_metrics | (always_include & set(common_metrics))))


def extract_metric_value(metric_entry: List) -> Any:
    """Extract the value from a metric entry regardless of its type."""
    if len(metric_entry) < 2:
        return None
    
    value_container = metric_entry[1]
    if isinstance(value_container, dict):
        # Check for different value types
        if "Boolean" in value_container:
            return value_container["Boolean"]
        elif "Integer" in value_container:
            return value_container["Integer"]
        elif "Float" in value_container:
            return value_container["Float"]
        elif "String" in value_container:
            return value_container["String"]
        else:
            # If we can't determine the type, return the first value
            return next(iter(value_container.values()), None)
    else:
        return value_container  # If it's not a dict, return as is


def get_metric_value(evaluation: Dict[str, Any], metric_name: str) -> Optional[Any]:
    """Get a specific metric value from an evaluation."""
    for metric_entry in evaluation.get("metrics", []):
        if len(metric_entry) >= 2 and metric_entry[0] == metric_name:
            return extract_metric_value(metric_entry)
    return None


def calculate_suite_success_rate(evaluation: Dict[str, Any]) -> float:
    """Calculate the success rate for boolean metrics in an evaluation."""
    total_boolean = 0
    passed_boolean = 0
    
    for metric_entry in evaluation.get("metrics", []):
        if len(metric_entry) >= 2:
            value = extract_metric_value(metric_entry)
            if isinstance(value, bool):
                total_boolean += 1
                if value:
                    passed_boolean += 1
    
    if total_boolean == 0:
        return 0.0
    
    return passed_boolean / total_boolean


def format_value(value: Any) -> str:
    """Format a value for display in the output table."""
    if isinstance(value, bool):
        return "✅" if value else "❌"
    elif isinstance(value, float):
        return f"{value:.2f}"
    else:
        return str(value)


def generate_csv(results: List[Dict[str, Any]], output_file: str) -> None:
    """Generate a CSV file with the aggregated benchmark results."""
    # Determine common metrics
    common_metrics = get_common_metrics(results)
    
    # Prepare data for all rows
    rows = []
    
    for result in results:
        provider = result.get("provider", "unknown")
        model = result.get("model", "unknown")
        timestamp = result.get("timestamp", "")
        suite_name = result.get("suite", "unknown")
        eval_name = result.get("evaluation", "unknown")
        has_errors = len(result.get("errors", [])) > 0
        success_rate = calculate_suite_success_rate(result)
        
        # Create the row with basic information
        row = {
            "Provider": provider,
            "Model": model,
            "Suite": suite_name,
            "Evaluation": eval_name,
            "Timestamp": timestamp,
            "Success Rate": f"{success_rate:.2f}",
            "Has Errors": "Yes" if has_errors else "No"
        }
        
        # Add all common metrics
        for metric_name in common_metrics:
            value = get_metric_value(result, metric_name)
            row[metric_name] = format_value(value) if value is not None else ""
        
        # Add all other metrics
        for metric_entry in result.get("metrics", []):
            if len(metric_entry) >= 2:
                metric_name = metric_entry[0]
                if metric_name not in common_metrics:
                    value = extract_metric_value(metric_entry)
                    row[f"{metric_name}"] = format_value(value) if value is not None else ""
        
        rows.append(row)
    
    # Write to CSV
    if rows:
        # Get all column names from all rows
        all_columns = set()
        for row in rows:
            all_columns.update(row.keys())
        
        # Ensure core columns come first
        ordered_columns = [
            "Provider", 
            "Model", 
            "Suite", 
            "Evaluation", 
            "Timestamp", 
            "Success Rate", 
            "Has Errors"
        ]
        
        # Add common metrics next
        for metric in common_metrics:
            if metric not in ordered_columns:
                ordered_columns.append(metric)
        
        # Add any remaining columns
        remaining_columns = sorted(col for col in all_columns if col not in ordered_columns)
        ordered_columns.extend(remaining_columns)
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=ordered_columns)
            writer.writeheader()
            writer.writerows(rows)
            
        print(f"CSV report written to {output_file}")
    else:
        print("No data to write to CSV")


def generate_summary_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate overall summary statistics for the benchmark results."""
    summary = {
        "total_results": len(results),
        "providers": set(),
        "models": set(),
        "suites": set(),
        "evaluations": set(),
        "timestamps": set(),
        "success_rates": [],
        "total_tokens": 0,
        "tool_calls": 0,
        "execution_time": 0.0,
        "boolean_metrics_total": 0,
        "boolean_metrics_passed": 0,
        "errors": 0
    }
    
    for result in results:
        summary["providers"].add(result.get("provider", "unknown"))
        summary["models"].add(result.get("model", "unknown"))
        summary["suites"].add(result.get("suite", "unknown"))
        summary["evaluations"].add(result.get("evaluation", "unknown"))
        summary["timestamps"].add(result.get("timestamp", "unknown"))
        
        # Count errors
        summary["errors"] += len(result.get("errors", []))
        
        # Calculate success rate
        success_rate = calculate_suite_success_rate(result)
        summary["success_rates"].append(success_rate)
        
        # Tally metrics
        for metric_entry in result.get("metrics", []):
            if len(metric_entry) >= 2:
                metric_name = metric_entry[0]
                value = extract_metric_value(metric_entry)
                
                if metric_name == "total_tokens" and isinstance(value, (int, float)):
                    summary["total_tokens"] += value
                elif metric_name == "total_tool_calls" and isinstance(value, (int, float)):
                    summary["tool_calls"] += value
                elif metric_name == "prompt_execution_time_seconds" and isinstance(value, (int, float)):
                    summary["execution_time"] += value
                
                if isinstance(value, bool):
                    summary["boolean_metrics_total"] += 1
                    if value:
                        summary["boolean_metrics_passed"] += 1
    
    # Calculate overall success rate
    if summary["boolean_metrics_total"] > 0:
        summary["overall_success_rate"] = summary["boolean_metrics_passed"] / summary["boolean_metrics_total"]
    else:
        summary["overall_success_rate"] = 0.0
        
    # Average success rate across all evaluations
    if summary["success_rates"]:
        summary["avg_success_rate"] = sum(summary["success_rates"]) / len(summary["success_rates"])
    else:
        summary["avg_success_rate"] = 0.0
    
    return summary


def generate_markdown(results: List[Dict[str, Any]], output_file: str) -> None:
    """Generate a markdown file with the aggregated benchmark results."""
    # Determine common metrics
    common_metrics = get_common_metrics(results)
    
    # Generate summary statistics
    summary = generate_summary_stats(results)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Goose Benchmark Results\n\n")
        f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Add summary section
        f.write("## Summary\n\n")
        f.write(f"- Total Results: {summary['total_results']}\n")
        f.write(f"- Providers: {len(summary['providers'])}\n")
        f.write(f"- Models: {len(summary['models'])}\n")
        f.write(f"- Suites: {len(summary['suites'])}\n")
        f.write(f"- Unique Evaluations: {len(summary['evaluations'])}\n")
        f.write(f"- Total Tokens: {summary['total_tokens']:,}\n")
        f.write(f"- Total Tool Calls: {summary['tool_calls']}\n")
        f.write(f"- Total Execution Time: {summary['execution_time']:.2f} seconds\n")
        f.write(f"- Overall Success Rate: {summary['overall_success_rate']:.2%}\n")
        f.write(f"- Average Evaluation Success Rate: {summary['avg_success_rate']:.2%}\n")
        f.write(f"- Total Errors: {summary['errors']}\n\n")
        
        # Add model comparison table
        f.write("### Model Comparison\n\n")
        f.write("| Provider | Model | Evaluations | Success Rate | Avg. Tokens | Avg. Time (s) |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        
        # Gather data per provider-model
        model_stats = defaultdict(lambda: {
            "count": 0, 
            "success_rates": [], 
            "tokens": [], 
            "times": []
        })
        
        for result in results:
            provider = result.get("provider", "unknown")
            model = result.get("model", "unknown")
            key = (provider, model)
            
            model_stats[key]["count"] += 1
            
            success_rate = calculate_suite_success_rate(result)
            model_stats[key]["success_rates"].append(success_rate)
            
            tokens = get_metric_value(result, "total_tokens")
            if tokens is not None:
                model_stats[key]["tokens"].append(tokens)
                
            time = get_metric_value(result, "prompt_execution_time_seconds")
            if time is not None:
                model_stats[key]["times"].append(time)
        
        # Add rows for each model
        for (provider, model), stats in sorted(model_stats.items()):
            avg_success = sum(stats["success_rates"]) / len(stats["success_rates"]) if stats["success_rates"] else 0
            avg_tokens = sum(stats["tokens"]) / len(stats["tokens"]) if stats["tokens"] else 0
            avg_time = sum(stats["times"]) / len(stats["times"]) if stats["times"] else 0
            
            f.write(f"| {provider} | {model} | {stats['count']} | {avg_success:.2%} | {avg_tokens:.1f} | {avg_time:.2f} |\n")
        
        f.write("\n")
        
        # Group by provider and model
        provider_model_data = defaultdict(list)
        
        for result in results:
            provider = result.get("provider", "unknown")
            model = result.get("model", "unknown")
            key = (provider, model)
            provider_model_data[key].append(result)
        
        # Process each provider-model combination
        for (provider, model), model_results in provider_model_data.items():
            f.write(f"## {provider} - {model}\n\n")
            
            # Group by timestamp (run)
            timestamp_data = defaultdict(list)
            for result in model_results:
                timestamp = result.get("timestamp", "unknown")
                timestamp_data[timestamp].append(result)
            
            for timestamp, timestamp_results in timestamp_data.items():
                f.write(f"### Run: {timestamp}\n\n")
                
                # Group by suite
                suite_data = defaultdict(list)
                for result in timestamp_results:
                    suite_name = result.get("suite", "unknown")
                    suite_data[suite_name].append(result)
                
                # Process each suite
                for suite_name, suite_results in suite_data.items():
                    f.write(f"#### Suite: {suite_name}\n\n")
                    
                    # Create table headers
                    headers = ["Evaluation", "Success Rate"]
                    headers.extend(common_metrics)
                    
                    f.write("| " + " | ".join(headers) + " |\n")
                    f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
                    
                    # Add rows for each evaluation
                    for evaluation in suite_results:
                        eval_name = evaluation.get("evaluation", "unknown")
                        success_rate = calculate_suite_success_rate(evaluation)
                        
                        row = [eval_name, f"{success_rate:.2f}"]
                        
                        # Add values for common metrics
                        for metric_name in common_metrics:
                            value = get_metric_value(evaluation, metric_name)
                            row.append(format_value(value) if value is not None else "")
                        
                        f.write("| " + " | ".join(row) + " |\n")
                    
                    f.write("\n")
                    
                    # Add a section for boolean metrics (success criteria)
                    f.write("##### Success Criteria\n\n")
                    f.write("| Evaluation | Metric | Result |\n")
                    f.write("| --- | --- | --- |\n")
                    
                    for evaluation in suite_results:
                        eval_name = evaluation.get("evaluation", "unknown")
                        
                        # Find boolean metrics
                        boolean_metrics = []
                        for metric_entry in evaluation.get("metrics", []):
                            if len(metric_entry) >= 2:
                                metric_name = metric_entry[0]
                                value = extract_metric_value(metric_entry)
                                if isinstance(value, bool):
                                    boolean_metrics.append((metric_name, value))
                        
                        # Add rows for each boolean metric
                        for metric_name, value in boolean_metrics:
                            row = [eval_name, metric_name, "✅" if value else "❌"]
                            f.write("| " + " | ".join(row) + " |\n")
                    
                    f.write("\n")
                    
                    # Add a section for errors
                    has_errors = any(len(evaluation.get("errors", [])) > 0 for evaluation in suite_results)
                    if has_errors:
                        f.write("##### Errors\n\n")
                        
                        for evaluation in suite_results:
                            eval_name = evaluation.get("evaluation", "unknown")
                            errors = evaluation.get("errors", [])
                            
                            if errors:
                                f.write(f"**{eval_name}**:\n\n")
                                for error in errors:
                                    error_msg = error.get("message", "Unknown error")
                                    error_level = error.get("level", "error")
                                    f.write(f"- [{error_level}] {error_msg}\n")
                                f.write("\n")
                    
                    f.write("\n---\n\n")
            
            f.write("\n\n")
        
        print(f"Markdown report written to {output_file}")


def process_benchmark_directories(base_dir: str) -> List[Dict[str, Any]]:
    """Process all benchmark directories and collect their results."""
    results = []
    benchmark_dirs = find_benchmark_directories(base_dir)
    
    if not benchmark_dirs:
        print(f"No benchmark directories found in {base_dir}")
        return []
    
    print(f"Found {len(benchmark_dirs)} benchmark directories")
    
    for benchmark_dir in benchmark_dirs:
        provider, model = parse_provider_model(benchmark_dir)
        print(f"Processing benchmark directory: {benchmark_dir} (Provider: {provider}, Model: {model})")
        
        eval_result_files = find_eval_result_files(benchmark_dir)
        print(f"  Found {len(eval_result_files)} eval_result.json files")
        
        # Process each eval result file
        for file_path in eval_result_files:
            result = load_eval_result_file(file_path)
            if result:
                # Augment the result with metadata
                timestamp = parse_timestamp_from_path(file_path)
                suite, eval_name = parse_suite_eval_from_path(file_path)
                
                # Create a complete result object
                augmented_result = {
                    "provider": provider,
                    "model": model,
                    "timestamp": timestamp,
                    "suite": suite,
                    "evaluation": eval_name,
                    "metrics": result.get("metrics", []),
                    "errors": result.get("errors", []),
                    "_file_path": file_path  # Keep original file path for reference
                }
                
                results.append(augmented_result)
    
    return results


def analyze_squirrel_census(base_dir: str, output_csv: str) -> None:
    """
    Analyze squirrel_census evaluation results from benchmark directories.
    
    For each provider/model, determine:
    - Whether any run had 'correct_results' = True (best_run)
    - Number of runs with correct results
    - Average tokens, tool calls, and execution time for successful runs
    """
    benchmark_dirs = find_benchmark_directories(base_dir)
    results_by_provider_model = defaultdict(list)
    
    print(f"Found {len(benchmark_dirs)} benchmark directories")
    
    for benchmark_dir in benchmark_dirs:
        provider, model = parse_provider_model(benchmark_dir)
        key = (provider, model)
        
        # Look specifically for squirrel_census eval results
        pattern = os.path.join(benchmark_dir, "**", "vibes", "squirrel_census", "eval_result.json")
        squirrel_results = glob.glob(pattern, recursive=True)
        
        if squirrel_results:
            print(f"Found {len(squirrel_results)} squirrel_census results for {provider}-{model}")
            
            for result_file in squirrel_results:
                data = load_eval_result_file(result_file)
                if data:
                    # Extract relevant metrics
                    metrics = data.get("metrics", [])
                    result_data = {
                        "file_path": result_file,
                        "correct_results": False,
                        "total_tokens": 0,
                        "total_tool_calls": 0,
                        "prompt_execution_time_seconds": 0
                    }
                    
                    # Get specific metric values
                    for metric_entry in metrics:
                        if len(metric_entry) >= 2:
                            metric_name = metric_entry[0]
                            value = extract_metric_value(metric_entry)
                            
                            if metric_name == "correct_results" and isinstance(value, bool):
                                result_data["correct_results"] = value
                            elif metric_name == "total_tokens" and isinstance(value, (int, float)):
                                result_data["total_tokens"] = value
                            elif metric_name == "total_tool_calls" and isinstance(value, (int, float)):
                                result_data["total_tool_calls"] = value
                            elif metric_name == "prompt_execution_time_seconds" and isinstance(value, (int, float)):
                                result_data["prompt_execution_time_seconds"] = value
                    
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
        
        print(f"Squirrel Census analysis complete. Results saved to {output_csv}")
    else:
        print("No squirrel_census evaluation results found")


def main():
    parser = argparse.ArgumentParser(description='Analyze Goose benchmark results')
    parser.add_argument('--base-dir', default='.', 
                        help='Base directory containing benchmark-provider-model directories')
    parser.add_argument('--output', default='benchmark-analysis', 
                        help='Output file name (without extension)')
    parser.add_argument('--format', choices=['csv', 'markdown', 'both'], default='both',
                        help='Output format (csv, markdown, or both)')
    parser.add_argument('--squirrel-census', action='store_true',
                        help='Generate analysis specific to the squirrel_census evaluation')
    parser.add_argument('--squirrel-output', default='squirrel-census-analysis.csv',
                        help='Output file for squirrel census analysis (CSV only)')
    
    args = parser.parse_args()

    if args.squirrel_census:
        analyze_squirrel_census(args.base_dir, args.squirrel_output)
        return
    
    # Process benchmark directories and collect results
    results = process_benchmark_directories(args.base_dir)
    
    if not results:
        print("No benchmark results found")
        sys.exit(1)
    
    print(f"Collected {len(results)} individual benchmark results")
    
    # Generate outputs in the requested format(s)
    if args.format in ['csv', 'both']:
        csv_output = f"{args.output}.csv"
        generate_csv(results, csv_output)
    
    if args.format in ['markdown', 'both']:
        md_output = f"{args.output}.md"
        generate_markdown(results, md_output)


if __name__ == "__main__":
    main()