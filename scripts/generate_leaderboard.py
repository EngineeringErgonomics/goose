#!/usr/bin/env python3
"""
Generate a leaderboard by aggregating model performance across all evaluations.

This script analyzes the combined evaluation results from all-evaluations.csv,
computes aggregated metrics for each model, and creates a ranked leaderboard CSV.
"""

import os
import pandas as pd
import argparse
from collections import defaultdict
from typing import Dict, List, Any

def generate_leaderboard(input_csv: str, output_csv: str) -> None:
    """
    Generate a leaderboard by aggregating model performance across all evals.
    
    Args:
        input_csv: Path to the combined evaluations CSV file
        output_csv: Path to save the leaderboard CSV
    """
    # Load the combined evaluations data
    try:
        df = pd.read_csv(input_csv)
        print(f"Loaded data from {input_csv}: {len(df)} rows")
    except Exception as e:
        print(f"Error loading {input_csv}: {e}")
        return
    
    # Validate required columns
    required_columns = [
        "provider", "model", "evaluation", 
        "n_successful_runs", "total_runs",
        "avg_tokens_successful", "avg_tool_calls_successful", "avg_exec_time_successful"
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return
    
    # Get unique providers and models
    providers = df['provider'].unique()
    models = df['model'].unique()
    evals = df['evaluation'].unique()
    
    print(f"Found {len(providers)} providers, {len(models)} models, and {len(evals)} evaluations")
    
    # Initialize dictionaries to store aggregated metrics by provider-model
    aggregated_metrics = defaultdict(lambda: {
        'provider': '',
        'model': '',
        'total_successful_runs': 0,
        'total_runs': 0,
        'avg_success_rate': 0.0,
        'eval_count': 0,
        'sum_tokens_successful': 0.0,
        'sum_tool_calls_successful': 0.0,
        'sum_exec_time_successful': 0.0,
        'avg_tokens_successful': 0.0,
        'avg_tool_calls_successful': 0.0,
        'avg_exec_time_successful': 0.0
    })
    
    # Aggregate metrics by provider and model
    for _, row in df.iterrows():
        provider = row['provider']
        model = row['model']
        key = f"{provider}-{model}"
        
        aggregated_metrics[key]['provider'] = provider
        aggregated_metrics[key]['model'] = model
        
        # Add to total successful runs and total runs
        aggregated_metrics[key]['total_successful_runs'] += row['n_successful_runs']
        aggregated_metrics[key]['total_runs'] += row['total_runs']
        
        # Only include metrics from evals with successful runs
        if row['n_successful_runs'] > 0:
            aggregated_metrics[key]['eval_count'] += 1
            aggregated_metrics[key]['sum_tokens_successful'] += row['avg_tokens_successful']
            aggregated_metrics[key]['sum_tool_calls_successful'] += row['avg_tool_calls_successful']
            aggregated_metrics[key]['sum_exec_time_successful'] += row['avg_exec_time_successful']
    
    # Calculate final averages and success rates
    for key, metrics in aggregated_metrics.items():
        if metrics['total_runs'] > 0:
            metrics['avg_success_rate'] = metrics['total_successful_runs'] / metrics['total_runs']
        
        if metrics['eval_count'] > 0:
            metrics['avg_tokens_successful'] = metrics['sum_tokens_successful'] / metrics['eval_count']
            metrics['avg_tool_calls_successful'] = metrics['sum_tool_calls_successful'] / metrics['eval_count']
            metrics['avg_exec_time_successful'] = metrics['sum_exec_time_successful'] / metrics['eval_count']
            
        # Remove intermediate sum fields
        del metrics['sum_tokens_successful']
        del metrics['sum_tool_calls_successful']
        del metrics['sum_exec_time_successful']
    
    # Convert to dataframe and sort by success rate
    leaderboard_df = pd.DataFrame(list(aggregated_metrics.values()))
    leaderboard_df = leaderboard_df.sort_values('avg_success_rate', ascending=False)
    
    # Add rank column
    leaderboard_df.insert(0, 'rank', range(1, len(leaderboard_df) + 1))
    
    # Reorder columns for readability
    column_order = [
        'rank', 'provider', 'model', 
        'avg_success_rate', 'total_successful_runs', 'total_runs', 'eval_count',
        'avg_tokens_successful', 'avg_tool_calls_successful', 'avg_exec_time_successful'
    ]
    leaderboard_df = leaderboard_df[column_order]
    
    # Round numeric columns
    leaderboard_df['avg_tokens_successful'] = leaderboard_df['avg_tokens_successful'].round(0).astype(int)
    leaderboard_df['avg_tool_calls_successful'] = leaderboard_df['avg_tool_calls_successful'].round(1)
    leaderboard_df['avg_exec_time_successful'] = leaderboard_df['avg_exec_time_successful'].round(1)
    
    # Save to CSV
    leaderboard_df.to_csv(output_csv, index=False)
    print(f"Leaderboard saved to {output_csv}")
    
    # Print top 5 models
    print("\nTop 5 Models by Success Rate:")
    top_5 = leaderboard_df.head(5)
    for idx, row in top_5.iterrows():
        print(f"{row['rank']}. {row['provider']}-{row['model']}: {row['success_rate_percent']} " +
              f"({row['total_successful_runs']}/{row['total_runs']} runs across {row['eval_count']} evals)")

def main():
    parser = argparse.ArgumentParser(description='Generate a model performance leaderboard')
    parser.add_argument('--input', default='./analysis-results/all-evaluations.csv', 
                        help='Path to the combined evaluations CSV file')
    parser.add_argument('--output', default='./analysis-results/leaderboard.csv',
                        help='Path to save the leaderboard CSV')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    generate_leaderboard(args.input, args.output)

if __name__ == "__main__":
    main()