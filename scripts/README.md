# Goose Benchmark Scripts

This directory contains scripts for running and analyzing Goose benchmarks.

## analyze_benchmark_results.py

This Python script aggregates and analyzes benchmark results from eval_result.json files across all benchmark runs.

### Prerequisites

- Python 3.6 or higher

### Usage

```bash
# Basic usage (scans current directory for benchmark-* directories)
./analyze_benchmark_results.py

# Specify a custom base directory
./analyze_benchmark_results.py --base-dir /path/to/benchmarks

# Choose output format (csv, markdown, or both)
./analyze_benchmark_results.py --format csv

# Specify a custom output filename (without extension)
./analyze_benchmark_results.py --output my-benchmark-report
```

### Output Files

The script generates two types of output files:

1. **CSV File**: Contains all metrics in a tabular format with one row per evaluation.
   - Useful for filtering, sorting, and additional analysis in spreadsheet applications.

2. **Markdown File**: Organizes results hierarchically by provider, model, timestamp, and suite.
   - Includes summary statistics and model comparison tables.
   - Shows success criteria and errors for each evaluation.

### Directory Structure

The script expects benchmark results in the following directory structure:

```
benchmark-[provider]-[model]/
  └── YYYY-MM-DD-HH:MM:SS/
      └── [suite]/
          └── [evaluation]/
              └── eval_result.json
```

Where:
- `provider`: The LLM provider (e.g., ollama, openai)
- `model`: The model name (e.g., mistral, claude-3-opus)
- The timestamp directory represents when the benchmark was run
- `suite`: The evaluation suite name (e.g., core, vibes)
- `evaluation`: The specific evaluation name (e.g., blog_summary, flappy_bird)

## run-benchmarks.sh

This script runs Goose benchmarks across multiple provider:model pairs and analyzes the results.

### Prerequisites

- Goose CLI must be built or installed
- `jq` command-line tool for JSON processing (optional, but recommended for result analysis)

### Usage

```bash
./scripts/run-benchmarks.sh [options]
```

#### Options

- `-p, --provider-models`: Comma-separated list of provider:model pairs (e.g., 'openai:gpt-4o,anthropic:claude-3-5-sonnet')
- `-s, --suites`: Comma-separated list of benchmark suites to run (e.g., 'core,small_models')
- `-o, --output-dir`: Directory to store benchmark results (default: './benchmark-results')
- `-d, --debug`: Use debug build instead of release build
- `-h, --help`: Show help message

#### Examples

```bash
# Run with release build (default)
./scripts/run-benchmarks.sh --provider-models 'openai:gpt-4o,anthropic:claude-3-5-sonnet' --suites 'core,small_models'

# Run with debug build
./scripts/run-benchmarks.sh --provider-models 'openai:gpt-4o' --suites 'core' --debug
```

### How It Works

The script:
1. Parses the provider:model pairs and benchmark suites
2. Determines whether to use the debug or release binary
3. For each provider:model pair:
   - Sets the `GOOSE_PROVIDER` and `GOOSE_MODEL` environment variables
   - Runs the benchmark with the specified suites
   - Analyzes the results for failures
4. Generates a summary of all benchmark runs

### Output

The script creates the following files in the output directory:

- `summary.md`: A summary of all benchmark results
- `{provider}-{model}.json`: Raw JSON output from each benchmark run
- `{provider}-{model}-analysis.txt`: Analysis of each benchmark run

### Exit Codes

- `0`: All benchmarks completed successfully
- `1`: One or more benchmarks failed

## parse-benchmark-results.sh

This script analyzes a single benchmark JSON result file and identifies any failures.

### Usage

```bash
./scripts/parse-benchmark-results.sh path/to/benchmark-results.json
```

### Output

The script outputs an analysis of the benchmark results to stdout, including:

- Basic information about the benchmark run
- Results for each evaluation in each suite
- Summary of passed and failed metrics

### Exit Codes

- `0`: All metrics passed successfully
- `1`: One or more metrics failed