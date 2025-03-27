#!/usr/bin/env python
"""
Run post-analysis for a Trimmed Match marketing experiment.

This script analyzes the results of a marketing experiment using the 
Trimmed Match methodology.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# Add parent directory to path so we can import geo_causal_inference
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from geo_causal_inference.post_analysis import ExperimentAnalyzer
from geo_causal_inference.utils import (
    TimeWindow, create_time_window, plot_time_series_comparison, 
    plot_pair_time_series, plot_correlation_matrix
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run post-analysis for a Trimmed Match marketing experiment"
    )
    
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to the data CSV file"
    )
    
    parser.add_argument(
        "--test-start", type=str, required=True,
        help="Test period start date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--test-end", type=str, required=True,
        help="Test period end date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--design-start", type=str, required=True,
        help="Design period start date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--design-end", type=str, required=True,
        help="Design period end date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--cooldown-end", type=str, default=None,
        help="Cooldown period end date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--response-col", type=str, default="response",
        help="Column name for response variable (default: response)"
    )
    
    parser.add_argument(
        "--spend-col", type=str, default="cost",
        help="Column name for spend variable (default: cost)"
    )
    
    parser.add_argument(
        "--aov", type=float, default=1.0,
        help="Average order value (default: 1.0)"
    )
    
    parser.add_argument(
        "--exclude-pairs", type=str, default="",
        help="Comma-separated list of pair IDs to exclude"
    )
    
    parser.add_argument(
        "--output-dir", type=str, default="output",
        help="Output directory (default: output)"
    )
    
    return parser.parse_args()


def main():
    """Run post-analysis for a Trimmed Match marketing experiment."""
    args = parse_args()
    
    # Create output directories
    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "postanalysis" / "plots"
    data_dir = output_dir / "postanalysis" / "data"
    
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse periods
    test_period = create_time_window(args.test_start, args.test_end)
    design_period = create_time_window(args.design_start, args.design_end)
    
    cooldown_period = None
    if args.cooldown_end:
        cooldown_period = create_time_window(args.test_end, args.cooldown_end)
    
    # Parse excluded pairs
    excluded_pairs = []
    if args.exclude_pairs:
        excluded_pairs = [int(pair.strip()) for pair in args.exclude_pairs.split(",")]
    
    # Load data
    print(f"Loading data from {args.data}...")
    data = pd.read_csv(args.data)
    
    # Create analyzer
    analyzer = ExperimentAnalyzer(
        data=data,
        test_period=test_period,
        design_period=design_period,
        cooldown_period=cooldown_period,
        response_col=args.response_col,
        spend_col=args.spend_col,
        average_order_value=args.aov
    )
    
    # Exclude pairs if specified
    if excluded_pairs:
        print(f"Excluding pairs: {excluded_pairs}")
        analyzer.exclude_pairs(excluded_pairs)
    
    # Get summary statistics
    print("Calculating summary statistics...")
    summary = analyzer.get_summary_stats()
    summary.to_csv(data_dir / "period_summary.csv", index=False)
    print(summary)
    
    # Calculate results without cooldown
    print("\nCalculating results (excluding cooldown period)...")
    summary_df, detailed_df = analyzer.report_results(exclude_cooldown=True)
    
    # Save results to CSV
    summary_df.to_csv(data_dir / "results_summary.csv", index=False)
    detailed_df.to_csv(data_dir / "results_detailed.csv", index=False)
    
    print("\nSummary Results:")
    print(summary_df)
    print("\nDetailed Results with Confidence Intervals:")
    print(detailed_df)
    
    # Calculate results with cooldown if available
    if cooldown_period:
        print("\nCalculating results (including cooldown period)...")
        summary_cd_df, detailed_cd_df = analyzer.report_results(exclude_cooldown=False)
        
        # Save results to CSV
        summary_cd_df.to_csv(data_dir / "results_summary_with_cooldown.csv", index=False)
        detailed_cd_df.to_csv(data_dir / "results_detailed_with_cooldown.csv", index=False)
        
        print("\nSummary Results (with cooldown):")
        print(summary_cd_df)
        print("\nDetailed Results with Confidence Intervals (with cooldown):")
        print(detailed_cd_df)
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Time series comparison for response
    fig = plot_time_series_comparison(
        analyzer.data, "response", test_period, design_period, cooldown_period
    )
    fig.savefig(plots_dir / "response_time_series.png", bbox_inches="tight")
    
    # Time series comparison for cost
    fig = plot_time_series_comparison(
        analyzer.data, "cost", test_period, design_period, cooldown_period
    )
    fig.savefig(plots_dir / "cost_time_series.png", bbox_inches="tight")
    
    # Pair-level time series
    pairs = analyzer.data["pair"].unique().tolist()
    g = plot_pair_time_series(
        analyzer.data, pairs, response_col="response",
        test_period=test_period, design_period=design_period, cooldown_period=cooldown_period
    )
    g.fig.savefig(plots_dir / "pair_time_series.png", bbox_inches="tight")
    
    # Correlation matrix
    fig = plot_correlation_matrix(
        analyzer.data, design_period, test_period, response_col="response"
    )
    fig.savefig(plots_dir / "correlation_matrix.png", bbox_inches="tight")
    
    print("\nPost-analysis complete!")
    print(f"Results saved to {output_dir}")
    print(f"Plots saved to {plots_dir}")
    print(f"Data files saved to {data_dir}")


if __name__ == "__main__":
    main()
