# Table of Contents
- __init__.py
- setup.py
- test_map.py
- scripts/run_postanalysis.py
- scripts/run_experiment.py
- design_trimmed_match/design_colab_for_trimmed_match_og.py
- design_trimmed_match/trimmed_match_postanalysis_colab.py
- src/data_pipeline/data_standardizer.py
- src/data_pipeline/__init__.py
- src/data_pipeline/data_joiner.py
- src/data_pipeline/geo_joiner.py
- src/data_pipeline/geo_reference_builder.py
- src/geo_causal_inference/config.py
- src/geo_causal_inference/post_analysis.py
- src/geo_causal_inference/data_loader.py
- src/geo_causal_inference/design.py
- src/geo_causal_inference/__init__.py
- src/geo_causal_inference/visualization.py
- src/geo_causal_inference/utils.py
- src/geo_causal_inference/validation.py
- src/examples/cost_response_join_example.py

## File: __init__.py

- Extension: .py
- Language: python
- Size: 156 bytes
- Created: 2025-03-27 11:17:55
- Modified: 2025-03-27 11:17:55

### Code

```python
"""
Geo Causal Inference - Top level package.

This makes it possible to import modules from the src directory.
"""

from src.geo_causal_inference import *

```

## File: setup.py

- Extension: .py
- Language: python
- Size: 122 bytes
- Created: 2025-03-27 11:18:23
- Modified: 2025-03-27 11:18:23

### Code

```python
from setuptools import setup

setup(
    name="geo_causal_inference",
    version="0.1.0",
    package_dir={"": "src"},
)

```

## File: test_map.py

- Extension: .py
- Language: python
- Size: 1120 bytes
- Created: 2025-04-09 09:03:04
- Modified: 2025-04-09 09:03:04

### Code

```python
#!/usr/bin/env python
"""
Test script for the geo map visualization.
"""

import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Ensure output directory exists
output_dir = os.path.join('/app/output', 'map_test')
os.makedirs(output_dir, exist_ok=True)

# Add the project to the path
sys.path.insert(0, '/app')

# Import the visualization module
from src.geo_causal_inference.visualization import plot_geo_map

# Path to the geo spine data
spine_path = '/app/data/reference/geo_spine.csv'

# Path to geo assignments
assignments_path = '/app/output/client1_weekly/design/data/geo_assignments.csv'

# Check if file exists
if not os.path.exists(assignments_path):
    print(f"File not found: {assignments_path}")
    sys.exit(1)

# Read geo assignments
geo_assignments = pd.read_csv(assignments_path)

# Save the map visualization
plot_geo_map(
    geo_assignments=geo_assignments,
    spine_path=spine_path,
    map_type='dma',
    debug=True,  # Add debug mode to get more verbose output
    output_path=os.path.join(output_dir, 'geo_map_test.png')
)

```

## File: scripts/run_postanalysis.py

- Extension: .py
- Language: python
- Size: 6572 bytes
- Created: 2025-03-27 11:21:07
- Modified: 2025-03-27 11:21:07

### Code

```python
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

```

## File: scripts/run_experiment.py

- Extension: .py
- Language: python
- Size: 17115 bytes
- Created: 2025-04-09 09:03:04
- Modified: 2025-04-09 09:03:04

### Code

```python
#!/usr/bin/env python
"""
Sample script to run a Trimmed Match experiment using the modular codebase.

This script demonstrates how to use the trimmed_match package
with a local CSV file.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

# Import from the package directly
from geo_causal_inference.data_loader import load_data
from geo_causal_inference.validation import validate_input_data, validate_experiment_periods, validate_geos
from geo_causal_inference.design import ExperimentDesigner
from geo_causal_inference.config import ExperimentConfig
from geo_causal_inference.utils import create_time_window, format_summary_table
from geo_causal_inference.visualization import plot_designs_comparison, plot_geo_time_series, plot_geo_map

from trimmed_match.design.common_classes import GeoXType, GeoAssignment


def main():
    """Main function to run the experiment."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run a Trimmed Match experiment using a CSV file.')
    parser.add_argument('--input', type=str, 
                       help='Path to input CSV file containing geo-level time series data')
    parser.add_argument('--client', type=str, default='example',
                       help='Client name used for output directory structure')
    parser.add_argument('--frequency', type=str, default='daily',
                       help='Data frequency: "daily" or "weekly"')
    args = parser.parse_args()
    
    # Define the path to the test data
    if args.input:
        test_data_path = os.path.abspath(args.input)
    else:
        # Use default example data if no input is provided
        test_data_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'raw_data',
            'example_data_for_design.csv'
        ))
    
    # Set client name
    client_name = args.client
    
    # Set data frequency
    data_frequency = args.frequency.lower()
    if data_frequency not in ['daily', 'weekly']:
        print("Warning: Invalid frequency specified. Defaulting to 'daily'.")
        data_frequency = 'daily'
    
    print(f"==================== DEBUG INFO ====================")
    print(f"Running experiment with:")
    print(f"- Input data: {test_data_path}")
    print(f"- Client name: {client_name}")
    print(f"- Data frequency: {data_frequency}")
    print(f"===================================================")
    
    # Add diagnostic prints
    raw_df = pd.read_csv(test_data_path)
    print(f"Raw CSV rows: {len(raw_df)}")
    
    # Load and validate the data
    geo_level_time_series = load_data(test_data_path)
    print(f"After load_data: {len(geo_level_time_series)}")
    
    geo_level_time_series = validate_input_data(geo_level_time_series)
    print(f"After validate_input_data: {len(geo_level_time_series)}")
    
    print(f"Loaded {len(geo_level_time_series)} rows of data")
    print(f"Unique geos: {geo_level_time_series['geo'].nunique()}")
    print(f"Date range: {geo_level_time_series['date'].min()} to {geo_level_time_series['date'].max()}")
    
    # Create a configuration
    config = ExperimentConfig(
        geox_type=GeoXType.HOLD_BACK,
        experiment_duration_weeks=4,
        experiment_budget=15000.0,
        alternative_budgets=[15000.0, 20000.0, 25000.0],
        minimum_detectable_iroas=3.0,
        average_order_value=256, # average order value in dollars / total sessions
        significance_level=0.10,
        power_level=0.80,
        use_cross_validation=True, # set to True if you want to use cross validation
        number_of_simulations=200, # number of simulations to run typically 200 but this wont finish
    )
    
    # Set dates based on the data
    min_date = geo_level_time_series['date'].min()
    max_date = geo_level_time_series['date'].max()
    
    # Design period covers all data
    config.design_start_date = min_date
    config.design_end_date = max_date
    
    # Calculate time delta based on frequency
    if data_frequency == 'weekly':
        # For weekly data, adjust time windows to use weeks instead of days
        eval_delta = pd.Timedelta(weeks=config.experiment_duration_weeks * 2)  # 8 weeks before end
        coverage_delta = pd.Timedelta(weeks=config.experiment_duration_weeks)  # 4 weeks
    else:
        # For daily data, continue using the original calculation
        eval_delta = pd.Timedelta(days=28*2)  # 8 weeks before end
        coverage_delta = pd.Timedelta(days=28)  # 4 weeks
    
    # Evaluation period starts at a reasonable point for a 4-week test
    eval_start = max_date - eval_delta
    config.eval_start_date = eval_start
    
    # Coverage test period is before evaluation
    config.coverage_test_start_date = eval_start - coverage_delta
    
    # Validate geos and ensure even number
    config.geos_exclude, warnings = validate_geos(geo_level_time_series)
    for warning in warnings:
        print(warning)
    
    # Validate experiment periods and get excluded days
    pass_checks, error_message, days_exclude = validate_experiment_periods(
        geo_level_time_series,
        config.eval_start_date,
        config.coverage_test_start_date,
        config.experiment_duration_weeks
    )
    
    if not pass_checks:
        print(f"Error in experiment periods: {error_message}")
        return
    
    # Remove excluded days
    geo_time_series = geo_level_time_series[~geo_level_time_series["date"].isin(days_exclude)]
    geo_time_series = geo_time_series[~geo_time_series["geo"].isin(config.geos_exclude)]
    
    # Get time windows 
    # Note: We'll handle different frequencies in the designer
    time_window_for_design, time_window_for_eval, coverage_test_window = config.get_time_windows()
    
    # Prepare data for design by excluding coverage test period
    data_without_coverage_test_period = geo_time_series[
        (geo_time_series["date"] < coverage_test_window.start_date) |
        (geo_time_series["date"] > coverage_test_window.end_date)
    ]
    
    # Create the experiment designer
    designer = ExperimentDesigner(
        geox_type=config.geox_type,
        data=data_without_coverage_test_period,
        time_window_for_design=time_window_for_design,
        time_window_for_eval=time_window_for_eval,
        response_col=config.response_col,
        spend_col=config.spend_col,
        matching_metrics=config.matching_metrics
    )
    
    # Calculate the optimal budget
    design_results = designer.calculate_optimal_budget(
        experiment_budget=config.experiment_budget,
        minimum_detectable_iroas=config.minimum_detectable_iroas,
        average_order_value=config.average_order_value,
        additional_budget=config.alternative_budgets,
        use_cross_validation=config.use_cross_validation,
        num_simulations=config.number_of_simulations
    )
    
    # Debug print for design results
    print(f"\n==================== DEBUG DESIGN RESULTS ====================")
    print(f"Type of design_results: {type(design_results)}")
    print(f"Keys in design_results: {design_results.keys()}")
    print(f"Optimal pair index: {design_results['optimal_pair_index']}")
    
    if 'results' in design_results:
        print(f"Results shape: {design_results['results'].shape}")
        print(f"Results columns: {design_results['results'].columns}")
        
        # Check for min RMSE cost adjusted
        if 'rmse_cost_adjusted' in design_results['results'].columns:
            min_rmse_idx = design_results['results']['rmse_cost_adjusted'].idxmin()
            min_rmse_pair = design_results['results'].loc[min_rmse_idx, 'pair_index']
            min_rmse = design_results['results'].loc[min_rmse_idx, 'rmse']
            min_rmse_cost_adj = design_results['results'].loc[min_rmse_idx, 'rmse_cost_adjusted']
            print(f"Min RMSE cost adjusted: {min_rmse_cost_adj} at pair_index {min_rmse_pair} with raw RMSE {min_rmse}")
        
        # Show values for optimal pair index
        optimal_idx = design_results['results']['pair_index'] == design_results['optimal_pair_index']
        if any(optimal_idx):
            opt_row = design_results['results'][optimal_idx].iloc[0]
            print(f"Optimal pair values:")
            print(f"  - pair_index: {opt_row['pair_index']}")
            print(f"  - rmse: {opt_row['rmse']}")
            print(f"  - rmse_cost_adjusted: {opt_row['rmse_cost_adjusted']}")
    print(f"==============================================================")
    
    # Get the optimal design
    optimal_pair_index = design_results["optimal_pair_index"]
    
    # Output the chosen design
    axes, geopairs, treatment_geo, control_geo = designer.get_optimal_design(
        pair_index=optimal_pair_index,
        confidence=1-config.significance_level
    )
    
    # Print summary of the design
    summary = format_summary_table(
        design_results,
        config.minimum_detectable_iroas,
        config.minimum_detectable_lift_in_response_metric
    )
    print("\nDesign Summary:")
    print(summary)
    
    # Debug print for summary creation
    print(f"\n==================== DEBUG SUMMARY ====================")
    print(f"Summary table content:")
    print(summary)
    print(f"Summary table data types:")
    print(summary.dtypes)
    print(f"=====================================================")
    
    # Print treatment and control geos
    print(f"\nTreatment Geos ({len(treatment_geo)}):")
    print(", ".join(map(str, sorted(treatment_geo))))
    
    print(f"\nControl Geos ({len(control_geo)}):")
    print(", ".join(map(str, sorted(control_geo))))
    
    # Create client-specific output directories
    plots_dir = os.path.join(os.path.dirname(__file__), '..', 'output', client_name, 'design', 'plots')
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'output', client_name, 'design', 'data')
    post_analysis_dir = os.path.join(os.path.dirname(__file__), '..', 'output', client_name, 'postanalysis')
    
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(post_analysis_dir, exist_ok=True)
    
    # Debug print for output directories
    print(f"\n==================== DEBUG OUTPUT PATHS ====================")
    print(f"Output directories:")
    print(f"- Plots: {plots_dir}")
    print(f"- Data: {data_dir}")
    print(f"- Post-analysis: {post_analysis_dir}")
    print(f"===========================================================")
    
    # Save design results
    design_results_df = design_results["results"]
    design_results_df.to_csv(os.path.join(data_dir, "design_results.csv"), index=False)
    
    # Find the optimal pair index for minimal RMSE
    optimal_pair_index = design_results["optimal_pair_index"]
    
    # Create summary of the optimal design
    optimal_design = design_results["results"].loc[
        design_results["results"]["pair_index"] == optimal_pair_index
    ].squeeze()
    
    # Create design summary dataframe
    design_summary = pd.DataFrame({
        "optimal_pair_index": [optimal_pair_index],
        "budget": [optimal_design["budget"]],
        "min_detectable_iroas": [config.minimum_detectable_iroas],
        "rmse": [optimal_design["rmse"]],
        "rmse_cost_adjusted": [optimal_design["rmse_cost_adjusted"]],
        "experiment_spend": [optimal_design["experiment_spend"]],
        "num_pairs": [optimal_design["num_pairs"]],
        "trim_rate": [optimal_design["trim_rate"]]
    })
    
    # Save design summary
    design_summary.to_csv(os.path.join(data_dir, "design_summary.csv"), index=False)
    
    # Generate and save trade-off visualizations
    visualization_paths = designer.add_trade_off_visualizations(
        design_results_df, 
        plots_dir
    )
    print(f"Design trade-off visualizations saved to {plots_dir}")
    
    # Plot the designs comparison
    fig_design = plot_designs_comparison(design_results["results"])
    fig_design.savefig(os.path.join(plots_dir, 'design_comparison.png'))
    
    # Plot the geo time series
    if data_frequency == 'weekly':
        eval_end_date = config.eval_start_date + pd.Timedelta(weeks=config.experiment_duration_weeks-1)
    else:
        eval_end_date = config.eval_start_date + pd.Timedelta(days=28-1)
    
    fig_timeseries = plot_geo_time_series(
        geo_level_time_series, 
        treatment_geos=treatment_geo, 
        control_geos=control_geo,
        eval_start_date=config.eval_start_date,
        eval_end_date=eval_end_date
    )
    fig_timeseries.savefig(os.path.join(plots_dir, 'geo_time_series.png'))
    
    # Save dataframes to CSV
    summary_path = os.path.join(data_dir, 'design_summary.csv')
    summary.to_csv(summary_path, index=False)
    print(f"\n==================== DEBUG FILE SAVING ====================")
    print(f"Saved summary to: {summary_path}")
    print(f"Summary content saved:")
    print(summary)
    print(f"==========================================================")
    
    # Save geo assignments
    geo_assignments = pd.DataFrame({
        'geo': sorted(treatment_geo + control_geo),
        'assignment': ['treatment' if geo in treatment_geo else 'control' 
                      for geo in sorted(treatment_geo + control_geo)]
    })
    geo_assignments.to_csv(os.path.join(data_dir, 'geo_assignments.csv'), index=False)
    
    # Create and save a geographic visualization of treatment/control assignments
    geo_spine_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'data',
        'reference',
        'geo_spine.csv'
    ))
    
    fig_geo_map = plot_geo_map(
        geo_assignments=geo_assignments,
        spine_path=geo_spine_path,
        map_type='dma',
        debug=True  # Enable debug mode for detailed logging
    )
    fig_geo_map.savefig(os.path.join(plots_dir, 'geo_assignment_map.png'))
    print(f"Saved geographic treatment/control map to: {os.path.join(plots_dir, 'geo_assignment_map.png')}")
    
    # Save design results
    results_path = os.path.join(data_dir, 'design_results.csv')
    design_results['results'].to_csv(results_path, index=False)
    print(f"Saved detailed design results to: {results_path}")
    
    # Create a complete dataset for post-analysis
    # First convert assignment to numeric values (1=Treatment, 2=Control)
    numeric_assignments = {
        geo: GeoAssignment.TREATMENT if geo in treatment_geo else GeoAssignment.CONTROL 
        for geo in treatment_geo + control_geo
    }
    
    # For post-analysis, we need to create proper 1:1 geo pairs (one treatment, one control per pair)
    # Sort the geos to ensure consistent pairing
    sorted_treatment_geos = sorted(treatment_geo)
    sorted_control_geos = sorted(control_geo)
    
    # Make sure we have equal numbers of treatment and control geos
    min_length = min(len(sorted_treatment_geos), len(sorted_control_geos))
    paired_treatment_geos = sorted_treatment_geos[:min_length]
    paired_control_geos = sorted_control_geos[:min_length]
    
    # Create a geo to pair mapping (pair_id will be 0, 1, 2, ...)
    geo_to_pair = {}
    for pair_id, (treat_geo, control_geo) in enumerate(zip(paired_treatment_geos, paired_control_geos)):
        geo_to_pair[treat_geo] = pair_id
        geo_to_pair[control_geo] = pair_id
    
    # Filter to only include the geos in our paired design
    post_analysis_data = geo_level_time_series[
        geo_level_time_series['geo'].isin(list(geo_to_pair.keys()))
    ].copy()
    
    # Add assignment and pair information
    post_analysis_data['assignment'] = post_analysis_data['geo'].map(numeric_assignments)
    post_analysis_data['pair'] = post_analysis_data['geo'].map(geo_to_pair)
    
    # Ensure we have all needed columns and in the right order
    post_analysis_data = post_analysis_data[['date', 'geo', 'pair', 'assignment', config.response_col, config.spend_col]]
    
    # Rename columns if needed to match post-analysis expectations
    if config.response_col != 'response':
        post_analysis_data = post_analysis_data.rename(columns={config.response_col: 'response'})
    if config.spend_col != 'cost':
        post_analysis_data = post_analysis_data.rename(columns={config.spend_col: 'cost'})
    
    # Save the post-analysis data
    post_analysis_data.to_csv(os.path.join(post_analysis_dir, 'experiment_data_for_postanalysis.csv'), index=False)
    
    print(f"\nPlots saved to: {plots_dir}")
    print(f"Data files saved to: {data_dir}")
    print(f"Post-analysis data file saved to: {os.path.join(post_analysis_dir, 'experiment_data_for_postanalysis.csv')}")
    
    # Print data frequency information
    print(f"\nExperiment designed with {data_frequency} data")
    if data_frequency == 'weekly':
        print(f"Evaluation period: {config.eval_start_date} to {eval_end_date} ({config.experiment_duration_weeks} weeks)")
    else:
        print(f"Evaluation period: {config.eval_start_date} to {eval_end_date} ({28} days)")


if __name__ == "__main__":
    main()

```

## File: design_trimmed_match/design_colab_for_trimmed_match_og.py

- Extension: .py
- Language: python
- Size: 32478 bytes
- Created: 2025-03-27 11:16:38
- Modified: 2025-03-25 14:23:34

### Code

```python
# -*- coding: utf-8 -*-
"""Design Colab for Trimmed Match.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/github/google/trimmed_match/blob/master/trimmed_match/notebook/design_colab_for_trimmed_match.ipynb

# **Colab to design an LC experiment with Trimmed Match**
"""

#@title **Getting Started**

#@markdown * Connect to the hosted runtime and run each cell after updating the necessary inputs
#@markdown * Download the file "example_data_for_design.csv" from the folder "example_datasets" in github.
#@markdown * Upload the csv file to your Google Drive and open it with Google Sheets
#@markdown * In the cell below, copy and paste the url of the sheet.

"""# Data input"""

#@title Load the libraries needed for the design

BAZEL_VERSION = '6.1.2'
!wget https://github.com/bazelbuild/bazel/releases/download/{BAZEL_VERSION}/bazel-{BAZEL_VERSION}-installer-linux-x86_64.sh
!chmod +x bazel-{BAZEL_VERSION}-installer-linux-x86_64.sh
!./bazel-{BAZEL_VERSION}-installer-linux-x86_64.sh
!sudo apt-get install python3-dev python3-setuptools git
!git clone https://github.com/google/trimmed_match
!python3 -m pip install ./trimmed_match
!pip install colorama
!pip install gspread-dataframe

"""Loading the necessary python modules."""
import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import re
from scipy import stats
import warnings

from IPython.display import display
from IPython.core.interactiveshell import InteractiveShell

import gspread
from colorama import Fore, Style
from gspread_dataframe import set_with_dataframe
from google import auth as google_auth
from google.colab import auth
from google.colab import data_table
from google.colab import widgets
from google.colab import drive
from trimmed_match.design.common_classes import GeoXType, TimeWindow
from trimmed_match.design.common_classes import GeoAssignment
from trimmed_match.design.trimmed_match_design import TrimmedMatchGeoXDesign
from trimmed_match.design.util import find_days_to_exclude, overlap_percent
from trimmed_match.design.util import check_time_periods, check_input_data
from trimmed_match.design.util import human_readable_number
from trimmed_match.design.util import expand_time_windows
from trimmed_match.design.util import CalculateMinDetectableIroas
from trimmed_match.design.util import format_design_table, create_output_table



warnings.filterwarnings('ignore')
InteractiveShell.ast_node_interactivity = "all"

#@markdown ---
#@markdown ### Enter the trix url for the sheet file containing the Client Sales Data:
#@markdown The spreadsheet should contain the mandatory columns:
#@markdown * date: date in the format YYYY-MM-DD
#@markdown * geo: the number which identifies the geo
#@markdown * response: variable on which you want to measure incrementality
#@markdown (e.g. sales, transactions)
#@markdown * cost: variable used as spend proxy (e.g. ad spend)
#@markdown * (optional) other columns can be present in the spreadsheet.

#@markdown ---

## load the trix in input

#@markdown Spreadsheet URL
client_sales_table = "https://docs.google.com/spreadsheets/d/1lkZQRCAJrlA49S1ld2Ad5LcjPQxdqnA8732bR3gMqhs/edit?gid=1997936658#gid=1997936658" #@param {type:"string"}
auth.authenticate_user()
creds, _ = google_auth.default()
gc = gspread.authorize(creds)
def read_trix(url: str):
  wks = gc.open_by_url(url).sheet1
  data = wks.get_all_values()
  headers = data.pop(0)
  return pd.DataFrame(data, columns=headers)
geo_level_time_series = read_trix(client_sales_table)

#@markdown ### [OPTIONAL] Enter the trix url for the sheet file containing the geo pairs:
#@markdown The spreadsheet should contain the mandatory columns:
#@markdown * geo: the number which identifies the geo
#@markdown * pair: the number which identifies the pair for the corresponding
#@markdown geo. Pair numbers should be 1,...,N for a pairing with N pairs.
#@markdown * assignment: a string, where "Filtered" indicates pairs excluded.
#@markdown Use any other value for the assignment column for pairs that
#@markdown should be included.

#@markdown NOTE: Avoid using the data from the chosen evaluation_period below
#@markdown for generating the pairing (if passed here) as this could lead to
#@markdown overfitting.


pairs_table = "" #@param {type:"string"}

geo_level_time_series = check_input_data(geo_level_time_series)

if pairs_table == "":
  pairing = None
  filtered_geos = []
else:
  pairs = read_trix(pairs_table)
  if not set(["pair", "geo"]).issubset(pairs.columns):
    raise ValueError("The sheet in input must have the columns " +
                     f"['pair', 'geo'], got {pairs.columns}")
  for colname in ["pair", "geo"]:
    pairs[colname] = pd.to_numeric(pairs[colname])

  pairs = pairs.loc[pairs["assignment"] != "Filtered"]
  pairs.sort_values(by="pair", inplace=True)
  pairing = [pd.DataFrame({"geo1": pairs["geo"].values[::2],
                           "geo2": pairs["geo"].values[1::2],
                           "pair": pairs["pair"].values[::2]})]

## set parameters in other cells of the colab based on the loaded data
number_of_weeks_test = 4 # length of the experiment in weeks
number_of_days_test = number_of_weeks_test * 7

"""# Select the parameters for the design"""

#@title Select the parameters for the design of the experiment

use_cross_validation = True

#@markdown Specification of the GeoXType
geox_type = 'HOLD_BACK' #@param['HOLD_BACK', 'HEAVY_UP', 'GO_DARK'] {type:'string'}
geox_type = GeoXType[geox_type]

#@markdown Minimum detectable iROAS
minimum_detectable_iROAS =  3 #@param{type: "number"}

#@markdown Average value per unit response: 1 if the response is sales/revenue, else the average value (e.g. 80 USD) per transactions/footfall/contracts/etc.
average_order_value =  1#@param{type: "number"}

#@markdown Design framework based on hypothesis testing H0: iROAS = 0 vs. H1: iROAS >= minimal_detectable_iROAS
significance_level = 0.10 #@param {type:"number"}
power_level = 0.80 #@param {type:"number"}
calc_min_detectable_iroas = CalculateMinDetectableIroas(significance_level,
                                                        power_level)

#@markdown Configuration of the test duration and pre-test data for the design
experiment_duration_in_weeks = 4 #@param {type:"integer"}
design_start_date = "\"2020-01-01\"" #@param {type:"date"}
design_end_date = "\"2020-12-29\"" #@param {type:"date"}
eval_start_date = "\"2020-12-02\"" #@param {type:"date"}
coverage_test_start_date = "\"2020-11-04\"" #@param {type:"date"}

#@markdown List the maximum budget for the experiment e.g. 300000
experiment_budget = "300000" #@param{type: "string"}
experiment_budget = float(experiment_budget)

#@markdown List any alternative budget which you would like to test separated
#@markdown by a comma, e.g. 125000, 150000
alternative_budget = "125000" #@param{type: "string"}
additional_budget = [float(re.sub(r"\W+", "", x)) for x in
                     alternative_budget.split(',') if alternative_budget != ""]

## Additional constraints which will be flagged in red if not met in
## the design

# upper bound on the minimal detectable relative lift
minimum_detectable_lift_in_response_metric = 0.1 * 100
# lower bound on the baseline revenue covered by the treatment group
minimum_revenue_covered_by_treatment = 0.05 * 100

#@markdown List the geo_id of the geos you want to exclude separated by
#@markdown a comma e.g. 100,200. Leave empty to select all geos.
geos_exclude = "2,13,14" #@param {type: "string"}
geos_exclude = [] if geos_exclude == "" else [re.sub(r"\W+", "", x) for x in
                                              geos_exclude.split(',')]

#@markdown List the days and time periods that you want to exclude separated by
#@markdown a comma e.g. 2019/10/10, 2010/10/11, 2018/10/20-2018/11/20. The format for time periods
#@markdown is "YYYY/MM/DD - YYYY/MM/DD", where the two dates specify the
#@markdown start and end date for the period. The format for
#@markdown day is "YYYY/MM/DD". Leave empty to use all days/weeks.
day_week_exclude = "" #@param {type: "string"}
day_week_exclude = [] if day_week_exclude == "" else [
    re.sub(r"\s+", "", x) for x in day_week_exclude.split(",")
]


## convert input dates to datetimes (needed due to interactive parameters)
design_start_date = pd.to_datetime(design_start_date.replace("\"",""))
design_end_date = pd.to_datetime(design_end_date.replace("\"",""))
eval_start_date = pd.to_datetime(eval_start_date.replace("\"",""))
coverage_test_start_date = pd.to_datetime(coverage_test_start_date.replace("\"",
                                                                           ""))

number_of_days_test = experiment_duration_in_weeks * 7
eval_end_date = eval_start_date + datetime.timedelta(days=number_of_days_test-1)
coverage_test_end_date = coverage_test_start_date + datetime.timedelta(
    days=number_of_days_test - 1)

design_start_date = min(design_start_date, coverage_test_start_date,
                        eval_start_date)
design_end_date = max(design_end_date, coverage_test_end_date, eval_end_date)

## Find all the days we should exclude from the analysis from the input
periods_to_exclude = find_days_to_exclude(day_week_exclude)
days_exclude = expand_time_windows(periods_to_exclude)

## remove the excluded days from the rest of the analysis
geo_time_series = geo_level_time_series.copy()
geo_time_series = geo_time_series[~geo_time_series["date"].isin(days_exclude)]


## check that the user doesn't attempt by mistake to remove
## days/weeks in the evaluation or AA test periods.
days_in_eval = [
    x for x in geo_level_time_series["date"].drop_duplicates()
    if x in pd.Interval(eval_start_date, eval_end_date, closed="both")
]

days_in_coverage_test = [
    x for x in geo_level_time_series["date"].drop_duplicates()
    if x in pd.Interval(coverage_test_start_date, coverage_test_end_date,
                        closed="both")]

percentage_overlap_eval = overlap_percent(days_exclude, days_in_eval)
if percentage_overlap_eval > 0:
  raise ValueError((f'{Fore.RED}WARNING: {percentage_overlap_eval:.2} % of  ' +
                    f'the evaluation time period overlaps with days/weeks ' +
                    f'excluded in input. Please change eval_start_date.' +
                    f'\n{Style.RESET_ALL}'))

percentage_overlap_coverage_test = overlap_percent(days_exclude,
                                                   days_in_coverage_test)
if percentage_overlap_coverage_test > 0:
  raise ValueError(
      f'{Fore.RED}WARNING: {percentage_overlap_coverage_test:.2} % of '
      f'the aa test  time period overlaps with days/weeks ' +
      f'excluded in input. Please change coverage_test_start_date.' +
      f'\n{Style.RESET_ALL}')


## check that the evaluation and AA test periods do not
## overlap (if the user has changed them)
percentage_overlap_eval_coverage_test = overlap_percent(days_in_eval,
                                                        days_in_coverage_test)
if percentage_overlap_eval_coverage_test > 0:
  raise ValueError(f'{Fore.RED}WARNING: part of the evaluation time period ' +
                   f'overlaps with the coverage test period. Please change ' +
                   f'eval_start_date.\n{Style.RESET_ALL}')

try:
  pass_checks = check_time_periods(geox_data=geo_level_time_series,
                       start_date_eval=eval_start_date,
                       start_date_aa_test=coverage_test_start_date,
                       experiment_duration_weeks=experiment_duration_in_weeks,
                       frequency="infer")
except Exception as e:
  print(f'{Fore.RED} ERROR: ' + str(e) + f'\n{Style.RESET_ALL}')
  error_raised = e
  pass_checks = False

## check that the number of geos is even
geos_exclude = [int(x) for x in geos_exclude]
all_geos = set(geo_level_time_series["geo"].to_list())
non_existing_geos = set(geos_exclude) - set(all_geos)
if non_existing_geos:
  geos_exclude = [x for x in geos_exclude if x not in non_existing_geos]
  print(f'{Fore.RED}WARNING: Attempting to exclude the geos ' +
        f'{non_existing_geos} which do not exist in ' +
        f'the input trix.\n{Style.RESET_ALL}')
num_geos = len(all_geos - set(geos_exclude))
if num_geos % 2 != 0:
  geo_level_data = geo_level_time_series.groupby(
      "geo", as_index=False)["response"].sum()
  largest_geo = geo_level_data.loc[geo_level_data["response"].idxmax()]
  print(f'\nSince the number of geos is odd, we have removed the following' +
  f' geo (the one with largest response):')
  largest_geo
  geos_exclude.append(largest_geo["geo"])

#@title Summary of the possible designs

if min([percentage_overlap_eval, percentage_overlap_coverage_test,
        percentage_overlap_eval_coverage_test]) > 0:
  raise ValueError(f'{Fore.RED}Either the evaluation time period or the AA ' +
                   f'test period overlaps with days/weeks excluded in input, ' +
                   f'or these two periods overlap. Please change them ' +
                   f'accordingly.\n{Style.RESET_ALL}')

if not pass_checks:
  raise ValueError(f'{Fore.RED} There is an error with the evaluation or' +
  f' aa test period in the previous colab cell.\nPlease correct that error' +
  f' and then rerun this cell.\n' +
  f'Previous error:\n' + str(error_raised) + '{Style.RESET_ALL}')

## set the number of excluded pairs to be tested
num_geos = len(set(geo_level_time_series["geo"].to_list()) - set(geos_exclude))
max_num_geopairs_trim = int(np.floor(num_geos/2 - 10))
# if any geo pairs was specified as filtered, we do not attempt further
# filtering

## set the number of simulation used to compute the RMSE
number_of_simulations = 200

### Evaluate the RMSE of the possible designs ###

## remove the AA test period to make sure it's not used in the evaluation
## of the RMSE or in the training.
data_without_coverage_test_period = geo_time_series[
    (geo_time_series["date"] < coverage_test_start_date) |
    (geo_time_series["date"] > coverage_test_end_date)]

data_for_design = data_without_coverage_test_period.copy()
data_for_design = data_for_design[~data_for_design["geo"].isin(geos_exclude)]

time_window_for_design = TimeWindow(design_start_date, design_end_date)
time_window_for_eval = TimeWindow(eval_start_date, eval_end_date)
## initialize the TrimmedMatchGeoxDesign
pretest = TrimmedMatchGeoXDesign(
    geox_type=geox_type,
    pretest_data=data_for_design,
    response="response",
    spend_proxy="cost",
    matching_metrics={"response": 1.0, "cost": 0.01},
    time_window_for_design=time_window_for_design,
    time_window_for_eval=time_window_for_eval,
    pairs=pairing)

## run a first design with budget equal to the max. budget
preliminary_results, prel_results_detailed = pretest.report_candidate_designs(
    budget_list=[experiment_budget],
    iroas_list=[0],
    use_cross_validation=use_cross_validation,
    num_simulations=number_of_simulations)

## calculate the minimum detectable iROAS for a design with max. budget
chosen_design = preliminary_results.loc[
    preliminary_results["rmse_cost_adjusted"].idxmin()].squeeze()
lowest_detectable_iroas = calc_min_detectable_iroas.at(chosen_design["rmse"])

## two cases are possible:
##   1) if the minimum detectable iROAS with max. budget is greater than
##      the minimum_detectable_iROAS in input, then calculate the budget needed
##      to reach the minimum_detectable_iROAS in input and run a design with
##      such budget. This is the code in the if clause below;
##   2) if the minimum detectable iROAS with max. budget is smaller than
##      the minimum_detectable_iROAS in input, then run designs with
##      budgets equal to the max. budget plus/minus 20%.
##      This is the code in the else clause below;
minimum_iroas_aov = minimum_detectable_iROAS / average_order_value
if lowest_detectable_iroas > minimum_iroas_aov:
  budget_to_reach_min_det_iroas = (experiment_budget * lowest_detectable_iroas
                                   / minimum_iroas_aov)
  additional_results, results_detailed = pretest.report_candidate_designs(
    budget_list=[budget_to_reach_min_det_iroas] + additional_budget,
    iroas_list=[0],
    use_cross_validation=use_cross_validation,
    num_simulations=number_of_simulations)

  results = pd.concat([preliminary_results, additional_results], sort=False)

else:
  optimal_budget = (experiment_budget * lowest_detectable_iroas /
                    minimum_iroas_aov)
  lower_budget = optimal_budget *  0.8
  upper_budget = optimal_budget * 1.2
  list_of_budgets = [lower_budget, optimal_budget, upper_budget
                    ] + additional_budget
  results, results_detailed = pretest.report_candidate_designs(
      budget_list=list_of_budgets,
      iroas_list=[0],
      use_cross_validation=use_cross_validation,
      num_simulations=number_of_simulations)


# these are numerical identifier used in the table in input to identify the two
# groups
group_treatment = GeoAssignment.TREATMENT
group_control = GeoAssignment.CONTROL
group_filtered = GeoAssignment.EXCLUDED

optimal_pair_index = preliminary_results.loc[
    preliminary_results["rmse_cost_adjusted"].idxmin(), "pair_index"]
axes_paired = pretest.output_chosen_design(pair_index=optimal_pair_index,
                                           base_seed=0,
                                           confidence=1-significance_level,
                                           group_control=group_control,
                                           group_treatment=group_treatment)
plt.close()

## assign geos to treatment and control groups
geopairs = pretest.geo_level_eval_data[optimal_pair_index]
geopairs.sort_values(by=["pair"], inplace=True)

geo_treatment = geopairs[geopairs["assignment"]==group_treatment]
geo_control = geopairs[geopairs["assignment"]==group_control]
treatment_geo = geo_treatment["geo"].to_list()
control_geo = geo_control["geo"].to_list()

budgets_for_design = results["budget"].drop_duplicates().to_list()

### AA test to check the coverage probability of the confidence interval ###

## remove the evaluation period to make sure it's not used in the evaluation
## of the RMSE or in the training.
data_without_eval_period = geo_time_series[
    (geo_time_series["date"] < eval_start_date) |
    (geo_time_series["date"] > eval_end_date)]


data_for_coverage_test = data_without_eval_period.copy()
data_for_coverage_test = data_for_coverage_test[
    ~data_for_coverage_test["geo"].isin(geos_exclude)]

time_window_for_coverage_test = TimeWindow(coverage_test_start_date,
                                           coverage_test_end_date)
## initialize the TrimmedMatchGeoxDesign
coverage_test_class = TrimmedMatchGeoXDesign(
    geox_type=geox_type,
    pretest_data=data_for_coverage_test,
    response="response",
    spend_proxy="cost",
    matching_metrics={"response": 1.0, "cost": 0.01},
    time_window_for_design=time_window_for_design,
    time_window_for_eval=time_window_for_coverage_test,
    pairs=pairing)

## calculate the point estimate for each simulation
aa_results, aa_results_detailed = coverage_test_class.report_candidate_designs(
    budget_list=[budgets_for_design[0]],
    iroas_list=[0],
    use_cross_validation=use_cross_validation,
    num_simulations=number_of_simulations)


## The code below this line only takes care of formatting the output

total_response = geo_level_time_series.loc[
    geo_level_time_series["date"].between(eval_start_date, eval_end_date),
    "response"].sum()
total_spend = geo_level_time_series.loc[
    geo_level_time_series["date"].between(eval_start_date, eval_end_date),
    "cost"].sum()

designs = create_output_table(results=results,
                              total_response=total_response,
                              total_spend=total_spend,
                              geo_treatment=geo_treatment,
                              budgets_for_design=budgets_for_design,
                              average_order_value=average_order_value,
                              num_geos=num_geos,
                              confidence_level=1-significance_level,
                              power_level=power_level)


designs_table = format_design_table(
    designs=designs,
    minimum_detectable_iroas=minimum_detectable_iROAS,
    minimum_lift_in_response_metric=minimum_detectable_lift_in_response_metric,
    minimum_revenue_covered_by_treatment=minimum_revenue_covered_by_treatment)

designs_table

#@title Select the design to be used in the experiment
#@markdown Select the design using the number as displayed in the table in
#@markdown the cell called "Summary of the possible designs".

selected_design =   1#@param {type:"integer"}

if selected_design not in designs.index:
  raise ValueError(f'the selected design must be one of {designs.index.to_list()}, got {selected_design}')

selected_design = int(selected_design)
final_design = designs[designs.index == selected_design]
selected_budget = final_design["Budget"].values[0]


## Uncomment the following line to override the automatic choice for
## the pairing to be used. For example, using optimal_pair_index = 5 will
## use the 5th pairing, and for the default pairing this means
## that we filter out the pairs 1, 2, 3, 4, 5.

# optimal_pair_index = 5

###


axes_paired = pretest.output_chosen_design(pair_index=optimal_pair_index,
                                           base_seed=0,
                                           confidence=1-significance_level,
                                           group_control=group_control,
                                           group_treatment=group_treatment)
plt.close()

#@title Scatterplot and time series comparison of different metrics for treatment vs. control groups

for ax in axes_paired[1]:
  ylim = ax.get_ylim()
  for period in periods_to_exclude:
    if period.first_day < period.last_day:
      useless=ax.fill_between([period.first_day, period.last_day], ylim[0], ylim[1],
                              facecolor="gray", alpha=0.5)
    else:
      useless=ax.vlines(period.first_day, ylim[0], ylim[1],
                         color="gray", alpha=0.5)

  handles, labels = ax.get_legend_handles_labels()
  patch = mpatches.Patch(color='grey', label='removed by the user')
  handles.append(patch)
  useless=ax.fill_between([coverage_test_start_date, coverage_test_end_date],
                          ylim[0], ylim[1], facecolor="g", alpha=0.5)
  patch = mpatches.Patch(color='g', label='left out for AA test')
  handles.append(patch)
  useless=ax.legend(handles=handles, loc='best')

axes_paired[1,1].figure

#@title Plot each pair of geos for comparison
g = pretest.plot_pair_by_pair_comparison(pair_index=optimal_pair_index,
                                         group_control=group_control,
                                         group_treatment=group_treatment)

"""# Summary of the design and save the pretest data, the geopairs, treatment and control stores in a trix."""

#@title Summary and Results

geopairs = pretest._geo_level_eval_data[optimal_pair_index]
geopairs.sort_values(by=["pair"], inplace=True)
treatment_geo = geopairs.loc[geopairs["assignment"] == group_treatment,
                             "geo"].to_list()
control_geo = geopairs.loc[geopairs["assignment"] == group_control,
                           "geo"].to_list()

temporary = geo_level_time_series[geo_level_time_series["geo"].isin(
    treatment_geo)]
treatment_time_series = temporary[temporary["date"].between(
    design_start_date, design_end_date)].groupby(
        "date", as_index=False)[["response", "cost"]].sum()

temporary = geo_level_time_series[geo_level_time_series["geo"].isin(
    control_geo)]
control_time_series = temporary[temporary["date"].between(
    design_start_date, design_end_date)].groupby(
        "date", as_index=False)[["response", "cost"]].sum()

eval_window = treatment_time_series["date"].between(eval_start_date,
                                                    eval_end_date)
baseline = treatment_time_series[eval_window]["response"].sum()

result_to_out = results[results["budget"] ==
                        budgets_for_design[selected_design]]

print("Data in input:\n")
print("-  {} Geos \n".format(
    len(geo_level_time_series["geo"].drop_duplicates().index)))

print("Output:\n")
print("The output contains two lists of geos: one for treatment" +
      " and the other for control\n")

human_baseline = human_readable_number(baseline)
cost_baseline = budgets_for_design[selected_design] * 100 / baseline
print("-  {} Geo pairs for the experiment\n".format(len(treatment_geo)))
print("    Baseline store response: ${} for treatment\n".format(human_baseline))
print("    Cost/baseline = ${} / ${} ~ {:.3}%\n".format(selected_budget,
                                                        human_baseline,
                                                        cost_baseline))

summary_rmse = result_to_out.loc[result_to_out["pair_index"]==
                                    optimal_pair_index, "rmse"].values[0]
summary_minimum_detectable_iroas = calc_min_detectable_iroas.at(summary_rmse)
summary_minimum_detectable_lift = (cost_baseline *
                                   summary_minimum_detectable_iroas)
summary_minimum_detectable_iroas_aov = (
    summary_minimum_detectable_iroas * average_order_value)
print(f'Minimum detectable iROAS = ' +
      f'{summary_minimum_detectable_iroas_aov:.3}')
print(f'Minimum detectable lift in % = ' +
      f'{summary_minimum_detectable_lift:.2f}')

print(f"The design has Power {100 * power_level:.3}+% with Type-I error " +
      f"{100 *(significance_level):.3}% for testing H0: iROAS=0 vs " +
      f"H1: iROAS >= {summary_minimum_detectable_iroas_aov:.3}")

#@title Report stores for treatment and control separately and write to trix

#@markdown ###Insert the name google sheets in which we will save the data.
#@markdown The trix contains 4 worksheets, named:
#@markdown * "pretest data", containing the geo level time series;
#@markdown * "geopairs", containing the pairs of geos and their assignment.
#@markdown * "treatment geos", contains the list of geos in the treatment;
#@markdown * "control geos", contains the geos in the control groups.
Client_Name = "Client_Name" #@param {type:"string"}
filename_design = Client_Name + "_design.csv" #@param {type:"string"}

geopairs_formatted = pretest._pretest_data[
    pretest._pretest_data['date'].between(
        time_window_for_eval.first_day, time_window_for_eval.last_day)].groupby(
            'geo', as_index=False).sum()
geopairs_formatted = geopairs_formatted.merge(
    geopairs[['geo', 'pair', 'assignment']], how='left', on='geo').fillna({
        'pair': group_filtered,
        'assignment': group_filtered
    }).sort_values(by=["pair", "geo"])[["geo", "pair", "response",
                                        "cost", "assignment"]]
geopairs_formatted["assignment"] = geopairs_formatted["assignment"].map({
    group_filtered: "Filtered",
    group_control: "Control",
    group_treatment: "Treatment"
})

geo_level_time_series["period"] = [
   # -3 indicates days excluded
   -3 if x in days_exclude else (
   # 0 indicates days in the evaluation period
   0 if x>=eval_start_date and x<=eval_end_date else (
   # -2 indicates days in the coverage_test period
   -2 if x >= coverage_test_start_date and x <= coverage_test_end_date
   # -1 indicates days in the training period
   else -1))
   for x in geo_level_time_series["date"]
]

tmp = geo_level_time_series[geo_level_time_series["geo"].isin(treatment_geo +
                                                              control_geo)]
design_data = tmp.merge(
    geopairs[["geo", "pair", "assignment"]], on="geo", how="left")

tmp_parameters = {
    "geox_type": str(geox_type).replace("GeoXType.", ""),
    "minimum_detectable_iROAS": minimum_detectable_iROAS,
    "average_order_value": average_order_value,
    "significance_level": significance_level,
    "power_level": power_level,
    "experiment_duration_in_weeks": experiment_duration_in_weeks,
    "design_start_date": design_start_date.strftime("%Y-%m-%d"),
    "design_end_date": design_end_date.strftime("%Y-%m-%d"),
    "eval_start_date": eval_start_date.strftime("%Y-%m-%d"),
    "coverage_test_start_date": coverage_test_start_date.strftime("%Y-%m-%d"),
    "experiment_budget": experiment_budget,
    "alternative_budget": alternative_budget,
    "geo_exclude": ", ".join(str(x) for x in geos_exclude),
    "day_week_exclude": ", ".join(day_week_exclude),
    "selected_design": selected_design,
    "pair_index": str(optimal_pair_index)
}

parameters = {"parameter": list(tmp_parameters.keys()),
              "value": list(tmp_parameters.values())}


sh = gc.create(filename_design)
wid = sh.add_worksheet("pretest data", rows=1, cols=1)
set_with_dataframe(wid, design_data)
wid = sh.add_worksheet("geopairs", rows=1, cols=1)
set_with_dataframe(wid, geopairs_formatted)
wid = sh.add_worksheet("treatment geos", rows=1, cols=1)
set_with_dataframe(wid, pd.DataFrame({"geo": treatment_geo}))
wid = sh.add_worksheet("control geos", rows=1, cols=1)
set_with_dataframe(wid, pd.DataFrame({"geo": control_geo}))
wid = sh.add_worksheet("parameters used in the design", rows=1, cols=1)
set_with_dataframe(wid, pd.DataFrame(parameters))
out = sh.del_worksheet(sh.sheet1)

"""# Appendix:"""

#@markdown The following cell is optional and show the graph/table behind the automatic designs presented above

#@title Plot of the RMSE as a function of the # of trimmed pairs

#@markdown The first graph below shows the RMSE of the iROAS estimator
#@markdown with respect to the number of excluded geo pairs,
#@markdown with the baseline store sales (treatment group + control group)
#@markdown displayed next to each point of (# excluded pairs, RMSE).

#@markdown The second graph below shows the proportion of confidence
#@markdown intervals that cointains zero in an A/A test.

coverage_test_result = []
for pair_index in aa_results["pair_index"].unique():
  temp_df = aa_results_detailed[(budgets_for_design[0], 0, pair_index)]
  temp_df["contains_zero"] = (temp_df["conf_interval_low"] <= 0) & (
    temp_df["conf_interval_up"] >= 0)
  coverage = temp_df['contains_zero'].mean()
  coverage_test_result.append({"coverage": coverage,
                               "pair_index": pair_index,
                               "budget": budgets_for_design[0],
                               "ci_level": temp_df["ci_level"][0]})

coverage_test_result = pd.DataFrame(coverage_test_result)

# call the function that creates all the axis
axes_dict = pretest.plot_candidate_design(results=results)
## display the results of the designs with different budgets in different tabs
list_of_str = ["budget = " + human_readable_number(budget)+"$"
               for budget in budgets_for_design]
tb = widgets.TabBar(list_of_str)
for ind in range(len(budgets_for_design)):

  result = results[results["budget"]==budgets_for_design[ind]].reset_index(
      drop=True)
  with tb.output_to(ind):
    ax = axes_dict[(budgets_for_design[ind], 0)].axes[0]
    labels = ax.get_yticks().tolist()
    labels = [str(round(float(x)*average_order_value, 2)) for x in labels]
    useless = ax.set_yticklabels(labels)
    if len(coverage_test_result) > 1:
      display(axes_dict[(budgets_for_design[ind], 0)])

    print_result = result.copy()
    print_result["rmse"] = print_result["rmse"] * average_order_value
    print_result["rmse_cost_adjusted"] = (print_result["rmse_cost_adjusted"]
                                          * average_order_value)
    data_table.DataTable(print_result[["pair_index", "num_pairs",
                 "rmse", "rmse_cost_adjusted",
                 "experiment_response"]],
                 include_index=False)

if len(coverage_test_result) > 1:
  fig1 = plt.figure(figsize=(15, 7.5))
  ax1 = fig1.add_subplot(1, 1, 1)
  useless=ax1.plot(
      coverage_test_result['pair_index'],
      coverage_test_result['coverage'], 'blue',
      label='Coverage')
  useless=ax1.hlines(
      y=coverage_test_result['ci_level'][0],
      xmin=min(coverage_test_result["pair_index"]),
      xmax=max(coverage_test_result["pair_index"]),
      colors='red',
      linestyles='dashed',
      label='Confidence level (nominal)')
  useless=ax1.set_xlabel('Pairing number')
  useless=ax1.set_ylabel('Coverage')
  useless=ax1.set_title('A/A test confidence interval coverage')
  useless=ax1.legend()
```

## File: design_trimmed_match/trimmed_match_postanalysis_colab.py

- Extension: .py
- Language: python
- Size: 11167 bytes
- Created: 2025-03-27 11:16:38
- Modified: 2025-03-26 11:21:26

### Code

```python
# -*- coding: utf-8 -*-
"""Trimmed Match PostAnalysis Colab.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/github/google/trimmed_match/blob/master/trimmed_match/notebook/post_analysis_colab_for_trimmed_match.ipynb
"""

#@markdown * Connect to the hosted runtime and run each cell after updating the necessary inputs
#@markdown * Download the file "example_data_for_post_analysis.csv" from the folder "example_datasets" in github.
#@markdown * Upload the csv file to your Google Drive and open it with Google Sheets
#@markdown * In the cell below, copy and paste the url of the sheet.

#@markdown ### Load the required packages, e.g. trimmed_match.

BAZEL_VERSION = '6.1.2'
!wget https://github.com/bazelbuild/bazel/releases/download/{BAZEL_VERSION}/bazel-{BAZEL_VERSION}-installer-linux-x86_64.sh
!chmod +x bazel-{BAZEL_VERSION}-installer-linux-x86_64.sh
!./bazel-{BAZEL_VERSION}-installer-linux-x86_64.sh
!sudo apt-get install python3-dev python3-setuptools git
!git clone https://github.com/google/trimmed_match
!python3 -m pip install ./trimmed_match

"""Loading the necessary python modules."""
import matplotlib.pyplot as plt
import pandas as pd
import re
import seaborn as sns

from IPython.display import display
from IPython.core.interactiveshell import InteractiveShell
from pandas.plotting import register_matplotlib_converters

import gspread
import warnings
from google import auth as google_auth
from google.colab import auth
from google.colab import data_table
from google.colab import drive
from trimmed_match.design.common_classes import GeoAssignment
from trimmed_match.design import plot_utilities
from trimmed_match.design import util
from trimmed_match.post_analysis import trimmed_match_post_analysis

warnings.filterwarnings('ignore')
register_matplotlib_converters()
InteractiveShell.ast_node_interactivity = "all"

#@markdown ### Enter the trix id for the sheet file containing the Data:
#@markdown The spreadsheet should contain the mandatory columns:
#@markdown * date: date in the format YYYY-MM-DD
#@markdown * geo: the number which identifies the geo
#@markdown * pair: the number which identifies the geo pair
#@markdown * assignment: geo assignment (1=Treatment, 2=Control)
#@markdown * response: variable on which you want to measure incrementality
#@markdown (e.g. sales, transactions)
#@markdown * cost: variable on ad spend

#@markdown ---

## load the trix in input
#@markdown Spreadsheet URL


experiment_table = "add your url here, which should look like https://docs.google.com/spreadsheets/d/???/edit#gid=???" #@param {type:"string"}
auth.authenticate_user()
creds, _ = google_auth.default()
gc = gspread.authorize(creds)
wks = gc.open_by_url(experiment_table).sheet1
data = wks.get_all_values()
headers = data.pop(0)
data = pd.DataFrame(data, columns=headers)

data["date"] = pd.to_datetime(data["date"])
for colname in ["geo", "pair", "assignment", "response", "cost"]:
  data[colname] = pd.to_numeric(data[colname])

#@title Summary of the data for the design, test, and test+cooldown period

test_start_date = "2020-11-04" #@param {type:"date"}
test_end_date = "2020-12-01" #@param {type:"date"}
cooldown_end_date = "2020-12-16" #@param {type:"date"}
design_eval_start_date = "2020-09-03" #@param {type:"date"}
design_eval_end_date = "2020-10-01" #@param {type:"date"}

#@markdown Use an average order value of 1 if the experiment is based on sales/revenue or an actual average order value (e.g. 80$) for an experiment based on transactions/footfall/contracts.
average_order_value =  1#@param{type: "number"}

test_start_date = pd.to_datetime(test_start_date)
test_end_date = pd.to_datetime(test_end_date)
cooldown_end_date = pd.to_datetime(cooldown_end_date)
design_eval_start_date = pd.to_datetime(design_eval_start_date)
design_eval_end_date = pd.to_datetime(design_eval_end_date)

#@markdown (OPTIONAL) List the pairs of geos you need to exclude separated by a comma e.g. 1,2. Leave empty to select all pairs.
pairs_exclude = "" #@param {type: "string"}
pairs_exclude = [] if pairs_exclude == "" else [
    int(re.sub(r"\W+", "", x)) for x in pairs_exclude.split(",")
]

# these are numerical identifier used in the table in input to identify the two
# groups
group_treatment = GeoAssignment.TREATMENT
group_control = GeoAssignment.CONTROL

geox_data = trimmed_match_post_analysis.check_input_data(
    data.copy(),
    group_control=group_control,
    group_treatment=group_treatment)
geox_data = geox_data[~geox_data["pair"].isin(pairs_exclude)]

geox_data["period"] = geox_data["date"].apply(
    lambda row: 0 if row in pd.Interval(
        design_eval_start_date, design_eval_end_date, closed="both") else
    (1 if row in pd.Interval(test_start_date, test_end_date, closed="both") else
     (2 if row in pd.Interval(test_end_date, cooldown_end_date, closed="right")
      else -1)))
geox_data = geox_data[["date", "geo", "pair", "assignment", "response", "cost",
       "period"]]
pairs = geox_data["pair"].sort_values().drop_duplicates().to_list()

total_cost = geox_data.loc[geox_data["period"]==1, "cost"].sum()
print("Total cost: {}".format(util.human_readable_number(total_cost)))

print("Total response and cost by period and group")
output_table = geox_data.loc[
    geox_data["period"].isin([0, 1]),
    ["period", "assignment", "response", "cost"]].groupby(
        ["period", "assignment"], as_index=False).sum()
output_table.assignment = output_table.assignment.map(
    {group_control: "Control", group_treatment: "Treatment"})
output_table.period = output_table.period.map({0: "Pretest", 1: "Test"})

data_table.DataTable(output_table, include_index=False)

tmp = geox_data[geox_data["period"].isin([0, 1])].groupby(
    ["period", "assignment", "pair"])["response"].sum()**0.5
tmp = tmp.reset_index()

pretreatment = (tmp["period"]==0) & (tmp["assignment"]==group_treatment)
precontrol = (tmp["period"]==0) & (tmp["assignment"]==group_control)
posttreatment = (tmp["period"]==1) & (tmp["assignment"]==group_treatment)
postcontrol = (tmp["period"]==1) & (tmp["assignment"]==group_control)

comp = pd.DataFrame({"pretreatment": tmp[pretreatment]["response"].to_list(),
                   "precontrol": tmp[precontrol]["response"].to_list(),
                   "posttreatment": tmp[posttreatment]["response"].to_list(),
                   "postcontrol": tmp[postcontrol]["response"].to_list()})


fig, ax = plt.subplots(4, 4, figsize=(15, 15))
label = ["pretreatment", "precontrol", "posttreatment", "postcontrol"]
min_ax = min(comp.min())
max_ax = max(comp.max())
for col_ind in range(4):
  for row_ind in range(4):
    if col_ind > row_ind:
      useless = ax[row_ind, col_ind].scatter(comp[label[col_ind]],
                                             comp[label[row_ind]])
      useless = ax[row_ind, col_ind].plot([min_ax*0.97, max_ax*1.03],
                                          [min_ax*0.97, max_ax*1.03], 'r')
      useless = ax[row_ind, col_ind].set_xlim([min_ax*0.97, max_ax*1.03])
      useless = ax[row_ind, col_ind].set_ylim([min_ax*0.97, max_ax*1.03])
    elif col_ind == row_ind:
      useless = ax[row_ind, col_ind].annotate(label[col_ind],
                                              size=20,
                                              xy=(0.15, 0.5),
                                              xycoords="axes fraction")
      useless = ax[row_ind, col_ind].set_xlim([min_ax*0.97, max_ax*1.03])
      useless = ax[row_ind, col_ind].set_ylim([min_ax*0.97, max_ax*1.03])
    else:
      useless = ax[row_ind, col_ind].axis("off")

#@title Visualization of experiment data.

geox_data = geox_data.sort_values(by="date")

def plot_ts_comparison(geox_data, metric):
  f, axes = plt.subplots(1,1, figsize=(15,7.5))
  treatment_time_series = geox_data[geox_data["assignment"] ==
                                    group_treatment].groupby(
                                        ["date"], as_index=False)[metric].sum()
  control_time_series = geox_data[geox_data["assignment"] ==
                                  group_control].groupby(
                                      ["date"], as_index=False)[metric].sum()
  axes.plot(treatment_time_series["date"], treatment_time_series[metric],
            label="treatment")
  axes.plot(control_time_series["date"], control_time_series[metric],
            label="control")
  axes.set_ylabel(metric)
  axes.set_xlabel("date")
  axes.axvline(x=test_end_date, color="black", ls="-",
               label='Experiment period')
  axes.axvline(x=design_eval_start_date, color="red", ls="--",
               label='Design evaluation period')
  axes.axvline(x=cooldown_end_date, color="black", ls="--",
               label='End of cooldown period')
  axes.axvline(x=test_start_date, color="black", ls="-")
  axes.axvline(x=design_eval_end_date, color="red", ls="--")
  axes.legend(bbox_to_anchor=(0.5,1.1), loc='center')

plot_ts_comparison(geox_data, "response")

plot_ts_comparison(geox_data, "cost")

def ts_plot(x,y, **kwargs):
  ax=plt.gca()
  data=kwargs.pop("data")
  data.plot(x=x, y=y, ax=ax, grid=False, **kwargs)

g = sns.FacetGrid(geox_data, col="pair", hue="assignment", col_wrap=3,
                  sharey=False,sharex=False, legend_out=False, height=5,
                  aspect=2)
g = (g.map_dataframe(ts_plot, "date", "response").add_legend())
for ind in range(len(g.axes)):
  cont=geox_data[(geox_data["pair"]==pairs[ind]) &
                 (geox_data["assignment"]==group_control)]["geo"].values[0]
  treat=geox_data[(geox_data["pair"]==pairs[ind]) &
                  (geox_data["assignment"]==group_treatment)]["geo"].values[0]
  useless = g.axes[ind].axvline(x=test_end_date, color="black", ls="-")
  useless = g.axes[ind].axvline(x=design_eval_start_date, color="red", ls="--")
  useless = g.axes[ind].axvline(x=cooldown_end_date, color="black", ls="--")
  useless = g.axes[ind].axvline(x=test_start_date, color="black", ls="-")
  useless = g.axes[ind].axvline(x=design_eval_end_date, color="red", ls="--")
  useless = g.axes[ind].legend(["treatment"+" (geo {})".format(treat),
                                "control"+" (geo {})".format(cont),
                                "Experiment period", "Design evaluation period",
                                "End of cooldown period"], loc="best")

#@title Exclude the cooling down period.

geo_data = trimmed_match_post_analysis.prepare_data_for_post_analysis(
    geox_data=geox_data,
    exclude_cooldown=True,
    group_control=group_control,
    group_treatment=group_treatment
)

results = trimmed_match_post_analysis.calculate_experiment_results(geo_data)
trimmed_match_post_analysis.report_experiment_results(results, average_order_value)

#@title Include the cooling down period

geo_data_including_cooldown = trimmed_match_post_analysis.prepare_data_for_post_analysis(
    geox_data=geox_data,
    exclude_cooldown=False,
    group_control=group_control,
    group_treatment=group_treatment
)

results_with_cd = trimmed_match_post_analysis.calculate_experiment_results(
    geo_data_including_cooldown)
trimmed_match_post_analysis.report_experiment_results(results_with_cd, average_order_value)
```

## File: src/data_pipeline/data_standardizer.py

- Extension: .py
- Language: python
- Size: 6837 bytes
- Created: 2025-03-27 12:26:15
- Modified: 2025-03-27 12:26:15

### Code

```python
"""
Data standardization utilities for the geo-causal-inference project.

This module provides classes and functions to standardize data from different sources
for consistent analysis and joining.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional


class DateStandardizer:
    """Class for standardizing date formats across different datasets."""
    
    def __init__(self, output_format: str = '%Y-%m-%d'):
        """
        Initialize the DateStandardizer.
        
        Args:
            output_format: The desired output date format (default: '%Y-%m-%d')
        """
        self.output_format = output_format
        
    def standardize(self, df: pd.DataFrame, date_col: str, input_format: Optional[str] = None) -> pd.DataFrame:
        """
        Standardize date format in a dataframe.
        
        Args:
            df: The dataframe containing the date column
            date_col: Name of the date column to standardize
            input_format: Format of the input date. If None, tries to infer format.
            
        Returns:
            DataFrame with standardized date column
        """
        df = df.copy()
        
        # If input is already datetime, just format it
        if pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = df[date_col].dt.strftime(self.output_format)
            return df
        
        # Handle numeric YYYYMMDD format (like in GA4)
        if input_format is None and pd.api.types.is_numeric_dtype(df[date_col]):
            date_val = str(df[date_col].iloc[0])
            if len(date_val) == 8:  # YYYYMMDD format
                df[date_col] = pd.to_datetime(df[date_col], format='%Y%m%d')
            else:
                # Try generic parsing for numeric dates
                df[date_col] = pd.to_datetime(df[date_col])
        # Handle string dates
        elif input_format is None and isinstance(df[date_col].iloc[0], str):
            # Handle M/D/YY format (like in TikTok)
            if '/' in df[date_col].iloc[0]:
                df[date_col] = pd.to_datetime(df[date_col], format='%m/%d/%y')
            else:
                # Use generic parsing for other string formats
                df[date_col] = pd.to_datetime(df[date_col])
        # Use specified format or try to infer
        else:
            if input_format:
                df[date_col] = pd.to_datetime(df[date_col], format=input_format)
            else:
                df[date_col] = pd.to_datetime(df[date_col])
        
        # Convert to output format
        df[date_col] = df[date_col].dt.strftime(self.output_format)
        return df


class GeoStandardizer:
    """Class for standardizing geographic data across different datasets."""
    
    def __init__(self, region_mappings: Optional[Dict[str, str]] = None):
        """
        Initialize the GeoStandardizer.
        
        Args:
            region_mappings: Optional dictionary mapping non-standard region names to standard ones
        """
        self.region_mappings = region_mappings or {}
        
    def standardize(self, 
                   df: pd.DataFrame, 
                   geo_cols: Union[List[str], str], 
                   output_col: str = 'geo',
                   geo_level: str = 'region') -> pd.DataFrame:
        """
        Standardize geographic data in a dataframe.
        
        Args:
            df: The dataframe containing geo columns
            geo_cols: Column name(s) to use for geo standardization
            output_col: Name of the standardized output column
            geo_level: Level of geographic granularity to standardize to ('region', 'city', etc.)
            
        Returns:
            DataFrame with standardized geo column
        """
        df = df.copy()
        
        if isinstance(geo_cols, str):
            geo_cols = [geo_cols]
        
        # Case 1: Use first valid geo column
        for col in geo_cols:
            if col in df.columns:
                df[output_col] = df[col].str.strip() if isinstance(df[col].iloc[0], str) else df[col]
                break
        
        # Standardize names - convert to uppercase for consistent matching
        if output_col in df.columns and isinstance(df[output_col].iloc[0], str):
            df[output_col] = df[output_col].str.strip().str.upper()
            
            # Handle special cases
            # Replace "(NOT SET)" with "UNKNOWN"
            df[output_col] = df[output_col].replace(r'\(NOT SET\)', 'UNKNOWN', regex=True)
            
            # Remove state/region codes in parentheses if present
            df[output_col] = df[output_col].str.replace(r'\s*\([A-Z]{2}\)$', '', regex=True)
            
            # Apply custom region mappings if available
            if self.region_mappings:
                df[output_col] = df[output_col].replace(self.region_mappings)
        
        return df


class CostStandardizer:
    """Class for standardizing cost/spend data across different datasets."""
    
    def standardize(self, df: pd.DataFrame, cost_col: str) -> pd.DataFrame:
        """
        Standardize cost/spend data in a dataframe.
        
        Args:
            df: The dataframe containing the cost column
            cost_col: Column name containing cost data
            
        Returns:
            DataFrame with standardized cost column
        """
        df = df.copy()
        
        # Ensure cost column exists
        if cost_col not in df.columns:
            raise ValueError(f"Cost column '{cost_col}' not found in dataframe")
        
        # Handle string values with currency symbols
        if df[cost_col].dtype == 'object':
            df[cost_col] = df[cost_col].replace('[$,]', '', regex=True)
        
        # Convert to float
        df[cost_col] = pd.to_numeric(df[cost_col], errors='coerce')
        
        # Fill NaN with 0
        df[cost_col] = df[cost_col].fillna(0)
        
        return df


class DataAggregator:
    """Class for aggregating data by specified dimensions."""
    
    def aggregate(self, 
                 df: pd.DataFrame, 
                 group_cols: List[str], 
                 value_cols: Union[List[str], str], 
                 agg_func: Union[str, Dict] = 'sum') -> pd.DataFrame:
        """
        Aggregate data by specified dimensions.
        
        Args:
            df: The dataframe to aggregate
            group_cols: Columns to group by
            value_cols: Column(s) containing the values to aggregate
            agg_func: Aggregation function to apply
            
        Returns:
            Aggregated dataframe
        """
        if isinstance(value_cols, str):
            value_cols = [value_cols]
            
        return df.groupby(group_cols)[value_cols].agg(agg_func).reset_index()

```

## File: src/data_pipeline/__init__.py

- Extension: .py
- Language: python
- Size: 499 bytes
- Created: 2025-03-27 12:17:56
- Modified: 2025-03-27 12:17:56

### Code

```python
"""
Data pipeline package for geo-causal-inference project.

This package provides utilities for standardizing, cleaning, and joining
marketing data from multiple sources for geo-causal analysis.
"""

from .data_standardizer import DateStandardizer, GeoStandardizer, CostStandardizer, DataAggregator
from .data_joiner import DataJoiner, DatasetCleaner

__all__ = [
    'DateStandardizer',
    'GeoStandardizer',
    'CostStandardizer',
    'DataAggregator',
    'DataJoiner',
    'DatasetCleaner'
]

```

## File: src/data_pipeline/data_joiner.py

- Extension: .py
- Language: python
- Size: 12123 bytes
- Created: 2025-03-27 15:28:34
- Modified: 2025-03-27 15:28:34

### Code

```python
"""
Data joining utilities for the geo-causal-inference project.

This module provides classes and functions to join datasets from different sources
into a unified dataset for analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Tuple


class DataJoiner:
    """Class for joining multiple datasets into a unified format."""
    
    def __init__(self, date_col: str = 'Date', geo_col: str = 'geo'):
        """
        Initialize the DataJoiner.
        
        Args:
            date_col: The name of the date column used for joining
            geo_col: The name of the geo column used for joining
        """
        self.date_col = date_col
        self.geo_col = geo_col
    
    def join_datasets(self, 
                     base_df: pd.DataFrame, 
                     datasets: List[Tuple[pd.DataFrame, str]], 
                     join_type: str = 'outer') -> pd.DataFrame:
        """
        Join multiple datasets to a base dataframe.
        
        Args:
            base_df: The base dataframe to join to
            datasets: List of tuples (dataframe, suffix) to join
            join_type: Type of join to perform ('left', 'right', 'outer', 'inner')
            
        Returns:
            Joined dataframe
        """
        result = base_df.copy()
        
        for df, suffix in datasets:
            # Ensure join columns exist in both dataframes
            if self.date_col not in df.columns or self.geo_col not in df.columns:
                raise ValueError(f"Join columns {self.date_col} and {self.geo_col} must exist in all dataframes")
            
            # Perform the join
            result = pd.merge(
                result,
                df,
                on=[self.date_col, self.geo_col],
                how=join_type,
                suffixes=('', f'_{suffix}')
            )
        
        return result
    
    def calculate_total_cost(self, 
                            df: pd.DataFrame, 
                            cost_cols: List[str], 
                            output_col: str = 'cost') -> pd.DataFrame:
        """
        Calculate total cost across multiple cost columns.
        
        Args:
            df: The dataframe containing cost columns
            cost_cols: List of cost column names to sum
            output_col: Name of the output total cost column
            
        Returns:
            Dataframe with added total cost column
        """
        df = df.copy()
        
        # Fill NaN values with 0 for cost columns
        df[cost_cols] = df[cost_cols].fillna(0)
        
        # Calculate total cost
        df[output_col] = df[cost_cols].sum(axis=1)
        
        return df


class DatasetCleaner:
    """Class for cleaning and standardizing specific marketing datasets."""
    
    def __init__(self, standardizers):
        """
        Initialize the DatasetCleaner with standardizers.
        
        Args:
            standardizers: Dictionary of standardizers for date, geo, and cost
        """
        self.standardizers = standardizers
        
        # Add DataAggregator if not provided
        if 'aggregator' not in self.standardizers:
            from src.data_pipeline.data_standardizer import DataAggregator
            self.standardizers['aggregator'] = DataAggregator()
        
    def clean_ga4_sessions(self, df: pd.DataFrame, geo_level: str = 'region') -> pd.DataFrame:
        """
        Clean and standardize GA4 sessions data.
        
        Args:
            df: GA4 sessions dataframe
            geo_level: Level of geographic granularity ('region', 'city')
            
        Returns:
            Cleaned dataframe
        """
        # Standardize date
        df = self.standardizers['date'].standardize(df, 'Date')
        
        # Make a copy to avoid modifying the original DataFrame
        result_df = df.copy()
        
        # Standardize geo based on specified level
        geo_cols = ['City'] if geo_level == 'city' else ['Region']
        result_df = self.standardizers['geo'].standardize(result_df, geo_cols, 'geo', geo_level)
        
        # Preserve Region information even when using city level
        if geo_level == 'city' and 'Region' in df.columns:
            # Standardize the Region column separately
            region_df = self.standardizers['geo'].standardize(df, ['Region'], 'Region', 'region')
            result_df['Region'] = region_df['Region']
        
        # Define grouping columns
        group_cols = ['Date', 'geo']
        
        # Add location ID if available and using city level
        if geo_level == 'city' and 'City ID' in df.columns:
            result_df['location_id'] = df['City ID']
            group_cols.append('location_id')
        
        # Add Region to grouping columns if present
        if 'Region' in result_df.columns:
            group_cols.append('Region')
        
        # Aggregate sessions by date, geo, and region
        result_df = self.standardizers['aggregator'].aggregate(
            result_df, 
            group_cols, 
            'Sessions'
        )
        
        # Rename columns to standard format
        result_df = result_df.rename(columns={'Sessions': 'response'})
        
        return result_df
        
    def clean_meta_spend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize Meta geo spend data.
        
        Args:
            df: Meta geo spend dataframe
            
        Returns:
            Cleaned dataframe
        """
        # Standardize date
        df = self.standardizers['date'].standardize(df, 'Day')
        
        # Make a copy
        result_df = df.copy()
        
        # Standardize geo (DMA region)
        result_df = self.standardizers['geo'].standardize(result_df, ['DMA region'], 'geo')
        
        # Clean and preserve original DMA name for joining
        # Handle special characters like commas and & in DMA names
        result_df['dma_name'] = result_df['DMA region'].str.replace(r'^"(.+)"$', r'\1', regex=True)  # Remove quotation marks
        result_df['dma_name'] = result_df['dma_name'].str.replace('&', 'AND')  # Standardize ampersands
        result_df['dma_name'] = result_df['dma_name'].str.strip().str.upper()
        
        # Extract state from the DMA name if it ends with a state code
        result_df['dma_state'] = result_df['dma_name'].str.extract(r'([A-Z]{2})$')
        
        # Standardize cost
        result_df = self.standardizers['cost'].standardize(result_df, 'Amount spent (USD)')
        
        # Aggregate by date, geo, and DMA info
        result_df = self.standardizers['aggregator'].aggregate(
            result_df, 
            ['Day', 'geo', 'dma_name', 'dma_state'], 
            'Amount spent (USD)'
        )
        
        # Rename columns to standard format
        result_df = result_df.rename(columns={'Day': 'Date', 'Amount spent (USD)': 'meta_cost'})
        
        return result_df
                
    def clean_tiktok_spend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize TikTok geo spend data.
        
        Args:
            df: TikTok geo spend dataframe
            
        Returns:
            Cleaned dataframe
        """
        # Standardize date
        df = self.standardizers['date'].standardize(df, 'By Day')
        
        # Make a copy
        result_df = df.copy()
        
        # Standardize geo (state level)
        result_df = self.standardizers['geo'].standardize(result_df, ['Subregion'], 'geo')
        
        # Preserve original state name for joining
        result_df['state'] = result_df['geo'].str.strip().str.upper()
        
        # Add state abbreviation for easier joining
        state_mapping = {
            'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR', 'CALIFORNIA': 'CA',
            'COLORADO': 'CO', 'CONNECTICUT': 'CT', 'DELAWARE': 'DE', 'FLORIDA': 'FL', 'GEORGIA': 'GA',
            'HAWAII': 'HI', 'IDAHO': 'ID', 'ILLINOIS': 'IL', 'INDIANA': 'IN', 'IOWA': 'IA',
            'KANSAS': 'KS', 'KENTUCKY': 'KY', 'LOUISIANA': 'LA', 'MAINE': 'ME', 'MARYLAND': 'MD',
            'MASSACHUSETTS': 'MA', 'MICHIGAN': 'MI', 'MINNESOTA': 'MN', 'MISSISSIPPI': 'MS', 'MISSOURI': 'MO',
            'MONTANA': 'MT', 'NEBRASKA': 'NE', 'NEVADA': 'NV', 'NEW HAMPSHIRE': 'NH', 'NEW JERSEY': 'NJ',
            'NEW MEXICO': 'NM', 'NEW YORK': 'NY', 'NORTH CAROLINA': 'NC', 'NORTH DAKOTA': 'ND', 'OHIO': 'OH',
            'OKLAHOMA': 'OK', 'OREGON': 'OR', 'PENNSYLVANIA': 'PA', 'RHODE ISLAND': 'RI', 'SOUTH CAROLINA': 'SC',
            'SOUTH DAKOTA': 'SD', 'TENNESSEE': 'TN', 'TEXAS': 'TX', 'UTAH': 'UT', 'VERMONT': 'VT',
            'VIRGINIA': 'VA', 'WASHINGTON': 'WA', 'WEST VIRGINIA': 'WV', 'WISCONSIN': 'WI', 'WYOMING': 'WY',
            'UNKNOWN': 'UNKNOWN'
        }
        result_df['state_abbrev'] = result_df['state'].map(state_mapping)
        
        # Handle "Unknown" value
        result_df['state_abbrev'] = result_df['state_abbrev'].fillna('UNKNOWN')
        
        # Standardize cost
        result_df = self.standardizers['cost'].standardize(result_df, 'Cost')
        
        # Aggregate by date and state information
        result_df = self.standardizers['aggregator'].aggregate(
            result_df, 
            ['By Day', 'geo', 'state', 'state_abbrev'], 
            'Cost'
        )
        
        # Rename columns to standard format
        result_df = result_df.rename(columns={'By Day': 'Date', 'Cost': 'tiktok_cost'})
        
        return result_df
                
    def clean_gads_spend(self, df: pd.DataFrame, geo_level: str = 'region') -> pd.DataFrame:
        """
        Clean and standardize Google Ads geo spend data.
        
        Args:
            df: Google Ads geo spend dataframe
            geo_level: Level of geographic granularity ('region', 'city')
            
        Returns:
            Cleaned dataframe
        """
        # Standardize date
        df = self.standardizers['date'].standardize(df, 'Day')
        
        # Make a copy to avoid modifying the original DataFrame
        result_df = df.copy()
        
        # Standardize geo based on specified level
        geo_cols = ['City (User location)'] if geo_level == 'city' else ['Region (User location)']
        result_df = self.standardizers['geo'].standardize(result_df, geo_cols, 'geo', geo_level)
        
        # Preserve Region information when using city level
        if geo_level == 'city' and 'Region (User location)' in df.columns:
            region_df = self.standardizers['geo'].standardize(
                df, ['Region (User location)'], 'Region', 'region'
            )
            result_df['Region'] = region_df['Region']
        
        # Preserve Metro area information for more complete geographic context
        if 'Metro area (User location)' in df.columns:
            result_df['dma_name'] = df['Metro area (User location)'].str.strip().str.upper()
            # Extract state from Metro area if needed (e.g., "Albany-Schenectady-Troy NY" -> "NY")
            result_df['dma_state'] = result_df['dma_name'].str.extract(r' ([A-Z]{2})$')
        
        # Standardize cost
        result_df = self.standardizers['cost'].standardize(result_df, 'Cost')
        
        # Define grouping columns
        group_cols = ['Day', 'geo']
        
        # Add additional geographic columns to grouping columns if present
        if 'Region' in result_df.columns:
            group_cols.append('Region')
        if 'dma_name' in result_df.columns:
            group_cols.append('dma_name')
        if 'dma_state' in result_df.columns:
            group_cols.append('dma_state')
        
        # Aggregate by date, geo, and region
        result_df = self.standardizers['aggregator'].aggregate(
            result_df, 
            group_cols, 
            'Cost'
        )
        
        # Rename columns to standard format
        result_df = result_df.rename(columns={'Day': 'Date', 'Cost': 'gads_cost'})
        
        return result_df
```

## File: src/data_pipeline/geo_joiner.py

- Extension: .py
- Language: python
- Size: 17151 bytes
- Created: 2025-03-27 15:41:57
- Modified: 2025-03-27 15:41:57

### Code

```python
"""
Geographic data joining utilities for the geo-causal-inference project.

This module helps join datasets with different geographic granularity levels
using the geographic spine table.
"""

import pandas as pd
import os
from typing import List, Dict, Union, Optional, Tuple


class GeoJoiner:
    """Class for joining datasets with different geographic granularity."""
    
    def __init__(self, reference_data_path: str = '../data/reference'):
        """
        Initialize the GeoJoiner.
        
        Args:
            reference_data_path: Path to the geographic reference data
        """
        self.reference_data_path = reference_data_path
        self.spine_table = None
        self.city_dma_mapping = None
        self.dma_state_mapping = None
        
        # Load reference data
        self._load_reference_data()
    
    def _load_reference_data(self) -> None:
        """
        Load geographic reference data from disk.
        If files don't exist, they will be created.
        """
        # Define file paths
        spine_path = os.path.join(self.reference_data_path, 'geo_spine.csv')
        city_dma_path = os.path.join(self.reference_data_path, 'city_dma_mapping.csv')
        dma_state_path = os.path.join(self.reference_data_path, 'dma_state_mapping.csv')
        
        # Check if reference data needs to be generated
        if not os.path.exists(spine_path):
            from .geo_reference_builder import GeoReferenceBuilder
            raw_data_path = os.path.join('raw_data', 'region_data')
            builder = GeoReferenceBuilder(raw_data_path, self.reference_data_path)
            builder.build_geo_spine_table()
            builder.build_city_dma_mapping()
            builder.build_dma_state_mapping()
        
        # Load spine table
        if os.path.exists(spine_path):
            self.spine_table = pd.read_csv(spine_path)
        
        # Load city-DMA mapping
        if os.path.exists(city_dma_path):
            self.city_dma_mapping = pd.read_csv(city_dma_path)
        
        # Load DMA-state mapping
        if os.path.exists(dma_state_path):
            self.dma_state_mapping = pd.read_csv(dma_state_path)

    def enrich_city_data(self, df: pd.DataFrame, city_col: str, state_col: Optional[str] = None) -> pd.DataFrame:
        """
        Enrich city-level data with DMA and state information.
        
        Args:
            df: DataFrame containing city-level data
            city_col: Name of the column containing city names
            state_col: Name of the column containing state names or abbreviations
            
        Returns:
            Enriched DataFrame with added DMA information
        """
        if self.city_dma_mapping is None:
            raise ValueError("City-DMA mapping not available. Please ensure reference data is generated.")
        
        # Make a copy of the input dataframe
        result_df = df.copy()
        
        # Standardize city names
        result_df[city_col] = result_df[city_col].str.strip().str.upper()
        
        # Print sample data for debugging
        print(f"Sample city values in input data: {result_df[city_col].head(5).tolist()}")
        print(f"Sample city values in mapping: {self.city_dma_mapping['city'].head(5).tolist()}")
        
        # If state column is provided, use it for more accurate matching
        if state_col and state_col in result_df.columns:
            # Standardize state names
            result_df[state_col] = result_df[state_col].str.strip().str.upper()
            
            # Print sample data for debugging
            print(f"Sample state values in input data: {result_df[state_col].head(5).tolist()}")
            print(f"Sample state values in mapping: {self.city_dma_mapping['state'].head(5).tolist()}")
            
            # Check if we need to convert full state names to abbreviations
            # Get a sample state value to check if it's a full name or abbreviation
            sample_state = result_df[state_col].iloc[0]
            
            # State name to abbreviation mapping
            state_mapping = {
                'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR', 'CALIFORNIA': 'CA',
                'COLORADO': 'CO', 'CONNECTICUT': 'CT', 'DELAWARE': 'DE', 'FLORIDA': 'FL', 'GEORGIA': 'GA',
                'HAWAII': 'HI', 'IDAHO': 'ID', 'ILLINOIS': 'IL', 'INDIANA': 'IN', 'IOWA': 'IA',
                'KANSAS': 'KS', 'KENTUCKY': 'KY', 'LOUISIANA': 'LA', 'MAINE': 'ME', 'MARYLAND': 'MD',
                'MASSACHUSETTS': 'MA', 'MICHIGAN': 'MI', 'MINNESOTA': 'MN', 'MISSISSIPPI': 'MS', 'MISSOURI': 'MO',
                'MONTANA': 'MT', 'NEBRASKA': 'NE', 'NEVADA': 'NV', 'NEW HAMPSHIRE': 'NH', 'NEW JERSEY': 'NJ',
                'NEW MEXICO': 'NM', 'NEW YORK': 'NY', 'NORTH CAROLINA': 'NC', 'NORTH DAKOTA': 'ND', 'OHIO': 'OH',
                'OKLAHOMA': 'OK', 'OREGON': 'OR', 'PENNSYLVANIA': 'PA', 'RHODE ISLAND': 'RI', 'SOUTH CAROLINA': 'SC',
                'SOUTH DAKOTA': 'SD', 'TENNESSEE': 'TN', 'TEXAS': 'TX', 'UTAH': 'UT', 'VERMONT': 'VT',
                'VIRGINIA': 'VA', 'WASHINGTON': 'WA', 'WEST VIRGINIA': 'WV', 'WISCONSIN': 'WI', 'WYOMING': 'WY'
            }
            
            # Check if the state column contains full names
            if len(sample_state) > 2 and sample_state in state_mapping:
                print(f"Converting full state names to abbreviations")
                result_df['state_abbrev'] = result_df[state_col].map(
                    lambda x: state_mapping.get(x, x)
                )
                state_col_for_join = 'state_abbrev'
            else:
                # If the state column already contains abbreviations, use it directly
                state_col_for_join = state_col
            
            # Join with city-DMA mapping on both city and state
            print(f"Joining on columns: {city_col} and {state_col_for_join}")
            joined_df = pd.merge(
                result_df,
                self.city_dma_mapping,
                left_on=[city_col, state_col_for_join],
                right_on=['city', 'state'],
                how='left'
            )
            
            # Print join statistics
            null_count = joined_df['dma_name'].isnull().sum() if 'dma_name' in joined_df.columns else len(joined_df)
            total_count = len(joined_df)
            print(f"Join results: {total_count - null_count} matches, {null_count} NaN values out of {total_count} total")
            
            return joined_df
        else:
            # Join with city-DMA mapping on city only
            joined_df = pd.merge(
                result_df,
                self.city_dma_mapping,
                left_on=city_col,
                right_on='city',
                how='left'
            )
            
            return joined_df
            
    def enrich_dma_data(self, df: pd.DataFrame, dma_col: str) -> pd.DataFrame:
        """
        Enrich DMA-level data with state information.
        
        Args:
            df: DataFrame containing DMA-level data
            dma_col: Name of the column containing DMA names
            
        Returns:
            Enriched DataFrame with added state information
        """
        if self.dma_state_mapping is None:
            raise ValueError("DMA-state mapping not available. Please ensure reference data is generated.")
        
        # Make a copy of the input dataframe
        result_df = df.copy()
        
        # Standardize DMA names
        result_df[dma_col] = result_df[dma_col].str.strip().str.upper()
        
        # Join with DMA-state mapping
        result_df = pd.merge(
            result_df,
            self.dma_state_mapping,
            left_on=dma_col,
            right_on='dma_name',
            how='left'
        )
        
        return result_df
    
    def join_mixed_geo_data(self, 
                           city_df: pd.DataFrame, 
                           dma_df: pd.DataFrame,
                           city_col: str,
                           state_col: Optional[str],
                           dma_col: str,
                           value_cols: Dict[str, str]) -> pd.DataFrame:
        """
        Join city-level data with DMA-level data.
        
        Args:
            city_df: DataFrame containing city-level data
            dma_df: DataFrame containing DMA-level data
            city_col: Name of the column containing city names in city_df
            state_col: Name of the column containing state abbreviations in city_df (optional)
            dma_col: Name of the column containing DMA names in dma_df
            value_cols: Dictionary mapping column names from source dataframes to destination names
            
        Returns:
            Joined DataFrame with data from both granularity levels
        """
        # Enrich city data with DMA information
        enriched_city_df = self.enrich_city_data(city_df, city_col, state_col)
        
        # Standardize DMA names in DMA dataframe
        dma_df_copy = dma_df.copy()
        dma_df_copy[dma_col] = dma_df_copy[dma_col].str.strip().str.upper()
        
        # Join city data with DMA data
        result_df = pd.merge(
            enriched_city_df,
            dma_df_copy,
            left_on='dma_name',
            right_on=dma_col,
            how='left',
            suffixes=('_city', '_dma')
        )
        
        # Select and rename relevant columns
        city_value_cols = {k: v for k, v in value_cols.items() if k in city_df.columns}
        dma_value_cols = {k: v for k, v in value_cols.items() if k in dma_df.columns}
        
        # Use selected columns from each dataframe
        selected_cols = [city_col]
        if state_col:
            selected_cols.append(state_col)
        
        # Add value columns
        for src_col, dst_col in city_value_cols.items():
            result_df[dst_col] = result_df[src_col]
            selected_cols.append(dst_col)
        
        for src_col, dst_col in dma_value_cols.items():
            if dst_col not in result_df.columns:
                result_df[dst_col] = result_df[src_col]
            else:
                # If column exists but has null values, fill with DMA values
                result_df[dst_col] = result_df[dst_col].fillna(result_df[src_col])
            
            if dst_col not in selected_cols:
                selected_cols.append(dst_col)
        
        return result_df[selected_cols]
    
    def distribute_dma_values_to_cities(self,
                                      dma_df: pd.DataFrame,
                                      dma_col: str,
                                      value_col: str,
                                      distribution_method: str = 'equal') -> pd.DataFrame:
        """
        Distribute DMA-level values to constituent cities.
        
        Args:
            dma_df: DataFrame containing DMA-level data
            dma_col: Name of the column containing DMA names
            value_col: Name of the column containing values to distribute
            distribution_method: Method for distributing values ('equal', 'proportional')
            
        Returns:
            DataFrame with city-level distributed values
        """
        if self.city_dma_mapping is None:
            raise ValueError("City-DMA mapping not available. Please ensure reference data is generated.")
        
        # Standardize DMA names
        dma_df_copy = dma_df.copy()
        dma_df_copy[dma_col] = dma_df_copy[dma_col].str.strip().str.upper()
        
        # Join DMA data with city-DMA mapping
        city_dma_values = pd.merge(
            self.city_dma_mapping,
            dma_df_copy,
            left_on='dma_name',
            right_on=dma_col,
            how='inner'
        )
        
        if distribution_method == 'equal':
            # Calculate number of cities per DMA for equal distribution
            cities_per_dma = city_dma_values.groupby('dma_name').size().reset_index(name='city_count')
            city_dma_values = pd.merge(city_dma_values, cities_per_dma, on='dma_name')
            
            # Distribute values equally among cities in each DMA
            city_dma_values[f'{value_col}_city'] = city_dma_values[value_col] / city_dma_values['city_count']
        else:
            # Default to equal distribution if method not recognized
            city_dma_values[f'{value_col}_city'] = city_dma_values[value_col]
        
        return city_dma_values[['city', 'state', 'dma_name', value_col, f'{value_col}_city']]


class GeoHierarchyJoiner:
    """
    Class for joining datasets with different geographic hierarchies.
    
    This is a higher-level class that abstracts away the complexity of joining
    datasets at different geographic granularity levels.
    """
    
    def __init__(self, reference_data_path: str = '../data/reference'):
        """
        Initialize the GeoHierarchyJoiner.
        
        Args:
            reference_data_path: Path to the geographic reference data
        """
        self.geo_joiner = GeoJoiner(reference_data_path)
    
    def join_datasets(self,
                     datasets: List[Tuple[pd.DataFrame, str, str]],
                     date_col: str = 'Date',
                     value_cols: List[str] = None) -> pd.DataFrame:
        """
        Join multiple datasets with different geographic granularity.
        
        Args:
            datasets: List of tuples (dataframe, geo_col, geo_level)
                geo_level can be 'city', 'dma', 'state'
            date_col: Name of the date column for joining
            value_cols: List of value columns to include in the result
            
        Returns:
            Joined DataFrame with consistent geography
        """
        if not datasets:
            raise ValueError("At least one dataset must be provided")
        
        # Identify base dataset (preferably city-level)
        city_datasets = [idx for idx, (_, _, level) in enumerate(datasets) if level == 'city']
        if city_datasets:
            base_idx = city_datasets[0]
        else:
            base_idx = 0
        
        base_df, base_geo_col, base_geo_level = datasets[base_idx]
        
        # Start with base dataset
        result_df = base_df.copy()
        
        # For each additional dataset
        for idx, (df, geo_col, geo_level) in enumerate(datasets):
            if idx == base_idx:
                continue
            
            # Handle different join scenarios
            if base_geo_level == 'city' and geo_level == 'dma':
                # City-to-DMA join
                enriched_base = self.geo_joiner.enrich_city_data(result_df, base_geo_col)
                df_copy = df.copy()
                df_copy[geo_col] = df_copy[geo_col].str.strip().str.upper()
                
                result_df = pd.merge(
                    enriched_base,
                    df_copy,
                    left_on=['dma_name', date_col],
                    right_on=[geo_col, date_col],
                    how='left',
                    suffixes=('', f'_{idx}')
                )
            
            elif base_geo_level == 'city' and geo_level == 'state':
                # City-to-state join
                if 'state' in result_df.columns:
                    df_copy = df.copy()
                    df_copy[geo_col] = df_copy[geo_col].str.strip().str.upper()
                    
                    result_df = pd.merge(
                        result_df,
                        df_copy,
                        left_on=['state', date_col],
                        right_on=[geo_col, date_col],
                        how='left',
                        suffixes=('', f'_{idx}')
                    )
            
            elif base_geo_level == 'dma' and geo_level == 'city':
                # DMA-to-city join - need to aggregate cities to DMA level
                df_copy = df.copy()
                df_copy = self.geo_joiner.enrich_city_data(df_copy, geo_col)
                
                # Aggregate city data to DMA level
                if value_cols:
                    agg_dict = {col: 'sum' for col in value_cols if col in df_copy.columns}
                    city_agg = df_copy.groupby(['dma_name', date_col]).agg(agg_dict).reset_index()
                    
                    result_df = pd.merge(
                        result_df,
                        city_agg,
                        left_on=[base_geo_col, date_col],
                        right_on=['dma_name', date_col],
                        how='left',
                        suffixes=('', f'_{idx}')
                    )
            
            else:
                # Default case: try direct join if levels match
                result_df = pd.merge(
                    result_df,
                    df,
                    left_on=[base_geo_col, date_col],
                    right_on=[geo_col, date_col],
                    how='left',
                    suffixes=('', f'_{idx}')
                )
        
        return result_df

```

## File: src/data_pipeline/geo_reference_builder.py

- Extension: .py
- Language: python
- Size: 15536 bytes
- Created: 2025-03-28 10:32:15
- Modified: 2025-03-28 10:32:15

### Code

```python
"""
Geographic reference data builder for the geo-causal-inference project.

This module creates a comprehensive geographic spine table that helps
join datasets with different levels of geographic granularity.
"""

import pandas as pd
import os
from typing import Optional


class GeoReferenceBuilder:
    """Builder for geographic reference data tables."""
    
    def __init__(self, raw_data_path: str, output_path: str):
        """
        Initialize the GeoReferenceBuilder.
        
        Args:
            raw_data_path: Path to the raw geographic data files
            output_path: Path where the generated reference tables will be saved
        """
        self.raw_data_path = raw_data_path
        self.output_path = output_path
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
    
    def build_geo_spine_table(self, 
                             zip_city_file: str = 'zip_city_detail.csv',
                             zip_dma_file: str = 'zip_to_dma.csv',
                             geo_zip_dim_file: str = 'geo_zip_dim.csv',
                             output_file: str = 'geo_spine.csv') -> pd.DataFrame:
        """
        Build a comprehensive geographic spine table.
        
        Args:
            zip_city_file: Filename for the zip-to-city mapping data
            zip_dma_file: Filename for the zip-to-DMA mapping data
            geo_zip_dim_file: Filename for the primary zip-to-DMA mapping data (more complete)
            output_file: Filename for the output spine table
            
        Returns:
            The created spine table DataFrame
        """
        # First, try to load the primary source - geo_zip_dim file with proper dtypes
        geo_zip_dim_path = os.path.join(self.raw_data_path, geo_zip_dim_file)
        if os.path.exists(geo_zip_dim_path):
            geo_zip_dim_df = pd.read_csv(geo_zip_dim_path, dtype={'zip_code': str, 'zip_code_leading_zero': str, 'dma_code': str})
            
            # Use zip_code as the primary key for consistency
            primary_df = geo_zip_dim_df.rename(columns={
                'dma_name': 'dma_name',
                'dma_code': 'dma_code'
            })
        else:
            # If file doesn't exist, create an empty DataFrame with required columns
            primary_df = pd.DataFrame(columns=['zip_code', 'dma_code', 'dma_name'])
            print(f"Warning: Primary source file {geo_zip_dim_path} not found. Proceeding with secondary sources only.")
        
        # Load zip-to-city data with proper dtypes to preserve leading zeros
        zip_city_path = os.path.join(self.raw_data_path, zip_city_file)
        zip_city_df = pd.read_csv(zip_city_path, dtype={'DELIVERY ZIPCODE': str})
        
        # Simplify and standardize zip-city data
        # We're interested in the delivery zipcode, city, and state
        city_df = zip_city_df[[
            'DELIVERY ZIPCODE', 
            'PHYSICAL CITY', 
            'PHYSICAL STATE'
        ]].copy()
        
        # Rename columns to standard format
        city_df = city_df.rename(columns={
            'DELIVERY ZIPCODE': 'zip_code',
            'PHYSICAL CITY': 'city',
            'PHYSICAL STATE': 'state'
        })
        
        # Handle duplicate zip codes (keep the first occurrence for simplicity)
        city_df = city_df.drop_duplicates(subset=['zip_code'])
        
        # Load zip-to-DMA data with proper dtypes to preserve leading zeros
        zip_dma_path = os.path.join(self.raw_data_path, zip_dma_file)
        dma_df = pd.read_csv(zip_dma_path, dtype={'zip_code': str, 'dma_code': str})
        
        # Standardize DMA data
        dma_df = dma_df.rename(columns={
            'zip_code': 'zip_code',
            'dma_code': 'dma_code',
            'dma_description': 'dma_name'
        })
        
        # Start building the spine table from the primary source
        if not primary_df.empty:
            # First, make sure we have the necessary columns
            if 'zip_code' not in primary_df.columns:
                # Use zip_code_leading_zero if zip_code is not available
                if 'zip_code_leading_zero' in primary_df.columns:
                    primary_df['zip_code'] = primary_df['zip_code_leading_zero']
                else:
                    raise ValueError("Primary source must have either 'zip_code' or 'zip_code_leading_zero' column")
            
            # Initialize spine with primary source
            spine_df = primary_df[['zip_code']].copy()
            
            # Add DMA info from primary source
            if 'dma_code' in primary_df.columns:
                spine_df['dma_code'] = primary_df['dma_code']
            
            if 'dma_name' in primary_df.columns:
                spine_df['dma_name'] = primary_df['dma_name']
            else:
                # Use Google Ads DMA name if available
                if 'dma_name_googleads' in primary_df.columns:
                    spine_df['dma_name'] = primary_df['dma_name_googleads']
                # Fallback to Facebook DMA name
                elif 'dma_name_facebook' in primary_df.columns:
                    spine_df['dma_name'] = primary_df['dma_name_facebook']
                else:
                    spine_df['dma_name'] = None
            
            # Merge with city data to get city and state info
            spine_df = pd.merge(
                spine_df,
                city_df,
                on='zip_code',
                how='left'
            )
            
            # Fill in missing DMA info from secondary source
            if 'dma_code' not in spine_df.columns or spine_df['dma_code'].isna().any():
                # Merge with DMA data to get missing DMA info
                spine_df = pd.merge(
                    spine_df,
                    dma_df[['zip_code', 'dma_code', 'dma_name']],
                    on='zip_code',
                    how='left',
                    suffixes=('', '_secondary')
                )
                
                # Fill missing dma_code values with secondary source
                if 'dma_code' in spine_df.columns:
                    if 'dma_code_secondary' in spine_df.columns:
                        spine_df['dma_code'] = spine_df['dma_code'].fillna(spine_df['dma_code_secondary'])
                        spine_df.drop('dma_code_secondary', axis=1, inplace=True)
                else:
                    spine_df['dma_code'] = spine_df['dma_code_secondary']
                    spine_df.drop('dma_code_secondary', axis=1, inplace=True)
                
                # Fill missing dma_name values with secondary source
                if 'dma_name' in spine_df.columns:
                    if 'dma_name_secondary' in spine_df.columns:
                        spine_df['dma_name'] = spine_df['dma_name'].fillna(spine_df['dma_name_secondary'])
                        spine_df.drop('dma_name_secondary', axis=1, inplace=True)
                else:
                    spine_df['dma_name'] = spine_df['dma_name_secondary']
                    spine_df.drop('dma_name_secondary', axis=1, inplace=True)
        else:
            # Fall back to the original approach if no primary source data
            # Join city and DMA data on zip code
            spine_df = pd.merge(
                city_df,
                dma_df,
                on='zip_code',
                how='left'
            )
        
        # Create state abbreviation - full name mapping
        state_mapping = self._create_state_mapping()
        if state_mapping is not None:
            # Add full state name
            spine_df['state_name'] = spine_df['state'].map(state_mapping)
        
        # Standardize column values
        spine_df['city'] = spine_df['city'].str.strip().str.upper() if 'city' in spine_df.columns else None
        spine_df['dma_name'] = spine_df['dma_name'].fillna('').str.strip().str.upper() if 'dma_name' in spine_df.columns else None
        
        # Add geographic hierarchies
        # This allows for rolling up or drilling down between different geo levels
        spine_df['geo_key_zip'] = spine_df['zip_code']
        spine_df['geo_key_city'] = spine_df.apply(
            lambda x: f"{x['city']}, {x['state']}" if pd.notna(x.get('city')) and pd.notna(x.get('state')) else None, 
            axis=1
        )
        spine_df['geo_key_dma'] = spine_df['dma_name']
        spine_df['geo_key_state'] = spine_df['state']
        
        # Ensure we have all expected columns
        expected_columns = [
            'zip_code', 'city', 'state', 'dma_code', 'dma_name', 
            'state_name', 'geo_key_zip', 'geo_key_city', 'geo_key_dma', 'geo_key_state'
        ]
        
        for col in expected_columns:
            if col not in spine_df.columns:
                spine_df[col] = None
        
        # Save the spine table
        output_path = os.path.join(self.output_path, output_file)
        spine_df.to_csv(output_path, index=False)
        print(f"Geographic spine table saved to {output_path}")
        
        return spine_df
    
    def _create_state_mapping(self) -> Optional[dict]:
        """
        Create a mapping of state abbreviations to full state names.
        
        Returns:
            Dictionary mapping state abbreviations to full names, or None if not available
        """
        # Static mapping of state abbreviations to full names
        state_mapping = {
            'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
            'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
            'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
            'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
            'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
            'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
            'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
            'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
            'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
            'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
            'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
            'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
            'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia',
            'PR': 'Puerto Rico', 'VI': 'Virgin Islands', 'GU': 'Guam'
        }
        return state_mapping
    
    def build_city_dma_mapping(self, 
                             spine_file: str = 'geo_spine.csv',
                             output_file: str = 'city_dma_mapping.csv') -> pd.DataFrame:
        """
        Build a city-to-DMA mapping table.
        
        Args:
            spine_file: Filename for the geographic spine table
            output_file: Filename for the output city-DMA mapping
            
        Returns:
            The created city-DMA mapping DataFrame
        """
        # Load the spine table
        spine_path = os.path.join(self.output_path, spine_file)
        
        # If the spine table doesn't exist yet, build it
        if not os.path.exists(spine_path):
            spine_df = self.build_geo_spine_table(output_file=spine_file)
        else:
            spine_df = pd.read_csv(spine_path, dtype={'zip_code': str, 'dma_code': str})
        
        # Extract city-DMA mapping
        city_dma_df = spine_df[[
            'city', 
            'state', 
            'dma_name', 
            'dma_code'
        ]].copy()
        
        # Filter out rows without DMA info
        city_dma_df = city_dma_df[city_dma_df['dma_name'].notna() & (city_dma_df['dma_name'] != '')]
        
        # Handle multiple DMAs per city by selecting the most frequent DMA for each city-state pair
        city_dma_counts = city_dma_df.groupby(['city', 'state', 'dma_name', 'dma_code']).size().reset_index(name='count')
        city_dma_top = city_dma_counts.sort_values('count', ascending=False).drop_duplicates(['city', 'state'])
        city_dma_mapping = city_dma_top.drop('count', axis=1)
        
        # Save the city-DMA mapping
        output_path = os.path.join(self.output_path, output_file)
        city_dma_mapping.to_csv(output_path, index=False)
        print(f"City-DMA mapping saved to {output_path}")
        
        return city_dma_mapping
    
    def build_dma_state_mapping(self,
                               spine_file: str = 'geo_spine.csv',
                               output_file: str = 'dma_state_mapping.csv') -> pd.DataFrame:
        """
        Build a DMA-to-state mapping table.
        
        Some DMAs span multiple states, this maps each DMA to its constituent states.
        
        Args:
            spine_file: Filename for the geographic spine table
            output_file: Filename for the output DMA-state mapping
            
        Returns:
            The created DMA-state mapping DataFrame
        """
        # Load the spine table
        spine_path = os.path.join(self.output_path, spine_file)
        
        # If the spine table doesn't exist yet, build it
        if not os.path.exists(spine_path):
            spine_df = self.build_geo_spine_table(output_file=spine_file)
        else:
            spine_df = pd.read_csv(spine_path, dtype={'zip_code': str, 'dma_code': str})
        
        # Create DMA-state mapping
        dma_state_df = spine_df[['dma_name', 'dma_code', 'state']].drop_duplicates()
        
        # Filter out rows without DMA info
        dma_state_df = dma_state_df[dma_state_df['dma_name'].notna() & (dma_state_df['dma_name'] != '')]
        
        # Calculate the percentage of zip codes in each state for each DMA
        dma_state_counts = spine_df.groupby(['dma_name', 'dma_code', 'state']).size().reset_index(name='zip_count')
        dma_totals = dma_state_counts.groupby('dma_name')['zip_count'].sum().reset_index(name='total_zips')
        dma_state_pct = pd.merge(dma_state_counts, dma_totals, on='dma_name')
        dma_state_pct['state_percentage'] = dma_state_pct['zip_count'] / dma_state_pct['total_zips'] * 100
        
        # Keep only state-DMA pairs with significant representation (e.g., > 5%)
        significant_dma_states = dma_state_pct[dma_state_pct['state_percentage'] > 5]
        
        # Create a list of states for each DMA
        dma_states = significant_dma_states.groupby(['dma_name', 'dma_code'])['state'].apply(list).reset_index()
        
        # Save the DMA-state mapping
        output_path = os.path.join(self.output_path, output_file)
        dma_states.to_csv(output_path, index=False)
        print(f"DMA-state mapping saved to {output_path}")
        
        return dma_states


def main():
    """Build all geographic reference tables."""
    # Set paths
    raw_data_path = os.path.join('raw_data', 'region_data')
    output_path = os.path.join('data', 'reference')
    
    # Initialize builder
    builder = GeoReferenceBuilder(raw_data_path, output_path)
    
    # Build spine table
    spine_df = builder.build_geo_spine_table()
    
    # Build city-DMA mapping
    city_dma_df = builder.build_city_dma_mapping()
    
    # Build DMA-state mapping
    dma_state_df = builder.build_dma_state_mapping()


if __name__ == '__main__':
    main()

```

## File: src/geo_causal_inference/config.py

- Extension: .py
- Language: python
- Size: 4577 bytes
- Created: 2025-03-29 13:25:07
- Modified: 2025-03-29 13:25:07

### Code

```python
"""
Configuration functionality for Trimmed Match marketing experiments.

This module provides configuration handling for experiment design.
"""

import pandas as pd
import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union

from trimmed_match.design.common_classes import GeoXType
from .utils import parse_date, get_geox_type, create_time_window, TimeWindow


@dataclass
class ExperimentConfig:
    """Configuration for experiment design."""
    
    # Data configuration
    response_col: str = "response"
    spend_col: str = "cost"
    geo_col: str = "geo"
    date_col: str = "date"
    
    # Experiment type and parameters
    geox_type: Union[str, GeoXType] = "HOLD_BACK"
    experiment_duration_weeks: int = 4
    design_start_date: Any = None
    design_end_date: Any = None
    eval_start_date: Any = None
    coverage_test_start_date: Any = None
    
    # Budget parameters
    experiment_budget: float = 300000.0
    alternative_budgets: List[float] = field(default_factory=list)
    
    # Matching parameters
    matching_metrics: Dict[str, float] = field(default_factory=lambda: {"response": 1.0, "cost": 0.01})
    
    # Statistical parameters
    minimum_detectable_iroas: float = 3.0
    average_order_value: float = 256.0 # average order value in dollars / total sessions
    significance_level: float = 0.10
    power_level: float = 0.80
    
    # Constraints
    minimum_detectable_lift_in_response_metric: float = 10.0  # percent
    minimum_revenue_covered_by_treatment: float = 5.0  # percent
    
    # Exclusions
    geos_exclude: List[int] = field(default_factory=list)
    days_exclude: List[Any] = field(default_factory=list)
    
    # Processing parameters trying to speed up 
    use_cross_validation: bool = False#True
    number_of_simulations: int = 20#200
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Convert geox_type to GeoXType enum
        if isinstance(self.geox_type, str):
            self.geox_type = get_geox_type(self.geox_type)
        
        # Convert dates to datetime objects
        for date_attr in ['design_start_date', 'design_end_date', 'eval_start_date', 'coverage_test_start_date']:
            if getattr(self, date_attr) is not None:
                setattr(self, date_attr, parse_date(getattr(self, date_attr)))
    
    def get_time_windows(self):
        """Get time windows for experiment design.
        
        Returns:
            tuple: (time_window_for_design, time_window_for_eval, coverage_test_window)
        """
        # Calculate end dates if needed
        number_of_days_test = self.experiment_duration_weeks * 7
        
        eval_end_date = (
            self.eval_start_date + datetime.timedelta(days=number_of_days_test - 1)
            if self.eval_start_date else None
        )
        
        coverage_test_end_date = (
            self.coverage_test_start_date + datetime.timedelta(days=number_of_days_test - 1)
            if self.coverage_test_start_date else None
        )
        
        # Set design start/end dates to encompass all periods if not specified
        if self.design_start_date is None:
            self.design_start_date = min(
                self.eval_start_date, 
                self.coverage_test_start_date
            )
        
        if self.design_end_date is None:
            self.design_end_date = max(
                eval_end_date,
                coverage_test_end_date
            )
        
        # Create time windows
        time_window_for_design = TimeWindow(self.design_start_date, self.design_end_date)
        time_window_for_eval = TimeWindow(self.eval_start_date, eval_end_date)
        coverage_test_window = TimeWindow(self.coverage_test_start_date, coverage_test_end_date)
        
        return time_window_for_design, time_window_for_eval, coverage_test_window
    
    def to_dict(self):
        """Convert config to dictionary.
        
        Returns:
            dict: Dictionary representation of config
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary.
        
        Args:
            config_dict: Dictionary with configuration parameters
            
        Returns:
            ExperimentConfig
        """
        return cls(**config_dict)
    
    @classmethod
    def default(cls):
        """Create default configuration.
        
        Returns:
            ExperimentConfig
        """
        return cls()

```

## File: src/geo_causal_inference/post_analysis.py

- Extension: .py
- Language: python
- Size: 9467 bytes
- Created: 2025-03-27 11:16:38
- Modified: 2025-03-26 11:24:50

### Code

```python
"""
Post-analysis functionality for Trimmed Match marketing experiments.

This module provides post-analysis tools for analyzing experiment results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union

from trimmed_match.design.common_classes import GeoAssignment
from trimmed_match.post_analysis import trimmed_match_post_analysis

from .utils import TimeWindow, human_readable_number


class ExperimentAnalyzer:
    """Analyzer for Trimmed Match experiment results."""
    
    def __init__(
        self,
        data: pd.DataFrame,
        test_period: TimeWindow,
        design_period: TimeWindow,
        cooldown_period: Optional[TimeWindow] = None,
        response_col: str = "response",
        spend_col: str = "cost",
        average_order_value: float = 1.0,
        group_treatment: int = GeoAssignment.TREATMENT,
        group_control: int = GeoAssignment.CONTROL
    ):
        """Initialize the ExperimentAnalyzer.
        
        Args:
            data: DataFrame with experiment data (date, geo, pair, assignment, response, cost)
            test_period: TimeWindow representing the experiment period
            design_period: TimeWindow representing the design period
            cooldown_period: Optional TimeWindow representing the cooldown period
            response_col: Column name for the response variable
            spend_col: Column name for the spend variable
            average_order_value: Average value per unit of response
            group_treatment: Value indicating treatment group
            group_control: Value indicating control group
        """
        self.data = data.copy()
        
        # Rename columns if needed
        col_mapping = {}
        if response_col != "response":
            col_mapping[response_col] = "response"
        if spend_col != "cost":
            col_mapping[spend_col] = "cost"
            
        if col_mapping:
            self.data = self.data.rename(columns=col_mapping)
        
        # Convert date to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(self.data["date"]):
            self.data["date"] = pd.to_datetime(self.data["date"])
            
        # Convert numeric columns
        for col in ["geo", "pair", "assignment", "response", "cost"]:
            if col in self.data.columns and not pd.api.types.is_numeric_dtype(self.data[col]):
                self.data[col] = pd.to_numeric(self.data[col])
        
        self.test_period = test_period
        self.design_period = design_period
        self.cooldown_period = cooldown_period
        self.average_order_value = average_order_value
        self.group_treatment = group_treatment
        self.group_control = group_control
        
        # Add period to the data
        self._add_period_column()
        
        # Check data validity
        self._validate_data()
        
    def _add_period_column(self):
        """Add a period column to the data.
        
        Periods:
        0 = Design period
        1 = Test period
        2 = Cooldown period
        -1 = Outside of any period
        """
        def determine_period(date):
            if date >= self.design_period.start_date and date <= self.design_period.end_date:
                return 0
            elif date >= self.test_period.start_date and date <= self.test_period.end_date:
                return 1
            elif (self.cooldown_period and 
                  date > self.test_period.end_date and 
                  date <= self.cooldown_period.end_date):
                return 2
            else:
                return -1
                
        self.data["period"] = self.data["date"].apply(determine_period)
        
    def _validate_data(self):
        """Validate the data for post-analysis."""
        # Check that we have the required columns
        required_cols = ["date", "geo", "pair", "assignment", "response", "cost", "period"]
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Check that we have data for design and test periods
        if len(self.data[self.data["period"] == 0]) == 0:
            raise ValueError("No data available for the design period.")
        if len(self.data[self.data["period"] == 1]) == 0:
            raise ValueError("No data available for the test period.")
            
        # Check that each geo has a pair and a valid assignment
        invalid_assignments = self.data[~self.data["assignment"].isin(
            [self.group_treatment, self.group_control])]
        if len(invalid_assignments) > 0:
            geos = invalid_assignments["geo"].unique()
            raise ValueError(f"Invalid assignments for geos: {geos}")
            
    def exclude_pairs(self, pairs_to_exclude: List[int]):
        """Exclude specific pairs from the analysis.
        
        Args:
            pairs_to_exclude: List of pair IDs to exclude
        """
        self.data = self.data[~self.data["pair"].isin(pairs_to_exclude)]
        
    def get_summary_stats(self) -> pd.DataFrame:
        """Get summary statistics by period and assignment.
        
        Returns:
            DataFrame with summary statistics
        """
        summary = self.data.loc[
            self.data["period"].isin([0, 1]),
            ["period", "assignment", "response", "cost"]
        ].groupby(["period", "assignment"], as_index=False).sum()
        
        # Map period and assignment to human-readable values
        summary["period"] = summary["period"].map({0: "Design", 1: "Test"})
        summary["assignment"] = summary["assignment"].map({
            self.group_control: "Control", 
            self.group_treatment: "Treatment"
        })
        
        return summary
        
    def prepare_data_for_analysis(self, exclude_cooldown: bool = True) -> pd.DataFrame:
        """Prepare data for post-analysis.
        
        Args:
            exclude_cooldown: Whether to exclude the cooldown period
            
        Returns:
            DataFrame prepared for post-analysis
        """
        # Use the trimmed_match implementation
        return trimmed_match_post_analysis.prepare_data_for_post_analysis(
            geox_data=self.data,
            exclude_cooldown=exclude_cooldown,
            group_control=self.group_control,
            group_treatment=self.group_treatment
        )
        
    def calculate_results(self, exclude_cooldown: bool = True) -> Dict:
        """Calculate experiment results.
        
        Args:
            exclude_cooldown: Whether to exclude the cooldown period
            
        Returns:
            Dict with experiment results
        """
        data = self.prepare_data_for_analysis(exclude_cooldown=exclude_cooldown)
        return trimmed_match_post_analysis.calculate_experiment_results(data)
    
    def report_results(self, exclude_cooldown: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate and report experiment results.
        
        Args:
            exclude_cooldown: Whether to exclude the cooldown period
            
        Returns:
            Tuple of (summary_df, detailed_results_df)
        """
        results = self.calculate_results(exclude_cooldown=exclude_cooldown)
        
        # Use the trimmed_match implementation to get the formatted report
        trimmed_match_post_analysis.report_experiment_results(
            results, self.average_order_value)
        
        # Convert the results to DataFrames for easier handling
        summary = pd.DataFrame({
            'Metric': [
                'iROAS (Response / Spend)',
                'Incremental Response',
                'Incremental Response Value',
                f'Incremental Response Value (AOV: {self.average_order_value})',
                'Total Spend',
                'Response Rate',
                'Incremental Response Rate'
            ],
            'Value': [
                f"{results['iroas']:.2f}",
                f"{results['incremental_response']:.2f}",
                f"{results['incremental_response']:.2f}",
                f"{results['incremental_response'] * self.average_order_value:.2f}",
                f"{results['total_spend']:.2f}",
                f"{results['response_rate']:.2%}",
                f"{results['incremental_response_rate']:.2%}"
            ]
        })
        
        # Create a DataFrame with confidence intervals
        detailed = pd.DataFrame({
            'Metric': [
                'Response Rate',
                'Incremental Response Rate',
                'iROAS'
            ],
            'Value': [
                f"{results['response_rate']:.4f}",
                f"{results['incremental_response_rate']:.4f}",
                f"{results['iroas']:.4f}"
            ],
            'CI Lower': [
                f"{results['response_rate_ci_lower']:.4f}",
                f"{results['incremental_response_rate_ci_lower']:.4f}",
                f"{results['iroas_ci_lower']:.4f}"
            ],
            'CI Upper': [
                f"{results['response_rate_ci_upper']:.4f}",
                f"{results['incremental_response_rate_ci_upper']:.4f}",
                f"{results['iroas_ci_upper']:.4f}"
            ]
        })
        
        return summary, detailed

```

## File: src/geo_causal_inference/data_loader.py

- Extension: .py
- Language: python
- Size: 4704 bytes
- Created: 2025-03-27 11:16:38
- Modified: 2025-03-25 14:56:08

### Code

```python
"""
Data loading functionality for Trimmed Match marketing experiments.

This module handles the loading of data from various sources including
Google Sheets, local CSV files, and pandas DataFrames.
"""

import pandas as pd
import gspread
from google import auth as google_auth
from gspread_dataframe import set_with_dataframe
import os


def authenticate_google():
    """Authenticate with Google services."""
    return google_auth.default()


def get_gspread_client(creds=None):
    """Get a gspread client.
    
    Args:
        creds: Google credentials. If None, authenticate first.
    
    Returns:
        gspread client
    """
    if creds is None:
        creds, _ = authenticate_google()
    return gspread.authorize(creds)


def read_trix(url, client=None):
    """Read data from a Google Sheet.
    
    Args:
        url: URL to the Google Sheet
        client: Authenticated gspread client. If None, one will be created.
    
    Returns:
        pandas DataFrame with the sheet contents
    """
    if client is None:
        client = get_gspread_client()
        
    wks = client.open_by_url(url).sheet1
    data = wks.get_all_values()
    headers = data.pop(0)
    return pd.DataFrame(data, columns=headers)


def write_trix(df, url, sheet_name="Sheet1", client=None):
    """Write a DataFrame to a Google Sheet.
    
    Args:
        df: pandas DataFrame to write
        url: URL to the Google Sheet
        sheet_name: Name of the sheet to write to
        client: Authenticated gspread client. If None, one will be created.
    """
    if client is None:
        client = get_gspread_client()
        
    workbook = client.open_by_url(url)
    try:
        worksheet = workbook.worksheet(sheet_name)
    except gspread.exceptions.WorksheetNotFound:
        worksheet = workbook.add_worksheet(sheet_name, rows=df.shape[0] + 10, cols=df.shape[1] + 5)
    
    worksheet.clear()
    set_with_dataframe(worksheet, df)
    
    return worksheet


def read_csv(file_path, **kwargs):
    """Read data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        **kwargs: Additional arguments to pass to pandas.read_csv
    
    Returns:
        pandas DataFrame with the file contents
    """
    df = pd.read_csv(file_path, **kwargs)
    
    # Convert columns to appropriate types
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    if 'geo' in df.columns:
        df['geo'] = pd.to_numeric(df['geo'], errors='coerce')
    
    if 'response' in df.columns:
        df['response'] = pd.to_numeric(df['response'], errors='coerce')
    
    if 'cost' in df.columns:
        df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
        
    return df


def write_csv(df, file_path, **kwargs):
    """Write data to a CSV file.
    
    Args:
        df: pandas DataFrame to write
        file_path: Path to the CSV file
        **kwargs: Additional arguments to pass to pandas.to_csv
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, **kwargs)


def prepare_dataframe(df):
    """Prepare a DataFrame for use with Trimmed Match.
    
    Args:
        df: pandas DataFrame to prepare
    
    Returns:
        Prepared pandas DataFrame
    """
    df = df.copy()
    
    # Convert columns to appropriate types
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    if 'geo' in df.columns:
        df['geo'] = pd.to_numeric(df['geo'], errors='coerce')
    
    if 'response' in df.columns:
        df['response'] = pd.to_numeric(df['response'], errors='coerce')
    
    if 'cost' in df.columns:
        df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
        
    return df


def load_data(source, **kwargs):
    """Load data from various sources.
    
    Args:
        source: Source of the data. Can be a pandas DataFrame, a path to a CSV file,
               or a URL to a Google Sheet.
        **kwargs: Additional arguments to pass to the specific loader
    
    Returns:
        pandas DataFrame with the loaded data
    """
    if isinstance(source, pd.DataFrame):
        return prepare_dataframe(source)
    elif isinstance(source, str):
        if source.startswith('http') and ('docs.google.com' in source or 'spreadsheets.google.com' in source):
            return prepare_dataframe(read_trix(source, **kwargs))
        elif os.path.isfile(source) and source.endswith('.csv'):
            return read_csv(source, **kwargs)
        else:
            raise ValueError(f"Unable to determine data source type for: {source}")
    else:
        raise TypeError(f"Unsupported data source type: {type(source)}")

```

## File: src/geo_causal_inference/design.py

- Extension: .py
- Language: python
- Size: 30778 bytes
- Created: 2025-04-04 12:50:01
- Modified: 2025-04-04 12:50:01

### Code

```python
"""
Core experiment design functionality for Trimmed Match marketing experiments.

This module handles the design of experiments using the Trimmed Match methodology.
"""

import pandas as pd
import numpy as np

from trimmed_match.design.common_classes import GeoXType, GeoAssignment
from geo_causal_inference.utils import TimeWindow
from trimmed_match.design.trimmed_match_design import TrimmedMatchGeoXDesign
from trimmed_match.design.util import CalculateMinDetectableIroas


def plot_cost_precision_tradeoff(design_results):
    """
    Create a plot showing the trade-off between experiment cost and precision.
    
    Args:
        design_results: DataFrame with design results, as returned by calculate_optimal_budget
    
    Returns:
        matplotlib.figure.Figure: The generated plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get unique budgets
    budgets = design_results['budget'].unique()
    
    # Use a more pleasing color palette
    palette = sns.color_palette("muted", n_colors=len(budgets))
    
    # Loop through each budget
    for i, budget in enumerate(budgets):
        budget_data = design_results[design_results['budget'] == budget].sort_values('num_pairs')
        
        # Plot the cost vs. rmse trade-off
        scatter = ax.scatter(
            budget_data['experiment_spend'], 
            budget_data['rmse'],
            s=budget_data['num_pairs']*2,  # Size represents number of pairs
            alpha=0.7,
            color=palette[i],
            label=f'Budget ${budget/1000:.1f}K'
        )
        
        # Connect points with a line
        ax.plot(
            budget_data['experiment_spend'],
            budget_data['rmse'],
            '-',
            alpha=0.5,
            color=palette[i]
        )
        
        # Annotate key points
        min_rmse_idx = budget_data['rmse'].idxmin()
        if min_rmse_idx in budget_data.index:
            min_point = budget_data.loc[min_rmse_idx]
            ax.annotate(
                f"Pairs: {min_point['num_pairs']:.0f}",
                (min_point['experiment_spend'], min_point['rmse']),
                xytext=(10, 0),
                textcoords='offset points',
                fontsize=8,
                arrowprops=dict(arrowstyle='->', color='gray')
            )
    
    # Set labels and title
    ax.set_xlabel('Experiment Cost ($)')
    ax.set_ylabel('Raw RMSE')
    ax.set_title('Cost vs. Precision Trade-off', fontsize=14)
    
    # Add legend and grid
    ax.legend(title='Budget Options')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Add a secondary x-axis showing the percentage of total budget
    if len(budgets) > 0:
        max_budget = max(budgets)
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        
        # Set tick positions based on percentages of max budget
        tick_positions = np.linspace(0, max_budget, 6)
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels([f'{100*x/max_budget:.0f}%' for x in tick_positions])
        ax2.set_xlabel('Percentage of Maximum Budget')
    
    # Add annotations explaining the trade-off
    bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8)
    ax.text(
        0.02, 0.02, 
        "Lower RMSE = Higher Precision\nLarger points = More geo pairs",
        transform=ax.transAxes,
        fontsize=10,
        bbox=bbox_props
    )
    
    fig.tight_layout()
    return fig


def plot_design_dashboard(design_results):
    """
    Create a comprehensive dashboard of design options.
    
    Args:
        design_results: DataFrame with design results, as returned by calculate_optimal_budget
    
    Returns:
        matplotlib.figure.Figure: The generated dashboard plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    
    # Create a figure with a grid of subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Define subplot grid
    gs = fig.add_gridspec(2, 3)
    
    # Get unique budgets and assign a more pleasing color palette
    budgets = design_results['budget'].unique()
    # Use a more harmonious color palette
    palette = sns.color_palette("muted", n_colors=len(budgets))
    
    # Plot 1: Cost vs RMSE
    ax1 = fig.add_subplot(gs[0, 0])
    for i, budget in enumerate(budgets):
        budget_data = design_results[design_results['budget'] == budget].sort_values('experiment_spend')
        ax1.plot(
            budget_data['experiment_spend'],
            budget_data['rmse'],
            'o-',
            alpha=0.7,
            color=palette[i],
            label=f'${budget/1000:.1f}K'
        )
    ax1.set_xlabel('Experiment Cost ($)')
    ax1.set_ylabel('RMSE')
    ax1.set_title('Cost vs Raw RMSE')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(title='Budget')
    
    # Plot 2: Number of Geo Pairs vs RMSE
    ax2 = fig.add_subplot(gs[0, 1])
    for i, budget in enumerate(budgets):
        # Sort by number of pairs to ensure the line connects points in order
        budget_data = design_results[design_results['budget'] == budget].sort_values('num_pairs')
        ax2.plot(
            budget_data['num_pairs'],
            budget_data['rmse'],
            'o-',
            alpha=0.7,
            color=palette[i],
            label=f'${budget/1000:.1f}K'
        )
    ax2.set_xlabel('Number of Geo Pairs')
    ax2.set_ylabel('RMSE')
    ax2.set_title('Geo Pairs vs RMSE')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(title='Budget')
    
    # Plot 3: Trim Rate vs RMSE
    ax3 = fig.add_subplot(gs[0, 2])
    for i, budget in enumerate(budgets):
        budget_data = design_results[design_results['budget'] == budget]
        sc = ax3.scatter(
            budget_data['trim_rate'] * 100,  # Convert to percentage
            budget_data['rmse'],
            alpha=0.7,
            s=50,
            c=budget_data['num_pairs'],
            cmap='viridis'
        )
    cbar = plt.colorbar(sc, ax=ax3)
    cbar.set_label('Number of Pairs')
    ax3.set_xlabel('Trim Rate (%)')
    ax3.set_ylabel('RMSE')
    ax3.set_title('Trim Rate vs RMSE')
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    # Plot 4: Elbow Curve - Number of Pairs vs RMSE Cost Adjusted
    ax4 = fig.add_subplot(gs[1, 0:2])
    for i, budget in enumerate(budgets):
        budget_data = design_results[design_results['budget'] == budget].sort_values('num_pairs')
        
        # Mark the "elbow point" using the point of maximum curvature
        x = budget_data['num_pairs'].values
        y = budget_data['rmse_cost_adjusted'].values
        
        ax4.plot(
            x, 
            y,
            'o-',
            alpha=0.7,
            color=palette[i],
            label=f'${budget/1000:.1f}K'
        )
        
        # Find the minimum point
        min_idx = budget_data['rmse_cost_adjusted'].idxmin()
        min_point = budget_data.loc[min_idx]
        ax4.scatter(
            min_point['num_pairs'],
            min_point['rmse_cost_adjusted'],
            s=100,
            edgecolor='black',
            color=palette[i],
            zorder=5,
            marker='*'
        )
        
    ax4.set_xlabel('Number of Geo Pairs')
    ax4.set_ylabel('RMSE Cost Adjusted')
    ax4.set_title('Elbow Curve: Finding Optimal Number of Pairs')
    ax4.grid(True, linestyle='--', alpha=0.5)
    ax4.legend(title='Budget')
    
    # Plot 5: Budget Comparison
    ax5 = fig.add_subplot(gs[1, 2])
    
    # Get the minimum RMSE cost adjusted for each budget
    min_rmse_by_budget = []
    for budget in budgets:
        budget_data = design_results[design_results['budget'] == budget]
        min_idx = budget_data['rmse_cost_adjusted'].idxmin()
        min_point = budget_data.loc[min_idx]
        min_rmse_by_budget.append({
            'budget': budget,
            'rmse_cost_adjusted': min_point['rmse_cost_adjusted'],
            'experiment_spend': min_point['experiment_spend'],
            'num_pairs': min_point['num_pairs']
        })
    
    df_min = pd.DataFrame(min_rmse_by_budget)
    
    # Bar chart with the new color palette
    bars = ax5.bar(
        range(len(df_min)),
        df_min['rmse_cost_adjusted'],
        color=palette[:len(df_min)]
    )
    
    # Add budget labels
    ax5.set_xticks(range(len(df_min)))
    ax5.set_xticklabels([f'${b/1000:.1f}K' for b in df_min['budget']])
    
    # Add value annotations
    for i, bar in enumerate(bars):
        ax5.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.001,
            f"Pairs: {df_min.iloc[i]['num_pairs']:.0f}\nCost: ${df_min.iloc[i]['experiment_spend']/1000:.1f}K",
            ha='center',
            va='bottom',
            fontsize=8
        )
    
    ax5.set_ylabel('RMSE Cost Adjusted')
    ax5.set_title('Optimal Design by Budget')
    ax5.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # Add overall title
    fig.suptitle('Experiment Design Trade-offs Dashboard', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig


def plot_bang_for_buck(design_results):
    """
    Create a "bang for the buck" visualization showing improvement in 
    precision per dollar spent.
    
    Args:
        design_results: DataFrame with design results, as returned by calculate_optimal_budget
    
    Returns:
        matplotlib.figure.Figure: The generated plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Get unique budgets
    budgets = design_results['budget'].unique()
    
    # Use a more harmonious color palette 
    palette = sns.color_palette("muted", n_colors=len(budgets))
    
    # Plot 1: RMSE reduction per dollar vs. budget
    for i, budget in enumerate(budgets):
        budget_data = design_results[design_results['budget'] == budget].sort_values('num_pairs', ascending=False)
        
        if len(budget_data) > 1:
            # Calculate RMSE reduction per dollar spent
            budget_data = budget_data.copy()
            # Get the worst (highest) RMSE value for this budget
            max_rmse = budget_data['rmse'].max()
            
            # Calculate improvement from the worst RMSE
            budget_data['rmse_improvement'] = max_rmse - budget_data['rmse']
            
            # Calculate bang for the buck (improvement per dollar)
            budget_data['bang_for_buck'] = budget_data['rmse_improvement'] / budget_data['experiment_spend']
            
            # Multiply by 10000 for readability
            budget_data['bang_for_buck'] *= 10000
            
            # Plot bang for buck vs. experiment cost
            ax1.plot(
                budget_data['experiment_spend'],
                budget_data['bang_for_buck'],
                'o-',
                alpha=0.7,
                color=palette[i],
                label=f'${budget/1000:.1f}K'
            )
            
            # Highlight the optimal point (maximum bang for the buck)
            max_idx = budget_data['bang_for_buck'].idxmax()
            max_point = budget_data.loc[max_idx]
            ax1.scatter(
                max_point['experiment_spend'],
                max_point['bang_for_buck'],
                s=120,
                edgecolor='black',
                color=palette[i],
                zorder=5,
                marker='*'
            )
    
    ax1.set_xlabel('Experiment Cost ($)')
    ax1.set_ylabel('RMSE Reduction per $10K Spent')
    ax1.set_title('Diminishing Returns on Additional Spend')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(title='Budget')
    
    # Plot 2: Relationship between number of pairs and RMSE improvement
    for i, budget in enumerate(budgets):
        budget_data = design_results[design_results['budget'] == budget].sort_values('num_pairs')
        
        if len(budget_data) > 1:
            # Calculate the rate of RMSE improvement vs. number of pairs            
            ax2.scatter(
                budget_data['num_pairs'],
                budget_data['rmse'],
                alpha=0.7,
                s=60,
                color=palette[i],
                label=f'${budget/1000:.1f}K'
            )
            
            # Plot regression line to show trend
            z = np.polyfit(budget_data['num_pairs'], budget_data['rmse'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(budget_data['num_pairs'].min(), budget_data['num_pairs'].max(), 100)
            ax2.plot(
                x_range, 
                p(x_range), 
                '--', 
                color=palette[i], 
                alpha=0.5
            )
    
    ax2.set_xlabel('Number of Geo Pairs')
    ax2.set_ylabel('RMSE')
    ax2.set_title('Relationship Between Geo Pairs and Precision')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(title='Budget')
    
    # Add annotations
    bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8)
    ax1.text(
        0.02, 0.02, 
        "Higher = Better value\nShows diminishing returns as spending increases",
        transform=ax1.transAxes,
        fontsize=10,
        bbox=bbox_props
    )
    
    ax2.text(
        0.02, 0.98, 
        "More pairs (left) generally leads\nto better precision (lower RMSE)",
        transform=ax2.transAxes,
        fontsize=10,
        va='top',
        bbox=bbox_props
    )
    
    fig.suptitle('Value Analysis: Finding the "Sweet Spot" for Experiment Design', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig


def create_interactive_design_explorer(design_results):
    """
    Create an interactive design explorer using ipywidgets.
    
    Note: This function is intended to be used in a Jupyter notebook environment.
    
    Args:
        design_results: DataFrame with design results, as returned by calculate_optimal_budget
    
    Returns:
        ipywidgets.Widget: The interactive dashboard
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create widgets
        budget_selector = widgets.SelectMultiple(
            options=sorted([f"${b/1000:.1f}K" for b in design_results['budget'].unique()]),
            description='Budget:',
            disabled=False
        )
        
        min_pairs = widgets.IntSlider(
            value=int(design_results['num_pairs'].min()),
            min=int(design_results['num_pairs'].min()),
            max=int(design_results['num_pairs'].max()),
            step=1,
            description='Min Pairs:',
            disabled=False
        )
        
        max_rmse = widgets.FloatSlider(
            value=float(design_results['rmse_cost_adjusted'].max()),
            min=float(design_results['rmse_cost_adjusted'].min()),
            max=float(design_results['rmse_cost_adjusted'].max()),
            step=0.001,
            description='Max RMSE:',
            disabled=False
        )
        
        max_cost = widgets.FloatSlider(
            value=float(design_results['experiment_spend'].max()),
            min=float(design_results['experiment_spend'].min()),
            max=float(design_results['experiment_spend'].max()),
            step=1000,
            description='Max Cost ($):',
            disabled=False
        )
        
        # Output widget for plotting
        output = widgets.Output()
        
        # Function to update the plot
        def update_plot(*args):
            with output:
                # Clear previous output
                output.clear_output(wait=True)
                
                # Get selected budgets
                selected_budgets = [float(b.replace('$', '').replace('K', '')) * 1000 for b in budget_selector.value]
                if not selected_budgets:
                    selected_budgets = design_results['budget'].unique()
                
                # Filter data
                filtered_data = design_results[
                    (design_results['budget'].isin(selected_budgets)) &
                    (design_results['num_pairs'] >= min_pairs.value) &
                    (design_results['rmse_cost_adjusted'] <= max_rmse.value) &
                    (design_results['experiment_spend'] <= max_cost.value)
                ]
                
                if len(filtered_data) == 0:
                    print("No data matches the current filters.")
                    return
                
                # Create plot
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Color map for different budgets
                colors = plt.cm.tab10(np.linspace(0, 1, len(selected_budgets)))
                
                # Plot each budget scenario
                for i, budget in enumerate(selected_budgets):
                    budget_data = filtered_data[filtered_data['budget'] == budget]
                    if len(budget_data) == 0:
                        continue
                        
                    scatter = ax.scatter(
                        budget_data['experiment_spend'],
                        budget_data['rmse_cost_adjusted'],
                        alpha=0.7,
                        s=budget_data['num_pairs'],
                        c=[colors[i]],
                        label=f'Budget: ${budget/1000:.1f}K'
                    )
                    
                    # Add annotations for key points
                    min_idx = budget_data['rmse_cost_adjusted'].idxmin()
                    if min_idx in budget_data.index:
                        min_point = budget_data.loc[min_idx]
                        ax.annotate(
                            f"Pairs: {min_point['num_pairs']}",
                            (min_point['experiment_spend'], min_point['rmse_cost_adjusted']),
                            xytext=(10, 0),
                            textcoords='offset points'
                        )
                
                # Set labels and title
                ax.set_xlabel('Experiment Cost ($)')
                ax.set_ylabel('RMSE (Cost Adjusted)')
                ax.set_title('Filtered Design Options')
                
                # Add grid and legend
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()
                
                # Show plot
                plt.tight_layout()
                plt.show()
                
                # Show table of best options
                print("Top 5 Designs by RMSE Cost Adjusted:")
                display(filtered_data.sort_values('rmse_cost_adjusted').head(5)[
                    ['budget', 'num_pairs', 'experiment_spend', 'rmse', 'rmse_cost_adjusted', 'trim_rate']
                ].reset_index(drop=True))
        
        # Connect widgets to update function
        budget_selector.observe(update_plot, names='value')
        min_pairs.observe(update_plot, names='value')
        max_rmse.observe(update_plot, names='value')
        max_cost.observe(update_plot, names='value')
        
        # Create layout
        controls = widgets.VBox([budget_selector, min_pairs, max_rmse, max_cost])
        dashboard = widgets.HBox([controls, output])
        
        # Initial update
        update_plot()
        
        return dashboard
    except ImportError:
        print("ipywidgets is required for the interactive explorer. Install with: pip install ipywidgets")
        return None


class ExperimentDesigner:
    """Design experiments using Trimmed Match methodology."""
    
    def __init__(
        self,
        geox_type,
        data,
        time_window_for_design,
        time_window_for_eval,
        response_col="response",
        spend_col="cost",
        matching_metrics=None,
        pairs=None
    ):
        """Initialize ExperimentDesigner.
        
        Args:
            geox_type: Type of experiment (HOLD_BACK, HEAVY_UP, GO_DARK)
            data: DataFrame with geo-level time series data
            time_window_for_design: TimeWindow for design
            time_window_for_eval: TimeWindow for evaluation
            response_col: Column name for response variable
            spend_col: Column name for spend variable
            matching_metrics: Dict of metrics and weights for matching
            pairs: List of pre-defined geo pairs
        """
        self.geox_type = geox_type
        self.data = data
        self.response_col = response_col
        self.spend_col = spend_col
        
        if matching_metrics is None:
            matching_metrics = {response_col: 1.0, spend_col: 0.01}
        
        self.time_window_for_design = time_window_for_design
        self.time_window_for_eval = time_window_for_eval
        self.matching_metrics = matching_metrics
        self.pairs = pairs
        
        # Initialize the TrimmedMatchGeoXDesign
        self.pretest = TrimmedMatchGeoXDesign(
            geox_type=self.geox_type,
            pretest_data=self.data,
            response=self.response_col,
            spend_proxy=self.spend_col,
            matching_metrics=self.matching_metrics,
            time_window_for_design=self.time_window_for_design,
            time_window_for_eval=self.time_window_for_eval,
            pairs=self.pairs
        )
    
    def get_candidate_designs(
        self,
        budget_list,
        iroas_list=[0],
        use_cross_validation=True,
        num_simulations=200
    ):
        """Get candidate experiment designs.
        
        Args:
            budget_list: List of budgets to evaluate
            iroas_list: List of iROAS values to evaluate
            use_cross_validation: Whether to use cross-validation
            num_simulations: Number of simulations for RMSE calculation
            
        Returns:
            tuple: (results_summary, results_detailed)
        """
        return self.pretest.report_candidate_designs(
            budget_list=budget_list,
            iroas_list=iroas_list,
            use_cross_validation=use_cross_validation,
            num_simulations=num_simulations
        )
    
    def calculate_optimal_budget(
        self,
        experiment_budget,
        minimum_detectable_iroas,
        average_order_value=1,
        additional_budget=None,
        use_cross_validation=True,
        num_simulations=200
    ):
        """Calculate the optimal budget for achieving the minimum detectable iROAS.
        
        Args:
            experiment_budget: Maximum experiment budget
            minimum_detectable_iroas: Target minimum detectable iROAS
            average_order_value: Average value per unit response
            additional_budget: List of additional budgets to evaluate
            use_cross_validation: Whether to use cross-validation
            num_simulations: Number of simulations for RMSE calculation
            
        Returns:
            dict: Results including optimal budget, designs, etc.
        """
        if additional_budget is None:
            additional_budget = []
            
        minimum_iroas_aov = minimum_detectable_iroas / average_order_value
        
        # Run a first design with budget equal to the max. budget
        preliminary_results, prel_results_detailed = self.get_candidate_designs(
            budget_list=[experiment_budget],
            iroas_list=[0],
            use_cross_validation=use_cross_validation,
            num_simulations=num_simulations
        )
        
        # Calculate the minimum detectable iROAS for a design with max. budget
        chosen_design = preliminary_results.loc[
            preliminary_results["rmse_cost_adjusted"].idxmin()].squeeze()
        
        calc_min_det_iroas = CalculateMinDetectableIroas(0.10, 0.80)  # Default values
        lowest_detectable_iroas = calc_min_det_iroas.at(chosen_design["rmse"])
        
        # Two cases:
        # 1) If min detectable iROAS with max budget > target min detectable iROAS
        if lowest_detectable_iroas > minimum_iroas_aov:
            budget_to_reach_min_det_iroas = (
                experiment_budget * lowest_detectable_iroas / minimum_iroas_aov
            )
            additional_results, results_detailed = self.get_candidate_designs(
                budget_list=[budget_to_reach_min_det_iroas] + additional_budget,
                iroas_list=[0],
                use_cross_validation=use_cross_validation,
                num_simulations=num_simulations
            )
            
            results = pd.concat([preliminary_results, additional_results], sort=False)
            
        # 2) If min detectable iROAS with max budget < target min detectable iROAS
        else:
            optimal_budget = (
                experiment_budget * lowest_detectable_iroas / minimum_iroas_aov
            )
            lower_budget = optimal_budget * 0.8
            upper_budget = optimal_budget * 1.2
            
            list_of_budgets = [
                lower_budget, optimal_budget, upper_budget
            ] + additional_budget
            
            results, results_detailed = self.get_candidate_designs(
                budget_list=list_of_budgets,
                iroas_list=[0],
                use_cross_validation=use_cross_validation,
                num_simulations=num_simulations
            )
        
        return {
            "results": results,
            "results_detailed": results_detailed,
            "preliminary_results": preliminary_results,
            "optimal_pair_index": preliminary_results.loc[
                preliminary_results["rmse_cost_adjusted"].idxmin(), "pair_index"
            ],
            "lowest_detectable_iroas": lowest_detectable_iroas,
            "minimum_iroas_aov": minimum_iroas_aov
        }
    
    def get_optimal_design(
        self,
        pair_index,
        base_seed=0,
        confidence=0.90,
        group_control=GeoAssignment.CONTROL,
        group_treatment=GeoAssignment.TREATMENT
    ):
        """Get the optimal experiment design.
        
        Args:
            pair_index: Index of the optimal pair
            base_seed: Base seed for random number generation
            confidence: Confidence level
            group_control: Control group assignment
            group_treatment: Treatment group assignment
            
        Returns:
            tuple: (axes, geopairs, treatments, controls)
        """
        axes = self.pretest.output_chosen_design(
            pair_index=pair_index,
            base_seed=base_seed,
            confidence=confidence,
            group_control=group_control,
            group_treatment=group_treatment
        )
        
        # Get the geo pairs and group assignments
        geopairs = self.pretest.geo_level_eval_data[pair_index].copy()
        
        # Determine treatment and control geos
        treatment_geo = geopairs.loc[
            geopairs["assignment"] == group_treatment, "geo"
        ].tolist()
        
        control_geo = geopairs.loc[
            geopairs["assignment"] == group_control, "geo"
        ].tolist()
        
        return axes, geopairs, treatment_geo, control_geo
    
    def run_coverage_test(
        self,
        data_coverage_test,
        optimal_pair_index,
        group_control=GeoAssignment.CONTROL,
        group_treatment=GeoAssignment.TREATMENT,
        confidence=0.90
    ):
        """Run coverage test for the chosen design.
        
        Args:
            data_coverage_test: Data for coverage test
            optimal_pair_index: Index of the optimal pair
            group_control: Control group assignment
            group_treatment: Treatment group assignment
            confidence: Confidence level
            
        Returns:
            DataFrame: Coverage test results
        """
        return self.pretest.run_aa_test(
            data=data_coverage_test,
            time_window_for_eval=self.time_window_for_eval,
            pair_index=optimal_pair_index,
            confidence=confidence,
            group_control=group_control,
            group_treatment=group_treatment
        )
    
    def plot_pair_comparison(
        self,
        pair_index,
        group_control=GeoAssignment.CONTROL,
        group_treatment=GeoAssignment.TREATMENT
    ):
        """Plot pair-by-pair comparison.
        
        Args:
            pair_index: Index of the pair to plot
            group_control: Control group assignment
            group_treatment: Treatment group assignment
            
        Returns:
            matplotlib.figure.Figure: Plot
        """
        return self.pretest.plot_pair_by_pair_comparison(
            pair_index=pair_index,
            group_control=group_control,
            group_treatment=group_treatment
        )

    def add_trade_off_visualizations(self, results_df, output_dir):
        """
        Generate and save visualizations showing the design trade-offs.
        
        Args:
            results_df: DataFrame with design results, typically from the 'results' key in 
                       the dictionary returned by calculate_optimal_budget
            output_dir: Directory where to save the visualizations
            
        Returns:
            dict: Dictionary with paths to the saved visualizations
        """
        import os
        import matplotlib.pyplot as plt
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate visualizations
        fig_pareto = plot_cost_precision_tradeoff(results_df)
        fig_dashboard = plot_design_dashboard(results_df)
        fig_returns = plot_bang_for_buck(results_df)
        
        # Save visualizations
        pareto_path = os.path.join(output_dir, 'design_pareto.png')
        dashboard_path = os.path.join(output_dir, 'design_dashboard.png')
        returns_path = os.path.join(output_dir, 'design_returns.png')
        
        fig_pareto.savefig(pareto_path, dpi=300, bbox_inches='tight')
        fig_dashboard.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        fig_returns.savefig(returns_path, dpi=300, bbox_inches='tight')
        
        # Close figures to free memory
        plt.close(fig_pareto)
        plt.close(fig_dashboard)
        plt.close(fig_returns)
        
        return {
            'pareto': pareto_path,
            'dashboard': dashboard_path,
            'returns': returns_path
        }

```

## File: src/geo_causal_inference/__init__.py

- Extension: .py
- Language: python
- Size: 239 bytes
- Created: 2025-03-27 11:21:22
- Modified: 2025-03-27 11:21:22

### Code

```python
"""
Geo Causal Inference - A modular implementation of Trimmed Match for marketing experiments.

This package provides tools for designing and analyzing marketing experiments
using the Trimmed Match methodology.
"""

__version__ = "0.1.0"

```

## File: src/geo_causal_inference/visualization.py

- Extension: .py
- Language: python
- Size: 16365 bytes
- Created: 2025-04-09 09:03:04
- Modified: 2025-04-09 09:03:04

### Code

```python
"""
Visualization functionality for Trimmed Match marketing experiments.

This module provides plotting and visualization functions for
experiment design and analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib.colors import ListedColormap


def plot_designs_comparison(results, metric='rmse_cost_adjusted'):
    """Plot comparison of different experiment designs.
    
    Args:
        results: DataFrame with design results
        metric: Metric to use for comparison
        
    Returns:
        matplotlib.figure.Figure: Plot of design comparison
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(results['budget'], results[metric], 'o-')
    ax.set_xlabel('Budget')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title('Design Comparison by Budget')
    ax.grid(True, alpha=0.3)
    
    # Annotate the best design
    best_idx = results[metric].idxmin()
    best_design = results.loc[best_idx]
    
    # Handle the case where best_design returns a DataFrame with multiple rows
    if isinstance(best_design, pd.DataFrame):
        # Use the first row
        best_design = best_design.iloc[0]
    
    # Extract scalar values for formatting
    budget_value = best_design['budget']
    metric_value = best_design[metric]
    
    ax.annotate(
        f'Best: {budget_value:.0f}',
        xy=(budget_value, metric_value),
        xytext=(10, -20),
        textcoords='offset points',
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2')
    )
    
    return fig


def plot_coverage_test_results(coverage_test_result):
    """Plot coverage test results.
    
    Args:
        coverage_test_result: DataFrame with coverage test results
        
    Returns:
        matplotlib.figure.Figure: Plot of coverage test results
    """
    if len(coverage_test_result) <= 1:
        return None
        
    fig = plt.figure(figsize=(15, 7.5))
    ax = fig.add_subplot(1, 1, 1)
    
    ax.plot(
        coverage_test_result['pair_index'],
        coverage_test_result['bias_cost_normalized'],
        'o-', 
        label='Cost'
    )
    
    ax.plot(
        coverage_test_result['pair_index'],
        coverage_test_result['bias_response_normalized'],
        'o-', 
        label='Response'
    )
    
    ax.axhspan(-1, 1, alpha=0.2, color='green')
    ax.legend(loc='best')
    ax.set_xlabel('Pair index (Number of pairs included)')
    ax.set_ylabel('Normalized Bias')
    ax.set_title('Coverage Test Results')
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_geo_time_series(geo_data, response_col='response', date_col='date', 
                          geo_col='geo', treatment_geos=None, control_geos=None,
                          eval_start_date=None, eval_end_date=None):
    """Plot geo-level time series.
    
    Args:
        geo_data: DataFrame with geo-level time series data
        response_col: Column name for response variable
        date_col: Column name for date variable
        geo_col: Column name for geo variable
        treatment_geos: List of treatment geos
        control_geos: List of control geos
        eval_start_date: Start date of evaluation period
        eval_end_date: End date of evaluation period
        
    Returns:
        matplotlib.figure.Figure: Plot of geo-level time series
    """
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Prepare data
    geo_data = geo_data.copy()
    geo_data[date_col] = pd.to_datetime(geo_data[date_col])
    
    # Aggregate data by date and group
    data_agg = []
    
    if treatment_geos:
        treatment_data = geo_data[geo_data[geo_col].isin(treatment_geos)].groupby(date_col)[response_col].sum().reset_index()
        treatment_data['group'] = 'Treatment'
        data_agg.append(treatment_data)
    
    if control_geos:
        control_data = geo_data[geo_data[geo_col].isin(control_geos)].groupby(date_col)[response_col].sum().reset_index()
        control_data['group'] = 'Control'
        data_agg.append(control_data)
    
    if data_agg:
        data_agg = pd.concat(data_agg)
        
        # Plot groups
        for group, group_data in data_agg.groupby('group'):
            ax.plot(group_data[date_col], group_data[response_col], 
                   label=group, marker='o' if len(group_data) < 30 else None,
                   linewidth=2)
    else:
        # If no groups specified, plot all geos
        for geo, geo_data in geo_data.groupby(geo_col):
            ax.plot(geo_data[date_col], geo_data[response_col], 
                   label=f'Geo {geo}', alpha=0.7)
    
    # Highlight evaluation period if specified
    if eval_start_date and eval_end_date:
        ax.axvspan(eval_start_date, eval_end_date, alpha=0.2, color='green', label='Evaluation Period')
    
    ax.set_xlabel('Date')
    ax.set_ylabel(response_col.replace('_', ' ').title())
    ax.set_title('Geo-Level Time Series')
    ax.grid(True, alpha=0.3)
    
    # Add legend with proper sizing
    if len(data_agg.groupby('group')) > 10:
        ax.legend(loc='best', fontsize='small', ncol=2)
    else:
        ax.legend(loc='best')
    
    # Format x-axis with appropriate date ticks
    fig.autofmt_xdate()
    
    return fig


def plot_design_summary(design_results, min_detectable_iroas, min_detectable_lift):
    """Plot summary of experiment design.
    
    Args:
        design_results: Dict with design results
        min_detectable_iroas: Minimum detectable iROAS
        min_detectable_lift: Minimum detectable lift
        
    Returns:
        matplotlib.figure.Figure: Plot of design summary
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot 1: RMSE vs Budget
    results = design_results["results"]
    ax1.plot(results['budget'], results['rmse'], 'o-', label='RMSE')
    ax1.plot(results['budget'], results['rmse_cost_adjusted'], 'o-', label='RMSE Cost Adjusted')
    ax1.set_xlabel('Budget')
    ax1.set_ylabel('RMSE')
    ax1.set_title('RMSE vs Budget')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Minimum Detectable Metrics
    best_design = results.loc[results['rmse_cost_adjusted'].idxmin()]
    metrics = ['Min. Detectable iROAS', 'Min. Detectable Lift (%)']
    values = [min_detectable_iroas, min_detectable_lift]
    
    ax2.bar(metrics, values, color=['blue', 'orange'])
    ax2.set_title('Minimum Detectable Metrics')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, v in enumerate(values):
        ax2.text(i, v + 0.1, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    return fig


def plot_geo_map(geo_assignments, spine_path, map_type='dma', debug=False, output_path=None):
    """
    Plot a choropleth map of the US by state showing treatment and control areas.
    
    Parameters
    ----------
    geo_assignments: pandas.DataFrame or list of dict
        DataFrame or list of dictionaries containing geo codes and their assignments.
        Must contain columns/keys 'geo' and 'assignment'.
    spine_path: str
        Path to the geo spine file.
    map_type: str, optional
        Type of map to plot. Currently supports 'dma'.
        Default is 'dma'.
    debug: bool, optional
        Whether to print debug information.
        Default is False.
    output_path: str, optional
        Path to save the map. If None, the figure is returned.
        Default is None.
    
    Returns
    -------
    str
        Path to the saved figure if output_path is provided, otherwise None.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import geopandas as gpd
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches
    
    # Read the geo spine
    geo_spine = pd.read_csv(spine_path)
    
    # Process geo_assignments if it's a list
    if isinstance(geo_assignments, list):
        if debug:
            print(f"Loaded {len(geo_assignments)} geo assignments")
            print(f"Sample: {geo_assignments[:5]}")
        
        # Convert to dataframe
        geo_assignments = pd.DataFrame(geo_assignments)
        
        if debug:
            print(f"Loaded {len(geo_assignments)} geo assignments")
            print(f"Sample: {geo_assignments.head()}")
    
    # Ensure the geo columns in geo_assignments and geo_spine match types
    if debug:
        print(f"Number of unique geos in assignments: {len(geo_assignments['geo'].unique())}")
        print(f"Sample geo values: {geo_assignments['geo'].unique()[:5]}")
        print(f"Number of unique DMAs in spine: {len(geo_spine['dma_code'].unique())}")
        print(f"Sample DMA values: {geo_spine['dma_code'].unique()[:5]}")
        print(f"DMA value types in assignments: {geo_assignments['geo'].dtype}")
        print(f"DMA value types in spine: {geo_spine['dma_code'].dtype}")
    
    # Check for states in the spine
    states = geo_spine['state'].unique()
    if debug:
        states_without_nan = [s for s in states if pd.notna(s)]
        print(f"States in spine data: {sorted(states_without_nan)}")
    
    # Create the figure and axis for plotting
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Filter the spine to DMAs we're using
    if map_type == 'dma':
        # Identify treatment and control DMAs
        treatment_dmas = geo_assignments[geo_assignments['assignment'] == 'treatment']['geo'].unique()
        control_dmas = geo_assignments[geo_assignments['assignment'] == 'control']['geo'].unique()
        
        if debug:
            print(f"Treatment DMAs: {treatment_dmas}")
            print(f"Control DMAs: {control_dmas}")
        
        # Create a mapping of DMAs to states
        dma_to_states = {}
        for dma in sorted(set(treatment_dmas) | set(control_dmas)):
            states = geo_spine[geo_spine['dma_code'] == dma]['state'].unique()
            states_filtered = [s for s in states if pd.notna(s)]
            dma_to_states[dma] = sorted(states_filtered)
        
        if debug:
            print("DMA to States mapping (sample first 5):")
            for i, (dma, states) in enumerate(list(dma_to_states.items())[:5]):
                print(f"  DMA {dma}: {states}")
        
        # Identify the treatment/control status for each state
        state_status = {}
        for state in sorted([s for s in geo_spine['state'].unique() if pd.notna(s)]):
            # Get all DMAs in this state
            dmas_in_state = geo_spine[geo_spine['state'] == state]['dma_code'].unique()
            
            # Check if all DMAs in the state are either treatment or control
            treatment_count = sum(1 for dma in dmas_in_state if dma in treatment_dmas)
            control_count = sum(1 for dma in dmas_in_state if dma in control_dmas)
            
            if debug:
                print(f"{state} ({state}): ", end="")
            
            if treatment_count > 0 and control_count == 0:
                # All DMAs in state are treatment
                state_status[state] = 'treatment'
                if debug:
                    print("treatment")
            elif control_count > 0 and treatment_count == 0:
                # All DMAs in state are control
                state_status[state] = 'control'
                if debug:
                    print("control")
            else:
                if treatment_count > 0 or control_count > 0:
                    # State has mixed treatment/control status
                    state_status[state] = 'mixed'
                    if debug:
                        print("mixed")
                else:
                    # State has no DMAs in the study
                    state_status[state] = 'unmapped'
                    if debug:
                        print("unmapped")
    
    # Load US States geometry from a GeoJSON URL
    states_geojson_url = "https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/us-states.json"
    us_states_gdf = gpd.read_file(states_geojson_url)

    # --- DEBUG CHECK: Verify loaded data ---
    if 'name' not in us_states_gdf.columns or not us_states_gdf['name'].iloc[0] == 'Alabama':
        print("\nERROR: Failed to load correct US States GeoJSON.")
        print("Loaded columns:", us_states_gdf.columns)
        print("Sample data:", us_states_gdf.head())
        # Handle error appropriately, maybe raise an exception or plot an error message
        ax.set_title("Error Loading State Map Data")
        ax.text(0.5, 0.5, "Failed to load state geometry data from URL.", ha='center', va='center')
        if output_path:
             plt.savefig(output_path)
             plt.close(fig)
        return output_path or fig
    # --- END DEBUG CHECK ---
    
    if debug:
        print("\nDEBUG: Successfully loaded US States GeoDataFrame from URL:")
        print(us_states_gdf[['id', 'name', 'geometry']].head())
        print(f"Columns: {us_states_gdf.columns}")

    # Filter out Alaska and Hawaii for a continental US view
    us_states_gdf = us_states_gdf[~us_states_gdf['name'].isin(['Alaska', 'Hawaii'])]
    
    # Use full state names from spine for merging
    state_name_map = geo_spine[['state', 'state_name']].drop_duplicates().set_index('state')['state_name']
    state_status_df = pd.DataFrame(list(state_status.items()), columns=['state_abbr', 'status'])
    state_status_df['STATE_NAME'] = state_status_df['state_abbr'].map(state_name_map)

    if debug:
        print("\nState Status DF with Names:")
        print(state_status_df.head())

    # Merge the status data with the GeoDataFrame
    merged_gdf = us_states_gdf.merge(state_status_df, left_on='name', right_on='STATE_NAME', how='left')

    # Fill states not in our assignment data with 'unmapped'
    merged_gdf['status'] = merged_gdf['status'].fillna('unmapped')

    if debug:
        print("\nMerged GeoDataFrame:")
        print(merged_gdf[['name', 'STATE_NAME', 'status', 'geometry']].head())
        print(f"Number of states in merged GDF: {len(merged_gdf)}")
        print(f"Status counts:\n{merged_gdf['status'].value_counts()}")

    # Define desired colors
    status_colors = {
        'treatment': '#1f77b4',  # Blue
        'control': '#ff7f0e',     # Orange
        'mixed': '#2ca02c',       # Green
        'unmapped': '#d3d3d3'     # Light gray
    }

    # Create cmap based on the actual sorted unique values in the status column
    sorted_statuses = sorted(merged_gdf['status'].unique())
    ordered_colors = [status_colors[status] for status in sorted_statuses]
    cmap = ListedColormap(ordered_colors)

    # Plot the choropleth map
    merged_gdf.plot(column='status', 
                    categorical=True, 
                    legend=False, # We create a custom legend below
                    cmap=cmap, 
                    linewidth=0.8, 
                    ax=ax, 
                    edgecolor='0.8',
                    missing_kwds={
                        "color": status_colors['unmapped'], # Use explicit color for missing
                        "edgecolor": "red",
                        "hatch": "///",
                        "label": "Missing values",
                    })

    # Set title and remove axis
    ax.set_title("US State Treatment/Control Assignment")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off() # Turn off the axis frame for a cleaner map

    # Create a custom legend using the defined colors
    legend_patches = [
        mpatches.Patch(color=status_colors['treatment'], label='Treatment'),
        mpatches.Patch(color=status_colors['control'], label='Control'),
        mpatches.Patch(color=status_colors['mixed'], label='Mixed'),
        mpatches.Patch(color=status_colors['unmapped'], label='Unmapped')
    ]
    ax.legend(handles=legend_patches, loc='lower right', title="Assignment Status")
    
    # Save or return the figure
    if output_path:
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Map saved to {output_path}")
        return output_path
    else:
        return fig

```

## File: src/geo_causal_inference/utils.py

- Extension: .py
- Language: python
- Size: 13532 bytes
- Created: 2025-04-04 11:18:05
- Modified: 2025-04-04 11:18:05

### Code

```python
"""
Utility functions for Trimmed Match marketing experiments.

This module provides common utilities used across the package.
"""

import pandas as pd
import numpy as np
import datetime
import re
import matplotlib.pyplot as plt
import seaborn as sns

from trimmed_match.design.common_classes import TimeWindow as OriginalTimeWindow, GeoAssignment, GeoXType
from trimmed_match.design.util import human_readable_number, format_design_table, create_output_table


class TimeWindow(OriginalTimeWindow):
    """Extension of TimeWindow class with start_date and end_date properties."""
    
    @property
    def start_date(self):
        """Alias for first_day."""
        return self.first_day
        
    @property
    def end_date(self):
        """Alias for last_day."""
        return self.last_day


def parse_date(date_str):
    """Parse date string to datetime.
    
    Args:
        date_str: Date string in various formats
        
    Returns:
        datetime.datetime
    """
    if isinstance(date_str, (datetime.datetime, pd.Timestamp)):
        return date_str
    
    # Remove quotes if present
    date_str = date_str.replace('"', '')
    
    return pd.to_datetime(date_str)


def format_budget(budget_value):
    """Format budget value to human readable format.
    
    Args:
        budget_value: Budget value to format
        
    Returns:
        str: Formatted budget value
    """
    return human_readable_number(budget_value)


def create_time_window(start_date, end_date=None, duration_days=None):
    """Create a time window for experiment design.
    
    Args:
        start_date: Start date of the time window
        end_date: End date of the time window
        duration_days: Duration of the time window in days
        
    Returns:
        TimeWindow
    """
    start_date = parse_date(start_date)
    
    if end_date is not None:
        end_date = parse_date(end_date)
    elif duration_days is not None:
        end_date = start_date + datetime.timedelta(days=duration_days)
    else:
        raise ValueError("Either end_date or duration_days must be specified")
        
    return TimeWindow(start_date, end_date)


def get_geox_type(type_str):
    """Get GeoXType from string.
    
    Args:
        type_str: String representation of GeoXType
        
    Returns:
        GeoXType
    """
    if isinstance(type_str, GeoXType):
        return type_str
        
    return GeoXType[type_str.upper()]


def format_summary_table(design_results, minimum_detectable_iroas, min_detectable_lift=None):
    """Format summary table for experiment design.
    
    Args:
        design_results: Dict with design results
        minimum_detectable_iroas: Minimum detectable iROAS
        min_detectable_lift: Minimum detectable lift
        
    Returns:
        pandas.DataFrame: Formatted summary table
    """
    results = design_results["results"]
    
    # Add debug print to understand results structure
    print("\n==== DEBUG: Design Results Info ====")
    print(f"Results shape: {results.shape}")
    
    # Find the row with minimum rmse_cost_adjusted
    min_rmse_idx = results['rmse_cost_adjusted'].idxmin()
    print(f"Min RMSE Cost Adjusted Index: {min_rmse_idx}")
    
    # Get the best design row
    best_design = results.loc[min_rmse_idx]
    print(f"Best design type: {type(best_design)}")
    
    # Check if we have multiple rows with the same index
    if isinstance(best_design, pd.DataFrame):
        print(f"Multiple design rows found with index {min_rmse_idx}, shape: {best_design.shape}")
        # Use the first row
        best_design = best_design.iloc[0]
        print("Using first row of best design")
    
    # Print key values for debugging
    print("\n==== Best Design Values ====")
    for col in ['budget', 'rmse', 'rmse_cost_adjusted', 'pair_index', 'num_pairs']:
        if col in best_design:
            print(f"{col}: {best_design[col]}")
    
    # Extract scalar values to avoid formatting errors with Series
    budget_value = best_design['budget']
    rmse = best_design['rmse']
    rmse_cost_adjusted = best_design['rmse_cost_adjusted']
    pair_index = best_design['pair_index']
    
    summary = pd.DataFrame([{
        'Budget': format_budget(budget_value),
        'Min Detectable iROAS': f"{minimum_detectable_iroas:.2f}",
        'RMSE': f"{rmse:.4f}",
        'RMSE (Cost Adjusted)': f"{rmse_cost_adjusted:.4f}",
        'Pair Index': int(pair_index),
    }])
    
    if min_detectable_lift is not None:
        summary['Min Detectable Lift (%)'] = f"{min_detectable_lift:.2f}%"
        
    return summary


def combine_design_outputs(geo_level_time_series, geopairs, treatment_geo, control_geo):
    """Combine geo pairs with time series data.
    
    Args:
        geo_level_time_series: DataFrame with geo-level time series data
        geopairs: DataFrame with geo pairs
        treatment_geo: List of treatment geos
        control_geo: List of control geos
        
    Returns:
        pandas.DataFrame: Combined data
    """
    tmp = geo_level_time_series[
        geo_level_time_series["geo"].isin(treatment_geo + control_geo)
    ]
    
    design_data = tmp.merge(
        geopairs[["geo", "pair", "assignment"]], 
        on="geo", 
        how="left"
    )
    
    return design_data


# Post-analysis visualization functions

def ts_plot(x, y, **kwargs):
    """Plot time series data.
    
    Args:
        x: X-axis values (date)
        y: Y-axis values (metric)
        **kwargs: Additional arguments
    """
    ax = plt.gca()
    data = kwargs.pop("data")
    data.plot(x=x, y=y, ax=ax, grid=False, **kwargs)


def plot_time_series_comparison(data, metric, test_period, design_period, cooldown_period=None, 
                                group_treatment=GeoAssignment.TREATMENT, 
                                group_control=GeoAssignment.CONTROL,
                                figsize=(15, 7.5)):
    """Plot time series comparison between treatment and control groups.
    
    Args:
        data: DataFrame with experiment data
        metric: Column name to plot (e.g., "response", "cost")
        test_period: TimeWindow for test period
        design_period: TimeWindow for design period
        cooldown_period: Optional TimeWindow for cooldown period
        group_treatment: Value indicating treatment group
        group_control: Value indicating control group
        figsize: Figure size as (width, height)
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Get treatment and control time series
    treatment_data = data[data["assignment"] == group_treatment]
    control_data = data[data["assignment"] == group_control]
    
    treatment_time_series = treatment_data.groupby(["date"], as_index=False)[metric].sum()
    control_time_series = control_data.groupby(["date"], as_index=False)[metric].sum()
    
    # Plot the data
    ax.plot(treatment_time_series["date"], treatment_time_series[metric], label="Treatment")
    ax.plot(control_time_series["date"], control_time_series[metric], label="Control")
    
    # Add vertical lines for periods
    ax.axvline(x=test_period.start_date, color="black", ls="-", label='Experiment period')
    ax.axvline(x=test_period.end_date, color="black", ls="-")
    
    ax.axvline(x=design_period.start_date, color="red", ls="--", label='Design evaluation period')
    ax.axvline(x=design_period.end_date, color="red", ls="--")
    
    if cooldown_period:
        ax.axvline(x=cooldown_period.end_date, color="black", ls="--", label='End of cooldown period')
    
    # Set labels and legend
    ax.set_ylabel(metric)
    ax.set_xlabel("Date")
    ax.legend(bbox_to_anchor=(0.5, 1.1), loc='center')
    
    plt.tight_layout()
    return fig


def plot_pair_time_series(data, pairs, response_col="response", 
                          group_treatment=GeoAssignment.TREATMENT,
                          group_control=GeoAssignment.CONTROL,
                          test_period=None, design_period=None, cooldown_period=None,
                          col_wrap=3, height=5, aspect=2):
    """Plot time series for each geo pair.
    
    Args:
        data: DataFrame with experiment data
        pairs: List of pair IDs
        response_col: Column name for response
        group_treatment: Value indicating treatment group
        group_control: Value indicating control group
        test_period: TimeWindow for test period
        design_period: TimeWindow for design period
        cooldown_period: Optional TimeWindow for cooldown period
        col_wrap: Number of columns in the facet grid
        height: Height of each facet
        aspect: Aspect ratio of each facet
        
    Returns:
        seaborn.FacetGrid: The created facet grid
    """
    # Create facet grid
    g = sns.FacetGrid(
        data, col="pair", hue="assignment", col_wrap=col_wrap,
        sharey=False, sharex=False, legend_out=False, height=height, aspect=aspect
    )
    
    # Plot time series
    g = g.map_dataframe(ts_plot, "date", response_col).add_legend()
    
    # Add vertical lines and legends for each pair
    for ind, ax in enumerate(g.axes):
        # Skip if we've run out of pairs
        if ind >= len(pairs):
            break
            
        pair_id = pairs[ind]
        
        # Get geo IDs for control and treatment
        control_geo = data[(data["pair"] == pair_id) & 
                          (data["assignment"] == group_control)]["geo"].values[0]
        treatment_geo = data[(data["pair"] == pair_id) & 
                            (data["assignment"] == group_treatment)]["geo"].values[0]
        
        # Add vertical lines for periods
        if test_period:
            ax.axvline(x=test_period.start_date, color="black", ls="-")
            ax.axvline(x=test_period.end_date, color="black", ls="-")
            
        if design_period:
            ax.axvline(x=design_period.start_date, color="red", ls="--")
            ax.axvline(x=design_period.end_date, color="red", ls="--")
            
        if cooldown_period:
            ax.axvline(x=cooldown_period.end_date, color="black", ls="--")
        
        # Add custom legend with geo IDs
        ax.legend([
            f"Treatment (geo {treatment_geo})",
            f"Control (geo {control_geo})",
            "Experiment period", 
            "Design evaluation period",
            "End of cooldown period"
        ], loc="best")
    
    plt.tight_layout()
    return g


def plot_correlation_matrix(data, design_period, test_period,
                           group_treatment=GeoAssignment.TREATMENT,
                           group_control=GeoAssignment.CONTROL,
                           response_col="response", figsize=(15, 15)):
    """Plot correlation matrix for geo pairs.
    
    Args:
        data: DataFrame with experiment data
        design_period: TimeWindow for design period
        test_period: TimeWindow for test period
        group_treatment: Value indicating treatment group
        group_control: Value indicating control group
        response_col: Column name for response
        figsize: Figure size as (width, height)
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Filter data for design and test periods
    period_data = data[data["period"].isin([0, 1])]
    
    # Aggregate data by period, assignment, and pair
    agg_data = period_data.groupby(["period", "assignment", "pair"])[response_col].sum() ** 0.5
    agg_data = agg_data.reset_index()
    
    # Create masks for each group
    pretreatment = (agg_data["period"] == 0) & (agg_data["assignment"] == group_treatment)
    precontrol = (agg_data["period"] == 0) & (agg_data["assignment"] == group_control)
    posttreatment = (agg_data["period"] == 1) & (agg_data["assignment"] == group_treatment)
    postcontrol = (agg_data["period"] == 1) & (agg_data["assignment"] == group_control)
    
    # Create comparison dataframe
    comp = pd.DataFrame({
        "pretreatment": agg_data[pretreatment][response_col].to_list(),
        "precontrol": agg_data[precontrol][response_col].to_list(),
        "posttreatment": agg_data[posttreatment][response_col].to_list(),
        "postcontrol": agg_data[postcontrol][response_col].to_list()
    })
    
    # Create the plot
    fig, ax = plt.subplots(4, 4, figsize=figsize)
    label = ["pretreatment", "precontrol", "posttreatment", "postcontrol"]
    min_ax = min(comp.min())
    max_ax = max(comp.max())
    
    for col_ind in range(4):
        for row_ind in range(4):
            if col_ind > row_ind:
                ax[row_ind, col_ind].scatter(comp[label[col_ind]], comp[label[row_ind]])
                ax[row_ind, col_ind].plot([min_ax * 0.97, max_ax * 1.03],
                                         [min_ax * 0.97, max_ax * 1.03], 'r')
                ax[row_ind, col_ind].set_xlim([min_ax * 0.97, max_ax * 1.03])
                ax[row_ind, col_ind].set_ylim([min_ax * 0.97, max_ax * 1.03])
            elif col_ind == row_ind:
                ax[row_ind, col_ind].annotate(label[col_ind], size=20, xy=(0.15, 0.5),
                                            xycoords="axes fraction")
                ax[row_ind, col_ind].set_xlim([min_ax * 0.97, max_ax * 1.03])
                ax[row_ind, col_ind].set_ylim([min_ax * 0.97, max_ax * 1.03])
            else:
                ax[row_ind, col_ind].axis("off")
                
    plt.tight_layout()
    return fig

```

## File: src/geo_causal_inference/validation.py

- Extension: .py
- Language: python
- Size: 7396 bytes
- Created: 2025-03-27 11:16:38
- Modified: 2025-03-25 14:40:51

### Code

```python
"""
Validation functionality for Trimmed Match marketing experiments.

This module handles validation of input data and parameters for
the experiment design process.
"""

import datetime
import re
import pandas as pd
import numpy as np
from colorama import Fore, Style

from trimmed_match.design.util import find_days_to_exclude, overlap_percent
from trimmed_match.design.util import check_time_periods, check_input_data


def validate_input_data(geo_level_time_series):
    """Validate input data structure and format.
    
    Args:
        geo_level_time_series: DataFrame with geo-level time series data
        
    Returns:
        Validated DataFrame
    
    Raises:
        ValueError: If the input data doesn't meet requirements
    """
    return check_input_data(geo_level_time_series)


def validate_date_format(date_str):
    """Convert string date to datetime.
    
    Args:
        date_str: Date string in the format "YYYY-MM-DD" possibly with quotes
        
    Returns:
        pandas Timestamp
    """
    return pd.to_datetime(date_str.replace("\"", ""))


def validate_experiment_periods(
    geo_level_time_series,
    eval_start_date,
    coverage_test_start_date,
    experiment_duration_weeks,
    day_week_exclude=None
):
    """Validate experiment time periods.
    
    Args:
        geo_level_time_series: DataFrame with geo-level time series data
        eval_start_date: Start date of evaluation period
        coverage_test_start_date: Start date of coverage test period
        experiment_duration_weeks: Duration of experiment in weeks
        day_week_exclude: List of days/weeks to exclude from analysis
        
    Returns:
        tuple: (pass_checks, error_message, days_exclude)
        
    Raises:
        ValueError: If validation fails
    """
    if day_week_exclude is None:
        day_week_exclude = []
        
    number_of_days_test = experiment_duration_weeks * 7
    eval_end_date = eval_start_date + datetime.timedelta(days=number_of_days_test-1)
    coverage_test_end_date = coverage_test_start_date + datetime.timedelta(
        days=number_of_days_test - 1)
    
    # Find all the days to exclude from the analysis
    periods_to_exclude = find_days_to_exclude(day_week_exclude)
    days_exclude = expand_time_windows(periods_to_exclude)
    
    # Get days in evaluation and coverage test periods
    days_in_eval = [
        x for x in geo_level_time_series["date"].drop_duplicates()
        if x in pd.Interval(eval_start_date, eval_end_date, closed="both")
    ]
    
    days_in_coverage_test = [
        x for x in geo_level_time_series["date"].drop_duplicates()
        if x in pd.Interval(coverage_test_start_date, coverage_test_end_date,
                            closed="both")
    ]
    
    # Check for overlaps
    percentage_overlap_eval = overlap_percent(days_exclude, days_in_eval)
    if percentage_overlap_eval > 0:
        raise ValueError(
            f'{Fore.RED}WARNING: {percentage_overlap_eval:.2f}% of the evaluation '
            f'time period overlaps with days/weeks excluded in input. '
            f'Please change eval_start_date.{Style.RESET_ALL}'
        )
    
    percentage_overlap_coverage_test = overlap_percent(days_exclude, days_in_coverage_test)
    if percentage_overlap_coverage_test > 0:
        raise ValueError(
            f'{Fore.RED}WARNING: {percentage_overlap_coverage_test:.2f}% of the '
            f'AA test time period overlaps with days/weeks excluded in input. '
            f'Please change coverage_test_start_date.{Style.RESET_ALL}'
        )
    
    # Check that evaluation and AA test periods don't overlap
    percentage_overlap_eval_coverage_test = overlap_percent(days_in_eval, days_in_coverage_test)
    if percentage_overlap_eval_coverage_test > 0:
        raise ValueError(
            f'{Fore.RED}WARNING: part of the evaluation time period overlaps with '
            f'the coverage test period. Please change eval_start_date.{Style.RESET_ALL}'
        )
    
    # Check time periods
    try:
        pass_checks = check_time_periods(
            geox_data=geo_level_time_series,
            start_date_eval=eval_start_date,
            start_date_aa_test=coverage_test_start_date,
            experiment_duration_weeks=experiment_duration_weeks,
            frequency="infer"
        )
        return pass_checks, None, days_exclude
    except Exception as e:
        return False, str(e), days_exclude


def validate_geos(geo_level_time_series, geos_exclude=None):
    """Validate geo exclusions and ensure even number of geos.
    
    Args:
        geo_level_time_series: DataFrame with geo-level time series data
        geos_exclude: List of geo IDs to exclude
        
    Returns:
        tuple: (validated_excluded_geos, warning_messages)
    """
    if geos_exclude is None:
        geos_exclude = []
        
    warnings = []
    geos_exclude = [int(x) for x in geos_exclude]
    all_geos = set(geo_level_time_series["geo"].to_list())
    non_existing_geos = set(geos_exclude) - set(all_geos)
    
    if non_existing_geos:
        geos_exclude = [x for x in geos_exclude if x not in non_existing_geos]
        warnings.append(
            f'{Fore.RED}WARNING: Attempting to exclude the geos {non_existing_geos} '
            f'which do not exist in the input data.{Style.RESET_ALL}'
        )
    
    # Ensure even number of geos
    num_geos = len(all_geos - set(geos_exclude))
    if num_geos % 2 != 0:
        geo_level_data = geo_level_time_series.groupby("geo", as_index=False)["response"].sum()
        largest_geo = geo_level_data.loc[geo_level_data["response"].idxmax()]
        
        warnings.append(
            f'\nSince the number of geos is odd, we have removed the following '
            f'geo (the one with largest response): {largest_geo["geo"]}'
        )
        geos_exclude.append(largest_geo["geo"])
    
    return geos_exclude, warnings


def expand_time_windows(time_windows):
    """Expand time windows into individual days.
    
    Args:
        time_windows: List of time windows
        
    Returns:
        List of individual dates
    """
    all_days = []
    for tw in time_windows:
        start_date = tw[0]
        end_date = tw[1]
        days = pd.date_range(start=start_date, end=end_date)
        all_days.extend(days)
    return all_days


def parse_budget_values(experiment_budget, alternative_budget):
    """Parse budget values from strings.
    
    Args:
        experiment_budget: Maximum budget string
        alternative_budget: String of comma-separated alternative budgets
        
    Returns:
        tuple: (experiment_budget_float, alternative_budget_list)
    """
    experiment_budget = float(experiment_budget)
    
    if alternative_budget and alternative_budget.strip():
        additional_budget = [
            float(re.sub(r"\W+", "", x)) 
            for x in alternative_budget.split(',')
        ]
    else:
        additional_budget = []
        
    return experiment_budget, additional_budget


def parse_excluded_geos(geos_exclude_str):
    """Parse excluded geos from string.
    
    Args:
        geos_exclude_str: String of comma-separated geo IDs to exclude
        
    Returns:
        List of geo IDs to exclude
    """
    if geos_exclude_str and geos_exclude_str.strip():
        return [re.sub(r"\W+", "", x) for x in geos_exclude_str.split(',')]
    return []

```

## File: src/examples/cost_response_join_example.py

- Extension: .py
- Language: python
- Size: 8889 bytes
- Created: 2025-03-27 13:29:26
- Modified: 2025-03-27 13:29:26

### Code

```python
"""
This example script demonstrates how to use the geographic spine table
to join marketing datasets at different geographic levels.

The steps in this script can be converted to a Jupyter notebook.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.append('..')

# Import data pipeline modules
from src.data_pipeline.data_standardizer import DateStandardizer, GeoStandardizer, CostStandardizer
from src.data_pipeline.data_joiner import DatasetCleaner
from src.data_pipeline.geo_joiner import GeoJoiner, GeoHierarchyJoiner

# Set paths
RAW_DATA_PATH = 'raw_data/'
REFERENCE_DATA_PATH = 'data/reference/'
OUTPUT_PATH = 'data/processed/'

# Ensure output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)


def load_datasets():
    """Load all raw datasets."""
    print("Loading datasets...")
    
    # Load GA4 Sessions data (city level)
    ga4_sessions = pd.read_csv(os.path.join(RAW_DATA_PATH, 'ga4_sessions.csv'))
    print(f"GA4 Sessions shape: {ga4_sessions.shape}")
    
    # Load Meta Geo Spend data (DMA level)
    meta_geo_spend = pd.read_csv(os.path.join(RAW_DATA_PATH, 'meta_geo_spend.csv'))
    print(f"Meta Geo Spend shape: {meta_geo_spend.shape}")
    
    # Load TikTok Geo Spend data (state level)
    tiktok_geo_spend = pd.read_csv(os.path.join(RAW_DATA_PATH, 'tiktok_geo_spend.csv'))
    print(f"TikTok Geo Spend shape: {tiktok_geo_spend.shape}")
    
    # Load Google Ads Geo Spend data (city level)
    gads_geo_spend = pd.read_csv(os.path.join(RAW_DATA_PATH, 'gads_geo_spend.csv'))
    print(f"Google Ads Geo Spend shape: {gads_geo_spend.shape}")
    
    return ga4_sessions, meta_geo_spend, tiktok_geo_spend, gads_geo_spend


def clean_datasets(ga4_sessions, meta_geo_spend, tiktok_geo_spend, gads_geo_spend):
    """Clean and standardize datasets."""
    print("\nCleaning datasets...")
    
    # Initialize standardizers
    date_standardizer = DateStandardizer()
    geo_standardizer = GeoStandardizer()
    cost_standardizer = CostStandardizer()
    
    # Initialize dataset cleaner
    dataset_cleaner = DatasetCleaner(
        standardizers={
            'date': date_standardizer,
            'geo': geo_standardizer,
            'cost': cost_standardizer
        }
    )
    
    # Clean GA4 sessions data at city level
    ga4_clean = dataset_cleaner.clean_ga4_sessions(ga4_sessions, geo_level='city')
    print(f"Cleaned GA4 Sessions shape: {ga4_clean.shape}")
    
    # Clean Meta spend data at DMA level
    meta_clean = dataset_cleaner.clean_meta_spend(meta_geo_spend)
    print(f"Cleaned Meta Spend shape: {meta_clean.shape}")
    
    # Clean TikTok spend data at state level
    tiktok_clean = dataset_cleaner.clean_tiktok_spend(tiktok_geo_spend)
    print(f"Cleaned TikTok Spend shape: {tiktok_clean.shape}")
    
    # Clean Google Ads spend data at city level
    gads_clean = dataset_cleaner.clean_gads_spend(gads_geo_spend, geo_level='city')
    print(f"Cleaned Google Ads Spend shape: {gads_clean.shape}")
    
    return ga4_clean, meta_clean, tiktok_clean, gads_clean


def enrich_with_geo_reference(ga4_clean, meta_clean, tiktok_clean, gads_clean):
    """Enrich datasets with geographic reference data."""
    print("\nEnriching datasets with geographic reference data...")
    
    # Initialize GeoJoiner with reference data path
    geo_joiner = GeoJoiner(reference_data_path=REFERENCE_DATA_PATH)
    
    # Enrich GA4 city-level data with DMA information
    ga4_enriched = geo_joiner.enrich_city_data(
        ga4_clean, 
        city_col='City', 
        state_col='state' if 'state' in ga4_clean.columns else None
    )
    print(f"Enriched GA4 Sessions shape: {ga4_enriched.shape}")
    
    # Enrich Google Ads city-level data with DMA information
    gads_enriched = geo_joiner.enrich_city_data(
        gads_clean, 
        city_col='City', 
        state_col='state' if 'state' in gads_clean.columns else None
    )
    print(f"Enriched Google Ads Spend shape: {gads_enriched.shape}")
    
    # Enrich Meta DMA-level data with state information
    meta_enriched = meta_clean
    if 'dma_name' in meta_clean.columns:
        meta_enriched = geo_joiner.enrich_dma_data(meta_clean, dma_col='dma_name')
    print(f"Enriched Meta Spend shape: {meta_enriched.shape}")
    
    return ga4_enriched, meta_enriched, tiktok_clean, gads_enriched


def distribute_dma_and_state_values(meta_enriched, tiktok_clean, geo_joiner):
    """Distribute DMA and state-level values to city level."""
    print("\nDistributing aggregated values to city level...")
    
    # Distribute Meta spend (DMA level) to constituent cities
    meta_city = None
    if 'dma_name' in meta_enriched.columns and 'Cost' in meta_enriched.columns:
        meta_city = geo_joiner.distribute_dma_values_to_cities(
            meta_enriched,
            dma_col='dma_name',
            value_col='Cost',
            distribution_method='equal'
        )
        print(f"Meta spend distributed to cities shape: {meta_city.shape}")
    
    # For TikTok data (state level), we need to distribute to DMAs first, then to cities
    # This is a simplistic approach; in practice, you might want more sophisticated distribution
    tiktok_dma = None
    # This would require additional logic to map states to DMAs and then to cities
    
    return meta_city, tiktok_dma


def join_datasets_by_geo_hierarchy(ga4_enriched, gads_enriched, meta_enriched, tiktok_clean):
    """Join datasets with different geographic hierarchies."""
    print("\nJoining datasets across geographic levels...")
    
    # Initialize the GeoHierarchyJoiner
    hierarchy_joiner = GeoHierarchyJoiner(reference_data_path=REFERENCE_DATA_PATH)
    
    # Prepare datasets with their geographic column and level
    datasets = [
        (ga4_enriched, 'City', 'city'),  # City-level GA4 data
        (gads_enriched, 'City', 'city'),  # City-level Google Ads data
        (meta_enriched, 'DMA Region', 'dma'),  # DMA-level Meta data
        (tiktok_clean, 'Region', 'state')  # State-level TikTok data
    ]
    
    # Join datasets
    joined_df = hierarchy_joiner.join_datasets(
        datasets=datasets,
        date_col='Date',
        value_cols=['Sessions', 'gads_cost', 'meta_cost', 'tiktok_cost']
    )
    
    print(f"Joined dataset shape: {joined_df.shape}")
    
    return joined_df


def create_final_dataset(joined_df):
    """Create the final cost-response dataset."""
    print("\nCreating final cost-response dataset...")
    
    # Rename columns for consistency
    final_df = joined_df.copy()
    
    # Add total cost column
    cost_cols = [col for col in final_df.columns if 'cost' in col.lower()]
    final_df['total_cost'] = final_df[cost_cols].sum(axis=1, skipna=True)
    
    # Ensure date is in datetime format
    if 'Date' in final_df.columns:
        final_df['Date'] = pd.to_datetime(final_df['Date'])
    
    # Sort by date and geography
    sort_cols = ['Date']
    if 'City' in final_df.columns:
        sort_cols.append('City')
    final_df = final_df.sort_values(sort_cols).reset_index(drop=True)
    
    # Save the final dataset
    output_file = os.path.join(OUTPUT_PATH, 'cost_response_data.csv')
    final_df.to_csv(output_file, index=False)
    print(f"Final dataset saved to {output_file}")
    
    return final_df


def main():
    """Main function to run the pipeline."""
    # Load raw datasets
    ga4_sessions, meta_geo_spend, tiktok_geo_spend, gads_geo_spend = load_datasets()
    
    # Clean datasets
    ga4_clean, meta_clean, tiktok_clean, gads_clean = clean_datasets(
        ga4_sessions, meta_geo_spend, tiktok_geo_spend, gads_geo_spend
    )
    
    # Initialize GeoJoiner
    geo_joiner = GeoJoiner(reference_data_path=REFERENCE_DATA_PATH)
    
    # Enrich datasets with geographic reference data
    ga4_enriched, meta_enriched, tiktok_clean, gads_enriched = enrich_with_geo_reference(
        ga4_clean, meta_clean, tiktok_clean, gads_clean
    )
    
    # Distribute DMA and state-level values to city level if needed
    meta_city, tiktok_dma = distribute_dma_and_state_values(
        meta_enriched, tiktok_clean, geo_joiner
    )
    
    # Join datasets across geographic hierarchies
    joined_df = join_datasets_by_geo_hierarchy(
        ga4_enriched, gads_enriched, meta_enriched, tiktok_clean
    )
    
    # Create final cost-response dataset
    final_df = create_final_dataset(joined_df)
    
    print("\nDataset creation complete! Here's a sample of the final dataset:")
    print(final_df.head())
    
    # Basic statistics
    print("\nBasic statistics of the final dataset:")
    print(final_df.describe())
    
    # Missing values check
    print("\nMissing values in the final dataset:")
    print(final_df.isnull().sum())
    
    print("\nProcess completed successfully!")


if __name__ == '__main__':
    main()

```

