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
from geo_causal_inference.visualization import plot_designs_comparison, plot_geo_time_series

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
    
    print(f"Loading data from: {test_data_path}")
    print(f"Using client name: {client_name}")
    print(f"Data frequency: {data_frequency}")
    
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
        experiment_budget=300000.0,
        minimum_detectable_iroas=3.0,
        average_order_value=256, # average order value in dollars / total sessions
        significance_level=0.10,
        power_level=0.80,
        use_cross_validation=False, # set to True if you want to use cross validation
        number_of_simulations=20, # number of simulations to run typically 200 but this wont finish
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
    summary.to_csv(os.path.join(data_dir, 'design_summary.csv'), index=False)
    
    # Save geo assignments
    geo_assignments = pd.DataFrame({
        'geo': sorted(treatment_geo + control_geo),
        'assignment': ['treatment' if geo in treatment_geo else 'control' 
                      for geo in sorted(treatment_geo + control_geo)]
    })
    geo_assignments.to_csv(os.path.join(data_dir, 'geo_assignments.csv'), index=False)
    
    # Save design results
    design_results['results'].to_csv(os.path.join(data_dir, 'design_results.csv'), index=False)
    
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
