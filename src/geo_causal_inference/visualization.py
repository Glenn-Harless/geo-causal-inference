"""
Visualization functionality for Trimmed Match marketing experiments.

This module provides plotting and visualization functions for
experiment design and analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


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
    best_design = results.loc[results[metric].idxmin()]
    ax.annotate(
        f'Best: {best_design["budget"]:.0f}',
        xy=(best_design['budget'], best_design[metric]),
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
