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
