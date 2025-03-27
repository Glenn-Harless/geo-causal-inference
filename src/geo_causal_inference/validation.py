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
