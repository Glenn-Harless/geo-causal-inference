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
