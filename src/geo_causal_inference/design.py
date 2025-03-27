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
