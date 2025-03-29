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
