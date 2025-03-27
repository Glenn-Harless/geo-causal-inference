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
