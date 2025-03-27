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