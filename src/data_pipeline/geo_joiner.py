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
