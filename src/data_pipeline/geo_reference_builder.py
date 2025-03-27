"""
Geographic reference data builder for the geo-causal-inference project.

This module creates a comprehensive geographic spine table that helps
join datasets with different levels of geographic granularity.
"""

import pandas as pd
import os
from typing import Optional


class GeoReferenceBuilder:
    """Builder for geographic reference data tables."""
    
    def __init__(self, raw_data_path: str, output_path: str):
        """
        Initialize the GeoReferenceBuilder.
        
        Args:
            raw_data_path: Path to the raw geographic data files
            output_path: Path where the generated reference tables will be saved
        """
        self.raw_data_path = raw_data_path
        self.output_path = output_path
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
    
    def build_geo_spine_table(self, 
                             zip_city_file: str = 'zip_city_detail.csv',
                             zip_dma_file: str = 'zip_to_dma.csv',
                             output_file: str = 'geo_spine.csv') -> pd.DataFrame:
        """
        Build a comprehensive geographic spine table.
        
        Args:
            zip_city_file: Filename for the zip-to-city mapping data
            zip_dma_file: Filename for the zip-to-DMA mapping data
            output_file: Filename for the output spine table
            
        Returns:
            The created spine table DataFrame
        """
        # Load zip-to-city data with proper dtypes to preserve leading zeros
        zip_city_path = os.path.join(self.raw_data_path, zip_city_file)
        zip_city_df = pd.read_csv(zip_city_path, dtype={'DELIVERY ZIPCODE': str})
        
        # Simplify and standardize zip-city data
        # We're interested in the delivery zipcode, city, and state
        city_df = zip_city_df[[
            'DELIVERY ZIPCODE', 
            'PHYSICAL CITY', 
            'PHYSICAL STATE'
        ]].copy()
        
        # Rename columns to standard format
        city_df = city_df.rename(columns={
            'DELIVERY ZIPCODE': 'zip_code',
            'PHYSICAL CITY': 'city',
            'PHYSICAL STATE': 'state'
        })
        
        # Handle duplicate zip codes (keep the first occurrence for simplicity)
        city_df = city_df.drop_duplicates(subset=['zip_code'])
        
        # Load zip-to-DMA data with proper dtypes to preserve leading zeros
        zip_dma_path = os.path.join(self.raw_data_path, zip_dma_file)
        dma_df = pd.read_csv(zip_dma_path, dtype={'zip_code': str, 'dma_code': str})
        
        # Standardize DMA data
        dma_df = dma_df.rename(columns={
            'zip_code': 'zip_code',
            'dma_code': 'dma_code',
            'dma_description': 'dma_name'
        })
        
        # Join city and DMA data on zip code
        spine_df = pd.merge(
            city_df,
            dma_df,
            on='zip_code',
            how='left'
        )
        
        # Create state abbreviation - full name mapping
        state_mapping = self._create_state_mapping()
        if state_mapping is not None:
            # Add full state name
            spine_df['state_name'] = spine_df['state'].map(state_mapping)
        
        # Standardize column values
        spine_df['city'] = spine_df['city'].str.strip().str.upper()
        spine_df['dma_name'] = spine_df['dma_name'].fillna('').str.strip().str.upper()
        
        # Add geographic hierarchies
        # This allows for rolling up or drilling down between different geo levels
        spine_df['geo_key_zip'] = spine_df['zip_code']
        spine_df['geo_key_city'] = spine_df['city'] + ', ' + spine_df['state']
        spine_df['geo_key_dma'] = spine_df['dma_name']
        spine_df['geo_key_state'] = spine_df['state']
        
        # Save the spine table
        output_path = os.path.join(self.output_path, output_file)
        spine_df.to_csv(output_path, index=False)
        print(f"Geographic spine table saved to {output_path}")
        
        return spine_df
    
    def _create_state_mapping(self) -> Optional[dict]:
        """
        Create a mapping of state abbreviations to full state names.
        
        Returns:
            Dictionary mapping state abbreviations to full names, or None if not available
        """
        # Static mapping of state abbreviations to full names
        state_mapping = {
            'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
            'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
            'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
            'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
            'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
            'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
            'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
            'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
            'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
            'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
            'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
            'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
            'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia',
            'PR': 'Puerto Rico', 'VI': 'Virgin Islands', 'GU': 'Guam'
        }
        return state_mapping
    
    def build_city_dma_mapping(self, 
                             spine_file: str = 'geo_spine.csv',
                             output_file: str = 'city_dma_mapping.csv') -> pd.DataFrame:
        """
        Build a city-to-DMA mapping table.
        
        Args:
            spine_file: Filename for the geographic spine table
            output_file: Filename for the output city-DMA mapping
            
        Returns:
            The created city-DMA mapping DataFrame
        """
        # Load the spine table
        spine_path = os.path.join(self.output_path, spine_file)
        
        # If the spine table doesn't exist yet, build it
        if not os.path.exists(spine_path):
            spine_df = self.build_geo_spine_table(output_file=spine_file)
        else:
            spine_df = pd.read_csv(spine_path, dtype={'zip_code': str, 'dma_code': str})
        
        # Extract city-DMA mapping
        city_dma_df = spine_df[[
            'city', 
            'state', 
            'dma_name', 
            'dma_code'
        ]].copy()
        
        # Filter out rows without DMA info
        city_dma_df = city_dma_df[city_dma_df['dma_name'].notna() & (city_dma_df['dma_name'] != '')]
        
        # Handle multiple DMAs per city by selecting the most frequent DMA for each city-state pair
        city_dma_counts = city_dma_df.groupby(['city', 'state', 'dma_name', 'dma_code']).size().reset_index(name='count')
        city_dma_top = city_dma_counts.sort_values('count', ascending=False).drop_duplicates(['city', 'state'])
        city_dma_mapping = city_dma_top.drop('count', axis=1)
        
        # Save the city-DMA mapping
        output_path = os.path.join(self.output_path, output_file)
        city_dma_mapping.to_csv(output_path, index=False)
        print(f"City-DMA mapping saved to {output_path}")
        
        return city_dma_mapping
    
    def build_dma_state_mapping(self,
                               spine_file: str = 'geo_spine.csv',
                               output_file: str = 'dma_state_mapping.csv') -> pd.DataFrame:
        """
        Build a DMA-to-state mapping table.
        
        Some DMAs span multiple states, this maps each DMA to its constituent states.
        
        Args:
            spine_file: Filename for the geographic spine table
            output_file: Filename for the output DMA-state mapping
            
        Returns:
            The created DMA-state mapping DataFrame
        """
        # Load the spine table
        spine_path = os.path.join(self.output_path, spine_file)
        
        # If the spine table doesn't exist yet, build it
        if not os.path.exists(spine_path):
            spine_df = self.build_geo_spine_table(output_file=spine_file)
        else:
            spine_df = pd.read_csv(spine_path, dtype={'zip_code': str, 'dma_code': str})
        
        # Create DMA-state mapping
        dma_state_df = spine_df[['dma_name', 'dma_code', 'state']].drop_duplicates()
        
        # Filter out rows without DMA info
        dma_state_df = dma_state_df[dma_state_df['dma_name'].notna() & (dma_state_df['dma_name'] != '')]
        
        # Calculate the percentage of zip codes in each state for each DMA
        dma_state_counts = spine_df.groupby(['dma_name', 'dma_code', 'state']).size().reset_index(name='zip_count')
        dma_totals = dma_state_counts.groupby('dma_name')['zip_count'].sum().reset_index(name='total_zips')
        dma_state_pct = pd.merge(dma_state_counts, dma_totals, on='dma_name')
        dma_state_pct['state_percentage'] = dma_state_pct['zip_count'] / dma_state_pct['total_zips'] * 100
        
        # Keep only state-DMA pairs with significant representation (e.g., > 5%)
        significant_dma_states = dma_state_pct[dma_state_pct['state_percentage'] > 5]
        
        # Create a list of states for each DMA
        dma_states = significant_dma_states.groupby(['dma_name', 'dma_code'])['state'].apply(list).reset_index()
        
        # Save the DMA-state mapping
        output_path = os.path.join(self.output_path, output_file)
        dma_states.to_csv(output_path, index=False)
        print(f"DMA-state mapping saved to {output_path}")
        
        return dma_states


def main():
    """Build all geographic reference tables."""
    # Set paths
    raw_data_path = os.path.join('raw_data', 'region_data')
    output_path = os.path.join('data', 'reference')
    
    # Initialize builder
    builder = GeoReferenceBuilder(raw_data_path, output_path)
    
    # Build spine table
    spine_df = builder.build_geo_spine_table()
    
    # Build city-DMA mapping
    city_dma_df = builder.build_city_dma_mapping()
    
    # Build DMA-state mapping
    dma_state_df = builder.build_dma_state_mapping()


if __name__ == '__main__':
    main()
