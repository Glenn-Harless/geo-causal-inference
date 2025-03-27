"""
This example script demonstrates how to use the geographic spine table
to join marketing datasets at different geographic levels.

The steps in this script can be converted to a Jupyter notebook.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.append('..')

# Import data pipeline modules
from src.data_pipeline.data_standardizer import DateStandardizer, GeoStandardizer, CostStandardizer
from src.data_pipeline.data_joiner import DatasetCleaner
from src.data_pipeline.geo_joiner import GeoJoiner, GeoHierarchyJoiner

# Set paths
RAW_DATA_PATH = 'raw_data/'
REFERENCE_DATA_PATH = 'data/reference/'
OUTPUT_PATH = 'data/processed/'

# Ensure output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)


def load_datasets():
    """Load all raw datasets."""
    print("Loading datasets...")
    
    # Load GA4 Sessions data (city level)
    ga4_sessions = pd.read_csv(os.path.join(RAW_DATA_PATH, 'ga4_sessions.csv'))
    print(f"GA4 Sessions shape: {ga4_sessions.shape}")
    
    # Load Meta Geo Spend data (DMA level)
    meta_geo_spend = pd.read_csv(os.path.join(RAW_DATA_PATH, 'meta_geo_spend.csv'))
    print(f"Meta Geo Spend shape: {meta_geo_spend.shape}")
    
    # Load TikTok Geo Spend data (state level)
    tiktok_geo_spend = pd.read_csv(os.path.join(RAW_DATA_PATH, 'tiktok_geo_spend.csv'))
    print(f"TikTok Geo Spend shape: {tiktok_geo_spend.shape}")
    
    # Load Google Ads Geo Spend data (city level)
    gads_geo_spend = pd.read_csv(os.path.join(RAW_DATA_PATH, 'gads_geo_spend.csv'))
    print(f"Google Ads Geo Spend shape: {gads_geo_spend.shape}")
    
    return ga4_sessions, meta_geo_spend, tiktok_geo_spend, gads_geo_spend


def clean_datasets(ga4_sessions, meta_geo_spend, tiktok_geo_spend, gads_geo_spend):
    """Clean and standardize datasets."""
    print("\nCleaning datasets...")
    
    # Initialize standardizers
    date_standardizer = DateStandardizer()
    geo_standardizer = GeoStandardizer()
    cost_standardizer = CostStandardizer()
    
    # Initialize dataset cleaner
    dataset_cleaner = DatasetCleaner(
        standardizers={
            'date': date_standardizer,
            'geo': geo_standardizer,
            'cost': cost_standardizer
        }
    )
    
    # Clean GA4 sessions data at city level
    ga4_clean = dataset_cleaner.clean_ga4_sessions(ga4_sessions, geo_level='city')
    print(f"Cleaned GA4 Sessions shape: {ga4_clean.shape}")
    
    # Clean Meta spend data at DMA level
    meta_clean = dataset_cleaner.clean_meta_spend(meta_geo_spend)
    print(f"Cleaned Meta Spend shape: {meta_clean.shape}")
    
    # Clean TikTok spend data at state level
    tiktok_clean = dataset_cleaner.clean_tiktok_spend(tiktok_geo_spend)
    print(f"Cleaned TikTok Spend shape: {tiktok_clean.shape}")
    
    # Clean Google Ads spend data at city level
    gads_clean = dataset_cleaner.clean_gads_spend(gads_geo_spend, geo_level='city')
    print(f"Cleaned Google Ads Spend shape: {gads_clean.shape}")
    
    return ga4_clean, meta_clean, tiktok_clean, gads_clean


def enrich_with_geo_reference(ga4_clean, meta_clean, tiktok_clean, gads_clean):
    """Enrich datasets with geographic reference data."""
    print("\nEnriching datasets with geographic reference data...")
    
    # Initialize GeoJoiner with reference data path
    geo_joiner = GeoJoiner(reference_data_path=REFERENCE_DATA_PATH)
    
    # Enrich GA4 city-level data with DMA information
    ga4_enriched = geo_joiner.enrich_city_data(
        ga4_clean, 
        city_col='City', 
        state_col='state' if 'state' in ga4_clean.columns else None
    )
    print(f"Enriched GA4 Sessions shape: {ga4_enriched.shape}")
    
    # Enrich Google Ads city-level data with DMA information
    gads_enriched = geo_joiner.enrich_city_data(
        gads_clean, 
        city_col='City', 
        state_col='state' if 'state' in gads_clean.columns else None
    )
    print(f"Enriched Google Ads Spend shape: {gads_enriched.shape}")
    
    # Enrich Meta DMA-level data with state information
    meta_enriched = meta_clean
    if 'dma_name' in meta_clean.columns:
        meta_enriched = geo_joiner.enrich_dma_data(meta_clean, dma_col='dma_name')
    print(f"Enriched Meta Spend shape: {meta_enriched.shape}")
    
    return ga4_enriched, meta_enriched, tiktok_clean, gads_enriched


def distribute_dma_and_state_values(meta_enriched, tiktok_clean, geo_joiner):
    """Distribute DMA and state-level values to city level."""
    print("\nDistributing aggregated values to city level...")
    
    # Distribute Meta spend (DMA level) to constituent cities
    meta_city = None
    if 'dma_name' in meta_enriched.columns and 'Cost' in meta_enriched.columns:
        meta_city = geo_joiner.distribute_dma_values_to_cities(
            meta_enriched,
            dma_col='dma_name',
            value_col='Cost',
            distribution_method='equal'
        )
        print(f"Meta spend distributed to cities shape: {meta_city.shape}")
    
    # For TikTok data (state level), we need to distribute to DMAs first, then to cities
    # This is a simplistic approach; in practice, you might want more sophisticated distribution
    tiktok_dma = None
    # This would require additional logic to map states to DMAs and then to cities
    
    return meta_city, tiktok_dma


def join_datasets_by_geo_hierarchy(ga4_enriched, gads_enriched, meta_enriched, tiktok_clean):
    """Join datasets with different geographic hierarchies."""
    print("\nJoining datasets across geographic levels...")
    
    # Initialize the GeoHierarchyJoiner
    hierarchy_joiner = GeoHierarchyJoiner(reference_data_path=REFERENCE_DATA_PATH)
    
    # Prepare datasets with their geographic column and level
    datasets = [
        (ga4_enriched, 'City', 'city'),  # City-level GA4 data
        (gads_enriched, 'City', 'city'),  # City-level Google Ads data
        (meta_enriched, 'DMA Region', 'dma'),  # DMA-level Meta data
        (tiktok_clean, 'Region', 'state')  # State-level TikTok data
    ]
    
    # Join datasets
    joined_df = hierarchy_joiner.join_datasets(
        datasets=datasets,
        date_col='Date',
        value_cols=['Sessions', 'gads_cost', 'meta_cost', 'tiktok_cost']
    )
    
    print(f"Joined dataset shape: {joined_df.shape}")
    
    return joined_df


def create_final_dataset(joined_df):
    """Create the final cost-response dataset."""
    print("\nCreating final cost-response dataset...")
    
    # Rename columns for consistency
    final_df = joined_df.copy()
    
    # Add total cost column
    cost_cols = [col for col in final_df.columns if 'cost' in col.lower()]
    final_df['total_cost'] = final_df[cost_cols].sum(axis=1, skipna=True)
    
    # Ensure date is in datetime format
    if 'Date' in final_df.columns:
        final_df['Date'] = pd.to_datetime(final_df['Date'])
    
    # Sort by date and geography
    sort_cols = ['Date']
    if 'City' in final_df.columns:
        sort_cols.append('City')
    final_df = final_df.sort_values(sort_cols).reset_index(drop=True)
    
    # Save the final dataset
    output_file = os.path.join(OUTPUT_PATH, 'cost_response_data.csv')
    final_df.to_csv(output_file, index=False)
    print(f"Final dataset saved to {output_file}")
    
    return final_df


def main():
    """Main function to run the pipeline."""
    # Load raw datasets
    ga4_sessions, meta_geo_spend, tiktok_geo_spend, gads_geo_spend = load_datasets()
    
    # Clean datasets
    ga4_clean, meta_clean, tiktok_clean, gads_clean = clean_datasets(
        ga4_sessions, meta_geo_spend, tiktok_geo_spend, gads_geo_spend
    )
    
    # Initialize GeoJoiner
    geo_joiner = GeoJoiner(reference_data_path=REFERENCE_DATA_PATH)
    
    # Enrich datasets with geographic reference data
    ga4_enriched, meta_enriched, tiktok_clean, gads_enriched = enrich_with_geo_reference(
        ga4_clean, meta_clean, tiktok_clean, gads_clean
    )
    
    # Distribute DMA and state-level values to city level if needed
    meta_city, tiktok_dma = distribute_dma_and_state_values(
        meta_enriched, tiktok_clean, geo_joiner
    )
    
    # Join datasets across geographic hierarchies
    joined_df = join_datasets_by_geo_hierarchy(
        ga4_enriched, gads_enriched, meta_enriched, tiktok_clean
    )
    
    # Create final cost-response dataset
    final_df = create_final_dataset(joined_df)
    
    print("\nDataset creation complete! Here's a sample of the final dataset:")
    print(final_df.head())
    
    # Basic statistics
    print("\nBasic statistics of the final dataset:")
    print(final_df.describe())
    
    # Missing values check
    print("\nMissing values in the final dataset:")
    print(final_df.isnull().sum())
    
    print("\nProcess completed successfully!")


if __name__ == '__main__':
    main()
