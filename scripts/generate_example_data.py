#!/usr/bin/env python3
"""Generate synthetic example data for testing the Geo Causal Inference framework."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

# Top DMAs by population
TOP_DMAS = [
    "NEW YORK", "LOS ANGELES", "CHICAGO", "PHILADELPHIA", "DALLAS-FT. WORTH",
    "SAN FRANCISCO-OAK-SAN JOSE", "ATLANTA", "HOUSTON", "WASHINGTON DC", "BOSTON",
    "DETROIT", "PHOENIX", "SEATTLE-TACOMA", "MINNEAPOLIS-ST. PAUL", "MIAMI-FT. LAUDERDALE",
    "DENVER", "ORLANDO-DAYTONA BEACH", "CLEVELAND-AKRON", "SACRAMENTO-STOCKTON-MODESTO",
    "ST. LOUIS"
]

def generate_example_data(
    start_date="2023-10-01",
    end_date="2023-12-31",
    n_geos=20,
    base_spend_range=(500, 2000),
    base_response_range=(1000, 5000),
    output_dir="data/example"
):
    """Generate synthetic cost-response data for example purposes."""
    
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Select DMAs
    selected_dmas = TOP_DMAS[:n_geos]
    
    # Generate data
    data = []
    for date in dates:
        for dma in selected_dmas:
            # Add some seasonality and random variation
            day_of_week = date.dayofweek
            week_of_year = date.isocalendar()[1]
            
            # Weekend effect (lower spend/response on weekends)
            weekend_factor = 0.7 if day_of_week >= 5 else 1.0
            
            # Seasonal trend (increasing towards end of year)
            seasonal_factor = 1 + (week_of_year - 40) * 0.02
            
            # DMA-specific baseline (larger markets have higher spend)
            dma_factor = 1 + (TOP_DMAS.index(dma) / len(TOP_DMAS)) * -0.5
            
            # Generate cost with random variation
            base_cost = np.random.uniform(*base_spend_range)
            cost = base_cost * weekend_factor * seasonal_factor * dma_factor
            cost = max(0, cost + np.random.normal(0, cost * 0.1))  # Add noise
            
            # Generate response (correlated with cost but with noise)
            cost_efficiency = np.random.uniform(2.0, 3.5)  # ROI factor
            response = cost * cost_efficiency
            response = max(0, response + np.random.normal(0, response * 0.15))  # Add noise
            
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'geo': dma,
                'cost': round(cost, 2),
                'response': int(response)
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    output_path = os.path.join(output_dir, 'example_cost_response_data.csv')
    df.to_csv(output_path, index=False)
    
    print(f"Generated example data with {len(df)} rows")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Number of DMAs: {n_geos}")
    print(f"Saved to: {output_path}")
    
    # Show sample
    print("\nSample of generated data:")
    print(df.head(10))
    
    return df

if __name__ == "__main__":
    # Generate example data
    df = generate_example_data()
    
    # Also generate a weekly aggregated version
    df['date'] = pd.to_datetime(df['date'])
    df['week'] = df['date'].dt.to_period('W').dt.start_time
    
    weekly_df = df.groupby(['week', 'geo']).agg({
        'cost': 'sum',
        'response': 'sum'
    }).reset_index()
    weekly_df.rename(columns={'week': 'date'}, inplace=True)
    weekly_df['date'] = weekly_df['date'].dt.strftime('%Y-%m-%d')
    
    weekly_path = os.path.join('data/example', 'example_cost_response_data_weekly.csv')
    weekly_df.to_csv(weekly_path, index=False)
    print(f"\nAlso generated weekly data: {weekly_path}")