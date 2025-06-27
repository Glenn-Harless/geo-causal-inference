# Example Data Structure

This directory shows the expected data format for the Geo Causal Inference framework.

## Required Input Format

Your input CSV should have these columns:
- `date`: Date in YYYY-MM-DD format
- `geo`: Geographic identifier (DMA name, city, or zip code)
- `cost`: Marketing spend in dollars
- `response`: Business metric (e.g., sessions, conversions)

## Example:

```csv
date,geo,cost,response
2024-01-01,NEW YORK,1000.50,2500
2024-01-01,LOS ANGELES,850.25,1800
2024-01-02,NEW YORK,1100.00,2700
```

## Geographic Levels Supported:
- DMA (Designated Market Area) - Preferred
- City, State (e.g., "HOUSTON, TX")
- ZIP Code (as string to preserve leading zeros)

## Data Sources:
The framework can integrate data from:
- Google Analytics 4 (sessions)
- Google Ads (spend by geography)
- Meta/Facebook Ads (spend by geography)
- TikTok Ads (spend by geography)

See `scripts/generate_example_data.py` for creating synthetic test data.