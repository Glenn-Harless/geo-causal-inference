# Geo Causal Inference

A framework for geographic data integration and causal inference analysis for marketing datasets.

## Overview

This project provides tools for handling the unique challenges of geographic data in marketing analytics and causal inference. It enables:

1. Integration of marketing datasets with different geographic granularities (zip, city, DMA, state)

2. Standardization and cleaning of geographic data from various marketing platforms

3. Creation of unified cost-response datasets for causal analysis

4. Visualization and analysis of geographic marketing data

## Key Components

### Geographic Data Integration

- **GeoReferenceBuilder**: Creates a comprehensive geographic spine table mapping between zip codes, cities, DMAs, and states

- **GeoJoiner**: Utilities for joining datasets at different geographic levels (city, DMA, state)

- **GeoHierarchyJoiner**: Higher-level abstraction for joining multiple datasets with different geographic granularities

### Data Pipeline

- **Data Standardization**: Functions for cleaning dates, geographic data, and costs

- **Data Joining**: Tools to integrate GA4 sessions (response metric) with Meta, TikTok, and Google Ads spend data

- **Cost-Response Dataset Creation**: Pipeline to create unified datasets with Date, Geo, Cost, and Response variables

### Experimental Design and Analysis

- **ExperimentDesigner**: Creates geographic treatment/control designs for marketing experiments

- **ExperimentAnalyzer**: Analyzes results from geographic experiments to calculate lift and iROAS

### Reference Data

- `geo_spine.csv`: Comprehensive mapping between geographic levels

- `city_dma_mapping.csv`: Mapping between cities and DMAs

- `dma_state_mapping.csv`: Mapping between DMAs and states

### Scripts

- **run_experiment.py**: Runs a Trimmed Match experiment design process
  - Takes input data, client name, and frequency (daily/weekly) as parameters
  - Creates optimal geographic treatment/control assignments
  - Generates experiment design summaries and visualizations
  - Exports data for post-analysis

- **run_postanalysis.py**: Analyzes results of a Trimmed Match marketing experiment
  - Calculates key metrics like incremental lift, iROAS, and confidence intervals
  - Generates time series comparisons and correlation visualizations
  - Supports analysis with and without cooldown periods
  - Exports detailed results for reporting

## Example Notebooks

- `notebooks/cost_response_dataset_join.ipynb`: Demonstrates joining marketing data from different platforms

- `notebooks/geo_cost_response_analysis.ipynb`: Analysis of geographic cost-response relationships

- `notebooks/weekly_daily_geo_assignment_comp.ipynb`: Comparison of weekly vs daily geographic assignments

## Setup

This project uses Docker for containerization. Make sure you have Docker and Docker Compose installed.

### Start the environment

```bash
docker-compose up
```

This will start a Jupyter Notebook server accessible at [http://localhost:9999](http://localhost:9999) (token will be displayed in the console).

### Run commands inside the container

```bash
docker-compose exec geo_causal_inference <command>
```

For example, to run a script:

```bash
docker-compose exec geo_causal_inference python scripts/run_experiment.py --input <path_to_data> --client <client_name>
```

## Important Notes

When working with geographic data like zip codes, always read them as strings (`dtype={'zip_code': str}`) rather than integers to preserve leading zeros and ensure proper join operations.

## Dependencies

Main dependencies (automatically installed in Docker container):

- pandas: For data manipulation and analysis

- jupyter: For interactive notebooks

- matplotlib/seaborn: For data visualization

- trimmed_match: Google's package for geo-based causal inference
