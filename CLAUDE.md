# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Geo Causal Inference is a framework for analyzing marketing effectiveness through geographic experiments. It integrates marketing data from multiple platforms (Meta, TikTok, Google Ads), standardizes geographic hierarchies, and uses Google's Trimmed Match methodology to design and analyze geo-based experiments.

## Key Commands

### Running Experiments
```bash
# Design experiment with daily data
python scripts/run_experiment.py --input data/processed/cost_response_data.csv --client client_name --frequency daily

# Design experiment with weekly data  
python scripts/run_experiment.py --input data/processed/cost_response_data_weekly.csv --client client_name --frequency weekly

# Run post-analysis
python scripts/run_postanalysis.py --data output/client_name/postanalysis/experiment_data_for_postanalysis.csv \
  --test-start 2024-01-01 --test-end 2024-01-31 \
  --design-start 2023-10-01 --design-end 2023-12-31
```

### Development Environment
```bash
# Start Docker environment
docker-compose up

# Run commands in container
docker-compose exec geo_causal_inference [command]

# Access Jupyter (http://localhost:9999)
docker-compose up
```

## Architecture Overview

The codebase follows a three-layer architecture:

1. **Data Pipeline Layer** (`src/data_pipeline/`)
   - `GeoReferenceBuilder`: Creates mapping between zip→city→DMA→state hierarchies
   - `DataStandardizer`: Normalizes dates, costs, and geo identifiers across platforms
   - `GeoJoiner/DataJoiner`: Joins datasets at appropriate geographic levels

2. **Experiment Framework** (`src/geo_causal_inference/`)
   - `ExperimentDesigner`: Uses Trimmed Match to create treatment/control groups
   - `ExperimentAnalyzer`: Calculates lift, iROAS, and confidence intervals
   - `Visualization`: Creates maps, time series, and trade-off dashboards

3. **Execution Layer** (`scripts/`)
   - `run_experiment.py`: End-to-end experiment design pipeline
   - `run_postanalysis.py`: Post-experiment analysis and reporting

## Data Standards

### Required Input Format
- **Columns**: `date`, `geo`, `cost`, `response`
- **Date Format**: ISO format (YYYY-MM-DD)
- **Geo Level**: DMA preferred, but supports zip/city/state
- **Cost**: Total spend in dollars
- **Response**: Business metric (e.g., sessions, orders)

### Geographic Hierarchy
```
zip_code → city → dma → state
```
Always read zip codes as strings to preserve leading zeros:
```python
pd.read_csv(file, dtype={'zip_code': str})
```

## Code Style Guidelines

- **Imports**: Standard library → third-party → local modules
- **Type Hints**: Required for function parameters and returns
- **Docstrings**: Google-style with Args/Returns sections
- **Naming**: snake_case functions, PascalCase classes
- **Error Handling**: Specific exceptions with context
- **File Organization**:
  - `src/`: Source code modules
  - `data/`: Input and processed datasets
  - `notebooks/`: Analysis and exploration
  - `scripts/`: Executable pipelines
  - `output/`: Experiment results by client

## Testing and Validation

- Data validation occurs automatically during pipeline execution
- Key validation checks:
  - Date format consistency
  - Geographic identifier completeness
  - Positive cost values
  - Response metric availability
- Use `validation.py` utilities for custom checks

## Output Structure

Experiments create structured outputs:
```
output/
└── {client_name}/
    ├── design/
    │   ├── data/          # Design results, assignments
    │   └── plots/         # Visualizations
    └── postanalysis/
        └── experiment_data_for_postanalysis.csv
```

## Common Workflows

### Adding New Data Source
1. Create standardization function in `data_standardizer.py`
2. Add to `DataJoiner` pipeline
3. Update validation rules if needed

### Running Full Analysis
1. Prepare cost-response dataset
2. Run experiment design
3. Execute experiment in production
4. Run post-analysis after test period

### Debugging Geographic Joins
- Check `data/reference/geo_spine.csv` for mapping coverage
- Use `notebooks/geo_overlap.png` to visualize coverage
- Validate with `GeoReferenceBuilder.validate_spine()`