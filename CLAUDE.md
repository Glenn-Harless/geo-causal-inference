# Geo Causal Inference Codebase Guidelines

## Commands
- Run experiments: `python scripts/run_experiment.py --input [CSV_PATH] --client [NAME] --frequency [daily|weekly]`
- Run post-analysis: `python scripts/run_postanalysis.py`
- Run Jupyter: `jupyter notebook` or `jupyter lab`
- Install dependencies: `pip install -r requirements.txt`

## Code Style
- **Imports**: Standard library first, then third-party, then local modules
- **Typing**: Use type hints for function parameters and return values
- **Docstrings**: Use Google-style docstrings with args/returns
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Error handling**: Use try/except for specific exceptions
- **File organization**: 
  - `src/` for source code
  - `data/` for datasets
  - `notebooks/` for analysis
  - `scripts/` for runnable scripts

## Data Pipeline
- Standardize dates to ISO format YYYY-MM-DD
- Geo data uses DMA (Designated Market Area) as standard reference
- Always validate data before experiment design.
