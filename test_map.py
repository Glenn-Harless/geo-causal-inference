#!/usr/bin/env python
"""
Test script for the geo map visualization.
"""

import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Ensure output directory exists
output_dir = os.path.join('/app/output', 'map_test')
os.makedirs(output_dir, exist_ok=True)

# Add the project to the path
sys.path.insert(0, '/app')

# Import the visualization module
from src.geo_causal_inference.visualization import plot_geo_map

# Path to the geo spine data
spine_path = '/app/data/reference/geo_spine.csv'

# Path to geo assignments
assignments_path = '/app/output/client1_weekly/design/data/geo_assignments.csv'

# Check if file exists
if not os.path.exists(assignments_path):
    print(f"File not found: {assignments_path}")
    sys.exit(1)

# Read geo assignments
geo_assignments = pd.read_csv(assignments_path)

# Save the map visualization
plot_geo_map(
    geo_assignments=geo_assignments,
    spine_path=spine_path,
    map_type='dma',
    debug=True,  # Add debug mode to get more verbose output
    output_path=os.path.join(output_dir, 'geo_map_test.png')
)
