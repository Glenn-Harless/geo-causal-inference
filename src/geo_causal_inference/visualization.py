"""
Visualization functionality for Trimmed Match marketing experiments.

This module provides plotting and visualization functions for
experiment design and analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib.colors import ListedColormap


def plot_designs_comparison(results, metric='rmse_cost_adjusted'):
    """Plot comparison of different experiment designs.
    
    Args:
        results: DataFrame with design results
        metric: Metric to use for comparison
        
    Returns:
        matplotlib.figure.Figure: Plot of design comparison
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(results['budget'], results[metric], 'o-')
    ax.set_xlabel('Budget')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title('Design Comparison by Budget')
    ax.grid(True, alpha=0.3)
    
    # Annotate the best design
    best_idx = results[metric].idxmin()
    best_design = results.loc[best_idx]
    
    # Handle the case where best_design returns a DataFrame with multiple rows
    if isinstance(best_design, pd.DataFrame):
        # Use the first row
        best_design = best_design.iloc[0]
    
    # Extract scalar values for formatting
    budget_value = best_design['budget']
    metric_value = best_design[metric]
    
    ax.annotate(
        f'Best: {budget_value:.0f}',
        xy=(budget_value, metric_value),
        xytext=(10, -20),
        textcoords='offset points',
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2')
    )
    
    return fig


def plot_coverage_test_results(coverage_test_result):
    """Plot coverage test results.
    
    Args:
        coverage_test_result: DataFrame with coverage test results
        
    Returns:
        matplotlib.figure.Figure: Plot of coverage test results
    """
    if len(coverage_test_result) <= 1:
        return None
        
    fig = plt.figure(figsize=(15, 7.5))
    ax = fig.add_subplot(1, 1, 1)
    
    ax.plot(
        coverage_test_result['pair_index'],
        coverage_test_result['bias_cost_normalized'],
        'o-', 
        label='Cost'
    )
    
    ax.plot(
        coverage_test_result['pair_index'],
        coverage_test_result['bias_response_normalized'],
        'o-', 
        label='Response'
    )
    
    ax.axhspan(-1, 1, alpha=0.2, color='green')
    ax.legend(loc='best')
    ax.set_xlabel('Pair index (Number of pairs included)')
    ax.set_ylabel('Normalized Bias')
    ax.set_title('Coverage Test Results')
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_geo_time_series(geo_data, response_col='response', date_col='date', 
                          geo_col='geo', treatment_geos=None, control_geos=None,
                          eval_start_date=None, eval_end_date=None):
    """Plot geo-level time series.
    
    Args:
        geo_data: DataFrame with geo-level time series data
        response_col: Column name for response variable
        date_col: Column name for date variable
        geo_col: Column name for geo variable
        treatment_geos: List of treatment geos
        control_geos: List of control geos
        eval_start_date: Start date of evaluation period
        eval_end_date: End date of evaluation period
        
    Returns:
        matplotlib.figure.Figure: Plot of geo-level time series
    """
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Prepare data
    geo_data = geo_data.copy()
    geo_data[date_col] = pd.to_datetime(geo_data[date_col])
    
    # Aggregate data by date and group
    data_agg = []
    
    if treatment_geos:
        treatment_data = geo_data[geo_data[geo_col].isin(treatment_geos)].groupby(date_col)[response_col].sum().reset_index()
        treatment_data['group'] = 'Treatment'
        data_agg.append(treatment_data)
    
    if control_geos:
        control_data = geo_data[geo_data[geo_col].isin(control_geos)].groupby(date_col)[response_col].sum().reset_index()
        control_data['group'] = 'Control'
        data_agg.append(control_data)
    
    if data_agg:
        data_agg = pd.concat(data_agg)
        
        # Plot groups
        for group, group_data in data_agg.groupby('group'):
            ax.plot(group_data[date_col], group_data[response_col], 
                   label=group, marker='o' if len(group_data) < 30 else None,
                   linewidth=2)
    else:
        # If no groups specified, plot all geos
        for geo, geo_data in geo_data.groupby(geo_col):
            ax.plot(geo_data[date_col], geo_data[response_col], 
                   label=f'Geo {geo}', alpha=0.7)
    
    # Highlight evaluation period if specified
    if eval_start_date and eval_end_date:
        ax.axvspan(eval_start_date, eval_end_date, alpha=0.2, color='green', label='Evaluation Period')
    
    ax.set_xlabel('Date')
    ax.set_ylabel(response_col.replace('_', ' ').title())
    ax.set_title('Geo-Level Time Series')
    ax.grid(True, alpha=0.3)
    
    # Add legend with proper sizing
    if len(data_agg.groupby('group')) > 10:
        ax.legend(loc='best', fontsize='small', ncol=2)
    else:
        ax.legend(loc='best')
    
    # Format x-axis with appropriate date ticks
    fig.autofmt_xdate()
    
    return fig


def plot_design_summary(design_results, min_detectable_iroas, min_detectable_lift):
    """Plot summary of experiment design.
    
    Args:
        design_results: Dict with design results
        min_detectable_iroas: Minimum detectable iROAS
        min_detectable_lift: Minimum detectable lift
        
    Returns:
        matplotlib.figure.Figure: Plot of design summary
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Plot 1: RMSE vs Budget
    results = design_results["results"]
    ax1.plot(results['budget'], results['rmse'], 'o-', label='RMSE')
    ax1.plot(results['budget'], results['rmse_cost_adjusted'], 'o-', label='RMSE Cost Adjusted')
    ax1.set_xlabel('Budget')
    ax1.set_ylabel('RMSE')
    ax1.set_title('RMSE vs Budget')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Minimum Detectable Metrics
    best_design = results.loc[results['rmse_cost_adjusted'].idxmin()]
    metrics = ['Min. Detectable iROAS', 'Min. Detectable Lift (%)']
    values = [min_detectable_iroas, min_detectable_lift]
    
    ax2.bar(metrics, values, color=['blue', 'orange'])
    ax2.set_title('Minimum Detectable Metrics')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, v in enumerate(values):
        ax2.text(i, v + 0.1, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    return fig


def plot_geo_map(geo_assignments, spine_path, map_type='dma', debug=False, output_path=None):
    """
    Plot a choropleth map of the US by state showing treatment and control areas.
    
    Parameters
    ----------
    geo_assignments: pandas.DataFrame or list of dict
        DataFrame or list of dictionaries containing geo codes and their assignments.
        Must contain columns/keys 'geo' and 'assignment'.
    spine_path: str
        Path to the geo spine file.
    map_type: str, optional
        Type of map to plot. Currently supports 'dma'.
        Default is 'dma'.
    debug: bool, optional
        Whether to print debug information.
        Default is False.
    output_path: str, optional
        Path to save the map. If None, the figure is returned.
        Default is None.
    
    Returns
    -------
    str
        Path to the saved figure if output_path is provided, otherwise None.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import geopandas as gpd
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches
    
    # Read the geo spine
    geo_spine = pd.read_csv(spine_path)
    
    # Process geo_assignments if it's a list
    if isinstance(geo_assignments, list):
        if debug:
            print(f"Loaded {len(geo_assignments)} geo assignments")
            print(f"Sample: {geo_assignments[:5]}")
        
        # Convert to dataframe
        geo_assignments = pd.DataFrame(geo_assignments)
        
        if debug:
            print(f"Loaded {len(geo_assignments)} geo assignments")
            print(f"Sample: {geo_assignments.head()}")
    
    # Ensure the geo columns in geo_assignments and geo_spine match types
    if debug:
        print(f"Number of unique geos in assignments: {len(geo_assignments['geo'].unique())}")
        print(f"Sample geo values: {geo_assignments['geo'].unique()[:5]}")
        print(f"Number of unique DMAs in spine: {len(geo_spine['dma_code'].unique())}")
        print(f"Sample DMA values: {geo_spine['dma_code'].unique()[:5]}")
        print(f"DMA value types in assignments: {geo_assignments['geo'].dtype}")
        print(f"DMA value types in spine: {geo_spine['dma_code'].dtype}")
    
    # Check for states in the spine
    states = geo_spine['state'].unique()
    if debug:
        states_without_nan = [s for s in states if pd.notna(s)]
        print(f"States in spine data: {sorted(states_without_nan)}")
    
    # Create the figure and axis for plotting
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Filter the spine to DMAs we're using
    if map_type == 'dma':
        # Identify treatment and control DMAs
        treatment_dmas = geo_assignments[geo_assignments['assignment'] == 'treatment']['geo'].unique()
        control_dmas = geo_assignments[geo_assignments['assignment'] == 'control']['geo'].unique()
        
        if debug:
            print(f"Treatment DMAs: {treatment_dmas}")
            print(f"Control DMAs: {control_dmas}")
        
        # Create a mapping of DMAs to states
        dma_to_states = {}
        for dma in sorted(set(treatment_dmas) | set(control_dmas)):
            states = geo_spine[geo_spine['dma_code'] == dma]['state'].unique()
            states_filtered = [s for s in states if pd.notna(s)]
            dma_to_states[dma] = sorted(states_filtered)
        
        if debug:
            print("DMA to States mapping (sample first 5):")
            for i, (dma, states) in enumerate(list(dma_to_states.items())[:5]):
                print(f"  DMA {dma}: {states}")
        
        # Identify the treatment/control status for each state
        state_status = {}
        for state in sorted([s for s in geo_spine['state'].unique() if pd.notna(s)]):
            # Get all DMAs in this state
            dmas_in_state = geo_spine[geo_spine['state'] == state]['dma_code'].unique()
            
            # Check if all DMAs in the state are either treatment or control
            treatment_count = sum(1 for dma in dmas_in_state if dma in treatment_dmas)
            control_count = sum(1 for dma in dmas_in_state if dma in control_dmas)
            
            if debug:
                print(f"{state} ({state}): ", end="")
            
            if treatment_count > 0 and control_count == 0:
                # All DMAs in state are treatment
                state_status[state] = 'treatment'
                if debug:
                    print("treatment")
            elif control_count > 0 and treatment_count == 0:
                # All DMAs in state are control
                state_status[state] = 'control'
                if debug:
                    print("control")
            else:
                if treatment_count > 0 or control_count > 0:
                    # State has mixed treatment/control status
                    state_status[state] = 'mixed'
                    if debug:
                        print("mixed")
                else:
                    # State has no DMAs in the study
                    state_status[state] = 'unmapped'
                    if debug:
                        print("unmapped")
    
    # Load US States geometry from a GeoJSON URL
    states_geojson_url = "https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/us-states.json"
    us_states_gdf = gpd.read_file(states_geojson_url)

    # --- DEBUG CHECK: Verify loaded data ---
    if 'name' not in us_states_gdf.columns or not us_states_gdf['name'].iloc[0] == 'Alabama':
        print("\nERROR: Failed to load correct US States GeoJSON.")
        print("Loaded columns:", us_states_gdf.columns)
        print("Sample data:", us_states_gdf.head())
        # Handle error appropriately, maybe raise an exception or plot an error message
        ax.set_title("Error Loading State Map Data")
        ax.text(0.5, 0.5, "Failed to load state geometry data from URL.", ha='center', va='center')
        if output_path:
             plt.savefig(output_path)
             plt.close(fig)
        return output_path or fig
    # --- END DEBUG CHECK ---
    
    if debug:
        print("\nDEBUG: Successfully loaded US States GeoDataFrame from URL:")
        print(us_states_gdf[['id', 'name', 'geometry']].head())
        print(f"Columns: {us_states_gdf.columns}")

    # Filter out Alaska and Hawaii for a continental US view
    us_states_gdf = us_states_gdf[~us_states_gdf['name'].isin(['Alaska', 'Hawaii'])]
    
    # Use full state names from spine for merging
    state_name_map = geo_spine[['state', 'state_name']].drop_duplicates().set_index('state')['state_name']
    state_status_df = pd.DataFrame(list(state_status.items()), columns=['state_abbr', 'status'])
    state_status_df['STATE_NAME'] = state_status_df['state_abbr'].map(state_name_map)

    if debug:
        print("\nState Status DF with Names:")
        print(state_status_df.head())

    # Merge the status data with the GeoDataFrame
    merged_gdf = us_states_gdf.merge(state_status_df, left_on='name', right_on='STATE_NAME', how='left')

    # Fill states not in our assignment data with 'unmapped'
    merged_gdf['status'] = merged_gdf['status'].fillna('unmapped')

    if debug:
        print("\nMerged GeoDataFrame:")
        print(merged_gdf[['name', 'STATE_NAME', 'status', 'geometry']].head())
        print(f"Number of states in merged GDF: {len(merged_gdf)}")
        print(f"Status counts:\n{merged_gdf['status'].value_counts()}")

    # Define desired colors
    status_colors = {
        'treatment': '#1f77b4',  # Blue
        'control': '#ff7f0e',     # Orange
        'mixed': '#2ca02c',       # Green
        'unmapped': '#d3d3d3'     # Light gray
    }

    # Create cmap based on the actual sorted unique values in the status column
    sorted_statuses = sorted(merged_gdf['status'].unique())
    ordered_colors = [status_colors[status] for status in sorted_statuses]
    cmap = ListedColormap(ordered_colors)

    # Plot the choropleth map
    merged_gdf.plot(column='status', 
                    categorical=True, 
                    legend=False, # We create a custom legend below
                    cmap=cmap, 
                    linewidth=0.8, 
                    ax=ax, 
                    edgecolor='0.8',
                    missing_kwds={
                        "color": status_colors['unmapped'], # Use explicit color for missing
                        "edgecolor": "red",
                        "hatch": "///",
                        "label": "Missing values",
                    })

    # Set title and remove axis
    ax.set_title("US State Treatment/Control Assignment")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off() # Turn off the axis frame for a cleaner map

    # Create a custom legend using the defined colors
    legend_patches = [
        mpatches.Patch(color=status_colors['treatment'], label='Treatment'),
        mpatches.Patch(color=status_colors['control'], label='Control'),
        mpatches.Patch(color=status_colors['mixed'], label='Mixed'),
        mpatches.Patch(color=status_colors['unmapped'], label='Unmapped')
    ]
    ax.legend(handles=legend_patches, loc='lower right', title="Assignment Status")
    
    # Save or return the figure
    if output_path:
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Map saved to {output_path}")
        return output_path
    else:
        return fig
