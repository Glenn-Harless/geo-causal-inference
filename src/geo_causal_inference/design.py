"""
Core experiment design functionality for Trimmed Match marketing experiments.

This module handles the design of experiments using the Trimmed Match methodology.
"""

import pandas as pd
import numpy as np

from trimmed_match.design.common_classes import GeoXType, GeoAssignment
from geo_causal_inference.utils import TimeWindow
from trimmed_match.design.trimmed_match_design import TrimmedMatchGeoXDesign
from trimmed_match.design.util import CalculateMinDetectableIroas


def plot_cost_precision_tradeoff(design_results):
    """
    Create a plot showing the trade-off between experiment cost and precision.
    
    Args:
        design_results: DataFrame with design results, as returned by calculate_optimal_budget
    
    Returns:
        matplotlib.figure.Figure: The generated plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get unique budgets
    budgets = design_results['budget'].unique()
    
    # Use a more pleasing color palette
    palette = sns.color_palette("muted", n_colors=len(budgets))
    
    # Loop through each budget
    for i, budget in enumerate(budgets):
        budget_data = design_results[design_results['budget'] == budget].sort_values('num_pairs')
        
        # Plot the cost vs. rmse trade-off
        scatter = ax.scatter(
            budget_data['experiment_spend'], 
            budget_data['rmse'],
            s=budget_data['num_pairs']*2,  # Size represents number of pairs
            alpha=0.7,
            color=palette[i],
            label=f'Budget ${budget/1000:.1f}K'
        )
        
        # Connect points with a line
        ax.plot(
            budget_data['experiment_spend'],
            budget_data['rmse'],
            '-',
            alpha=0.5,
            color=palette[i]
        )
        
        # Annotate key points
        min_rmse_idx = budget_data['rmse'].idxmin()
        if min_rmse_idx in budget_data.index:
            min_point = budget_data.loc[min_rmse_idx]
            ax.annotate(
                f"Pairs: {min_point['num_pairs']:.0f}",
                (min_point['experiment_spend'], min_point['rmse']),
                xytext=(10, 0),
                textcoords='offset points',
                fontsize=8,
                arrowprops=dict(arrowstyle='->', color='gray')
            )
    
    # Set labels and title
    ax.set_xlabel('Experiment Cost ($)')
    ax.set_ylabel('Raw RMSE')
    ax.set_title('Cost vs. Precision Trade-off', fontsize=14)
    
    # Add legend and grid
    ax.legend(title='Budget Options')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Add a secondary x-axis showing the percentage of total budget
    if len(budgets) > 0:
        max_budget = max(budgets)
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        
        # Set tick positions based on percentages of max budget
        tick_positions = np.linspace(0, max_budget, 6)
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels([f'{100*x/max_budget:.0f}%' for x in tick_positions])
        ax2.set_xlabel('Percentage of Maximum Budget')
    
    # Add annotations explaining the trade-off
    bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8)
    ax.text(
        0.02, 0.02, 
        "Lower RMSE = Higher Precision\nLarger points = More geo pairs",
        transform=ax.transAxes,
        fontsize=10,
        bbox=bbox_props
    )
    
    fig.tight_layout()
    return fig


def plot_design_dashboard(design_results):
    """
    Create a comprehensive dashboard of design options.
    
    Args:
        design_results: DataFrame with design results, as returned by calculate_optimal_budget
    
    Returns:
        matplotlib.figure.Figure: The generated dashboard plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    
    # Create a figure with a grid of subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Define subplot grid
    gs = fig.add_gridspec(2, 3)
    
    # Get unique budgets and assign a more pleasing color palette
    budgets = design_results['budget'].unique()
    # Use a more harmonious color palette
    palette = sns.color_palette("muted", n_colors=len(budgets))
    
    # Plot 1: Cost vs RMSE
    ax1 = fig.add_subplot(gs[0, 0])
    for i, budget in enumerate(budgets):
        budget_data = design_results[design_results['budget'] == budget].sort_values('experiment_spend')
        ax1.plot(
            budget_data['experiment_spend'],
            budget_data['rmse'],
            'o-',
            alpha=0.7,
            color=palette[i],
            label=f'${budget/1000:.1f}K'
        )
    ax1.set_xlabel('Experiment Cost ($)')
    ax1.set_ylabel('RMSE')
    ax1.set_title('Cost vs Raw RMSE')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(title='Budget')
    
    # Plot 2: Number of Geo Pairs vs RMSE
    ax2 = fig.add_subplot(gs[0, 1])
    for i, budget in enumerate(budgets):
        # Sort by number of pairs to ensure the line connects points in order
        budget_data = design_results[design_results['budget'] == budget].sort_values('num_pairs')
        ax2.plot(
            budget_data['num_pairs'],
            budget_data['rmse'],
            'o-',
            alpha=0.7,
            color=palette[i],
            label=f'${budget/1000:.1f}K'
        )
    ax2.set_xlabel('Number of Geo Pairs')
    ax2.set_ylabel('RMSE')
    ax2.set_title('Geo Pairs vs RMSE')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(title='Budget')
    
    # Plot 3: Trim Rate vs RMSE
    ax3 = fig.add_subplot(gs[0, 2])
    for i, budget in enumerate(budgets):
        budget_data = design_results[design_results['budget'] == budget]
        sc = ax3.scatter(
            budget_data['trim_rate'] * 100,  # Convert to percentage
            budget_data['rmse'],
            alpha=0.7,
            s=50,
            c=budget_data['num_pairs'],
            cmap='viridis'
        )
    cbar = plt.colorbar(sc, ax=ax3)
    cbar.set_label('Number of Pairs')
    ax3.set_xlabel('Trim Rate (%)')
    ax3.set_ylabel('RMSE')
    ax3.set_title('Trim Rate vs RMSE')
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    # Plot 4: Elbow Curve - Number of Pairs vs RMSE Cost Adjusted
    ax4 = fig.add_subplot(gs[1, 0:2])
    for i, budget in enumerate(budgets):
        budget_data = design_results[design_results['budget'] == budget].sort_values('num_pairs')
        
        # Mark the "elbow point" using the point of maximum curvature
        x = budget_data['num_pairs'].values
        y = budget_data['rmse_cost_adjusted'].values
        
        ax4.plot(
            x, 
            y,
            'o-',
            alpha=0.7,
            color=palette[i],
            label=f'${budget/1000:.1f}K'
        )
        
        # Find the minimum point
        min_idx = budget_data['rmse_cost_adjusted'].idxmin()
        min_point = budget_data.loc[min_idx]
        ax4.scatter(
            min_point['num_pairs'],
            min_point['rmse_cost_adjusted'],
            s=100,
            edgecolor='black',
            color=palette[i],
            zorder=5,
            marker='*'
        )
        
    ax4.set_xlabel('Number of Geo Pairs')
    ax4.set_ylabel('RMSE Cost Adjusted')
    ax4.set_title('Elbow Curve: Finding Optimal Number of Pairs')
    ax4.grid(True, linestyle='--', alpha=0.5)
    ax4.legend(title='Budget')
    
    # Plot 5: Budget Comparison
    ax5 = fig.add_subplot(gs[1, 2])
    
    # Get the minimum RMSE cost adjusted for each budget
    min_rmse_by_budget = []
    for budget in budgets:
        budget_data = design_results[design_results['budget'] == budget]
        min_idx = budget_data['rmse_cost_adjusted'].idxmin()
        min_point = budget_data.loc[min_idx]
        min_rmse_by_budget.append({
            'budget': budget,
            'rmse_cost_adjusted': min_point['rmse_cost_adjusted'],
            'experiment_spend': min_point['experiment_spend'],
            'num_pairs': min_point['num_pairs']
        })
    
    df_min = pd.DataFrame(min_rmse_by_budget)
    
    # Bar chart with the new color palette
    bars = ax5.bar(
        range(len(df_min)),
        df_min['rmse_cost_adjusted'],
        color=palette[:len(df_min)]
    )
    
    # Add budget labels
    ax5.set_xticks(range(len(df_min)))
    ax5.set_xticklabels([f'${b/1000:.1f}K' for b in df_min['budget']])
    
    # Add value annotations
    for i, bar in enumerate(bars):
        ax5.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.001,
            f"Pairs: {df_min.iloc[i]['num_pairs']:.0f}\nCost: ${df_min.iloc[i]['experiment_spend']/1000:.1f}K",
            ha='center',
            va='bottom',
            fontsize=8
        )
    
    ax5.set_ylabel('RMSE Cost Adjusted')
    ax5.set_title('Optimal Design by Budget')
    ax5.grid(True, axis='y', linestyle='--', alpha=0.5)
    
    # Add overall title
    fig.suptitle('Experiment Design Trade-offs Dashboard', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig


def plot_bang_for_buck(design_results):
    """
    Create a "bang for the buck" visualization showing improvement in 
    precision per dollar spent.
    
    Args:
        design_results: DataFrame with design results, as returned by calculate_optimal_budget
    
    Returns:
        matplotlib.figure.Figure: The generated plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Get unique budgets
    budgets = design_results['budget'].unique()
    
    # Use a more harmonious color palette 
    palette = sns.color_palette("muted", n_colors=len(budgets))
    
    # Plot 1: RMSE reduction per dollar vs. budget
    for i, budget in enumerate(budgets):
        budget_data = design_results[design_results['budget'] == budget].sort_values('num_pairs', ascending=False)
        
        if len(budget_data) > 1:
            # Calculate RMSE reduction per dollar spent
            budget_data = budget_data.copy()
            # Get the worst (highest) RMSE value for this budget
            max_rmse = budget_data['rmse'].max()
            
            # Calculate improvement from the worst RMSE
            budget_data['rmse_improvement'] = max_rmse - budget_data['rmse']
            
            # Calculate bang for the buck (improvement per dollar)
            budget_data['bang_for_buck'] = budget_data['rmse_improvement'] / budget_data['experiment_spend']
            
            # Multiply by 10000 for readability
            budget_data['bang_for_buck'] *= 10000
            
            # Plot bang for buck vs. experiment cost
            ax1.plot(
                budget_data['experiment_spend'],
                budget_data['bang_for_buck'],
                'o-',
                alpha=0.7,
                color=palette[i],
                label=f'${budget/1000:.1f}K'
            )
            
            # Highlight the optimal point (maximum bang for the buck)
            max_idx = budget_data['bang_for_buck'].idxmax()
            max_point = budget_data.loc[max_idx]
            ax1.scatter(
                max_point['experiment_spend'],
                max_point['bang_for_buck'],
                s=120,
                edgecolor='black',
                color=palette[i],
                zorder=5,
                marker='*'
            )
    
    ax1.set_xlabel('Experiment Cost ($)')
    ax1.set_ylabel('RMSE Reduction per $10K Spent')
    ax1.set_title('Diminishing Returns on Additional Spend')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(title='Budget')
    
    # Plot 2: Relationship between number of pairs and RMSE improvement
    for i, budget in enumerate(budgets):
        budget_data = design_results[design_results['budget'] == budget].sort_values('num_pairs')
        
        if len(budget_data) > 1:
            # Calculate the rate of RMSE improvement vs. number of pairs            
            ax2.scatter(
                budget_data['num_pairs'],
                budget_data['rmse'],
                alpha=0.7,
                s=60,
                color=palette[i],
                label=f'${budget/1000:.1f}K'
            )
            
            # Plot regression line to show trend
            z = np.polyfit(budget_data['num_pairs'], budget_data['rmse'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(budget_data['num_pairs'].min(), budget_data['num_pairs'].max(), 100)
            ax2.plot(
                x_range, 
                p(x_range), 
                '--', 
                color=palette[i], 
                alpha=0.5
            )
    
    ax2.set_xlabel('Number of Geo Pairs')
    ax2.set_ylabel('RMSE')
    ax2.set_title('Relationship Between Geo Pairs and Precision')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(title='Budget')
    
    # Add annotations
    bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8)
    ax1.text(
        0.02, 0.02, 
        "Higher = Better value\nShows diminishing returns as spending increases",
        transform=ax1.transAxes,
        fontsize=10,
        bbox=bbox_props
    )
    
    ax2.text(
        0.02, 0.98, 
        "More pairs (left) generally leads\nto better precision (lower RMSE)",
        transform=ax2.transAxes,
        fontsize=10,
        va='top',
        bbox=bbox_props
    )
    
    fig.suptitle('Value Analysis: Finding the "Sweet Spot" for Experiment Design', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig


def create_interactive_design_explorer(design_results):
    """
    Create an interactive design explorer using ipywidgets.
    
    Note: This function is intended to be used in a Jupyter notebook environment.
    
    Args:
        design_results: DataFrame with design results, as returned by calculate_optimal_budget
    
    Returns:
        ipywidgets.Widget: The interactive dashboard
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create widgets
        budget_selector = widgets.SelectMultiple(
            options=sorted([f"${b/1000:.1f}K" for b in design_results['budget'].unique()]),
            description='Budget:',
            disabled=False
        )
        
        min_pairs = widgets.IntSlider(
            value=int(design_results['num_pairs'].min()),
            min=int(design_results['num_pairs'].min()),
            max=int(design_results['num_pairs'].max()),
            step=1,
            description='Min Pairs:',
            disabled=False
        )
        
        max_rmse = widgets.FloatSlider(
            value=float(design_results['rmse_cost_adjusted'].max()),
            min=float(design_results['rmse_cost_adjusted'].min()),
            max=float(design_results['rmse_cost_adjusted'].max()),
            step=0.001,
            description='Max RMSE:',
            disabled=False
        )
        
        max_cost = widgets.FloatSlider(
            value=float(design_results['experiment_spend'].max()),
            min=float(design_results['experiment_spend'].min()),
            max=float(design_results['experiment_spend'].max()),
            step=1000,
            description='Max Cost ($):',
            disabled=False
        )
        
        # Output widget for plotting
        output = widgets.Output()
        
        # Function to update the plot
        def update_plot(*args):
            with output:
                # Clear previous output
                output.clear_output(wait=True)
                
                # Get selected budgets
                selected_budgets = [float(b.replace('$', '').replace('K', '')) * 1000 for b in budget_selector.value]
                if not selected_budgets:
                    selected_budgets = design_results['budget'].unique()
                
                # Filter data
                filtered_data = design_results[
                    (design_results['budget'].isin(selected_budgets)) &
                    (design_results['num_pairs'] >= min_pairs.value) &
                    (design_results['rmse_cost_adjusted'] <= max_rmse.value) &
                    (design_results['experiment_spend'] <= max_cost.value)
                ]
                
                if len(filtered_data) == 0:
                    print("No data matches the current filters.")
                    return
                
                # Create plot
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Color map for different budgets
                colors = plt.cm.tab10(np.linspace(0, 1, len(selected_budgets)))
                
                # Plot each budget scenario
                for i, budget in enumerate(selected_budgets):
                    budget_data = filtered_data[filtered_data['budget'] == budget]
                    if len(budget_data) == 0:
                        continue
                        
                    scatter = ax.scatter(
                        budget_data['experiment_spend'],
                        budget_data['rmse_cost_adjusted'],
                        alpha=0.7,
                        s=budget_data['num_pairs'],
                        c=[colors[i]],
                        label=f'Budget: ${budget/1000:.1f}K'
                    )
                    
                    # Add annotations for key points
                    min_idx = budget_data['rmse_cost_adjusted'].idxmin()
                    if min_idx in budget_data.index:
                        min_point = budget_data.loc[min_idx]
                        ax.annotate(
                            f"Pairs: {min_point['num_pairs']}",
                            (min_point['experiment_spend'], min_point['rmse_cost_adjusted']),
                            xytext=(10, 0),
                            textcoords='offset points'
                        )
                
                # Set labels and title
                ax.set_xlabel('Experiment Cost ($)')
                ax.set_ylabel('RMSE (Cost Adjusted)')
                ax.set_title('Filtered Design Options')
                
                # Add grid and legend
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()
                
                # Show plot
                plt.tight_layout()
                plt.show()
                
                # Show table of best options
                print("Top 5 Designs by RMSE Cost Adjusted:")
                display(filtered_data.sort_values('rmse_cost_adjusted').head(5)[
                    ['budget', 'num_pairs', 'experiment_spend', 'rmse', 'rmse_cost_adjusted', 'trim_rate']
                ].reset_index(drop=True))
        
        # Connect widgets to update function
        budget_selector.observe(update_plot, names='value')
        min_pairs.observe(update_plot, names='value')
        max_rmse.observe(update_plot, names='value')
        max_cost.observe(update_plot, names='value')
        
        # Create layout
        controls = widgets.VBox([budget_selector, min_pairs, max_rmse, max_cost])
        dashboard = widgets.HBox([controls, output])
        
        # Initial update
        update_plot()
        
        return dashboard
    except ImportError:
        print("ipywidgets is required for the interactive explorer. Install with: pip install ipywidgets")
        return None


class ExperimentDesigner:
    """Design experiments using Trimmed Match methodology."""
    
    def __init__(
        self,
        geox_type,
        data,
        time_window_for_design,
        time_window_for_eval,
        response_col="response",
        spend_col="cost",
        matching_metrics=None,
        pairs=None
    ):
        """Initialize ExperimentDesigner.
        
        Args:
            geox_type: Type of experiment (HOLD_BACK, HEAVY_UP, GO_DARK)
            data: DataFrame with geo-level time series data
            time_window_for_design: TimeWindow for design
            time_window_for_eval: TimeWindow for evaluation
            response_col: Column name for response variable
            spend_col: Column name for spend variable
            matching_metrics: Dict of metrics and weights for matching
            pairs: List of pre-defined geo pairs
        """
        self.geox_type = geox_type
        self.data = data
        self.response_col = response_col
        self.spend_col = spend_col
        
        if matching_metrics is None:
            matching_metrics = {response_col: 1.0, spend_col: 0.01}
        
        self.time_window_for_design = time_window_for_design
        self.time_window_for_eval = time_window_for_eval
        self.matching_metrics = matching_metrics
        self.pairs = pairs
        
        # Initialize the TrimmedMatchGeoXDesign
        self.pretest = TrimmedMatchGeoXDesign(
            geox_type=self.geox_type,
            pretest_data=self.data,
            response=self.response_col,
            spend_proxy=self.spend_col,
            matching_metrics=self.matching_metrics,
            time_window_for_design=self.time_window_for_design,
            time_window_for_eval=self.time_window_for_eval,
            pairs=self.pairs
        )
    
    def get_candidate_designs(
        self,
        budget_list,
        iroas_list=[0],
        use_cross_validation=True,
        num_simulations=200
    ):
        """Get candidate experiment designs.
        
        Args:
            budget_list: List of budgets to evaluate
            iroas_list: List of iROAS values to evaluate
            use_cross_validation: Whether to use cross-validation
            num_simulations: Number of simulations for RMSE calculation
            
        Returns:
            tuple: (results_summary, results_detailed)
        """
        return self.pretest.report_candidate_designs(
            budget_list=budget_list,
            iroas_list=iroas_list,
            use_cross_validation=use_cross_validation,
            num_simulations=num_simulations
        )
    
    def calculate_optimal_budget(
        self,
        experiment_budget,
        minimum_detectable_iroas,
        average_order_value=1,
        additional_budget=None,
        use_cross_validation=True,
        num_simulations=200
    ):
        """Calculate the optimal budget for achieving the minimum detectable iROAS.
        
        Args:
            experiment_budget: Maximum experiment budget
            minimum_detectable_iroas: Target minimum detectable iROAS
            average_order_value: Average value per unit response
            additional_budget: List of additional budgets to evaluate
            use_cross_validation: Whether to use cross-validation
            num_simulations: Number of simulations for RMSE calculation
            
        Returns:
            dict: Results including optimal budget, designs, etc.
        """
        if additional_budget is None:
            additional_budget = []
            
        minimum_iroas_aov = minimum_detectable_iroas / average_order_value
        
        # Run a first design with budget equal to the max. budget
        preliminary_results, prel_results_detailed = self.get_candidate_designs(
            budget_list=[experiment_budget],
            iroas_list=[0],
            use_cross_validation=use_cross_validation,
            num_simulations=num_simulations
        )
        
        # Calculate the minimum detectable iROAS for a design with max. budget
        chosen_design = preliminary_results.loc[
            preliminary_results["rmse_cost_adjusted"].idxmin()].squeeze()
        
        calc_min_det_iroas = CalculateMinDetectableIroas(0.10, 0.80)  # Default values
        lowest_detectable_iroas = calc_min_det_iroas.at(chosen_design["rmse"])
        
        # Two cases:
        # 1) If min detectable iROAS with max budget > target min detectable iROAS
        if lowest_detectable_iroas > minimum_iroas_aov:
            budget_to_reach_min_det_iroas = (
                experiment_budget * lowest_detectable_iroas / minimum_iroas_aov
            )
            additional_results, results_detailed = self.get_candidate_designs(
                budget_list=[budget_to_reach_min_det_iroas] + additional_budget,
                iroas_list=[0],
                use_cross_validation=use_cross_validation,
                num_simulations=num_simulations
            )
            
            results = pd.concat([preliminary_results, additional_results], sort=False)
            
        # 2) If min detectable iROAS with max budget < target min detectable iROAS
        else:
            optimal_budget = (
                experiment_budget * lowest_detectable_iroas / minimum_iroas_aov
            )
            lower_budget = optimal_budget * 0.8
            upper_budget = optimal_budget * 1.2
            
            list_of_budgets = [
                lower_budget, optimal_budget, upper_budget
            ] + additional_budget
            
            results, results_detailed = self.get_candidate_designs(
                budget_list=list_of_budgets,
                iroas_list=[0],
                use_cross_validation=use_cross_validation,
                num_simulations=num_simulations
            )
        
        return {
            "results": results,
            "results_detailed": results_detailed,
            "preliminary_results": preliminary_results,
            "optimal_pair_index": preliminary_results.loc[
                preliminary_results["rmse_cost_adjusted"].idxmin(), "pair_index"
            ],
            "lowest_detectable_iroas": lowest_detectable_iroas,
            "minimum_iroas_aov": minimum_iroas_aov
        }
    
    def get_optimal_design(
        self,
        pair_index,
        base_seed=0,
        confidence=0.90,
        group_control=GeoAssignment.CONTROL,
        group_treatment=GeoAssignment.TREATMENT
    ):
        """Get the optimal experiment design.
        
        Args:
            pair_index: Index of the optimal pair
            base_seed: Base seed for random number generation
            confidence: Confidence level
            group_control: Control group assignment
            group_treatment: Treatment group assignment
            
        Returns:
            tuple: (axes, geopairs, treatments, controls)
        """
        axes = self.pretest.output_chosen_design(
            pair_index=pair_index,
            base_seed=base_seed,
            confidence=confidence,
            group_control=group_control,
            group_treatment=group_treatment
        )
        
        # Get the geo pairs and group assignments
        geopairs = self.pretest.geo_level_eval_data[pair_index].copy()
        
        # Determine treatment and control geos
        treatment_geo = geopairs.loc[
            geopairs["assignment"] == group_treatment, "geo"
        ].tolist()
        
        control_geo = geopairs.loc[
            geopairs["assignment"] == group_control, "geo"
        ].tolist()
        
        return axes, geopairs, treatment_geo, control_geo
    
    def run_coverage_test(
        self,
        data_coverage_test,
        optimal_pair_index,
        group_control=GeoAssignment.CONTROL,
        group_treatment=GeoAssignment.TREATMENT,
        confidence=0.90
    ):
        """Run coverage test for the chosen design.
        
        Args:
            data_coverage_test: Data for coverage test
            optimal_pair_index: Index of the optimal pair
            group_control: Control group assignment
            group_treatment: Treatment group assignment
            confidence: Confidence level
            
        Returns:
            DataFrame: Coverage test results
        """
        return self.pretest.run_aa_test(
            data=data_coverage_test,
            time_window_for_eval=self.time_window_for_eval,
            pair_index=optimal_pair_index,
            confidence=confidence,
            group_control=group_control,
            group_treatment=group_treatment
        )
    
    def plot_pair_comparison(
        self,
        pair_index,
        group_control=GeoAssignment.CONTROL,
        group_treatment=GeoAssignment.TREATMENT
    ):
        """Plot pair-by-pair comparison.
        
        Args:
            pair_index: Index of the pair to plot
            group_control: Control group assignment
            group_treatment: Treatment group assignment
            
        Returns:
            matplotlib.figure.Figure: Plot
        """
        return self.pretest.plot_pair_by_pair_comparison(
            pair_index=pair_index,
            group_control=group_control,
            group_treatment=group_treatment
        )

    def add_trade_off_visualizations(self, results_df, output_dir):
        """
        Generate and save visualizations showing the design trade-offs.
        
        Args:
            results_df: DataFrame with design results, typically from the 'results' key in 
                       the dictionary returned by calculate_optimal_budget
            output_dir: Directory where to save the visualizations
            
        Returns:
            dict: Dictionary with paths to the saved visualizations
        """
        import os
        import matplotlib.pyplot as plt
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate visualizations
        fig_pareto = plot_cost_precision_tradeoff(results_df)
        fig_dashboard = plot_design_dashboard(results_df)
        fig_returns = plot_bang_for_buck(results_df)
        
        # Save visualizations
        pareto_path = os.path.join(output_dir, 'design_pareto.png')
        dashboard_path = os.path.join(output_dir, 'design_dashboard.png')
        returns_path = os.path.join(output_dir, 'design_returns.png')
        
        fig_pareto.savefig(pareto_path, dpi=300, bbox_inches='tight')
        fig_dashboard.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        fig_returns.savefig(returns_path, dpi=300, bbox_inches='tight')
        
        # Close figures to free memory
        plt.close(fig_pareto)
        plt.close(fig_dashboard)
        plt.close(fig_returns)
        
        return {
            'pareto': pareto_path,
            'dashboard': dashboard_path,
            'returns': returns_path
        }
