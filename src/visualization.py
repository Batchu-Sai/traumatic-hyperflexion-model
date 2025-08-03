"""
Visualization Module for Spinal Cord Model
==========================================

This module generates all figures referenced in the manuscript with exact specifications:
- Figure 1: Quasi-static spinal cord strain vs applied moment
- Figure 2: Dynamic response to step input moment
- Figure 3: Sensitivity analysis
- Figure 4: Model schematic

Author: Sai Batchu
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, Arrow
import matplotlib.patches as mpatches
from typing import Dict, Any, Optional
import seaborn as sns

from .spinal_model import SpinalCordModel


class SpinalModelVisualizer:
    """Visualization class for the spinal cord model."""
    
    def __init__(self, model: SpinalCordModel):
        """
        Initialize the visualizer with a spinal cord model.
        
        Parameters
        ----------
        model : SpinalCordModel
            The spinal cord model instance
        """
        self.model = model
        self.setup_plotting_style()
    
    def setup_plotting_style(self):
        """Set up consistent plotting style for all figures."""
        plt.style.use('default')
        
        # Set publication-quality parameters
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['lines.linewidth'] = 2
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['grid.linestyle'] = '--'
        plt.rcParams['grid.color'] = '#D3D3D3'
        plt.rcParams['axes.linewidth'] = 1
        plt.rcParams['axes.edgecolor'] = 'black'
    
    def plot_quasi_static_analysis(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Generate Figure 1: Quasi-static spinal cord strain vs applied moment.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The generated figure
        """
        # Generate data points
        phi_range = np.linspace(0, 0.8, 200)  # Up to ~46 degrees
        moments = []
        strains = []
        
        for phi in phi_range:
            M_ext, epsilon = self.model.quasi_static_analysis(phi)
            moments.append(M_ext)
            strains.append(epsilon)
        
        moments = np.array(moments)
        strains = np.array(strains)
        
        # Create figure with exact specifications
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot strain response curve (orange)
        ax.plot(moments, strains, color='#FFA500', linewidth=2, 
                label='Predicted Nonlinear Strain Response')
        
        # Add biological failure region (shaded)
        ax.axhspan(0.08, 0.12, alpha=0.5, color='#D3D3D3', 
                  label='Biological Failure Range (ε = 0.10 ± 0.02)')
        
        # Add mean failure line (dashed)
        ax.axhline(0.10, color='#A9A9A9', linestyle='--', linewidth=1,
                  label='Mean Failure Strain (ε = 0.10)')
        
        # Mark failure point
        M_fail = self.model.get_quasi_static_failure_moment()
        ax.plot([M_fail], [0.10], 'ro', markersize=8, 
               label=f'Predicted Failure ({M_fail:.1f} Nm)')
        
        # Customize axes exactly as specified
        ax.set_xlabel('Applied External Moment (Nm)', fontsize=12)
        ax.set_ylabel('Spinal Cord Strain', fontsize=12)
        ax.set_title('Quasi-Static Spinal Cord Strain Response', fontsize=14, fontweight='bold')
        
        # Set exact axis ranges and ticks
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 0.15)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.set_yticks([0, 0.05, 0.10, 0.15])
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', color='#D3D3D3')
        
        # Add legend
        ax.legend(loc='upper left')
        
        # Add border
        for spine in ax.spines.values():
            spine.set_linewidth(1)
            spine.set_color('black')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_dynamic_response(self, M_0: float = 30.0, t_final: float = 0.1, 
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Generate Figure 2: Dynamic response to step input moment.
        
        Parameters
        ----------
        M_0 : float
            Step moment magnitude [Nm] - increased to ensure failure
        t_final : float
            Final simulation time [s]
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The generated figure
        """
        # Run simulation
        M_ext_func = self.model.step_input_moment(M_0)
        results = self.model.simulate_dynamics(M_ext_func, t_final)
        
        # Create figure with exact specifications
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot dynamic strain curve (orange)
        ax.plot(results['time'] * 1000, results['epsilon'], 
                color='#FFA500', linewidth=2, 
                label='Dynamic Strain Response')
        
        # Add mean failure line (dashed crimson red)
        ax.axhline(0.10, color='#DC143C', linestyle='--', linewidth=1.5,
                  label='Mean Failure Strain (ε = 0.10)')
        
        # Add biological failure region (shaded)
        ax.axhspan(0.08, 0.12, alpha=0.5, color='#D3D3D3',
                  label='Failure Range (ε = 0.10 ± 0.02)')
        
        # Mark failure point if it occurs
        if results['failure_time'] is not None:
            failure_idx = np.argmin(np.abs(results['time'] - results['failure_time']))
            failure_strain = results['epsilon'][failure_idx]
            ax.plot(results['failure_time'] * 1000, failure_strain, 
                   'o', color='#DC143C', markersize=6,
                   label=f'Failure Onset ({results["failure_time"]*1000:.1f} ms)')
        
        # Customize axes exactly as specified
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Spinal Cord Strain', fontsize=12)
        ax.set_title(f'Dynamic Response to Step Input Moment (M₀ = {M_0} Nm)', 
                    fontsize=14, fontweight='bold')
        
        # Set exact axis ranges and ticks
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 0.15)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.set_yticks([0, 0.05, 0.10, 0.15])
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', color='#D3D3D3')
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Add border
        for spine in ax.spines.values():
            spine.set_linewidth(1)
            spine.set_color('black')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_sensitivity_analysis(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Generate Figure 3: Sensitivity analysis.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The generated figure
        """
        # Parameters to analyze - include all key parameters as specified in manuscript
        param_names = ['K_theta', 'k_c', 'C_theta', 'r_c', 'epsilon_fail']
        param_labels = ['$K_θ$', '$k_c$', '$C_θ$', '$r_c$', '$\\epsilon_{fail}$']
        
        # Create figure with optimal spacing for 5 parameters
        fig, ax = plt.subplots(figsize=(18, 8))
        
        M_ext = 30.0  # Fixed external moment - increased to ensure failure
        
        # Get baseline results with nominal parameters
        baseline_M_ext_func = self.model.step_input_moment(M_ext)
        baseline_results = self.model.simulate_dynamics(baseline_M_ext_func, t_final=0.1)
        baseline_max_strain = np.max(baseline_results['epsilon'])
        baseline_failure_time = baseline_results['failure_time']
        
        # Calculate data for each parameter
        all_strain_data = []
        all_time_data = []
        all_no_failure = []
        
        for param_name in param_names:
            # Create variation range (±50% from nominal)
            nominal_value = self.model.params[param_name]
            variation_range = [0.5 * nominal_value, 1.5 * nominal_value]
            
            # Run sensitivity analysis
            sensitivity_results = self.model.sensitivity_analysis(
                param_name, np.array(variation_range), M_ext
            )
            
            # Calculate percent changes relative to baseline
            strain_changes = []
            time_changes = []
            no_failure = []
            
            for i, param_val in enumerate(sensitivity_results['parameter_values']):
                max_strain = sensitivity_results['max_strains'][i]
                failure_time = sensitivity_results['failure_times'][i]
                
                # Calculate percent change in max strain relative to baseline
                if baseline_max_strain > 0:
                    strain_change = ((max_strain - baseline_max_strain) / baseline_max_strain) * 100
                else:
                    strain_change = 0
                
                # Calculate percent change in failure time relative to baseline
                if failure_time is not None and baseline_failure_time is not None:
                    time_change = ((failure_time - baseline_failure_time) / baseline_failure_time) * 100
                elif failure_time is None and baseline_failure_time is not None:
                    time_change = -100  # No failure
                else:
                    time_change = 0
                
                strain_changes.append(strain_change)
                time_changes.append(time_change)
                no_failure.append(failure_time is None)
            
            all_strain_data.append(strain_changes)
            all_time_data.append(time_changes)
            all_no_failure.append(no_failure)
        
        # Create proper grouped bar chart
        n_params = len(param_names)
        n_bars_per_group = 4  # -50% strain, +50% strain, -50% time, +50% time
        
        # Calculate bar positions
        bar_width = 0.15
        group_spacing = 0.3  # Space between parameter groups
        bar_spacing = 0.02   # Space between bars within a group
        
        # Calculate x positions for each group
        group_positions = np.arange(n_params) * (n_bars_per_group * bar_width + group_spacing)
        
        # Calculate positions for each bar within a group
        bar_positions = []
        for group_pos in group_positions:
            group_bars = []
            for i in range(n_bars_per_group):
                bar_pos = group_pos + i * (bar_width + bar_spacing)
                group_bars.append(bar_pos)
            bar_positions.append(group_bars)
        
        # Colors for the paired scheme
        strain_colors = ['#87CEEB', '#4682B4']  # Light Blue, Dark Blue
        time_colors = ['#90EE90', '#2E8B57']    # Light Green, Dark Green
        
        # Scale k_c values to make them visible (they're very small compared to other parameters)
        # k_c changes are ~0.0165% while others are ~50%, so scale k_c by 100x
        k_c_scale_factor = 100.0
        
        # Plot bars for each parameter group
        bars = []
        bar_labels = []
        
        for i, (strain_data, time_data, no_failure_data) in enumerate(zip(all_strain_data, all_time_data, all_no_failure)):
            group_bars = bar_positions[i]
            
            # Scale k_c values to make them visible (k_c is the second parameter, index 1)
            if i == 1:  # k_c parameter
                scaled_strain_data = [v * k_c_scale_factor for v in strain_data]
                scaled_time_data = [v * k_c_scale_factor for v in time_data]
            else:
                scaled_strain_data = strain_data
                scaled_time_data = time_data
            
            # Bar 1: Max Strain -50%
            if abs(scaled_strain_data[0]) > 0.001:  # Only plot if value is significant
                bar1 = ax.bar(group_bars[0], scaled_strain_data[0], bar_width, 
                             color=strain_colors[0], edgecolor='black', linewidth=0.5, alpha=0.9)
                bars.append(bar1)
                bar_labels.append('Max Strain, -50%')
            else:
                # Add small marker for zero/negligible values
                ax.text(group_bars[0], 0.1, '●', color='#444444', fontsize=12, 
                       ha='center', va='bottom')
            
            # Bar 2: Max Strain +50%
            if abs(scaled_strain_data[1]) > 0.001:  # Only plot if value is significant
                bar2 = ax.bar(group_bars[1], scaled_strain_data[1], bar_width, 
                             color=strain_colors[1], edgecolor='black', linewidth=0.5, alpha=0.9)
                bars.append(bar2)
                bar_labels.append('Max Strain, +50%')
            else:
                # Add small marker for zero/negligible values
                ax.text(group_bars[1], 0.1, '●', color='#444444', fontsize=12, 
                       ha='center', va='bottom')
            
            # Bar 3: Time to Failure -50%
            if not no_failure_data[0]:  # Only plot if there's a failure
                if abs(scaled_time_data[0]) > 0.001:  # Only plot if value is significant
                    bar3 = ax.bar(group_bars[2], scaled_time_data[0], bar_width, 
                                 color=time_colors[0], edgecolor='black', linewidth=0.5, alpha=0.9)
                    bars.append(bar3)
                    bar_labels.append('Time to Failure, -50%')
                else:
                    # Add small marker for negligible time changes
                    ax.text(group_bars[2], 0.1, '●', color='#444444', fontsize=12, 
                           ha='center', va='bottom')
            else:
                # Add red X for no failure
                ax.text(group_bars[2], -5, '×', color='#DC143C', fontsize=22, 
                       fontweight='bold', ha='center', va='top')
            
            # Bar 4: Time to Failure +50%
            if not no_failure_data[1]:  # Only plot if there's a failure
                if abs(scaled_time_data[1]) > 0.001:  # Only plot if value is significant
                    bar4 = ax.bar(group_bars[3], scaled_time_data[1], bar_width, 
                                 color=time_colors[1], edgecolor='black', linewidth=0.5, alpha=0.9)
                    bars.append(bar4)
                    bar_labels.append('Time to Failure, +50%')
                else:
                    # Add small marker for negligible time changes
                    ax.text(group_bars[3], 0.1, '●', color='#444444', fontsize=12, 
                           ha='center', va='bottom')
            else:
                # Add red X for no failure
                ax.text(group_bars[3], -5, '×', color='#DC143C', fontsize=22, 
                       fontweight='bold', ha='center', va='top')
        
        # Add zero line for better readability
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
        
        # Customize axes
        ax.set_xlabel('Model Parameter', fontsize=20)
        ax.set_ylabel('Percent Change in Output (%)', fontsize=20)
        ax.set_title('Sensitivity Analysis (±50% Parameter Changes)', 
                    fontsize=22, fontweight='bold', pad=20)
        
        # Set x-axis labels
        group_centers = [np.mean(bar_positions[i]) for i in range(n_params)]
        ax.set_xticks(group_centers)
        ax.set_xticklabels(param_labels, fontsize=18, fontweight='bold', fontfamily='serif')
        
        # Create comprehensive legend
        from matplotlib.patches import Patch
        
        # Create legend handles with proper symbols
        from matplotlib.lines import Line2D
        
        legend_handles = [
            Patch(color=strain_colors[0], label='Max Strain, -50% Change'),
            Patch(color=strain_colors[1], label='Max Strain, +50% Change'),
            Patch(color=time_colors[0], label='Time to Failure, -50% Change'),
            Patch(color=time_colors[1], label='Time to Failure, +50% Change'),
            Line2D([0], [0], marker='x', color='#DC143C', linestyle='None', 
                   markersize=10, label='No Failure Predicted'),
            Line2D([0], [0], marker='o', color='#444444', linestyle='None', 
                   markersize=8, label='Negligible Change (<0.1%)')
        ]
        
        # Add legend inside the plot
        ax.legend(handles=legend_handles, loc='upper right',
                frameon=True, fancybox=False, shadow=False, fontsize=16)
        

        
        # Add grid
        ax.grid(True, alpha=0.2, linestyle='-', color='#D3D3D3', axis='y')
        
        # Set tick label sizes for better readability
        ax.tick_params(axis='both', which='major', labelsize=16)
        
        # Add footnote about k_c scaling
        ax.text(0.98, 0.02, '*k_c values scaled by 100× for visibility', 
               transform=ax.transAxes, fontsize=15, ha='right', va='bottom',
               style='italic', color='#666666')
        
        # Add border
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('black')
        
        # Set y-axis limits for better visual balance
        all_values = []
        for strain_data in all_strain_data:
            all_values.extend(strain_data)
        for time_data in all_time_data:
            all_values.extend([v for v in time_data if v != -100])  # Exclude no-failure markers
        
        if all_values:
            y_min = min(all_values) - 10
            y_max = max(all_values) + 10
            
            # Ensure k_c bars are visible by expanding y-axis if needed
            # k_c changes are very small (±0.0165%), so we need to ensure they're visible
            if abs(y_min) < 1 and abs(y_max) < 1:
                # If all values are small, expand the y-axis to show small changes
                y_min = -2
                y_max = 2
            else:
                # For mixed large and small values, ensure small values are visible
                # Check if k_c values are being cut off
                k_c_strain_values = all_strain_data[1]  # k_c is the second parameter
                if any(abs(v) < 0.1 for v in k_c_strain_values):
                    # k_c has small values, ensure they're visible
                    y_min = min(y_min, -1)
                    y_max = max(y_max, 1)
            
            ax.set_ylim(y_min, y_max)
        
        # Adjust layout to prevent overlap
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_model_schematic(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Generate Figure 4: Model schematic.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save the figure
            
        Returns
        -------
        plt.Figure
            The generated figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set up coordinate system
        ax.set_xlim(-0.1, 0.3)
        ax.set_ylim(-0.1, 0.3)
        ax.set_aspect('equal')
        
        # Draw vertebrae (rectangles) - T_n and T_{n+1}
        vertebra1 = Rectangle((0.05, 0.1), 0.08, 0.15, 
                            facecolor='#D3D3D3', edgecolor='black', linewidth=1.5)
        vertebra2 = Rectangle((0.05, 0.35), 0.08, 0.15, 
                            facecolor='#D3D3D3', edgecolor='black', linewidth=1.5)
        
        ax.add_patch(vertebra1)
        ax.add_patch(vertebra2)
        
        # Add vertebra labels
        ax.text(0.09, 0.175, 'T$_n$', fontsize=12, ha='center', va='center', 
               weight='bold', fontfamily='sans-serif')
        ax.text(0.09, 0.425, 'T$_{n+1}$', fontsize=12, ha='center', va='center', 
               weight='bold', fontfamily='sans-serif')
        
        # Draw ICR (Instantaneous Center of Rotation)
        icr = Circle((0.02, 0.25), 0.008, facecolor='black', edgecolor='black', linewidth=1.5)
        ax.add_patch(icr)
        ax.text(0.02, 0.25, 'ICR', fontsize=10, ha='center', va='center', 
               color='white', weight='bold', fontfamily='sans-serif')
        
        # Draw disc-ligament complex (spring-damper)
        spring_x = [0.02, 0.15]
        spring_y = [0.25, 0.25]
        ax.plot(spring_x, spring_y, 'b-', linewidth=3, 
               label='Disc-Ligament Complex (M$_{disc-lig}$)')
        
        # Add spring coils
        for i in range(1, 6):
            x = 0.02 + i * 0.026
            ax.plot([x, x], [0.24, 0.26], 'b-', linewidth=2)
        
        # Draw spinal cord (tension element) - medium slate blue
        cord_x = [0.02, 0.15]
        cord_y = [0.25, 0.25]
        ax.plot(cord_x, cord_y, color='#7B68EE', linewidth=3, 
               label='Spinal Cord Element (F$_{cord}$)')
        
        # Add cord cross-section
        cord_circle = Circle((0.15, 0.25), 0.015, facecolor='#7B68EE', 
                           edgecolor='#7B68EE', linewidth=2)
        ax.add_patch(cord_circle)
        
        # Draw moment arm (r_c)
        moment_arm_x = [0.02, 0.15]
        moment_arm_y = [0.25, 0.25]
        ax.plot(moment_arm_x, moment_arm_y, 'k--', linewidth=2, alpha=0.7)
        ax.text(0.085, 0.27, '$r_c$', fontsize=10, ha='center', va='center', 
               color='black', weight='bold', fontfamily='sans-serif')
        
        # Add arrow for rotation (φ)
        rotation_arrow = patches.FancyArrowPatch((0.02, 0.3), (0.02, 0.35),
                                               arrowstyle='->', mutation_scale=20,
                                               color='orange', linewidth=3)
        ax.add_patch(rotation_arrow)
        ax.text(0.02, 0.37, 'φ', fontsize=12, ha='center', va='center', 
               color='orange', weight='bold', fontfamily='sans-serif')
        
        # Add external moment arrow (M_ext)
        moment_arrow = patches.FancyArrowPatch((0.15, 0.4), (0.15, 0.35),
                                             arrowstyle='->', mutation_scale=20,
                                             color='purple', linewidth=3)
        ax.add_patch(moment_arrow)
        ax.text(0.15, 0.42, '$M_{\\mathrm{ext}}$', fontsize=12, ha='center', va='center', 
               color='purple', weight='bold', fontfamily='sans-serif')
        
        # Add cord force arrows (F_cord)
        cord_force1 = patches.FancyArrowPatch((0.15, 0.25), (0.12, 0.25),
                                             arrowstyle='<-', mutation_scale=15,
                                             color='#7B68EE', linewidth=2)
        cord_force2 = patches.FancyArrowPatch((0.02, 0.25), (0.05, 0.25),
                                             arrowstyle='<-', mutation_scale=15,
                                             color='#7B68EE', linewidth=2)
        ax.add_patch(cord_force1)
        ax.add_patch(cord_force2)
        ax.text(0.12, 0.23, '$F_{cord}$', fontsize=10, ha='center', va='center', 
               color='#7B68EE', weight='bold', fontfamily='sans-serif')
        
        # Add parameter annotations
        ax.text(0.25, 0.2, 'Parameters:', fontsize=12, weight='bold', fontfamily='sans-serif')
        ax.text(0.25, 0.18, f'K_θ = {self.model.params["K_theta"]} Nm/rad', fontsize=10, fontfamily='sans-serif')
        ax.text(0.25, 0.16, f'C_θ = {self.model.params["C_theta"]} Nms/rad', fontsize=10, fontfamily='sans-serif')
        ax.text(0.25, 0.14, f'k_c = {self.model.params["k_c"]:.1f} N/m', fontsize=10, fontfamily='sans-serif')
        ax.text(0.25, 0.12, f'r_c = {self.model.params["r_c"]*1000:.1f} mm', fontsize=10, fontfamily='sans-serif')
        ax.text(0.25, 0.10, f'L_0 = {self.model.params["L_0"]*1000:.1f} mm', fontsize=10, fontfamily='sans-serif')
        
        # Customize plot
        ax.set_xlabel('Position (m)', fontsize=12)
        ax.set_ylabel('Position (m)', fontsize=12)
        ax.set_title('Lumped-Parameter Spinal Cord Model Schematic', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--', color='#D3D3D3')
        
        # Remove axis ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add border
        for spine in ax.spines.values():
            spine.set_linewidth(1)
            spine.set_color('black')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def generate_all_figures(self, output_dir: str = "figures"):
        """
        Generate all figures from the manuscript.
        
        Parameters
        ----------
        output_dir : str
            Directory to save the figures
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate all figures
        self.plot_quasi_static_analysis(f"{output_dir}/quasi_static_analysis.png")
        self.plot_dynamic_response(save_path=f"{output_dir}/dynamic_response.png")
        self.plot_sensitivity_analysis(f"{output_dir}/sensitivity_analysis.png")
        self.plot_model_schematic(f"{output_dir}/model_schematic.png")
        
        print(f"All figures saved to {output_dir}/")
    
    def create_validation_table(self) -> str:
        """
        Create the validation table from the manuscript.
        
        Returns
        -------
        str
            Formatted table string
        """
        table = """
Comparison of Model Predictions with Experimental Data
====================================================

Metric                    | Model Prediction | Experimental Data Range
-------------------------|------------------|------------------------
Failure Strain           | 0.10            | 0.08-0.12 [Yamada 1970]
Failure Moment (Nm)      | ≈40.0           | 23-53 [Lopez-Valdes 2011]
        """
        return table 