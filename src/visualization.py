"""
Visualization Module for Spinal Cord Model
==========================================

This module generates all figures referenced in the manuscript:
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
        sns.set_palette("husl")
        
        # Set default figure parameters
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['lines.linewidth'] = 2
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
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
        phi_range = np.linspace(0, 0.8, 100)  # Up to ~46 degrees
        moments = []
        strains = []
        
        for phi in phi_range:
            M_ext, epsilon = self.model.quasi_static_analysis(phi)
            moments.append(M_ext)
            strains.append(epsilon)
        
        moments = np.array(moments)
        strains = np.array(strains)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot strain vs moment
        ax.plot(moments, strains, 'b-', linewidth=2.5, label='Model Prediction')
        
        # Add failure threshold region
        epsilon_fail = self.model.params['epsilon_fail']
        ax.axhspan(epsilon_fail - 0.02, epsilon_fail + 0.02, 
                   alpha=0.3, color='red', label='Biological Failure Range')
        ax.axhline(epsilon_fail, color='red', linestyle='--', alpha=0.7, 
                  label=f'Mean Failure Strain ({epsilon_fail})')
        
        # Mark failure point
        M_fail = self.model.get_quasi_static_failure_moment()
        ax.plot([M_fail], [epsilon_fail], 'ro', markersize=10, 
               label=f'Predicted Failure ({M_fail:.1f} Nm)')
        
        # Customize plot
        ax.set_xlabel('Applied External Moment (Nm)')
        ax.set_ylabel('Spinal Cord Strain')
        ax.set_title('Quasi-Static Spinal Cord Strain Response')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set reasonable limits
        ax.set_xlim(0, 60)
        ax.set_ylim(0, 0.15)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_dynamic_response(self, M_0: float = 20.0, t_final: float = 0.1, 
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Generate Figure 2: Dynamic response to step input moment.
        
        Parameters
        ----------
        M_0 : float
            Step moment magnitude [Nm]
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
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Strain vs time
        ax1.plot(results['time'] * 1000, results['epsilon'], 'b-', linewidth=2.5, 
                label='Dynamic Strain Response')
        
        # Add failure threshold
        epsilon_fail = self.model.params['epsilon_fail']
        ax1.axhspan(epsilon_fail - 0.02, epsilon_fail + 0.02, 
                    alpha=0.3, color='red', label='Failure Range')
        ax1.axhline(epsilon_fail, color='red', linestyle='--', alpha=0.7,
                   label=f'Mean Failure Strain ({epsilon_fail})')
        
        # Mark failure point if it occurs
        if results['failure_time'] is not None:
            failure_strain = self.model.params['r_c'] * results['phi'][results['time'] == results['failure_time']] / self.model.params['L_0']
            ax1.plot(results['failure_time'] * 1000, failure_strain, 'ro', markersize=10,
                    label=f'Failure Onset ({results["failure_time"]*1000:.1f} ms)')
        
        ax1.set_ylabel('Spinal Cord Strain')
        ax1.set_title(f'Dynamic Response to Step Input Moment (M₀ = {M_0} Nm)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_ylim(0, 0.15)
        
        # Plot 2: Moment vs time
        ax2.plot(results['time'] * 1000, results['M_ext'], 'g-', linewidth=2.5,
                label='Applied External Moment')
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('External Moment (Nm)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
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
        # Parameters to analyze
        param_names = ['K_theta', 'k_c', 'C_theta', 'K_nl']
        param_labels = ['Disc-Ligament\nStiffness (K_θ)', 'Cord Stiffness\n(k_c)', 
                       'Damping\n(C_θ)', 'Nonlinear\nStiffness (K_nl)']
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        M_ext = 20.0  # Fixed external moment
        
        for i, (param_name, param_label) in enumerate(zip(param_names, param_labels)):
            ax = axes[i]
            
            # Create variation range (±50% from nominal)
            nominal_value = self.model.params[param_name]
            variation_range = np.linspace(0.5 * nominal_value, 1.5 * nominal_value, 20)
            
            # Run sensitivity analysis
            sensitivity_results = self.model.sensitivity_analysis(
                param_name, variation_range, M_ext
            )
            
            # Plot max strain vs parameter value
            ax.plot(sensitivity_results['parameter_values'], 
                   sensitivity_results['max_strains'], 'bo-', linewidth=2, markersize=6)
            
            # Mark nominal value
            ax.axvline(nominal_value, color='red', linestyle='--', alpha=0.7,
                      label='Nominal Value')
            
            # Mark failure threshold
            epsilon_fail = self.model.params['epsilon_fail']
            ax.axhline(epsilon_fail, color='red', alpha=0.5, linestyle='-',
                      label=f'Failure Threshold ({epsilon_fail})')
            
            # Mark points where failure occurs
            failure_mask = np.array(sensitivity_results['failure_times']) is not None
            if np.any(failure_mask):
                ax.plot(sensitivity_results['parameter_values'][failure_mask],
                       sensitivity_results['max_strains'][failure_mask], 'rx', 
                       markersize=8, label='Failure Predicted')
            
            ax.set_xlabel(param_label)
            ax.set_ylabel('Maximum Strain')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
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
        
        # Draw vertebrae (rectangles)
        vertebra1 = Rectangle((0.05, 0.1), 0.08, 0.15, 
                            facecolor='lightgray', edgecolor='black', linewidth=2)
        vertebra2 = Rectangle((0.05, 0.35), 0.08, 0.15, 
                            facecolor='lightgray', edgecolor='black', linewidth=2)
        
        ax.add_patch(vertebra1)
        ax.add_patch(vertebra2)
        
        # Add vertebra labels
        ax.text(0.09, 0.175, 'T$_n$', fontsize=14, ha='center', va='center', weight='bold')
        ax.text(0.09, 0.425, 'T$_{n+1}$', fontsize=14, ha='center', va='center', weight='bold')
        
        # Draw ICR (Instantaneous Center of Rotation)
        icr = Circle((0.02, 0.25), 0.008, facecolor='red', edgecolor='darkred', linewidth=2)
        ax.add_patch(icr)
        ax.text(0.02, 0.25, 'ICR', fontsize=10, ha='center', va='center', color='white', weight='bold')
        
        # Draw disc-ligament complex (spring-damper)
        spring_x = [0.02, 0.15]
        spring_y = [0.25, 0.25]
        ax.plot(spring_x, spring_y, 'b-', linewidth=3, label='Disc-Ligament Complex\n(Spring-Damper)')
        
        # Add spring coils
        for i in range(1, 6):
            x = 0.02 + i * 0.026
            ax.plot([x, x], [0.24, 0.26], 'b-', linewidth=2)
        
        # Draw spinal cord (tension element)
        cord_x = [0.02, 0.15]
        cord_y = [0.25, 0.25]
        ax.plot(cord_x, cord_y, 'g-', linewidth=3, label='Spinal Cord\n(Tension Element)')
        
        # Add cord cross-section
        cord_circle = Circle((0.15, 0.25), 0.015, facecolor='lightgreen', 
                           edgecolor='darkgreen', linewidth=2)
        ax.add_patch(cord_circle)
        
        # Draw moment arm
        moment_arm_x = [0.02, 0.15]
        moment_arm_y = [0.25, 0.25]
        ax.plot(moment_arm_x, moment_arm_y, 'r--', linewidth=2, alpha=0.7)
        ax.text(0.085, 0.27, '$r_c$', fontsize=12, ha='center', va='center', 
               color='red', weight='bold')
        
        # Add arrow for rotation
        rotation_arrow = patches.FancyArrowPatch((0.02, 0.3), (0.02, 0.35),
                                               arrowstyle='->', mutation_scale=20,
                                               color='orange', linewidth=3)
        ax.add_patch(rotation_arrow)
        ax.text(0.02, 0.37, 'φ(t)', fontsize=12, ha='center', va='center', 
               color='orange', weight='bold')
        
        # Add external moment arrow
        moment_arrow = patches.FancyArrowPatch((0.15, 0.4), (0.15, 0.35),
                                             arrowstyle='->', mutation_scale=20,
                                             color='purple', linewidth=3)
        ax.add_patch(moment_arrow)
        ax.text(0.15, 0.42, 'M_ext(t)', fontsize=12, ha='center', va='center', 
               color='purple', weight='bold')
        
        # Add parameter annotations
        ax.text(0.25, 0.2, 'Parameters:', fontsize=12, weight='bold')
        ax.text(0.25, 0.18, f'K_θ = {self.model.params["K_theta"]} Nm/rad', fontsize=10)
        ax.text(0.25, 0.16, f'C_θ = {self.model.params["C_theta"]} Nms/rad', fontsize=10)
        ax.text(0.25, 0.14, f'k_c = {self.model.params["k_c"]:.1f} N/m', fontsize=10)
        ax.text(0.25, 0.12, f'r_c = {self.model.params["r_c"]*1000:.1f} mm', fontsize=10)
        ax.text(0.25, 0.10, f'L_0 = {self.model.params["L_0"]*1000:.1f} mm', fontsize=10)
        
        # Customize plot
        ax.set_xlabel('Position (m)')
        ax.set_ylabel('Position (m)')
        ax.set_title('Lumped-Parameter Spinal Cord Model Schematic')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Remove axis ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        
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