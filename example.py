#!/usr/bin/env python3
"""
Simple Example: Spinal Cord Hyperflexion Model
==============================================

This script demonstrates basic usage of the spinal cord model.

Author: Sai Batchu
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append('src')

from src.spinal_model import SpinalCordModel
from src.visualization import SpinalModelVisualizer


def main():
    """Simple example demonstrating the model."""
    print("Spinal Cord Model - Simple Example")
    print("=" * 40)
    
    # 1. Initialize the model
    model = SpinalCordModel()
    print("Model initialized with default parameters")
    
    # 2. Quasi-static analysis
    print(f"\nQuasi-static failure moment: {model.get_quasi_static_failure_moment():.1f} Nm")
    
    # 3. Dynamic simulation
    M_ext = 25.0  # External moment
    M_ext_func = model.step_input_moment(M_ext)
    results = model.simulate_dynamics(M_ext_func, t_final=0.1)
    
    print(f"Dynamic simulation with M_ext = {M_ext} Nm:")
    print(f"Maximum strain: {np.max(results['epsilon']):.4f}")
    
    if results['failure_time'] is not None:
        print(f"Failure predicted at: {results['failure_time']*1000:.1f} ms")
    else:
        print("No failure predicted")
    
    # 4. Generate a simple plot
    plt.figure(figsize=(10, 6))
    plt.plot(results['time'] * 1000, results['epsilon'], 'b-', linewidth=2, label='Cord Strain')
    plt.axhline(model.params['epsilon_fail'], color='red', linestyle='--', 
               label=f'Failure Threshold ({model.params["epsilon_fail"]})')
    plt.xlabel('Time (ms)')
    plt.ylabel('Spinal Cord Strain')
    plt.title(f'Dynamic Response (M_ext = {M_ext} Nm)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('example_output.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'example_output.png'")
    
    # Show the plot
    plt.show()
    
    print("\nExample completed!")


if __name__ == "__main__":
    main() 