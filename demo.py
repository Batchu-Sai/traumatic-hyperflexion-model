#!/usr/bin/env python3
"""
Spinal Cord Hyperflexion Model Demo
===================================

This script demonstrates the complete functionality of the spinal cord model,
reproducing all results and figures from the manuscript.

Author: Sai Batchu
Institution: Cooper University Health Systems
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
    """Main demonstration function."""
    print("Spinal Cord Hyperflexion Model Demo")
    print("=" * 50)
    print()
    
    # Initialize model
    print("Initializing spinal cord model...")
    model = SpinalCordModel()
    model.print_parameters()
    
    # Initialize visualizer
    visualizer = SpinalModelVisualizer(model)
    
    # Demo 1: Quasi-static analysis
    print("\n" + "="*50)
    print("DEMO 1: Quasi-Static Analysis")
    print("="*50)
    
    # Calculate failure moment
    M_fail = model.get_quasi_static_failure_moment()
    print(f"Quasi-static failure moment: {M_fail:.1f} Nm")
    
    # Test various moments
    test_moments = [10, 20, 30, 40, 50]
    print("\nQuasi-static analysis results:")
    print("Moment (Nm) | Strain")
    print("-" * 20)
    
    for M in test_moments:
        # Solve for phi using Newton's method (simplified)
        phi_guess = M / (model.params['K_theta'] + model.params['k_c_r_c_squared'])
        M_calc, epsilon = model.quasi_static_analysis(phi_guess)
        print(f"{M:10.1f} | {epsilon:.4f}")
    
    # Demo 2: Dynamic simulation
    print("\n" + "="*50)
    print("DEMO 2: Dynamic Simulation")
    print("="*50)
    
    # Test different moment magnitudes
    M_test_values = [15, 20, 25, 30]
    
    for M_test in M_test_values:
        print(f"\nSimulating with M_ext = {M_test} Nm:")
        M_ext_func = model.step_input_moment(M_test)
        results = model.simulate_dynamics(M_ext_func, t_final=0.1)
        
        max_strain = np.max(results['epsilon'])
        print(f"  Maximum strain: {max_strain:.4f}")
        
        if results['failure_time'] is not None:
            print(f"  Failure predicted at: {results['failure_time']*1000:.1f} ms")
        else:
            print("  No failure predicted")
    
    # Demo 3: Sensitivity Analysis
    print("\n" + "="*50)
    print("DEMO 3: Sensitivity Analysis")
    print("="*50)
    
    # Test sensitivity to key parameters
    param_names = ['K_theta', 'k_c', 'C_theta']
    M_ext = 20.0
    
    for param_name in param_names:
        print(f"\nSensitivity to {param_name}:")
        nominal_value = model.params[param_name]
        variation_range = np.linspace(0.5 * nominal_value, 1.5 * nominal_value, 5)
        
        sensitivity_results = model.sensitivity_analysis(param_name, variation_range, M_ext)
        
        for i, param_val in enumerate(sensitivity_results['parameter_values']):
            max_strain = sensitivity_results['max_strains'][i]
            failure_time = sensitivity_results['failure_times'][i]
            
            status = "FAILURE" if failure_time is not None else "SAFE"
            print(f"  {param_val:8.3g} -> Max strain: {max_strain:.4f} ({status})")
    
    # Demo 4: Generate all figures
    print("\n" + "="*50)
    print("DEMO 4: Generating Figures")
    print("="*50)
    
    # Create figures directory
    os.makedirs("figures", exist_ok=True)
    
    # Generate all figures
    print("Generating quasi-static analysis figure...")
    fig1 = visualizer.plot_quasi_static_analysis("figures/quasi_static_analysis.png")
    
    print("Generating dynamic response figure...")
    fig2 = visualizer.plot_dynamic_response(save_path="figures/dynamic_response.png")
    
    print("Generating sensitivity analysis figure...")
    fig3 = visualizer.plot_sensitivity_analysis("figures/sensitivity_analysis.png")
    
    print("Generating model schematic...")
    fig4 = visualizer.plot_model_schematic("figures/model_schematic.png")
    
    print("\nAll figures saved to figures/ directory")
    
    # Demo 5: Validation
    print("\n" + "="*50)
    print("DEMO 5: Model Validation")
    print("="*50)
    
    print(visualizer.create_validation_table())
    
    # Demo 6: Advanced analysis
    print("\n" + "="*50)
    print("DEMO 6: Advanced Analysis")
    print("="*50)
    
    # Find critical moment for failure
    print("Finding critical moment for dynamic failure...")
    M_range = np.linspace(15, 35, 21)
    failure_moments = []
    
    for M in M_range:
        M_ext_func = model.step_input_moment(M)
        results = model.simulate_dynamics(M_ext_func, t_final=0.1)
        
        if results['failure_time'] is not None:
            failure_moments.append(M)
    
    if failure_moments:
        critical_moment = min(failure_moments)
        print(f"Critical moment for dynamic failure: {critical_moment:.1f} Nm")
        print(f"Quasi-static failure moment: {M_fail:.1f} Nm")
        print(f"Dynamic amplification factor: {M_fail/critical_moment:.2f}")
    else:
        print("No dynamic failure found in tested range")
    
    # Show plots
    print("\n" + "="*50)
    print("DEMO 7: Displaying Results")
    print("="*50)
    
    print("Displaying generated figures...")
    plt.show()
    
    print("\nDemo completed successfully!")
    print("\nFiles generated:")
    print("- figures/quasi_static_analysis.png")
    print("- figures/dynamic_response.png") 
    print("- figures/sensitivity_analysis.png")
    print("- figures/model_schematic.png")


def run_tests():
    """Run basic tests to verify model functionality."""
    print("Running model tests...")
    
    # Test 1: Parameter validation
    model = SpinalCordModel()
    assert model.params['k_c'] > 0, "Cord stiffness should be positive"
    assert model.params['phi_fail'] > 0, "Failure angle should be positive"
    
    # Test 2: Quasi-static analysis
    M_ext, epsilon = model.quasi_static_analysis(0.1)
    assert M_ext > 0, "External moment should be positive"
    assert epsilon > 0, "Strain should be positive"
    
    # Test 3: Dynamic simulation
    M_ext_func = model.step_input_moment(20.0)
    results = model.simulate_dynamics(M_ext_func, t_final=0.01)
    assert len(results['time']) > 0, "Simulation should produce results"
    assert np.all(results['epsilon'] >= 0), "Strain should be non-negative"
    
    print("All tests passed!")


if __name__ == "__main__":
    # Run tests first
    run_tests()
    
    # Run main demo
    main() 