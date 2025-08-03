#!/usr/bin/env python3
import numpy as np
from src.spinal_model import SpinalCordModel

def debug_bar_values():
    model = SpinalCordModel()
    
    # Parameters to analyze
    param_names = ['K_theta', 'k_c', 'C_theta', 'r_c', 'epsilon_fail']
    
    M_ext = 30.0
    
    # Get baseline results
    baseline_M_ext_func = model.step_input_moment(M_ext)
    baseline_results = model.simulate_dynamics(baseline_M_ext_func, t_final=0.1)
    baseline_max_strain = np.max(baseline_results['epsilon'])
    baseline_failure_time = baseline_results['failure_time']
    
    print(f"Baseline max strain: {baseline_max_strain}")
    print(f"Baseline failure time: {baseline_failure_time}")
    
    for i, param_name in enumerate(param_names):
        print(f"\n=== {param_name} ===")
        
        # Create variation range (±50% from nominal)
        nominal_value = model.params[param_name]
        variation_range = [0.5 * nominal_value, 1.5 * nominal_value]
        
        # Run sensitivity analysis
        sensitivity_results = model.sensitivity_analysis(
            param_name, np.array(variation_range), M_ext
        )
        
        # Calculate percent changes relative to baseline
        strain_changes = []
        time_changes = []
        no_failure = []
        
        for j, param_val in enumerate(sensitivity_results['parameter_values']):
            max_strain = sensitivity_results['max_strains'][j]
            failure_time = sensitivity_results['failure_times'][j]
            
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
        
        print(f"Strain changes: {strain_changes}")
        print(f"Time changes: {time_changes}")
        print(f"No failure flags: {no_failure}")
        
        # Check scaling for k_c
        if i == 1:  # k_c parameter
            k_c_scale_factor = 100.0
            scaled_strain = [v * k_c_scale_factor for v in strain_changes]
            scaled_time = [v * k_c_scale_factor for v in time_changes]
            print(f"Scaled strain changes: {scaled_strain}")
            print(f"Scaled time changes: {scaled_time}")

if __name__ == "__main__":
    debug_bar_values()
