"""
Spinal Cord Hyperflexion Model
==============================

A lumped-parameter model for simulating spinal cord injury during hyperflexion trauma.
This implementation matches the mathematical formulation described in the manuscript.

Author: Sai Batchu
Institution: Cooper University Health Systems
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, Any
import warnings


class SpinalCordModel:
    """
    Lumped-parameter model of spinal cord failure in hyperflexion trauma.
    
    This model integrates rigid vertebral dynamics with a viscoelastic representation
    of spinal soft tissues to investigate transient strain patterns during high-speed trauma.
    """
    
    def __init__(self, parameters: Optional[Dict[str, float]] = None):
        """
        Initialize the spinal cord model with parameters.
        
        Parameters
        ----------
        parameters : dict, optional
            Dictionary of model parameters. If None, uses default values from manuscript.
        """
        # Default parameters from manuscript (Table 1)
        self.default_params = {
            'K_theta': 60.0,           # Disc-ligament stiffness [Nm/rad]
            'C_theta': 0.12,           # Damping coefficient [Nms/rad]
            'E_c': 0.3e6,              # Cord modulus [Pa]
            'A_c': 3.2e-5,             # Cord area [m²]
            'L_0': 0.102,              # Cord rest length [m]
            'r_c': 0.019,              # Cord moment arm [m]
            'epsilon_fail': 0.10,      # Failure strain [-]
            'I_theta': 4.8e-4,        # Rotational inertia [kg·m²]
            'K_nl': 50.0               # Nonlinear stiffness [Nm/rad³]
        }
        
        # Set parameters
        if parameters is not None:
            self.params = {**self.default_params, **parameters}
        else:
            self.params = self.default_params.copy()
        
        # Calculate derived parameters
        self._calculate_derived_params()
        
    def _calculate_derived_params(self):
        """Calculate derived parameters from the base parameters."""
        # Cord axial stiffness: k_c = E_c * A_c / L_0
        self.params['k_c'] = self.params['E_c'] * self.params['A_c'] / self.params['L_0']
        
        # Cord moment contribution: k_c * r_c²
        self.params['k_c_r_c_squared'] = self.params['k_c'] * self.params['r_c']**2
        
        # Quasi-static failure angle
        self.params['phi_fail'] = (self.params['epsilon_fail'] * self.params['L_0'] / 
                                  self.params['r_c'])
        
    def quasi_static_analysis(self, phi: float) -> Tuple[float, float]:
        """
        Perform quasi-static analysis for a given flexion angle.
        
        Parameters
        ----------
        phi : float
            Flexion angle [rad]
            
        Returns
        -------
        tuple
            (external_moment, cord_strain)
        """
        # Calculate cord strain
        epsilon = self.params['r_c'] * phi / self.params['L_0']
        
        # Calculate external moment (equilibrium equation)
        M_ext = ((self.params['K_theta'] + self.params['k_c_r_c_squared']) * phi + 
                 self.params['K_nl'] * phi**3)
        
        return M_ext, epsilon
    
    def get_quasi_static_failure_moment(self) -> float:
        """
        Calculate the quasi-static failure moment.
        
        Returns
        -------
        float
            Failure moment [Nm]
        """
        M_fail, _ = self.quasi_static_analysis(self.params['phi_fail'])
        return M_fail
    
    def dynamic_equation(self, phi: float, phi_dot: float, M_ext: float) -> float:
        """
        Calculate the angular acceleration from the dynamic equation.
        
        Parameters
        ----------
        phi : float
            Flexion angle [rad]
        phi_dot : float
            Angular velocity [rad/s]
        M_ext : float
            External moment [Nm]
            
        Returns
        -------
        float
            Angular acceleration [rad/s²]
        """
        # Dynamic equation: I_theta * phi_ddot + C_theta * phi_dot + (K_theta + k_c*r_c²)*phi + K_nl*phi³ = M_ext
        phi_ddot = (M_ext - self.params['C_theta'] * phi_dot - 
                   (self.params['K_theta'] + self.params['k_c_r_c_squared']) * phi - 
                   self.params['K_nl'] * phi**3) / self.params['I_theta']
        
        return phi_ddot
    
    def rk4_step(self, t: float, phi: float, phi_dot: float, M_ext_func, dt: float) -> Tuple[float, float]:
        """
        Perform one step of fourth-order Runge-Kutta integration.
        
        Parameters
        ----------
        t : float
            Current time [s]
        phi : float
            Current flexion angle [rad]
        phi_dot : float
            Current angular velocity [rad/s]
        M_ext_func : callable
            Function that returns external moment at time t
        dt : float
            Time step [s]
            
        Returns
        -------
        tuple
            (new_phi, new_phi_dot)
        """
        
        # Calculate k1
        M_ext = M_ext_func(t)
        k1_phi = dt * phi_dot
        k1_phi_dot = dt * self.dynamic_equation(phi, phi_dot, M_ext)
        
        # Calculate k2
        M_ext_mid = M_ext_func(t + 0.5 * dt)
        k2_phi = dt * (phi_dot + 0.5 * k1_phi_dot)
        k2_phi_dot = dt * self.dynamic_equation(phi + 0.5 * k1_phi, 
                                               phi_dot + 0.5 * k1_phi_dot, 
                                               M_ext_mid)
        
        # Calculate k3
        k3_phi = dt * (phi_dot + 0.5 * k2_phi_dot)
        k3_phi_dot = dt * self.dynamic_equation(phi + 0.5 * k2_phi, 
                                               phi_dot + 0.5 * k2_phi_dot, 
                                               M_ext_mid)
        
        # Calculate k4
        M_ext_next = M_ext_func(t + dt)
        k4_phi = dt * (phi_dot + k3_phi_dot)
        k4_phi_dot = dt * self.dynamic_equation(phi + k3_phi, 
                                               phi_dot + k3_phi_dot, 
                                               M_ext_next)
        
        # Update state
        new_phi = phi + (k1_phi + 2*k2_phi + 2*k3_phi + k4_phi) / 6
        new_phi_dot = phi_dot + (k1_phi_dot + 2*k2_phi_dot + 2*k3_phi_dot + k4_phi_dot) / 6
        
        return new_phi, new_phi_dot
    
    def simulate_dynamics(self, M_ext_func, t_final: float = 0.1, dt: float = 0.0001,
                         initial_conditions: Optional[Tuple[float, float]] = None) -> Dict[str, np.ndarray]:
        """
        Simulate the dynamic response using RK4 integration.
        
        Parameters
        ----------
        M_ext_func : callable
            Function that returns external moment at time t
        t_final : float
            Final simulation time [s]
        dt : float
            Time step [s]
        initial_conditions : tuple, optional
            (initial_phi, initial_phi_dot). If None, uses (0, 0)
            
        Returns
        -------
        dict
            Dictionary containing time, phi, phi_dot, epsilon, and M_ext arrays
        """
        self.dt = dt
        
        # Initialize arrays
        n_steps = int(t_final / dt) + 1
        t_array = np.linspace(0, t_final, n_steps)
        phi_array = np.zeros(n_steps)
        phi_dot_array = np.zeros(n_steps)
        epsilon_array = np.zeros(n_steps)
        M_ext_array = np.zeros(n_steps)
        
        # Set initial conditions
        if initial_conditions is None:
            phi_array[0] = 0.0
            phi_dot_array[0] = 0.0
        else:
            phi_array[0], phi_dot_array[0] = initial_conditions
        
        # Calculate initial strain and moment
        epsilon_array[0] = self.params['r_c'] * phi_array[0] / self.params['L_0']
        try:
            M_ext_array[0] = M_ext_func(0.0)
        except Exception as e:
            # Fallback if function call fails
            M_ext_array[0] = M_ext_func(0) if callable(M_ext_func) else 0.0
        
        # Integration loop
        failure_time = None
        for i in range(1, n_steps):
            t = t_array[i-1]
            phi = phi_array[i-1]
            phi_dot = phi_dot_array[i-1]
            
            # RK4 step
            new_phi, new_phi_dot = self.rk4_step(t, phi, phi_dot, M_ext_func, dt)
            
            # Update arrays
            phi_array[i] = new_phi
            phi_dot_array[i] = new_phi_dot
            epsilon_array[i] = self.params['r_c'] * new_phi / self.params['L_0']
            try:
                M_ext_array[i] = M_ext_func(t_array[i])
            except Exception as e:
                # Fallback if function call fails
                M_ext_array[i] = M_ext_func(float(t_array[i])) if callable(M_ext_func) else 0.0
            
            # Check for failure
            if epsilon_array[i] >= self.params['epsilon_fail'] and failure_time is None:
                failure_time = t_array[i]
                warnings.warn(f"Cord failure predicted at t = {failure_time:.4f} s")
        
        return {
            'time': t_array,
            'phi': phi_array,
            'phi_dot': phi_dot_array,
            'epsilon': epsilon_array,
            'M_ext': M_ext_array,
            'failure_time': failure_time
        }
    
    def step_input_moment(self, M_0: float) -> callable:
        """
        Create a step input moment function.
        
        Parameters
        ----------
        M_0 : float
            Magnitude of the step moment [Nm]
            
        Returns
        -------
        callable
            Function that returns M_0 for all t >= 0
        """
        def step_func(t):
            """Step function that returns M_0 for t >= 0, 0 otherwise."""
            if isinstance(t, (int, float, np.number)):
                return M_0 if t >= 0 else 0.0
            else:
                return M_0  # Default to M_0 for non-numeric inputs
        return step_func
    
    def sensitivity_analysis(self, parameter_name: str, variation_range: np.ndarray,
                           M_ext: float = 20.0, t_final: float = 0.1) -> Dict[str, np.ndarray]:
        """
        Perform sensitivity analysis for a given parameter.
        
        Parameters
        ----------
        parameter_name : str
            Name of the parameter to vary
        variation_range : np.ndarray
            Array of parameter values to test
        M_ext : float
            External moment magnitude [Nm]
        t_final : float
            Final simulation time [s]
            
        Returns
        -------
        dict
            Dictionary containing parameter values and corresponding max strains
        """
        if parameter_name not in self.params:
            raise ValueError(f"Parameter '{parameter_name}' not found in model parameters")
        
        max_strains = []
        failure_times = []
        
        for param_value in variation_range:
            # Create modified parameters
            modified_params = self.params.copy()
            modified_params[parameter_name] = param_value
            
            # Handle k_c specially - modify E_c to achieve desired k_c
            if parameter_name == 'k_c':
                # Calculate required E_c to achieve desired k_c
                # k_c = E_c * A_c / L_0, so E_c = k_c * L_0 / A_c
                required_E_c = param_value * modified_params['L_0'] / modified_params['A_c']
                modified_params['E_c'] = required_E_c
            
            # Create new model instance
            modified_model = SpinalCordModel(modified_params)
            
            # Run simulation
            M_ext_func = modified_model.step_input_moment(M_ext)
            results = modified_model.simulate_dynamics(M_ext_func, t_final)
            
            # Record results
            max_strains.append(np.max(results['epsilon']))
            failure_times.append(results['failure_time'])
        
        return {
            'parameter_values': variation_range,
            'max_strains': np.array(max_strains),
            'failure_times': failure_times
        }
    
    def get_parameters(self) -> Dict[str, float]:
        """Get current model parameters."""
        return self.params.copy()
    
    def print_parameters(self):
        """Print current model parameters."""
        print("Spinal Cord Model Parameters:")
        print("=" * 40)
        for key, value in self.params.items():
            if isinstance(value, float):
                print(f"{key:15s}: {value:10.6g}")
            else:
                print(f"{key:15s}: {value}")
        print()
        print(f"Quasi-static failure moment: {self.get_quasi_static_failure_moment():.1f} Nm")
        print(f"Quasi-static failure angle: {self.params['phi_fail']:.3f} rad ({np.degrees(self.params['phi_fail']):.1f}°)") 