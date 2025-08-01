"""
Test suite for the Spinal Cord Hyperflexion Model
================================================

This module contains comprehensive tests for all model functionality.

Author: Sai Batchu
"""

import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.spinal_model import SpinalCordModel
from src.visualization import SpinalModelVisualizer


class TestSpinalCordModel(unittest.TestCase):
    """Test cases for the SpinalCordModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = SpinalCordModel()
    
    def test_parameter_initialization(self):
        """Test that all parameters are properly initialized."""
        required_params = [
            'K_theta', 'C_theta', 'E_c', 'A_c', 'L_0', 'r_c', 
            'epsilon_fail', 'I_theta', 'K_nl', 'k_c', 'k_c_r_c_squared', 'phi_fail'
        ]
        
        for param in required_params:
            self.assertIn(param, self.model.params)
            self.assertIsNotNone(self.model.params[param])
    
    def test_derived_parameters(self):
        """Test that derived parameters are calculated correctly."""
        # Test cord stiffness calculation
        expected_k_c = self.model.params['E_c'] * self.model.params['A_c'] / self.model.params['L_0']
        self.assertAlmostEqual(self.model.params['k_c'], expected_k_c, places=6)
        
        # Test cord moment contribution
        expected_k_c_r_c_squared = self.model.params['k_c'] * self.model.params['r_c']**2
        self.assertAlmostEqual(self.model.params['k_c_r_c_squared'], expected_k_c_r_c_squared, places=6)
        
        # Test failure angle
        expected_phi_fail = (self.model.params['epsilon_fail'] * self.model.params['L_0'] / 
                           self.model.params['r_c'])
        self.assertAlmostEqual(self.model.params['phi_fail'], expected_phi_fail, places=6)
    
    def test_quasi_static_analysis(self):
        """Test quasi-static analysis functionality."""
        # Test at zero angle
        M_ext, epsilon = self.model.quasi_static_analysis(0.0)
        self.assertEqual(M_ext, 0.0)
        self.assertEqual(epsilon, 0.0)
        
        # Test at small angle
        phi = 0.1
        M_ext, epsilon = self.model.quasi_static_analysis(phi)
        self.assertGreater(M_ext, 0.0)
        self.assertGreater(epsilon, 0.0)
        
        # Test strain calculation
        expected_epsilon = self.model.params['r_c'] * phi / self.model.params['L_0']
        self.assertAlmostEqual(epsilon, expected_epsilon, places=6)
    
    def test_quasi_static_failure_moment(self):
        """Test failure moment calculation."""
        M_fail = self.model.get_quasi_static_failure_moment()
        self.assertGreater(M_fail, 0.0)
        
        # Verify that this moment produces the failure strain
        M_ext, epsilon = self.model.quasi_static_analysis(self.model.params['phi_fail'])
        self.assertAlmostEqual(M_ext, M_fail, places=1)
        self.assertAlmostEqual(epsilon, self.model.params['epsilon_fail'], places=6)
    
    def test_dynamic_equation(self):
        """Test the dynamic equation calculation."""
        phi = 0.1
        phi_dot = 0.5
        M_ext = 20.0
        
        phi_ddot = self.model.dynamic_equation(phi, phi_dot, M_ext)
        self.assertIsInstance(phi_ddot, float)
    
    def test_rk4_step(self):
        """Test RK4 integration step."""
        t = 0.0
        phi = 0.1
        phi_dot = 0.5
        M_ext_func = self.model.step_input_moment(20.0)
        
        new_phi, new_phi_dot = self.model.rk4_step(t, phi, phi_dot, M_ext_func, 0.0001)
        
        self.assertIsInstance(new_phi, float)
        self.assertIsInstance(new_phi_dot, float)
        self.assertNotEqual(new_phi, phi)  # Should change
        self.assertNotEqual(new_phi_dot, phi_dot)  # Should change
    
    def test_simulation_dynamics(self):
        """Test dynamic simulation."""
        M_ext_func = self.model.step_input_moment(20.0)
        results = self.model.simulate_dynamics(M_ext_func, t_final=0.01)
        
        # Check that all required keys are present
        required_keys = ['time', 'phi', 'phi_dot', 'epsilon', 'M_ext', 'failure_time']
        for key in required_keys:
            self.assertIn(key, results)
        
        # Check array lengths
        self.assertEqual(len(results['time']), len(results['phi']))
        self.assertEqual(len(results['time']), len(results['phi_dot']))
        self.assertEqual(len(results['time']), len(results['epsilon']))
        self.assertEqual(len(results['time']), len(results['M_ext']))
        
        # Check that time starts at 0
        self.assertEqual(results['time'][0], 0.0)
        
        # Check that strain is non-negative
        self.assertTrue(np.all(results['epsilon'] >= 0))
    
    def test_step_input_moment(self):
        """Test step input moment function."""
        M_0 = 25.0
        M_ext_func = self.model.step_input_moment(M_0)
        
        # Test at negative time
        self.assertEqual(M_ext_func(-1.0), 0.0)
        
        # Test at zero time
        self.assertEqual(M_ext_func(0.0), M_0)
        
        # Test at positive time
        self.assertEqual(M_ext_func(1.0), M_0)
    
    def test_sensitivity_analysis(self):
        """Test sensitivity analysis functionality."""
        param_name = 'K_theta'
        variation_range = np.linspace(30.0, 90.0, 5)
        
        results = self.model.sensitivity_analysis(param_name, variation_range, M_ext=20.0)
        
        # Check required keys
        required_keys = ['parameter_values', 'max_strains', 'failure_times']
        for key in required_keys:
            self.assertIn(key, results)
        
        # Check array lengths
        self.assertEqual(len(results['parameter_values']), len(results['max_strains']))
        self.assertEqual(len(results['parameter_values']), len(results['failure_times']))
        
        # Check that max strains are non-negative
        self.assertTrue(np.all(results['max_strains'] >= 0))


class TestSpinalModelVisualizer(unittest.TestCase):
    """Test cases for the SpinalModelVisualizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = SpinalCordModel()
        self.visualizer = SpinalModelVisualizer(self.model)
    
    def test_visualizer_initialization(self):
        """Test visualizer initialization."""
        self.assertIsNotNone(self.visualizer.model)
        self.assertEqual(self.visualizer.model, self.model)
    
    def test_quasi_static_plot(self):
        """Test quasi-static analysis plot generation."""
        fig = self.visualizer.plot_quasi_static_analysis()
        self.assertIsNotNone(fig)
        self.assertTrue(hasattr(fig, 'savefig'))
    
    def test_dynamic_response_plot(self):
        """Test dynamic response plot generation."""
        fig = self.visualizer.plot_dynamic_response()
        self.assertIsNotNone(fig)
        self.assertTrue(hasattr(fig, 'savefig'))
    
    def test_sensitivity_analysis_plot(self):
        """Test sensitivity analysis plot generation."""
        fig = self.visualizer.plot_sensitivity_analysis()
        self.assertIsNotNone(fig)
        self.assertTrue(hasattr(fig, 'savefig'))
    
    def test_model_schematic_plot(self):
        """Test model schematic plot generation."""
        fig = self.visualizer.plot_model_schematic()
        self.assertIsNotNone(fig)
        self.assertTrue(hasattr(fig, 'savefig'))
    
    def test_validation_table(self):
        """Test validation table generation."""
        table = self.visualizer.create_validation_table()
        self.assertIsInstance(table, str)
        self.assertIn("Model Prediction", table)
        self.assertIn("Experimental Data", table)


class TestModelValidation(unittest.TestCase):
    """Test cases for model validation against literature data."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = SpinalCordModel()
    
    def test_failure_strain_validation(self):
        """Test that failure strain matches literature value."""
        # Literature value: 0.10 (Yamada 1970)
        expected_epsilon_fail = 0.10
        actual_epsilon_fail = self.model.params['epsilon_fail']
        self.assertAlmostEqual(actual_epsilon_fail, expected_epsilon_fail, places=2)
    
    def test_failure_moment_validation(self):
        """Test that failure moment falls within experimental range."""
        # Literature range: 23-53 Nm (Lopez-Valdes 2011)
        M_fail = self.model.get_quasi_static_failure_moment()
        self.assertGreaterEqual(M_fail, 20.0)  # Should be reasonable
        self.assertLessEqual(M_fail, 60.0)     # Should be reasonable
    
    def test_parameter_sources(self):
        """Test that parameters match literature sources."""
        # Test key parameters
        self.assertEqual(self.model.params['K_theta'], 60.0)  # Wilke et al. 2017
        self.assertEqual(self.model.params['epsilon_fail'], 0.10)  # Yamada 1970
        self.assertEqual(self.model.params['E_c'], 0.3e6)  # Bartlett et al. 2016


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2) 