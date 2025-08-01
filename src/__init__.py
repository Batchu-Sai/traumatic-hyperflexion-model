"""
Spinal Cord Hyperflexion Model Package
=====================================

A lumped-parameter model for simulating spinal cord injury during hyperflexion trauma.

Author: Sai Batchu
Institution: Cooper University Health Systems
"""

from .spinal_model import SpinalCordModel
from .visualization import SpinalModelVisualizer

__version__ = "1.0.0"
__author__ = "Sai Batchu"
__email__ = "batchu-sai@cooperhealth.edu"

__all__ = ['SpinalCordModel', 'SpinalModelVisualizer'] 