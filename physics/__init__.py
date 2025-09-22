"""
Physics module for black hole simulation.

This module contains the physics simulation components:
- Physical constants and units
- 2D and 3D null geodesic calculations
- Numerical integration methods
- Schwarzschild metric implementations
"""

from .constants import *
from .geodesic2d import Ray2D
from .geodesic3d import Ray3D
from .integrators import RK4Integrator, RK45Integrator

__all__ = [
    "C",
    "G", 
    "SAGITTARIUS_A_MASS",
    "Ray2D",
    "Ray3D",
    "RK4Integrator",
    "RK45Integrator",
]
