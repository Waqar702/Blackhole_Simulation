"""
Black Hole Simulation - Python Implementation

A Python implementation of black hole simulation featuring ray tracing,
gravitational lensing, and spacetime visualization.

This package provides:
- 2D and 3D null geodesic ray tracing
- GPU-accelerated rendering with OpenGL compute shaders
- Interactive camera controls and spacetime visualization
- Educational demos and examples

Author: Black Hole Simulation Team
Version: 0.1.0
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Black Hole Simulation Team"
__license__ = "MIT"

# Import main modules for easy access
from . import engine
from . import physics
from . import scene
from . import demos

__all__ = [
    "__version__",
    "__author__", 
    "__license__",
    "engine",
    "physics", 
    "scene",
    "demos",
]
