"""
Scene management module for black hole simulation.

This module contains scene objects, accretion disk, and uniform buffer management:
- Scene objects with position, color, and mass
- Accretion disk geometry and properties
- Uniform buffer object management for GPU
"""

from .objects import SceneObject, SceneManager
from .disk import AccretionDisk
from .ubos import UBOManager

__all__ = [
    "SceneObject",
    "SceneManager",
    "AccretionDisk",
    "UBOManager",
]
