"""
Engine module for black hole simulation.

This module contains the core rendering engine components:
- Window management and OpenGL context
- Shader compilation and management
- Camera system with orbit controls
- GPU compute pipeline
- Spacetime grid rendering
"""

from .window import Window
from .shaders import ShaderManager
from .camera import Camera
from .grid import GridRenderer
from .gpu import GPUComputePipeline

__all__ = [
    "Window",
    "ShaderManager", 
    "Camera",
    "GridRenderer",
    "GPUComputePipeline",
]
