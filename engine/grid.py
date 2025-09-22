"""
Spacetime grid rendering.

This module provides the GridRenderer class for generating and rendering
a warped spacetime grid that visualizes the curvature effects around
the black hole.
"""

import numpy as np
import moderngl
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class GridRenderer:
    """Renders a warped spacetime grid showing gravitational curvature."""
    
    def __init__(self, ctx: moderngl.Context):
        """
        Initialize grid renderer.
        
        Args:
            ctx: ModernGL context
        """
        self.ctx = ctx
        self.vao: Optional[moderngl.VertexArray] = None
        self.vbo: Optional[moderngl.Buffer] = None
        self.ebo: Optional[moderngl.Buffer] = None
        self.program: Optional[moderngl.Program] = None
        self.index_count = 0
        
        # Grid parameters
        self.grid_size = 25
        self.spacing = 1e10
        self.color = (0.5, 0.5, 0.5, 0.7)  # Translucent gray
        
        # Physical constants
        self.c = 299792458.0  # Speed of light
        self.G = 6.67430e-11  # Gravitational constant
    
    def create_program(self, vertex_source: str, fragment_source: str) -> None:
        """
        Create the grid rendering shader program.
        
        Args:
            vertex_source: Vertex shader source code
            fragment_source: Fragment shader source code
        """
        try:
            self.program = self.ctx.program(
                vertex_shader=vertex_source,
                fragment_shader=fragment_source,
            )
            logger.info("Grid shader program created")
        except Exception as e:
            logger.error(f"Failed to create grid shader program: {e}")
            raise
    
    def generate_grid(self, objects: List[dict]) -> None:
        """
        Generate grid vertices warped by Schwarzschild geometry.
        
        Args:
            objects: List of objects with position, mass, and radius
        """
        vertices = []
        indices = []
        
        # Generate grid vertices
        for z in range(self.grid_size + 1):
            for x in range(self.grid_size + 1):
                # Calculate world position
                world_x = (x - self.grid_size / 2) * self.spacing
                world_z = (z - self.grid_size / 2) * self.spacing
                
                # Initial height (flat plane)
                y = 0.0
                
                # Apply Schwarzschild geometry warping
                for obj in objects:
                    obj_pos = np.array(obj.get('position', [0.0, 0.0, 0.0]))
                    mass = obj.get('mass', 0.0)
                    
                    if mass <= 0:
                        continue
                    
                    # Calculate Schwarzschild radius
                    r_s = 2.0 * self.G * mass / (self.c * self.c)
                    
                    # Distance from object center
                    dx = world_x - obj_pos[0]
                    dz = world_z - obj_pos[2]
                    dist = np.sqrt(dx * dx + dz * dz)
                    
                    # Apply warping
                    if dist > r_s:
                        # Outside event horizon - apply curvature
                        delta_y = 2.0 * np.sqrt(r_s * (dist - r_s))
                        y += delta_y - 3e10  # Offset for visual clarity
                    else:
                        # Inside or at event horizon - create deep pit
                        y += 2.0 * np.sqrt(r_s * r_s) - 3e10
                
                vertices.extend([world_x, y, world_z])
        
        # Generate line indices
        for z in range(self.grid_size):
            for x in range(self.grid_size):
                i = z * (self.grid_size + 1) + x
                
                # Horizontal lines
                indices.extend([i, i + 1])
                
                # Vertical lines
                indices.extend([i, i + self.grid_size + 1])
        
        # Convert to numpy arrays
        vertices_array = np.array(vertices, dtype=np.float32)
        indices_array = np.array(indices, dtype=np.uint32)
        
        # Create buffers
        if self.vbo is not None:
            self.vbo.release()
        if self.ebo is not None:
            self.ebo.release()
        if self.vao is not None:
            self.vao.release()
        
        self.vbo = self.ctx.buffer(vertices_array.tobytes())
        self.ebo = self.ctx.buffer(indices_array.tobytes())
        
        # Create vertex array
        self.vao = self.ctx.vertex_array(
            self.program,
            [(self.vbo, '3f', 'aPos')],
            self.ebo,
        )
        
        self.index_count = len(indices)
        
        logger.info(f"Generated grid with {len(vertices)//3} vertices and {self.index_count} indices")
    
    def render(self, view_proj_matrix: np.ndarray) -> None:
        """
        Render the grid.
        
        Args:
            view_proj_matrix: 4x4 view-projection matrix
        """
        if not self.vao or not self.program:
            return
        
        # Enable blending for transparency
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        
        # Disable depth test for grid overlay
        self.ctx.disable(moderngl.DEPTH_TEST)
        
        # Set uniforms
        self.program['viewProj'].write(view_proj_matrix.astype(np.float32).tobytes())
        self.program['gridColor'].value = self.color
        
        # Render grid lines
        self.vao.render(moderngl.LINES)
        
        # Restore state
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)
    
    def set_grid_size(self, size: int) -> None:
        """
        Set grid resolution.
        
        Args:
            size: Number of grid cells per dimension
        """
        self.grid_size = size
    
    def set_spacing(self, spacing: float) -> None:
        """
        Set grid spacing.
        
        Args:
            spacing: Distance between grid points
        """
        self.spacing = spacing
    
    def set_color(self, color: Tuple[float, float, float, float]) -> None:
        """
        Set grid color.
        
        Args:
            color: RGBA color tuple (0.0 to 1.0)
        """
        self.color = color
    
    def cleanup(self) -> None:
        """Clean up grid resources."""
        if self.vao:
            self.vao.release()
        if self.vbo:
            self.vbo.release()
        if self.ebo:
            self.ebo.release()
        if self.program:
            self.program.release()
        
        self.vao = None
        self.vbo = None
        self.ebo = None
        self.program = None
        
        logger.info("Grid renderer cleaned up")
