"""
GPU compute pipeline for ray tracing.

This module provides the GPUComputePipeline class for managing
OpenGL compute shaders, uniform buffers, and texture rendering.
"""

import numpy as np
import moderngl
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class GPUComputePipeline:
    """Manages GPU compute shader pipeline for ray tracing."""
    
    def __init__(self, ctx: moderngl.Context):
        """
        Initialize GPU compute pipeline.
        
        Args:
            ctx: ModernGL context
        """
        self.ctx = ctx
        
        # Shaders
        self.compute_program: Optional[moderngl.Program] = None
        self.quad_program: Optional[moderngl.Program] = None
        
        # Buffers
        self.camera_ubo: Optional[moderngl.Buffer] = None
        self.disk_ubo: Optional[moderngl.Buffer] = None
        self.objects_ubo: Optional[moderngl.Buffer] = None
        
        # Textures and rendering
        self.output_texture: Optional[moderngl.Texture] = None
        self.quad_vao: Optional[moderngl.VertexArray] = None
        
        # Compute parameters
        self.compute_width = 200
        self.compute_height = 150
        self.render_width = 800
        self.render_height = 600
        
        # Workgroup size (matches GLSL local_size_x/y)
        self.workgroup_size = 16
    
    def create_compute_program(self, compute_source: str) -> None:
        """
        Create compute shader program.
        
        Args:
            compute_source: Compute shader source code
        """
        try:
            self.compute_program = self.ctx.compute_shader(compute_source)
            logger.info("Compute shader program created")
        except Exception as e:
            logger.error(f"Failed to create compute shader program: {e}")
            raise
    
    def create_quad_program(self, vertex_source: str, fragment_source: str) -> None:
        """
        Create screen quad rendering program.
        
        Args:
            vertex_source: Vertex shader source code
            fragment_source: Fragment shader source code
        """
        try:
            self.quad_program = self.ctx.program(
                vertex_shader=vertex_source,
                fragment_shader=fragment_source,
            )
            
            # Create screen quad
            self._create_screen_quad()
            
            logger.info("Quad rendering program created")
        except Exception as e:
            logger.error(f"Failed to create quad program: {e}")
            raise
    
    def _create_screen_quad(self) -> None:
        """Create full-screen quad for rendering compute results."""
        # Quad vertices (position + texture coordinates)
        quad_vertices = np.array([
            # positions   # texCoords
            -1.0,  1.0,  0.0, 1.0,  # top left
            -1.0, -1.0,  0.0, 0.0,  # bottom left
             1.0, -1.0,  1.0, 0.0,  # bottom right
            -1.0,  1.0,  0.0, 1.0,  # top left
             1.0, -1.0,  1.0, 0.0,  # bottom right
             1.0,  1.0,  1.0, 1.0   # top right
        ], dtype=np.float32)
        
        # Create vertex buffer
        vbo = self.ctx.buffer(quad_vertices.tobytes())
        
        # Create vertex array
        self.quad_vao = self.ctx.vertex_array(
            self.quad_program,
            [(vbo, '2f 2f', 'aPos', 'aTexCoord')],
        )
    
    def create_buffers(self) -> None:
        """Create uniform buffer objects."""
        try:
            # Camera UBO (128 bytes - matches GLSL layout)
            self.camera_ubo = self.ctx.buffer(reserve=128)
            
            # Disk UBO (16 bytes - 4 floats)
            self.disk_ubo = self.ctx.buffer(reserve=16)
            
            # Objects UBO (large buffer for multiple objects)
            # sizeof(int) + padding + 16×(vec4 posRadius + vec4 color) + 16×float mass
            objects_size = 4 + 12 + 16 * (16 + 16) + 16 * 4  # ~560 bytes
            self.objects_ubo = self.ctx.buffer(reserve=objects_size)
            
            logger.info("Uniform buffers created")
        except Exception as e:
            logger.error(f"Failed to create uniform buffers: {e}")
            raise
    
    def create_output_texture(self, width: int, height: int) -> None:
        """
        Create output texture for compute shader.
        
        Args:
            width: Texture width
            height: Texture height
        """
        try:
            if self.output_texture:
                self.output_texture.release()
            
            self.output_texture = self.ctx.texture(
                (width, height),
                4,  # RGBA
                dtype='f1',
            )
            
            # Set texture parameters
            self.output_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            
            self.compute_width = width
            self.compute_height = height
            
            logger.info(f"Output texture created: {width}x{height}")
        except Exception as e:
            logger.error(f"Failed to create output texture: {e}")
            raise
    
    def update_camera_ubo(
        self,
        position: np.ndarray,
        right: np.ndarray,
        up: np.ndarray,
        forward: np.ndarray,
        tan_half_fov: float,
        aspect: float,
        moving: bool,
    ) -> None:
        """
        Update camera uniform buffer.
        
        Args:
            position: Camera position
            right: Camera right vector
            up: Camera up vector
            forward: Camera forward vector
            tan_half_fov: Tangent of half field of view
            aspect: Aspect ratio
            moving: Whether camera is moving
        """
        if not self.camera_ubo:
            return
        
        # Pack data to match GLSL std140 layout
        data = np.zeros(32, dtype=np.float32)  # 32 floats = 128 bytes
        
        # Position (vec3 + padding)
        data[0:3] = position.astype(np.float32)
        
        # Right vector (vec3 + padding)
        data[4:7] = right.astype(np.float32)
        
        # Up vector (vec3 + padding)
        data[8:11] = up.astype(np.float32)
        
        # Forward vector (vec3 + padding)
        data[12:15] = forward.astype(np.float32)
        
        # tanHalfFov, aspect, moving, padding
        data[16] = tan_half_fov
        data[17] = aspect
        data[18] = 1.0 if moving else 0.0
        
        # Upload to GPU
        self.camera_ubo.write(data.tobytes())
    
    def update_disk_ubo(
        self,
        inner_radius: float,
        outer_radius: float,
        num_rays: float,
        thickness: float,
    ) -> None:
        """
        Update disk uniform buffer.
        
        Args:
            inner_radius: Inner disk radius
            outer_radius: Outer disk radius
            num_rays: Number of rays
            thickness: Disk thickness
        """
        if not self.disk_ubo:
            return
        
        data = np.array([inner_radius, outer_radius, num_rays, thickness], dtype=np.float32)
        self.disk_ubo.write(data.tobytes())
    
    def update_objects_ubo(self, objects: List[dict]) -> None:
        """
        Update objects uniform buffer.
        
        Args:
            objects: List of objects with position, color, mass
        """
        if not self.objects_ubo:
            return
        
        # Limit to 16 objects (matches shader)
        count = min(len(objects), 16)
        
        # Create buffer data
        data = np.zeros(140, dtype=np.float32)  # Large enough for 16 objects
        
        # Number of objects + padding
        data[0] = float(count)
        
        # Object data
        for i in range(count):
            obj = objects[i]
            
            # Position and radius (vec4)
            pos = obj.get('position', [0.0, 0.0, 0.0])
            radius = obj.get('radius', 1.0)
            data[4 + i*4:8 + i*4] = [pos[0], pos[1], pos[2], radius]
            
            # Color (vec4)
            color = obj.get('color', [1.0, 1.0, 1.0, 1.0])
            data[68 + i*4:72 + i*4] = color
            
            # Mass (float)
            mass = obj.get('mass', 1.0)
            data[132 + i] = mass
        
        self.objects_ubo.write(data.tobytes())
    
    def dispatch_compute(self, width: int, height: int) -> None:
        """
        Dispatch compute shader.
        
        Args:
            width: Compute resolution width
            height: Compute resolution height
        """
        if not self.compute_program or not self.output_texture:
            return
        
        # Resize texture if needed
        if width != self.compute_width or height != self.compute_height:
            self.create_output_texture(width, height)
        
        # Bind uniform buffers
        if self.camera_ubo:
            self.camera_ubo.bind_to_uniform_block(1)  # binding = 1
        if self.disk_ubo:
            self.disk_ubo.bind_to_uniform_block(2)    # binding = 2
        if self.objects_ubo:
            self.objects_ubo.bind_to_uniform_block(3) # binding = 3
        
        # Bind output texture as image
        self.output_texture.bind_to_image(0, 0, write=True)
        
        # Calculate workgroups
        groups_x = (width + self.workgroup_size - 1) // self.workgroup_size
        groups_y = (height + self.workgroup_size - 1) // self.workgroup_size
        
        # Dispatch compute shader
        self.compute_program.run(groups_x, groups_y, 1)
        
        # Wait for completion
        self.ctx.finish()
    
    def render_quad(self) -> None:
        """Render compute results to screen."""
        if not self.quad_program or not self.quad_vao or not self.output_texture:
            return
        
        # Disable depth test for full-screen quad
        self.ctx.disable(moderngl.DEPTH_TEST)
        
        # Bind output texture
        self.output_texture.use(0)
        self.quad_program['screenTexture'].value = 0
        
        # Render quad
        self.quad_vao.render()
        
        # Restore depth test
        self.ctx.enable(moderngl.DEPTH_TEST)
    
    def set_render_resolution(self, width: int, height: int) -> None:
        """
        Set render resolution.
        
        Args:
            width: Render width
            height: Render height
        """
        self.render_width = width
        self.render_height = height
    
    def cleanup(self) -> None:
        """Clean up GPU resources."""
        if self.camera_ubo:
            self.camera_ubo.release()
        if self.disk_ubo:
            self.disk_ubo.release()
        if self.objects_ubo:
            self.objects_ubo.release()
        if self.output_texture:
            self.output_texture.release()
        if self.quad_vao:
            self.quad_vao.release()
        if self.compute_program:
            self.compute_program.release()
        if self.quad_program:
            self.quad_program.release()
        
        self.camera_ubo = None
        self.disk_ubo = None
        self.objects_ubo = None
        self.output_texture = None
        self.quad_vao = None
        self.compute_program = None
        self.quad_program = None
        
        logger.info("GPU compute pipeline cleaned up")
