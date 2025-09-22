"""
Uniform Buffer Object management.

This module provides the UBOManager class for managing uniform buffers
that are used to pass data to GPU shaders.
"""

import numpy as np
import moderngl
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class UBOManager:
    """Manages uniform buffer objects for GPU rendering."""
    
    def __init__(self, ctx: moderngl.Context):
        """
        Initialize UBO manager.
        
        Args:
            ctx: ModernGL context
        """
        self.ctx = ctx
        self.buffers: Dict[str, moderngl.Buffer] = {}
        self.buffer_sizes: Dict[str, int] = {}
        
        # Standard buffer bindings
        self.bindings = {
            'camera': 1,
            'disk': 2,
            'objects': 3,
        }
        
        self._create_buffers()
    
    def _create_buffers(self) -> None:
        """Create standard uniform buffers."""
        try:
            # Camera UBO (128 bytes - matches GLSL std140 layout)
            self.create_buffer('camera', 128)
            
            # Disk UBO (16 bytes - 4 floats)
            self.create_buffer('disk', 16)
            
            # Objects UBO (large buffer for multiple objects)
            # sizeof(int) + padding + 16×(vec4 posRadius + vec4 color) + 16×float mass
            objects_size = 4 + 12 + 16 * (16 + 16) + 16 * 4  # ~560 bytes
            self.create_buffer('objects', objects_size)
            
            logger.info("Standard uniform buffers created")
        except Exception as e:
            logger.error(f"Failed to create uniform buffers: {e}")
            raise
    
    def create_buffer(self, name: str, size: int) -> None:
        """
        Create a uniform buffer.
        
        Args:
            name: Buffer name
            size: Buffer size in bytes
        """
        if name in self.buffers:
            self.buffers[name].release()
        
        self.buffers[name] = self.ctx.buffer(reserve=size)
        self.buffer_sizes[name] = size
        
        logger.debug(f"Created UBO '{name}' with size {size} bytes")
    
    def get_buffer(self, name: str) -> Optional[moderngl.Buffer]:
        """
        Get a uniform buffer by name.
        
        Args:
            name: Buffer name
            
        Returns:
            Buffer or None if not found
        """
        return self.buffers.get(name)
    
    def bind_buffer(self, name: str, binding: int) -> bool:
        """
        Bind a buffer to a uniform block.
        
        Args:
            name: Buffer name
            binding: Binding point
            
        Returns:
            True if successful
        """
        if name not in self.buffers:
            logger.error(f"Buffer '{name}' not found")
            return False
        
        try:
            self.buffers[name].bind_to_uniform_block(binding)
            logger.debug(f"Bound buffer '{name}' to binding {binding}")
            return True
        except Exception as e:
            logger.error(f"Failed to bind buffer '{name}': {e}")
            return False
    
    def bind_standard_buffers(self) -> None:
        """Bind all standard buffers to their default binding points."""
        for name, binding in self.bindings.items():
            self.bind_buffer(name, binding)
    
    def update_camera_buffer(
        self,
        position: Tuple[float, float, float],
        right: Tuple[float, float, float],
        up: Tuple[float, float, float],
        forward: Tuple[float, float, float],
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
        if 'camera' not in self.buffers:
            logger.error("Camera buffer not found")
            return
        
        # Pack data to match GLSL std140 layout
        data = np.zeros(32, dtype=np.float32)  # 32 floats = 128 bytes
        
        # Position (vec3 + padding)
        data[0:3] = position
        
        # Right vector (vec3 + padding)
        data[4:7] = right
        
        # Up vector (vec3 + padding)
        data[8:11] = up
        
        # Forward vector (vec3 + padding)
        data[12:15] = forward
        
        # tanHalfFov, aspect, moving, padding
        data[16] = tan_half_fov
        data[17] = aspect
        data[18] = 1.0 if moving else 0.0
        
        # Upload to GPU
        self.buffers['camera'].write(data.tobytes())
    
    def update_disk_buffer(
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
        if 'disk' not in self.buffers:
            logger.error("Disk buffer not found")
            return
        
        data = np.array([inner_radius, outer_radius, num_rays, thickness], dtype=np.float32)
        self.buffers['disk'].write(data.tobytes())
    
    def update_objects_buffer(self, objects: List[Dict[str, Any]]) -> None:
        """
        Update objects uniform buffer.
        
        Args:
            objects: List of objects with position, color, mass
        """
        if 'objects' not in self.buffers:
            logger.error("Objects buffer not found")
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
        
        self.buffers['objects'].write(data.tobytes())
    
    def update_buffer(self, name: str, data: np.ndarray) -> bool:
        """
        Update a buffer with raw data.
        
        Args:
            name: Buffer name
            data: Data to write
            
        Returns:
            True if successful
        """
        if name not in self.buffers:
            logger.error(f"Buffer '{name}' not found")
            return False
        
        if name not in self.buffer_sizes:
            logger.error(f"Buffer size not known for '{name}'")
            return False
        
        expected_size = self.buffer_sizes[name]
        actual_size = data.nbytes
        
        if actual_size > expected_size:
            logger.error(f"Data size {actual_size} exceeds buffer size {expected_size}")
            return False
        
        try:
            self.buffers[name].write(data.tobytes())
            logger.debug(f"Updated buffer '{name}' with {actual_size} bytes")
            return True
        except Exception as e:
            logger.error(f"Failed to update buffer '{name}': {e}")
            return False
    
    def get_buffer_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all buffers.
        
        Returns:
            Dictionary with buffer information
        """
        info = {}
        for name, buffer in self.buffers.items():
            info[name] = {
                'size': self.buffer_sizes.get(name, 0),
                'binding': self.bindings.get(name, -1),
                'exists': buffer is not None,
            }
        return info
    
    def cleanup(self) -> None:
        """Clean up all uniform buffers."""
        for name, buffer in self.buffers.items():
            if buffer:
                buffer.release()
        
        self.buffers.clear()
        self.buffer_sizes.clear()
        
        logger.info("UBO manager cleaned up")
