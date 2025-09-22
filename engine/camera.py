"""
Camera system with orbit controls.

This module provides the Camera class for managing 3D camera positioning,
orbit controls around a target point, and view matrix calculations.
"""

import math
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class Camera:
    """3D camera with orbit controls around a target point."""
    
    def __init__(
        self,
        target: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        radius: float = 6.34194e10,
        azimuth: float = 0.0,
        elevation: float = math.pi / 2.0,
        fov: float = 60.0,
        near: float = 1e9,
        far: float = 1e14,
    ):
        """
        Initialize the camera.
        
        Args:
            target: Target point to orbit around (x, y, z)
            radius: Distance from target
            azimuth: Horizontal angle in radians
            elevation: Vertical angle in radians (0 = up, π = down)
            fov: Field of view in degrees
            near: Near clipping plane distance
            far: Far clipping plane distance
        """
        self.target = np.array(target, dtype=np.float64)
        self.radius = radius
        self.azimuth = azimuth
        self.elevation = elevation
        self.fov = fov
        self.near = near
        self.far = far
        
        # Constraints
        self.min_radius = 1e10
        self.max_radius = 1e12
        self.min_elevation = 0.01
        self.max_elevation = math.pi - 0.01
        
        # Control parameters
        self.orbit_speed = 0.01
        self.zoom_speed = 25e9
        self.pan_speed = 0.01
        
        # State
        self.dragging = False
        self.panning = False
        self.moving = False
        self.last_x = 0.0
        self.last_y = 0.0
        
        # Update position
        self._update_position()
    
    def _update_position(self) -> None:
        """Update camera position based on spherical coordinates."""
        # Clamp elevation to avoid gimbal lock
        self.elevation = np.clip(self.elevation, self.min_elevation, self.max_elevation)
        
        # Calculate position in world space
        x = self.radius * math.sin(self.elevation) * math.cos(self.azimuth)
        y = self.radius * math.cos(self.elevation)
        z = self.radius * math.sin(self.elevation) * math.sin(self.azimuth)
        
        self.position = np.array([x, y, z], dtype=np.float64)
        
        # Update movement state
        self.moving = self.dragging or self.panning
    
    def get_position(self) -> np.ndarray:
        """
        Get camera position.
        
        Returns:
            Camera position as numpy array
        """
        return self.position.copy()
    
    def get_target(self) -> np.ndarray:
        """
        Get camera target.
        
        Returns:
            Camera target as numpy array
        """
        return self.target.copy()
    
    def get_view_matrix(self) -> np.ndarray:
        """
        Get view matrix.
        
        Returns:
            4x4 view matrix as numpy array
        """
        # Calculate camera basis vectors
        forward = self.target - self.position
        forward = forward / np.linalg.norm(forward)
        
        # Use world up vector (0, 1, 0) as reference
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        # Build view matrix (look-at matrix)
        view_matrix = np.eye(4, dtype=np.float64)
        
        # Translation part
        view_matrix[0, 3] = -np.dot(right, self.position)
        view_matrix[1, 3] = -np.dot(up, self.position)
        view_matrix[2, 3] = -np.dot(forward, self.position)
        
        # Rotation part
        view_matrix[0, 0] = right[0]
        view_matrix[1, 0] = right[1]
        view_matrix[2, 0] = right[2]
        
        view_matrix[0, 1] = up[0]
        view_matrix[1, 1] = up[1]
        view_matrix[2, 1] = up[2]
        
        view_matrix[0, 2] = -forward[0]
        view_matrix[1, 2] = -forward[1]
        view_matrix[2, 2] = -forward[2]
        
        return view_matrix
    
    def get_projection_matrix(self, aspect_ratio: float) -> np.ndarray:
        """
        Get projection matrix.
        
        Args:
            aspect_ratio: Width / height ratio
            
        Returns:
            4x4 projection matrix as numpy array
        """
        fov_rad = math.radians(self.fov)
        tan_half_fov = math.tan(fov_rad / 2.0)
        
        proj_matrix = np.zeros((4, 4), dtype=np.float64)
        
        # Perspective projection matrix
        proj_matrix[0, 0] = 1.0 / (aspect_ratio * tan_half_fov)
        proj_matrix[1, 1] = 1.0 / tan_half_fov
        proj_matrix[2, 2] = -(self.far + self.near) / (self.far - self.near)
        proj_matrix[2, 3] = -1.0
        proj_matrix[3, 2] = -(2.0 * self.far * self.near) / (self.far - self.near)
        
        return proj_matrix
    
    def get_view_projection_matrix(self, aspect_ratio: float) -> np.ndarray:
        """
        Get combined view-projection matrix.
        
        Args:
            aspect_ratio: Width / height ratio
            
        Returns:
            4x4 view-projection matrix as numpy array
        """
        view_matrix = self.get_view_matrix()
        proj_matrix = self.get_projection_matrix(aspect_ratio)
        return proj_matrix @ view_matrix
    
    def process_mouse_move(self, x: float, y: float) -> None:
        """
        Process mouse movement for camera controls.
        
        Args:
            x: Mouse x position
            y: Mouse y position
        """
        if not self.dragging:
            self.last_x = x
            self.last_y = y
            return
        
        dx = x - self.last_x
        dy = y - self.last_y
        
        if self.panning:
            # Pan: move target in camera plane
            # (Disabled in original C++ code to keep centered on black hole)
            pass
        else:
            # Orbit: change azimuth and elevation
            self.azimuth += dx * self.orbit_speed
            self.elevation -= dy * self.orbit_speed
            self.elevation = np.clip(self.elevation, self.min_elevation, self.max_elevation)
        
        self.last_x = x
        self.last_y = y
        self._update_position()
    
    def process_mouse_button(self, button: int, action: int, modifiers: int) -> None:
        """
        Process mouse button events.
        
        Args:
            button: Mouse button (0=left, 1=right, 2=middle)
            action: Button action (0=release, 1=press)
            modifiers: Modifier keys
        """
        if button == 0 or button == 2:  # Left or middle mouse
            if action == 1:  # Press
                self.dragging = True
                self.panning = False  # Disabled to keep centered on black hole
            elif action == 0:  # Release
                self.dragging = False
                self.panning = False
    
    def process_scroll(self, x_offset: float, y_offset: float) -> None:
        """
        Process scroll wheel events.
        
        Args:
            x_offset: Horizontal scroll offset
            y_offset: Vertical scroll offset
        """
        self.radius -= y_offset * self.zoom_speed
        self.radius = np.clip(self.radius, self.min_radius, self.max_radius)
        self._update_position()
    
    def process_key(self, key: int, scancode: int, action: int, modifiers: int) -> bool:
        """
        Process keyboard events.
        
        Args:
            key: Key code
            scancode: Platform-specific scan code
            action: Key action (0=release, 1=press, 2=repeat)
            modifiers: Modifier keys
            
        Returns:
            True if key was handled, False otherwise
        """
        if action == 1:  # Press
            if key == 71:  # G key - toggle gravity (placeholder)
                logger.info("Gravity toggle requested")
                return True
            elif key == 82:  # R key - reset camera
                self.radius = 6.34194e10
                self.azimuth = 0.0
                self.elevation = math.pi / 2.0
                self._update_position()
                logger.info("Camera reset")
                return True
        
        return False
    
    def set_target(self, target: Tuple[float, float, float]) -> None:
        """
        Set camera target.
        
        Args:
            target: New target position (x, y, z)
        """
        self.target = np.array(target, dtype=np.float64)
        self._update_position()
    
    def set_radius(self, radius: float) -> None:
        """
        Set camera radius.
        
        Args:
            radius: New radius distance
        """
        self.radius = np.clip(radius, self.min_radius, self.max_radius)
        self._update_position()
    
    def set_angles(self, azimuth: float, elevation: float) -> None:
        """
        Set camera angles.
        
        Args:
            azimuth: Horizontal angle in radians
            elevation: Vertical angle in radians
        """
        self.azimuth = azimuth
        self.elevation = np.clip(elevation, self.min_elevation, self.max_elevation)
        self._update_position()
    
    def get_basis_vectors(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get camera basis vectors.
        
        Returns:
            Tuple of (right, up, forward) vectors
        """
        forward = self.target - self.position
        forward = forward / np.linalg.norm(forward)
        
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        return right, up, forward
