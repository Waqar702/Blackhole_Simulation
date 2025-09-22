"""
3D null geodesic calculations for Schwarzschild metric.

This module implements 3D ray tracing in the Schwarzschild spacetime,
porting the functionality from the original CPU-geodesic.cpp.
"""

import numpy as np
import math
import logging
from typing import List, Tuple, Optional

from .constants import C, G

logger = logging.getLogger(__name__)


class Ray3D:
    """3D null geodesic ray in Schwarzschild spacetime."""
    
    def __init__(
        self,
        position: Tuple[float, float, float],
        direction: Tuple[float, float, float],
        black_hole_mass: float,
        schwarzschild_radius: Optional[float] = None,
    ):
        """
        Initialize 3D ray.
        
        Args:
            position: Initial position (x, y, z) in meters
            direction: Initial direction (dx, dy, dz) in m/s
            black_hole_mass: Black hole mass in kg
            schwarzschild_radius: Schwarzschild radius (calculated if None)
        """
        # Cartesian coordinates
        self.x = float(position[0])
        self.y = float(position[1])
        self.z = float(position[2])
        self.mass = black_hole_mass
        
        # Calculate Schwarzschild radius if not provided
        if schwarzschild_radius is None:
            self.r_s = 2.0 * G * black_hole_mass / (C * C)
        else:
            self.r_s = schwarzschild_radius
        
        # Convert to spherical coordinates
        self.r = math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)
        self.theta = math.acos(self.z / self.r) if self.r > 0 else 0.0
        self.phi = math.atan2(self.y, self.x)
        
        # Convert direction to spherical basis
        dx, dy, dz = direction
        self.dr = (
            math.sin(self.theta) * math.cos(self.phi) * dx +
            math.sin(self.theta) * math.sin(self.phi) * dy +
            math.cos(self.theta) * dz
        )
        self.dtheta = (
            (math.cos(self.theta) * math.cos(self.phi) * dx +
             math.cos(self.theta) * math.sin(self.phi) * dy -
             math.sin(self.theta) * dz) / self.r
        )
        self.dphi = (
            (-math.sin(self.phi) * dx + math.cos(self.phi) * dy) / 
            (self.r * math.sin(self.theta))
        )
        
        # Calculate conserved quantities
        self.L = self.r * self.r * math.sin(self.theta) * self.dphi  # Angular momentum
        f = 1.0 - self.r_s / self.r
        dt_dlambda = math.sqrt(
            (self.dr * self.dr) / f + 
            self.r * self.r * (self.dtheta * self.dtheta + 
                              math.sin(self.theta) * math.sin(self.theta) * self.dphi * self.dphi)
        )
        self.E = f * dt_dlambda  # Energy
        
        logger.debug(f"Ray3D initialized: r={self.r:.2e}, theta={self.theta:.3f}, phi={self.phi:.3f}")
        logger.debug(f"E={self.E:.2e}, L={self.L:.2e}")
    
    def step(self, dlambda: float) -> bool:
        """
        Integrate ray forward by one step.
        
        Args:
            dlambda: Integration step size
            
        Returns:
            True if step was successful, False if ray hit event horizon or became invalid
        """
        if self.r <= self.r_s:
            return False  # Stop if inside event horizon
        
        # Adaptive step sizing based on distance to event horizon
        min_step = self.r_s * 1e-8  # Minimum step near event horizon
        max_step = min(dlambda, self.r * 0.01)  # Much smaller maximum step
        adaptive_step = min(max_step, max(min_step, self.r * 0.0001))  # Even smaller relative step
        
        # Store current state for RK4
        y0 = np.array([self.r, self.theta, self.phi, self.dr, self.dtheta, self.dphi], dtype=np.float64)
        
        try:
            # RK4 integration with error checking
            k1 = self._geodesic_rhs(y0)
            if not self._is_valid_state(y0 + 0.5 * adaptive_step * k1):
                return False
                
            k2 = self._geodesic_rhs(y0 + 0.5 * adaptive_step * k1)
            if not self._is_valid_state(y0 + 0.5 * adaptive_step * k2):
                return False
                
            k3 = self._geodesic_rhs(y0 + 0.5 * adaptive_step * k2)
            if not self._is_valid_state(y0 + adaptive_step * k3):
                return False
                
            k4 = self._geodesic_rhs(y0 + adaptive_step * k3)
            
            # Update state
            y_new = y0 + (adaptive_step / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            # Validate new state
            if not self._is_valid_state(y_new):
                return False
            
            self.r = y_new[0]
            self.theta = y_new[1]
            self.phi = y_new[2]
            self.dr = y_new[3]
            self.dtheta = y_new[4]
            self.dphi = y_new[5]
            
            # Convert back to Cartesian coordinates
            self.x = self.r * math.sin(self.theta) * math.cos(self.phi)
            self.y = self.r * math.sin(self.theta) * math.sin(self.phi)
            self.z = self.r * math.cos(self.theta)
            
            return True
            
        except (ValueError, OverflowError, ZeroDivisionError):
            logger.warning(f"Numerical error in geodesic integration at r={self.r:.2e}")
            return False
    
    def _is_valid_state(self, y: np.ndarray) -> bool:
        """
        Check if a state vector is valid.
        
        Args:
            y: State vector [r, theta, phi, dr, dtheta, dphi]
            
        Returns:
            True if state is valid
        """
        r, theta, phi, dr, dtheta, dphi = y
        
        # Check for invalid values
        if not np.all(np.isfinite(y)):
            return False
        
        # Radius must be positive and greater than event horizon
        if r <= self.r_s or r <= 0:
            return False
        
        # Theta must be in valid range
        if theta < 0 or theta > math.pi:
            return False
        
        # Prevent extreme values (more lenient)
        if r > 1e16 or abs(dr) > 1e12 or abs(dtheta) > 1e6 or abs(dphi) > 1e6:
            return False
        
        return True
    
    def _geodesic_rhs(self, y: np.ndarray) -> np.ndarray:
        """
        Right-hand side of geodesic equations for 3D Schwarzschild.
        
        Args:
            y: State vector [r, theta, phi, dr, dtheta, dphi]
            
        Returns:
            Derivative vector [dr/dlambda, dtheta/dlambda, dphi/dlambda, d2r/dlambda2, d2theta/dlambda2, d2phi/dlambda2]
        """
        r, theta, phi, dr, dtheta, dphi = y[0], y[1], y[2], y[3], y[4], y[5]
        
        # Check for division by zero
        if r <= self.r_s or r <= 0:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Schwarzschild metric function
        f = 1.0 - self.r_s / r
        
        # Avoid division by zero near event horizon
        if abs(f) < 1e-10:
            return np.array([dr, dtheta, dphi, 0.0, 0.0, 0.0])
        
        # First derivatives
        dr_dlambda = dr
        dtheta_dlambda = dtheta
        dphi_dlambda = dphi
        
        # Second derivatives from Schwarzschild null geodesics
        dt_dlambda = self.E / f
        
        d2r_dlambda2 = (
            -(self.r_s / (2 * r * r)) * f * (dt_dlambda * dt_dlambda)
            + (self.r_s / (2 * r * r * f)) * (dr * dr)
            + r * (dtheta * dtheta + math.sin(theta) * math.sin(theta) * dphi * dphi)
        )
        
        d2theta_dlambda2 = (
            -2.0 * dr * dtheta / r + 
            math.sin(theta) * math.cos(theta) * dphi * dphi
        )
        
        # Handle singularity at theta = 0 or pi
        if abs(math.sin(theta)) < 1e-10:
            d2phi_dlambda2 = 0.0
        else:
            d2phi_dlambda2 = (
                -2.0 * dr * dphi / r - 
                2.0 * math.cos(theta) / math.sin(theta) * dtheta * dphi
            )
        
        return np.array([dr_dlambda, dtheta_dlambda, dphi_dlambda, 
                        d2r_dlambda2, d2theta_dlambda2, d2phi_dlambda2])
    
    def get_position(self) -> Tuple[float, float, float]:
        """
        Get current position.
        
        Returns:
            Current position (x, y, z)
        """
        return (self.x, self.y, self.z)
    
    def get_spherical_position(self) -> Tuple[float, float, float]:
        """
        Get current position in spherical coordinates.
        
        Returns:
            Current position (r, theta, phi)
        """
        return (self.r, self.theta, self.phi)
    
    def get_velocity(self) -> Tuple[float, float, float]:
        """
        Get current velocity in Cartesian coordinates.
        
        Returns:
            Current velocity (dx, dy, dz)
        """
        # Convert spherical velocities to Cartesian
        dx = (
            self.dr * math.sin(self.theta) * math.cos(self.phi) +
            self.r * self.dtheta * math.cos(self.theta) * math.cos(self.phi) -
            self.r * self.dphi * math.sin(self.theta) * math.sin(self.phi)
        )
        dy = (
            self.dr * math.sin(self.theta) * math.sin(self.phi) +
            self.r * self.dtheta * math.cos(self.theta) * math.sin(self.phi) +
            self.r * self.dphi * math.sin(self.theta) * math.cos(self.phi)
        )
        dz = (
            self.dr * math.cos(self.theta) -
            self.r * self.dtheta * math.sin(self.theta)
        )
        return (dx, dy, dz)
    
    def is_inside_event_horizon(self) -> bool:
        """
        Check if ray is inside the event horizon.
        
        Returns:
            True if inside event horizon
        """
        return self.r <= self.r_s
    
    def has_escaped(self, escape_radius: float = 1e14) -> bool:
        """
        Check if ray has escaped to infinity.
        
        Args:
            escape_radius: Radius at which ray is considered escaped
            
        Returns:
            True if ray has escaped
        """
        return self.r > escape_radius
    
    def get_conserved_quantities(self) -> Tuple[float, float]:
        """
        Get conserved quantities.
        
        Returns:
            Tuple of (energy, angular_momentum)
        """
        return (self.E, self.L)
    
    def intercept_sphere(
        self,
        center: Tuple[float, float, float],
        radius: float,
    ) -> bool:
        """
        Check if ray intercepts a sphere.
        
        Args:
            center: Sphere center (x, y, z)
            radius: Sphere radius
            
        Returns:
            True if ray intercepts sphere
        """
        dx = self.x - center[0]
        dy = self.y - center[1]
        dz = self.z - center[2]
        dist_squared = dx*dx + dy*dy + dz*dz
        return dist_squared <= radius * radius


def create_ray_from_camera(
    camera_pos: Tuple[float, float, float],
    screen_pos: Tuple[float, float],
    black_hole_mass: float,
    fov: float = 60.0,
    aspect_ratio: float = 1.0,
) -> Ray3D:
    """
    Create a ray from camera position through screen position.
    
    Args:
        camera_pos: Camera position (x, y, z)
        screen_pos: Screen position in normalized coordinates (-1 to 1)
        black_hole_mass: Black hole mass in kg
        fov: Field of view in degrees
        aspect_ratio: Aspect ratio (width/height)
        
    Returns:
        New Ray3D instance
    """
    # Calculate ray direction
    u, v = screen_pos
    tan_half_fov = math.tan(math.radians(fov) / 2.0)
    
    # Direction in camera space (assuming camera looks down -z axis)
    dir_x = u * aspect_ratio * tan_half_fov
    dir_y = -v * tan_half_fov  # Flip Y for screen coordinates
    dir_z = -1.0  # Forward direction
    
    # Normalize and scale by speed of light
    direction_length = math.sqrt(dir_x*dir_x + dir_y*dir_y + dir_z*dir_z)
    direction = (
        dir_x / direction_length * C,
        dir_y / direction_length * C,
        dir_z / direction_length * C
    )
    
    return Ray3D(camera_pos, direction, black_hole_mass)
