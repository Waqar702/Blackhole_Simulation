"""
2D null geodesic calculations for Schwarzschild metric.

This module implements 2D ray tracing in the Schwarzschild spacetime,
porting the functionality from the original 2D_lensing.cpp.
"""

import numpy as np
import math
import logging
from typing import List, Tuple, Optional
from .constants import C, G

logger = logging.getLogger(__name__)


class Ray2D:
    """2D null geodesic ray in Schwarzschild spacetime."""
    
    def __init__(
        self,
        position: Tuple[float, float],
        direction: Tuple[float, float],
        black_hole_mass: float,
        schwarzschild_radius: Optional[float] = None,
    ):
        """
        Initialize 2D ray.
        
        Args:
            position: Initial position (x, y) in meters
            direction: Initial direction (dx, dy) in m/s
            black_hole_mass: Black hole mass in kg
            schwarzschild_radius: Schwarzschild radius (calculated if None)
        """
        self.x = float(position[0])
        self.y = float(position[1])
        self.mass = black_hole_mass
        
        # Calculate Schwarzschild radius if not provided
        if schwarzschild_radius is None:
            self.r_s = 2.0 * G * black_hole_mass / (C * C)
        else:
            self.r_s = schwarzschild_radius
        
        # Convert to polar coordinates
        self.r = math.sqrt(self.x * self.x + self.y * self.y)
        self.phi = math.atan2(self.y, self.x)
        
        # Convert direction to polar basis
        dx, dy = direction
        self.dr = dx * math.cos(self.phi) + dy * math.sin(self.phi)
        self.dphi = (-dx * math.sin(self.phi) + dy * math.cos(self.phi)) / self.r
        
        # Calculate conserved quantities
        self.L = self.r * self.r * self.dphi  # Angular momentum
        f = 1.0 - self.r_s / self.r
        dt_dlambda = math.sqrt((self.dr * self.dr) / (f * f) + (self.r * self.r * self.dphi * self.dphi) / f)
        self.E = f * dt_dlambda  # Energy
        
        # Trail for visualization
        self.trail: List[Tuple[float, float]] = [(self.x, self.y)]
        
        logger.debug(f"Ray2D initialized: r={self.r:.2e}, phi={self.phi:.3f}, E={self.E:.2e}, L={self.L:.2e}")
    
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
        y0 = np.array([self.r, self.phi, self.dr, self.dphi], dtype=np.float64)
        
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
            self.phi = y_new[1]
            self.dr = y_new[2]
            self.dphi = y_new[3]
            
            # Convert back to Cartesian coordinates
            self.x = self.r * math.cos(self.phi)
            self.y = self.r * math.sin(self.phi)
            
            # Add to trail
            self.trail.append((self.x, self.y))
            
            # Limit trail length to prevent memory issues
            if len(self.trail) > 10000:
                self.trail = self.trail[-5000:]
                
            return True
            
        except (ValueError, OverflowError, ZeroDivisionError):
            logger.warning(f"Numerical error in geodesic integration at r={self.r:.2e}")
            return False
    
    def _is_valid_state(self, y: np.ndarray) -> bool:
        """
        Check if a state vector is valid.
        
        Args:
            y: State vector [r, phi, dr, dphi]
            
        Returns:
            True if state is valid
        """
        r, phi, dr, dphi = y
        
        # Check for invalid values
        if not np.all(np.isfinite(y)):
            return False
        
        # Radius must be positive and greater than event horizon
        if r <= self.r_s or r <= 0:
            return False
        
        # Prevent extreme values (more lenient)
        if r > 1e16 or abs(dr) > 1e12 or abs(dphi) > 1e6:
            return False
        
        return True
    
    def _geodesic_rhs(self, y: np.ndarray) -> np.ndarray:
        """
        Right-hand side of geodesic equations for 2D Schwarzschild.
        
        Args:
            y: State vector [r, phi, dr, dphi]
            
        Returns:
            Derivative vector [dr/dlambda, dphi/dlambda, d2r/dlambda2, d2phi/dlambda2]
        """
        r, phi, dr, dphi = y
        
        # Check for division by zero
        if r <= self.r_s or r <= 0:
            return np.array([0.0, 0.0, 0.0, 0.0])
        
        # Schwarzschild metric function
        f = 1.0 - self.r_s / r
        
        # Avoid division by zero near event horizon
        if abs(f) < 1e-10:
            return np.array([dr, dphi, 0.0, 0.0])
        
        # First derivatives
        dr_dlambda = dr
        dphi_dlambda = dphi
        
        # Second derivatives from Schwarzschild null geodesics
        # Use more numerically stable formulation
        dt_dlambda = self.E / f
        d2r_dlambda2 = (
            -(self.r_s / (2 * r * r)) * f * (dt_dlambda * dt_dlambda)
            + (self.r_s / (2 * r * r * f)) * (dr * dr)
            + (r - self.r_s) * (dphi * dphi)
        )
        
        d2phi_dlambda2 = -2.0 * dr * dphi / r
        
        return np.array([dr_dlambda, dphi_dlambda, d2r_dlambda2, d2phi_dlambda2])
    
    def get_position(self) -> Tuple[float, float]:
        """
        Get current position.
        
        Returns:
            Current position (x, y)
        """
        return (self.x, self.y)
    
    def get_polar_position(self) -> Tuple[float, float]:
        """
        Get current position in polar coordinates.
        
        Returns:
            Current position (r, phi)
        """
        return (self.r, self.phi)
    
    def get_velocity(self) -> Tuple[float, float]:
        """
        Get current velocity in Cartesian coordinates.
        
        Returns:
            Current velocity (dx, dy)
        """
        dx = self.dr * math.cos(self.phi) - self.r * self.dphi * math.sin(self.phi)
        dy = self.dr * math.sin(self.phi) + self.r * self.dphi * math.cos(self.phi)
        return (dx, dy)
    
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
    
    def get_trail(self) -> List[Tuple[float, float]]:
        """
        Get ray trail for visualization.
        
        Returns:
            List of (x, y) positions
        """
        return self.trail.copy()
    
    def clear_trail(self) -> None:
        """Clear the ray trail."""
        self.trail = [(self.x, self.y)]


def create_ray_from_camera(
    camera_pos: Tuple[float, float],
    screen_pos: Tuple[float, float],
    black_hole_mass: float,
    fov: float = 60.0,
    aspect_ratio: float = 1.0,
) -> Ray2D:
    """
    Create a ray from camera position through screen position.
    
    Args:
        camera_pos: Camera position (x, y)
        screen_pos: Screen position in normalized coordinates (-1 to 1)
        black_hole_mass: Black hole mass in kg
        fov: Field of view in degrees
        aspect_ratio: Aspect ratio (width/height)
        
    Returns:
        New Ray2D instance
    """
    # Calculate ray direction
    u, v = screen_pos
    tan_half_fov = math.tan(math.radians(fov) / 2.0)
    
    # Direction in camera space
    dir_x = u * aspect_ratio * tan_half_fov
    dir_y = v * tan_half_fov
    
    # For 2D, we'll assume the ray goes in the z=0 plane
    # and the direction represents the velocity
    direction = (dir_x * C, dir_y * C)  # Scale by speed of light
    
    return Ray2D(camera_pos, direction, black_hole_mass)
