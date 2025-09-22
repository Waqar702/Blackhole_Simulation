"""
Accretion disk implementation.

This module provides the AccretionDisk class for modeling and rendering
accretion disks around black holes with realistic physics and appearance.
"""

import numpy as np
import math
from typing import Tuple, List, Optional, Dict, Any
import logging

from physics.constants import C, G, schwarzschild_radius

logger = logging.getLogger(__name__)


class AccretionDisk:
    """Models an accretion disk around a black hole."""
    
    def __init__(
        self,
        black_hole_mass: float,
        inner_radius_factor: float = 2.2,
        outer_radius_factor: float = 5.2,
        thickness: float = 1e9,
        temperature_profile: str = "standard",
    ):
        """
        Initialize accretion disk.
        
        Args:
            black_hole_mass: Mass of the central black hole (kg)
            inner_radius_factor: Inner radius as multiple of Schwarzschild radius
            outer_radius_factor: Outer radius as multiple of Schwarzschild radius
            thickness: Disk thickness (m)
            temperature_profile: Temperature profile type ("standard" or "custom")
        """
        self.black_hole_mass = black_hole_mass
        self.inner_radius_factor = inner_radius_factor
        self.outer_radius_factor = outer_radius_factor
        self.thickness = thickness
        self.temperature_profile = temperature_profile
        
        # Calculate radii
        self.r_s = schwarzschild_radius(black_hole_mass)
        self.inner_radius = self.inner_radius_factor * self.r_s
        self.outer_radius = self.outer_radius_factor * self.r_s
        
        # Disk properties
        self.is_emitting = True
        self.opacity = 0.8
        
        logger.info(f"Accretion disk created: inner={self.inner_radius:.2e}m, outer={self.outer_radius:.2e}m")
    
    def get_radius_at_angle(self, angle: float, radius: float) -> float:
        """
        Get disk radius at given angle and distance.
        
        Args:
            angle: Angular position (0 to 2π)
            radius: Distance from center
            
        Returns:
            Effective radius at that position
        """
        # For now, return the input radius
        # Could add warping effects here
        return radius
    
    def get_temperature_at_radius(self, radius: float) -> float:
        """
        Get temperature at given radius.
        
        Args:
            radius: Distance from center
            
        Returns:
            Temperature in Kelvin
        """
        if radius < self.inner_radius:
            return 0.0
        
        if self.temperature_profile == "standard":
            # Standard Shakura-Sunyaev disk temperature profile
            # T ∝ r^(-3/4) for optically thick disk
            inner_temp = 1e7  # Temperature at inner radius (K)
            return inner_temp * (radius / self.inner_radius) ** (-3/4)
        else:
            # Simple linear temperature profile
            max_temp = 1e7
            min_temp = 1e3
            temp_factor = (radius - self.inner_radius) / (self.outer_radius - self.inner_radius)
            return max_temp - temp_factor * (max_temp - min_temp)
    
    def get_color_at_radius(self, radius: float) -> Tuple[float, float, float, float]:
        """
        Get color at given radius based on temperature.
        
        Args:
            radius: Distance from center
            
        Returns:
            RGBA color tuple
        """
        if radius < self.inner_radius:
            return (0.0, 0.0, 0.0, 0.0)  # Transparent
        
        # Normalize radius for color calculation
        r_norm = (radius - self.inner_radius) / (self.outer_radius - self.inner_radius)
        r_norm = np.clip(r_norm, 0.0, 1.0)
        
        # Color based on radius (hotter = bluer/whiter, cooler = redder)
        if r_norm < 0.2:
            # Inner region - hot, blue-white
            color = (0.8, 0.9, 1.0, self.opacity)
        elif r_norm < 0.5:
            # Mid region - white-yellow
            color = (1.0, 1.0, 0.8, self.opacity)
        else:
            # Outer region - red
            color = (1.0, r_norm, 0.2, self.opacity)
        
        return color
    
    def intersects_ray(
        self,
        ray_origin: Tuple[float, float, float],
        ray_direction: Tuple[float, float, float],
    ) -> Optional[Tuple[float, Tuple[float, float, float]]]:
        """
        Check if ray intersects with the accretion disk.
        
        Args:
            ray_origin: Ray starting position
            ray_direction: Ray direction (normalized)
            
        Returns:
            Tuple of (intersection_distance, intersection_point) or None
        """
        # Check intersection with disk plane (y = 0)
        if abs(ray_direction[1]) < 1e-10:
            return None  # Ray is parallel to disk plane
        
        # Calculate intersection with y = 0 plane
        t = -ray_origin[1] / ray_direction[1]
        if t <= 0:
            return None  # Intersection is behind ray origin
        
        # Intersection point
        intersection_point = (
            ray_origin[0] + t * ray_direction[0],
            ray_origin[1] + t * ray_direction[1],
            ray_origin[2] + t * ray_direction[2],
        )
        
        # Check if intersection is within disk bounds
        r = math.sqrt(intersection_point[0]**2 + intersection_point[2]**2)
        if self.inner_radius <= r <= self.outer_radius:
            return (t, intersection_point)
        
        return None
    
    def crosses_equatorial_plane(
        self,
        old_pos: Tuple[float, float, float],
        new_pos: Tuple[float, float, float],
    ) -> bool:
        """
        Check if ray crossed the equatorial plane within disk bounds.
        
        Args:
            old_pos: Previous ray position
            new_pos: Current ray position
            
        Returns:
            True if ray crossed equatorial plane in disk region
        """
        # Check if ray crossed y = 0 plane
        if old_pos[1] * new_pos[1] >= 0:
            return False
        
        # Calculate radius at intersection
        r = math.sqrt(new_pos[0]**2 + new_pos[2]**2)
        
        # Check if within disk bounds
        return self.inner_radius <= r <= self.outer_radius
    
    def get_disk_parameters(self) -> Dict[str, float]:
        """
        Get disk parameters for GPU rendering.
        
        Returns:
            Dictionary with disk parameters
        """
        return {
            'inner_radius': self.inner_radius,
            'outer_radius': self.outer_radius,
            'thickness': self.thickness,
            'opacity': self.opacity,
        }
    
    def set_inner_radius_factor(self, factor: float) -> None:
        """
        Set inner radius factor.
        
        Args:
            factor: Inner radius as multiple of Schwarzschild radius
        """
        self.inner_radius_factor = factor
        self.inner_radius = factor * self.r_s
        logger.info(f"Inner radius factor set to {factor}")
    
    def set_outer_radius_factor(self, factor: float) -> None:
        """
        Set outer radius factor.
        
        Args:
            factor: Outer radius as multiple of Schwarzschild radius
        """
        self.outer_radius_factor = factor
        self.outer_radius = factor * self.r_s
        logger.info(f"Outer radius factor set to {factor}")
    
    def set_opacity(self, opacity: float) -> None:
        """
        Set disk opacity.
        
        Args:
            opacity: Opacity value (0.0 to 1.0)
        """
        self.opacity = np.clip(opacity, 0.0, 1.0)
        logger.info(f"Disk opacity set to {self.opacity}")
    
    def set_emitting(self, emitting: bool) -> None:
        """
        Enable or disable disk emission.
        
        Args:
            emitting: Whether disk should emit light
        """
        self.is_emitting = emitting
        logger.info(f"Disk emission {'enabled' if emitting else 'disabled'}")
    
    def update_black_hole_mass(self, new_mass: float) -> None:
        """
        Update black hole mass and recalculate disk parameters.
        
        Args:
            new_mass: New black hole mass
        """
        self.black_hole_mass = new_mass
        self.r_s = schwarzschild_radius(new_mass)
        self.inner_radius = self.inner_radius_factor * self.r_s
        self.outer_radius = self.outer_radius_factor * self.r_s
        
        logger.info(f"Black hole mass updated to {new_mass:.2e} kg")
    
    def get_info(self) -> str:
        """
        Get disk information string.
        
        Returns:
            Formatted information string
        """
        return (
            f"Accretion Disk:\n"
            f"  Inner radius: {self.inner_radius:.2e} m ({self.inner_radius_factor:.1f} r_s)\n"
            f"  Outer radius: {self.outer_radius:.2e} m ({self.outer_radius_factor:.1f} r_s)\n"
            f"  Thickness: {self.thickness:.2e} m\n"
            f"  Opacity: {self.opacity:.2f}\n"
            f"  Emitting: {self.is_emitting}"
        )
