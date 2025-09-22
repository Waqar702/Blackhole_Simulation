"""
Physical constants and units for black hole simulation.

This module defines all the physical constants used in the simulation,
matching the values from the original C++ implementation.
"""

import numpy as np

# Fundamental constants
C = 299792458.0  # Speed of light (m/s)
G = 6.67430e-11  # Gravitational constant (m^3/kg/s^2)

# Black hole parameters
SAGITTARIUS_A_MASS = 8.54e36  # Sagittarius A* mass (kg)

# Derived constants
SAGITTARIUS_A_SCHWARZSCHILD_RADIUS = 2.0 * G * SAGITTARIUS_A_MASS / (C * C)

# Simulation parameters
DEFAULT_STEP_SIZE = 1e7  # Default integration step size
MAX_INTEGRATION_STEPS = 80000  # Maximum number of integration steps
ESCAPE_RADIUS = 1e14  # Radius at which rays are considered escaped

# Camera parameters (from C++ implementation)
DEFAULT_CAMERA_RADIUS = 6.34194e10  # Default camera distance
MIN_CAMERA_RADIUS = 1e10
MAX_CAMERA_RADIUS = 1e12

# Grid parameters
DEFAULT_GRID_SIZE = 25
DEFAULT_GRID_SPACING = 1e10

# Rendering parameters
DEFAULT_COMPUTE_WIDTH = 200
DEFAULT_COMPUTE_HEIGHT = 150
DEFAULT_RENDER_WIDTH = 800
DEFAULT_RENDER_HEIGHT = 600

# Integration tolerances
RK4_DEFAULT_STEP = 1e7
RK45_REL_TOL = 1e-6
RK45_ABS_TOL = 1e-8

# Physical units
METER = 1.0
KILOMETER = 1000.0
ASTRONOMICAL_UNIT = 1.496e11  # m
LIGHT_YEAR = 9.461e15  # m
SOLAR_MASS = 1.989e30  # kg

# Useful conversions
def meters_to_km(meters: float) -> float:
    """Convert meters to kilometers."""
    return meters / KILOMETER

def meters_to_au(meters: float) -> float:
    """Convert meters to astronomical units."""
    return meters / ASTRONOMICAL_UNIT

def meters_to_ly(meters: float) -> float:
    """Convert meters to light years."""
    return meters / LIGHT_YEAR

def kg_to_solar_mass(kg: float) -> float:
    """Convert kilograms to solar masses."""
    return kg / SOLAR_MASS

def schwarzschild_radius(mass: float) -> float:
    """
    Calculate Schwarzschild radius for given mass.
    
    Args:
        mass: Mass in kilograms
        
    Returns:
        Schwarzschild radius in meters
    """
    return 2.0 * G * mass / (C * C)

def gravitational_radius_to_mass(radius: float) -> float:
    """
    Calculate mass from Schwarzschild radius.
    
    Args:
        radius: Schwarzschild radius in meters
        
    Returns:
        Mass in kilograms
    """
    return radius * C * C / (2.0 * G)
