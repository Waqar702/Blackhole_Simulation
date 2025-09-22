"""
Physics tests for black hole simulation.

This module contains tests for geodesic calculations, integration methods,
and physical constants validation.
"""

import pytest
import numpy as np
import math
from typing import Tuple

from blackhole_python.physics import (
    Ray2D, Ray3D, RK4Integrator, RK45Integrator,
    C, G, SAGITTARIUS_A_MASS, schwarzschild_radius
)


class TestConstants:
    """Test physical constants."""
    
    def test_speed_of_light(self):
        """Test speed of light constant."""
        assert C == 299792458.0
    
    def test_gravitational_constant(self):
        """Test gravitational constant."""
        assert abs(G - 6.67430e-11) < 1e-16
    
    def test_sagittarius_a_mass(self):
        """Test Sagittarius A* mass."""
        assert SAGITTARIUS_A_MASS == 8.54e36
    
    def test_schwarzschild_radius_calculation(self):
        """Test Schwarzschild radius calculation."""
        mass = 1.0  # kg
        expected_r_s = 2.0 * G * mass / (C * C)
        calculated_r_s = schwarzschild_radius(mass)
        
        assert abs(calculated_r_s - expected_r_s) < 1e-15


class TestRay2D:
    """Test 2D ray geodesics."""
    
    def test_ray_initialization(self):
        """Test ray initialization."""
        position = (1e11, 0.0)
        direction = (C, 0.0)
        mass = SAGITTARIUS_A_MASS
        
        ray = Ray2D(position, direction, mass)
        
        assert ray.x == position[0]
        assert ray.y == position[1]
        assert ray.mass == mass
        assert ray.r > 0
        assert abs(ray.E) > 0  # Energy should be non-zero
        assert abs(ray.L) >= 0  # Angular momentum should be non-negative
    
    def test_conserved_quantities(self):
        """Test that conserved quantities are maintained."""
        position = (1e11, 0.0)
        direction = (C * 0.8, C * 0.6)
        mass = SAGITTARIUS_A_MASS
        
        ray = Ray2D(position, direction, mass)
        initial_E = ray.E
        initial_L = ray.L
        
        # Take several steps
        for _ in range(100):
            ray.step(1e6)
        
        # Check conservation (within numerical precision)
        assert abs(ray.E - initial_E) / abs(initial_E) < 1e-10
        assert abs(ray.L - initial_L) / abs(initial_L) < 1e-10
    
    def test_event_horizon_detection(self):
        """Test event horizon detection."""
        mass = SAGITTARIUS_A_MASS
        r_s = schwarzschild_radius(mass)
        
        # Ray inside event horizon
        position = (r_s * 0.5, 0.0)
        direction = (0.0, C)
        ray = Ray2D(position, direction, mass)
        
        assert ray.is_inside_event_horizon()
        
        # Ray outside event horizon
        position = (r_s * 2.0, 0.0)
        ray = Ray2D(position, direction, mass)
        
        assert not ray.is_inside_event_horizon()
    
    def test_escape_detection(self):
        """Test escape radius detection."""
        position = (1e12, 0.0)  # Far from black hole
        direction = (C, 0.0)
        mass = SAGITTARIUS_A_MASS
        
        ray = Ray2D(position, direction, mass)
        
        assert ray.has_escaped(1e11)  # Should have escaped small radius
        assert not ray.has_escaped(1e13)  # Should not have escaped large radius


class TestRay3D:
    """Test 3D ray geodesics."""
    
    def test_ray_initialization(self):
        """Test 3D ray initialization."""
        position = (1e11, 0.0, 0.0)
        direction = (C, 0.0, 0.0)
        mass = SAGITTARIUS_A_MASS
        
        ray = Ray3D(position, direction, mass)
        
        assert ray.x == position[0]
        assert ray.y == position[1]
        assert ray.z == position[2]
        assert ray.r > 0
        assert 0 <= ray.theta <= math.pi
        assert -math.pi <= ray.phi <= math.pi
        assert abs(ray.E) > 0
        assert abs(ray.L) >= 0
    
    def test_coordinate_conversion(self):
        """Test Cartesian to spherical coordinate conversion."""
        position = (1e11, 1e11, 0.0)
        direction = (C, 0.0, 0.0)
        mass = SAGITTARIUS_A_MASS
        
        ray = Ray3D(position, direction, mass)
        
        # Check that conversion is consistent
        expected_r = math.sqrt(2) * 1e11
        expected_theta = math.pi / 2.0
        expected_phi = math.pi / 4.0
        
        assert abs(ray.r - expected_r) < 1e-6
        assert abs(ray.theta - expected_theta) < 1e-6
        assert abs(ray.phi - expected_phi) < 1e-6
    
    def test_sphere_intersection(self):
        """Test sphere intersection detection."""
        position = (0.0, 0.0, 1e11)
        direction = (0.0, 0.0, -C)
        mass = SAGITTARIUS_A_MASS
        
        ray = Ray3D(position, direction, mass)
        
        # Should intersect sphere at origin with radius 5e10
        center = (0.0, 0.0, 0.0)
        radius = 5e10
        
        assert ray.intercept_sphere(center, radius)
        
        # Should not intersect sphere far away
        center = (1e12, 0.0, 0.0)
        assert not ray.intercept_sphere(center, radius)


class TestIntegrators:
    """Test numerical integration methods."""
    
    def test_rk4_integrator(self):
        """Test RK4 integrator."""
        integrator = RK4Integrator(step_size=0.1)
        
        # Test with simple harmonic oscillator: dy/dt = -y
        def rhs(y):
            return np.array([-y[0]])
        
        y0 = np.array([1.0])
        y_final = integrator.integrate(rhs, y0, steps=100)
        
        # Solution should be y(t) = exp(-t)
        expected = math.exp(-10.0)  # t = 0.1 * 100 = 10
        assert abs(y_final[0] - expected) < 1e-6
    
    def test_rk45_integrator(self):
        """Test adaptive RK45 integrator."""
        integrator = RK45Integrator(
            initial_step_size=0.1,
            rel_tol=1e-6,
            abs_tol=1e-8,
        )
        
        # Test with simple harmonic oscillator
        def rhs(y, t):
            return np.array([-y[0]])
        
        y0 = np.array([1.0])
        y_final, t_final, success = integrator.integrate(rhs, y0, t_end=10.0)
        
        assert success
        expected = math.exp(-10.0)
        assert abs(y_final[0] - expected) < 1e-5
    
    def test_integrator_creation(self):
        """Test integrator creation function."""
        rk4 = create_integrator("rk4", step_size=0.1)
        assert isinstance(rk4, RK4Integrator)
        
        rk45 = create_integrator("rk45", initial_step_size=0.1)
        assert isinstance(rk45, RK45Integrator)
        
        with pytest.raises(ValueError):
            create_integrator("invalid_method")


# Utility function for tests
def create_integrator(method: str, **kwargs):
    """Create an integrator instance for testing."""
    if method.lower() == "rk4":
        return RK4Integrator(**kwargs)
    elif method.lower() == "rk45":
        return RK45Integrator(**kwargs)
    else:
        raise ValueError(f"Unknown integration method: {method}")


if __name__ == "__main__":
    pytest.main([__file__])
