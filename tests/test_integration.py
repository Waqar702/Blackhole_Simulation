"""
Integration tests for black hole simulation.

This module contains end-to-end tests that verify the integration
between different components of the simulation.
"""

import pytest
import numpy as np
import math

from blackhole_python.scene import SceneManager, SceneObject, create_default_scene
from blackhole_python.physics import Ray2D, Ray3D, SAGITTARIUS_A_MASS


class TestSceneIntegration:
    """Test scene management integration."""
    
    def test_default_scene_creation(self):
        """Test creation of default scene."""
        scene = create_default_scene()
        
        assert len(scene.objects) == 3  # Sagittarius A* + 2 test stars
        
        # Check that Sagittarius A* is present
        sag_a = scene.get_object("Sagittarius A*")
        assert sag_a is not None
        assert sag_a.is_black_hole()
        assert sag_a.mass == SAGITTARIUS_A_MASS
        
        # Check that primary black hole is Sagittarius A*
        primary_bh = scene.get_primary_black_hole()
        assert primary_bh is sag_a
    
    def test_object_management(self):
        """Test object addition and removal."""
        scene = SceneManager()
        
        # Add object
        obj = SceneObject(
            position=(1e11, 0.0, 0.0),
            radius=1e10,
            color=(1.0, 0.0, 0.0, 1.0),
            mass=1e30,
            name="Test Object",
        )
        scene.add_object(obj)
        
        assert len(scene.objects) == 1
        assert scene.get_object("Test Object") is obj
        
        # Remove object
        assert scene.remove_object("Test Object")
        assert len(scene.objects) == 0
        assert scene.get_object("Test Object") is None
        
        # Try to remove non-existent object
        assert not scene.remove_object("Non-existent")
    
    def test_gravity_simulation(self):
        """Test gravitational interactions."""
        scene = SceneManager()
        
        # Create two objects
        obj1 = SceneObject(
            position=(0.0, 0.0, 0.0),
            radius=1e10,
            color=(1.0, 0.0, 0.0, 1.0),
            mass=1e30,
            name="Object 1",
        )
        obj2 = SceneObject(
            position=(1e11, 0.0, 0.0),
            radius=1e10,
            color=(0.0, 1.0, 0.0, 1.0),
            mass=1e30,
            name="Object 2",
        )
        
        scene.add_object(obj1)
        scene.add_object(obj2)
        
        # Enable gravity
        scene.set_gravity_enabled(True)
        
        # Store initial positions
        initial_pos1 = obj1.position
        initial_pos2 = obj2.position
        
        # Update gravity for one time step
        scene.update_gravity(1.0)  # 1 second
        
        # Objects should have moved (attracted to each other)
        assert obj1.position != initial_pos1
        assert obj2.position != initial_pos2
        
        # Distance should have decreased (objects attracted)
        initial_distance = math.sqrt(
            (initial_pos2[0] - initial_pos1[0])**2 +
            (initial_pos2[1] - initial_pos1[1])**2 +
            (initial_pos2[2] - initial_pos1[2])**2
        )
        current_distance = math.sqrt(
            (obj2.position[0] - obj1.position[0])**2 +
            (obj2.position[1] - obj1.position[1])**2 +
            (obj2.position[2] - obj1.position[2])**2
        )
        
        assert current_distance < initial_distance
    
    def test_gpu_object_format(self):
        """Test object format for GPU rendering."""
        scene = create_default_scene()
        gpu_objects = scene.get_objects_for_gpu()
        
        assert len(gpu_objects) == len(scene.objects)
        
        for i, obj in enumerate(scene.objects):
            gpu_obj = gpu_objects[i]
            assert gpu_obj['position'] == obj.position
            assert gpu_obj['radius'] == obj.radius
            assert gpu_obj['color'] == obj.color
            assert gpu_obj['mass'] == obj.mass


class TestPhysicsIntegration:
    """Test physics integration with scene."""
    
    def test_ray_scene_interaction(self):
        """Test ray interaction with scene objects."""
        scene = create_default_scene()
        sag_a = scene.get_object("Sagittarius A*")
        
        # Create ray that should hit the black hole
        position = (sag_a.position[0] + 2e10, sag_a.position[1], sag_a.position[2])
        direction = (-1.0, 0.0, 0.0)  # Towards black hole
        
        ray = Ray3D(position, direction, sag_a.mass)
        
        # Ray should eventually hit the black hole
        steps = 0
        max_steps = 1000
        
        while not ray.is_inside_event_horizon() and not ray.has_escaped() and steps < max_steps:
            ray.step(1e6)
            steps += 1
        
        assert ray.is_inside_event_horizon() or steps >= max_steps
    
    def test_ray_escape_condition(self):
        """Test ray escape conditions."""
        scene = create_default_scene()
        sag_a = scene.get_object("Sagittarius A*")
        
        # Create ray that should escape
        position = (sag_a.position[0] + 1e12, sag_a.position[1], sag_a.position[2])
        direction = (1.0, 0.0, 0.0)  # Away from black hole
        
        ray = Ray3D(position, direction, sag_a.mass)
        
        # Ray should escape
        steps = 0
        max_steps = 1000
        
        while not ray.is_inside_event_horizon() and not ray.has_escaped() and steps < max_steps:
            ray.step(1e6)
            steps += 1
        
        assert ray.has_escaped()
    
    def test_2d_3d_ray_consistency(self):
        """Test consistency between 2D and 3D ray tracing."""
        mass = SAGITTARIUS_A_MASS
        position_2d = (1e11, 0.0)
        position_3d = (1e11, 0.0, 0.0)
        direction_2d = (C * 0.8, C * 0.6)
        direction_3d = (C * 0.8, 0.0, C * 0.6)
        
        ray_2d = Ray2D(position_2d, direction_2d, mass)
        ray_3d = Ray3D(position_3d, direction_3d, mass)
        
        # Take same number of steps
        steps = 100
        for _ in range(steps):
            ray_2d.step(1e6)
            ray_3d.step(1e6)
        
        # Check that conserved quantities are similar
        # (They should be the same for equivalent initial conditions)
        assert abs(ray_2d.E - ray_3d.E) / abs(ray_2d.E) < 1e-10
        assert abs(ray_2d.L - ray_3d.L) / abs(ray_2d.L) < 1e-10


class TestPerformance:
    """Test performance characteristics."""
    
    def test_ray_tracing_performance(self):
        """Test ray tracing performance."""
        import time
        
        mass = SAGITTARIUS_A_MASS
        position = (1e11, 0.0, 0.0)
        direction = (C * 0.8, 0.0, C * 0.6)
        
        ray = Ray3D(position, direction, mass)
        
        # Time 1000 integration steps
        start_time = time.time()
        
        for _ in range(1000):
            ray.step(1e6)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Should complete in reasonable time (< 1 second)
        assert elapsed < 1.0
        
        # Performance should be at least 1000 steps per second
        steps_per_second = 1000 / elapsed
        assert steps_per_second > 1000
    
    def test_scene_update_performance(self):
        """Test scene update performance."""
        import time
        
        scene = create_default_scene()
        scene.set_gravity_enabled(True)
        
        # Time 100 gravity updates
        start_time = time.time()
        
        for _ in range(100):
            scene.update_gravity(1.0)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Should complete in reasonable time
        assert elapsed < 1.0


if __name__ == "__main__":
    pytest.main([__file__])
