"""
Scene objects for black hole simulation.

This module defines scene objects including black holes, stars, and other
celestial bodies that can be rendered in the simulation.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from physics.constants import C, G, schwarzschild_radius

logger = logging.getLogger(__name__)


@dataclass
class SceneObject:
    """Represents a scene object with physical properties."""
    
    position: Tuple[float, float, float]
    radius: float
    color: Tuple[float, float, float, float]
    mass: float
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    name: str = ""
    
    def __post_init__(self):
        """Validate object properties after initialization."""
        if self.mass < 0:
            raise ValueError("Mass cannot be negative")
        if self.radius < 0:
            raise ValueError("Radius cannot be negative")
        if any(c < 0 or c > 1 for c in self.color):
            logger.warning("Color values should be in range [0, 1]")
    
    def get_schwarzschild_radius(self) -> float:
        """
        Get Schwarzschild radius for this object.
        
        Returns:
            Schwarzschild radius in meters
        """
        return schwarzschild_radius(self.mass)
    
    def is_black_hole(self) -> bool:
        """
        Check if this object is a black hole.
        
        Returns:
            True if radius is less than or equal to Schwarzschild radius
        """
        return self.radius <= self.get_schwarzschild_radius()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert object to dictionary representation.
        
        Returns:
            Dictionary with object properties
        """
        return {
            'position': self.position,
            'radius': self.radius,
            'color': self.color,
            'mass': self.mass,
            'velocity': self.velocity,
            'name': self.name,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SceneObject':
        """
        Create object from dictionary representation.
        
        Args:
            data: Dictionary with object properties
            
        Returns:
            SceneObject instance
        """
        return cls(
            position=data['position'],
            radius=data['radius'],
            color=data['color'],
            mass=data['mass'],
            velocity=data.get('velocity', (0.0, 0.0, 0.0)),
            name=data.get('name', ''),
        )


class SceneManager:
    """Manages scene objects and their interactions."""
    
    def __init__(self):
        """Initialize scene manager."""
        self.objects: List[SceneObject] = []
        self.gravity_enabled = False
        
    def add_object(self, obj: SceneObject) -> None:
        """
        Add an object to the scene.
        
        Args:
            obj: Scene object to add
        """
        self.objects.append(obj)
        logger.info(f"Added object '{obj.name}' to scene")
    
    def remove_object(self, name: str) -> bool:
        """
        Remove an object from the scene by name.
        
        Args:
            name: Object name
            
        Returns:
            True if object was found and removed
        """
        for i, obj in enumerate(self.objects):
            if obj.name == name:
                del self.objects[i]
                logger.info(f"Removed object '{name}' from scene")
                return True
        return False
    
    def get_object(self, name: str) -> Optional[SceneObject]:
        """
        Get an object by name.
        
        Args:
            name: Object name
            
        Returns:
            SceneObject or None if not found
        """
        for obj in self.objects:
            if obj.name == name:
                return obj
        return None
    
    def get_black_holes(self) -> List[SceneObject]:
        """
        Get all black holes in the scene.
        
        Returns:
            List of black hole objects
        """
        return [obj for obj in self.objects if obj.is_black_hole()]
    
    def get_primary_black_hole(self) -> Optional[SceneObject]:
        """
        Get the primary (most massive) black hole.
        
        Returns:
            Primary black hole or None if no black holes exist
        """
        black_holes = self.get_black_holes()
        if not black_holes:
            return None
        return max(black_holes, key=lambda obj: obj.mass)
    
    def update_gravity(self, dt: float) -> None:
        """
        Update object positions based on gravitational interactions.
        
        Args:
            dt: Time step in seconds
        """
        if not self.gravity_enabled:
            return
        
        # Simple gravitational force calculation (N-body)
        for i, obj1 in enumerate(self.objects):
            total_force = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            
            for j, obj2 in enumerate(self.objects):
                if i == j:
                    continue
                
                # Calculate gravitational force
                pos1 = np.array(obj1.position, dtype=np.float64)
                pos2 = np.array(obj2.position, dtype=np.float64)
                
                r_vec = pos2 - pos1
                r_mag = np.linalg.norm(r_vec)
                
                if r_mag > 0:
                    # Gravitational force magnitude
                    force_mag = G * obj1.mass * obj2.mass / (r_mag * r_mag)
                    
                    # Force direction
                    force_dir = r_vec / r_mag
                    
                    # Total force on obj1
                    total_force += force_mag * force_dir
            
            # Update velocity (F = ma, so a = F/m)
            if obj1.mass > 0:
                acceleration = total_force / obj1.mass
                velocity = np.array(obj1.velocity, dtype=np.float64)
                velocity += acceleration * dt
                
                # Update position
                position = np.array(obj1.position, dtype=np.float64)
                position += velocity * dt
                
                # Update object
                obj1.velocity = tuple(velocity)
                obj1.position = tuple(position)
    
    def set_gravity_enabled(self, enabled: bool) -> None:
        """
        Enable or disable gravitational interactions.
        
        Args:
            enabled: Whether to enable gravity
        """
        self.gravity_enabled = enabled
        logger.info(f"Gravity {'enabled' if enabled else 'disabled'}")
    
    def clear(self) -> None:
        """Remove all objects from the scene."""
        self.objects.clear()
        logger.info("Scene cleared")
    
    def get_objects_for_gpu(self) -> List[Dict[str, Any]]:
        """
        Get objects in format suitable for GPU rendering.
        
        Returns:
            List of object dictionaries for GPU
        """
        gpu_objects = []
        for obj in self.objects:
            gpu_objects.append({
                'position': obj.position,
                'radius': obj.radius,
                'color': obj.color,
                'mass': obj.mass,
            })
        return gpu_objects
    
    def save_scene(self, filename: str) -> None:
        """
        Save scene to file.
        
        Args:
            filename: Output filename
        """
        import json
        
        scene_data = {
            'objects': [obj.to_dict() for obj in self.objects],
            'gravity_enabled': self.gravity_enabled,
        }
        
        with open(filename, 'w') as f:
            json.dump(scene_data, f, indent=2)
        
        logger.info(f"Scene saved to {filename}")
    
    def load_scene(self, filename: str) -> None:
        """
        Load scene from file.
        
        Args:
            filename: Input filename
        """
        import json
        
        with open(filename, 'r') as f:
            scene_data = json.load(f)
        
        self.clear()
        
        for obj_data in scene_data['objects']:
            obj = SceneObject.from_dict(obj_data)
            self.add_object(obj)
        
        self.gravity_enabled = scene_data.get('gravity_enabled', False)
        
        logger.info(f"Scene loaded from {filename}")


def create_default_scene() -> SceneManager:
    """
    Create a default scene with Sagittarius A* and test objects.
    
    Returns:
        SceneManager with default objects
    """
    scene = SceneManager()
    
    # Sagittarius A* black hole
    sag_a = SceneObject(
        position=(0.0, 0.0, 0.0),
        radius=schwarzschild_radius(8.54e36),  # Sagittarius A* mass
        color=(0.0, 0.0, 0.0, 1.0),  # Black
        mass=8.54e36,
        name="Sagittarius A*",
    )
    scene.add_object(sag_a)
    
    # Test star 1
    star1 = SceneObject(
        position=(4e11, 0.0, 0.0),
        radius=4e10,
        color=(1.0, 1.0, 0.0, 1.0),  # Yellow
        mass=1.98892e30,  # Solar mass
        name="Test Star 1",
    )
    scene.add_object(star1)
    
    # Test star 2
    star2 = SceneObject(
        position=(0.0, 0.0, 4e11),
        radius=4e10,
        color=(1.0, 0.0, 0.0, 1.0),  # Red
        mass=1.98892e30,  # Solar mass
        name="Test Star 2",
    )
    scene.add_object(star2)
    
    return scene
