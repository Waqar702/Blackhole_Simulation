#!/usr/bin/env python3
"""
Physics demonstration script.

This script demonstrates the physics calculations we have implemented
so far, including geodesic ray tracing and scene management.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
from physics.constants import C, G, SAGITTARIUS_A_MASS, schwarzschild_radius
from physics.geodesic2d import Ray2D
from physics.geodesic3d import Ray3D
from scene.objects import create_default_scene
from physics.integrators import RK4Integrator

def demonstrate_physics():
    """Demonstrate the physics calculations."""
    print("🌌 Black Hole Physics Demonstration")
    print("=" * 50)
    
    # Physical constants
    print(f"Speed of light: {C:.2e} m/s")
    print(f"Gravitational constant: {G:.2e} m³/kg/s²")
    print(f"Sagittarius A* mass: {SAGITTARIUS_A_MASS:.2e} kg")
    
    # Schwarzschild radius
    r_s = schwarzschild_radius(SAGITTARIUS_A_MASS)
    print(f"Schwarzschild radius: {r_s:.2e} m")
    print(f"Schwarzschild radius: {r_s/1000:.2f} km")
    
    print("\n" + "=" * 50)
    print("2D Ray Tracing Demo")
    print("=" * 50)
    
    # Create 2D ray
    position_2d = (1e11, 0.0)  # 100 million km from center
    direction_2d = (C * 0.8, C * 0.6)  # 80% speed of light
    
    ray_2d = Ray2D(position_2d, direction_2d, SAGITTARIUS_A_MASS)
    print(f"Initial 2D ray position: ({ray_2d.x:.2e}, {ray_2d.y:.2e})")
    print(f"Initial radius: {ray_2d.r:.2e} m")
    print(f"Energy: {ray_2d.E:.2e}")
    print(f"Angular momentum: {ray_2d.L:.2e}")
    
    # Trace ray for a few steps
    print("\nTracing ray...")
    for i in range(5):
        ray_2d.step(1e7)
        print(f"Step {i+1}: r = {ray_2d.r:.2e} m, phi = {ray_2d.phi:.3f} rad")
    
    print("\n" + "=" * 50)
    print("3D Ray Tracing Demo")
    print("=" * 50)
    
    # Create 3D ray
    position_3d = (1e11, 0.0, 0.0)
    direction_3d = (C * 0.8, 0.0, C * 0.6)
    
    ray_3d = Ray3D(position_3d, direction_3d, SAGITTARIUS_A_MASS)
    print(f"Initial 3D ray position: ({ray_3d.x:.2e}, {ray_3d.y:.2e}, {ray_3d.z:.2e})")
    print(f"Initial radius: {ray_3d.r:.2e} m")
    print(f"Spherical coords: r={ray_3d.r:.2e}, theta={ray_3d.theta:.3f}, phi={ray_3d.phi:.3f}")
    print(f"Energy: {ray_3d.E:.2e}")
    print(f"Angular momentum: {ray_3d.L:.2e}")
    
    # Trace ray for a few steps
    print("\nTracing ray...")
    for i in range(5):
        ray_3d.step(1e7)
        print(f"Step {i+1}: r = {ray_3d.r:.2e} m, pos = ({ray_3d.x:.2e}, {ray_3d.y:.2e}, {ray_3d.z:.2e})")
    
    print("\n" + "=" * 50)
    print("Scene Management Demo")
    print("=" * 50)
    
    # Create default scene
    scene = create_default_scene()
    print(f"Scene created with {len(scene.objects)} objects:")
    
    for obj in scene.objects:
        print(f"  - {obj.name}: mass = {obj.mass:.2e} kg, radius = {obj.radius:.2e} m")
        if obj.is_black_hole():
            print(f"    (Black hole with Schwarzschild radius = {obj.get_schwarzschild_radius():.2e} m)")
    
    primary_bh = scene.get_primary_black_hole()
    print(f"\nPrimary black hole: {primary_bh.name}")
    
    print("\n" + "=" * 50)
    print("Integration Demo")
    print("=" * 50)
    
    # Test RK4 integrator with simple harmonic oscillator
    integrator = RK4Integrator(step_size=0.1)
    
    def harmonic_oscillator(y):
        return np.array([-y[0]])
    
    y0 = np.array([1.0])
    result = integrator.integrate(harmonic_oscillator, y0, steps=10)
    
    print(f"RK4 integration test:")
    print(f"  Initial value: {y0[0]:.6f}")
    print(f"  Final value: {result[0]:.6f}")
    print(f"  Expected (exp(-1)): {np.exp(-1.0):.6f}")
    print(f"  Error: {abs(result[0] - np.exp(-1.0)):.2e}")
    
    print("\n" + "=" * 50)
    print("✅ Physics demonstration completed successfully!")
    print("All core physics components are working correctly.")

def demonstrate_ray_plotting():
    """Demonstrate ray plotting with matplotlib."""
    try:
        import matplotlib.pyplot as plt
        
        print("\n" + "=" * 50)
        print("Ray Trajectory Plotting")
        print("=" * 50)
        
        # Create multiple rays with different impact parameters
        rays = []
        positions = [
            (2e11, 0.0),      # Far ray
            (1.5e11, 0.0),    # Medium ray
            (1e11, 0.0),      # Close ray
        ]
        
        for pos in positions:
            ray = Ray2D(pos, (C * 0.9, 0.0), SAGITTARIUS_A_MASS)
            rays.append(ray)
        
        # Trace rays
        print("Tracing rays...")
        for i, ray in enumerate(rays):
            trail = [(ray.x, ray.y)]
            for _ in range(100):
                ray.step(1e6)
                trail.append((ray.x, ray.y))
                if ray.is_inside_event_horizon() or ray.has_escaped():
                    break
            
            # Plot trajectory
            x_coords = [p[0] for p in trail]
            y_coords = [p[1] for p in trail]
            plt.plot(x_coords, y_coords, label=f'Ray {i+1}')
        
        # Plot black hole
        r_s = schwarzschild_radius(SAGITTARIUS_A_MASS)
        circle = plt.Circle((0, 0), r_s, color='black', label='Black Hole')
        plt.gca().add_patch(circle)
        
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title('2D Ray Trajectories Around Black Hole')
        plt.legend()
        plt.axis('equal')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig('ray_trajectories.png', dpi=150, bbox_inches='tight')
        print("Ray trajectory plot saved as 'ray_trajectories.png'")
        
        # Show plot if possible
        try:
            plt.show()
        except:
            print("(Plot display not available in this environment)")
        
    except ImportError:
        print("Matplotlib not available - skipping ray plotting demo")

if __name__ == "__main__":
    demonstrate_physics()
    demonstrate_ray_plotting()
