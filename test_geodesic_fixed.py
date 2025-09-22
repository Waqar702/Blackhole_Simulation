#!/usr/bin/env python3
"""
Test the improved geodesic integration.

This script demonstrates the fixed numerical stability issues
in the geodesic integration.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib.pyplot as plt
from physics.constants import C, G, SAGITTARIUS_A_MASS, schwarzschild_radius
from physics.geodesic2d import Ray2D
from physics.geodesic3d import Ray3D

def test_2d_geodesic():
    """Test 2D geodesic integration with improved stability."""
    print("🔬 Testing 2D Geodesic Integration")
    print("=" * 50)
    
    # Create a ray at a safe distance
    position = (5e11, 0.0)  # 500 million km from center
    direction = (C * 0.8, C * 0.6)  # 80% speed of light
    
    ray = Ray2D(position, direction, SAGITTARIUS_A_MASS)
    print(f"Initial position: ({ray.x:.2e}, {ray.y:.2e})")
    print(f"Initial radius: {ray.r:.2e} m")
    print(f"Schwarzschild radius: {ray.r_s:.2e} m")
    print(f"Energy: {ray.E:.2e}")
    print(f"Angular momentum: {ray.L:.2e}")
    
    # Trace ray with smaller steps
    print("\nTracing ray with improved integration...")
    steps = 0
    max_steps = 1000
    
    while steps < max_steps:
        success = ray.step(1e6)  # 1 million meter steps
        steps += 1
        
        if not success:
            print(f"Ray stopped at step {steps}")
            break
        
        if ray.is_inside_event_horizon():
            print(f"Ray hit event horizon at step {steps}")
            break
        
        if ray.has_escaped():
            print(f"Ray escaped at step {steps}")
            break
        
        # Print progress every 100 steps
        if steps % 100 == 0:
            print(f"Step {steps}: r = {ray.r:.2e} m, phi = {ray.phi:.3f} rad")
    
    print(f"Final position: ({ray.x:.2e}, {ray.y:.2e})")
    print(f"Final radius: {ray.r:.2e} m")
    print(f"Trail length: {len(ray.trail)} points")
    
    return ray

def test_3d_geodesic():
    """Test 3D geodesic integration with improved stability."""
    print("\n🔬 Testing 3D Geodesic Integration")
    print("=" * 50)
    
    # Create a ray at a safe distance
    position = (5e11, 0.0, 0.0)  # 500 million km from center
    direction = (C * 0.8, 0.0, C * 0.6)  # 80% speed of light
    
    ray = Ray3D(position, direction, SAGITTARIUS_A_MASS)
    print(f"Initial position: ({ray.x:.2e}, {ray.y:.2e}, {ray.z:.2e})")
    print(f"Initial radius: {ray.r:.2e} m")
    print(f"Spherical: r={ray.r:.2e}, θ={ray.theta:.3f}, φ={ray.phi:.3f}")
    print(f"Energy: {ray.E:.2e}")
    print(f"Angular momentum: {ray.L:.2e}")
    
    # Trace ray with smaller steps
    print("\nTracing ray with improved integration...")
    steps = 0
    max_steps = 1000
    
    while steps < max_steps:
        success = ray.step(1e6)  # 1 million meter steps
        steps += 1
        
        if not success:
            print(f"Ray stopped at step {steps}")
            break
        
        if ray.is_inside_event_horizon():
            print(f"Ray hit event horizon at step {steps}")
            break
        
        if ray.has_escaped():
            print(f"Ray escaped at step {steps}")
            break
        
        # Print progress every 100 steps
        if steps % 100 == 0:
            print(f"Step {steps}: r = {ray.r:.2e} m, pos = ({ray.x:.2e}, {ray.y:.2e}, {ray.z:.2e})")
    
    print(f"Final position: ({ray.x:.2e}, {ray.y:.2e}, {ray.z:.2e})")
    print(f"Final radius: {ray.r:.2e} m")
    
    return ray

def test_multiple_rays():
    """Test multiple rays with different impact parameters."""
    print("\n🔬 Testing Multiple Rays")
    print("=" * 50)
    
    # Create rays at different distances
    distances = [2e11, 3e11, 4e11, 5e11]  # Different impact parameters
    rays = []
    
    for i, distance in enumerate(distances):
        position = (distance, 0.0)
        direction = (C * 0.9, 0.0)  # Radial approach
        
        ray = Ray2D(position, direction, SAGITTARIUS_A_MASS)
        rays.append(ray)
        print(f"Ray {i+1}: distance = {distance:.2e} m, r_s ratio = {distance/ray.r_s:.2f}")
    
    # Trace all rays
    print("\nTracing all rays...")
    for i, ray in enumerate(rays):
        steps = 0
        max_steps = 500
        
        while steps < max_steps:
            success = ray.step(1e6)
            steps += 1
            
            if not success or ray.is_inside_event_horizon() or ray.has_escaped():
                break
        
        print(f"Ray {i+1}: {steps} steps, final r = {ray.r:.2e} m")
    
    return rays

def plot_ray_trajectories(rays_2d, rays_3d, multiple_rays):
    """Plot ray trajectories."""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 2D ray trajectory
        if rays_2d and len(rays_2d.trail) > 1:
            trail = np.array(rays_2d.trail)
            ax1.plot(trail[:, 0], trail[:, 1], 'b-', linewidth=2, label='2D Ray')
            ax1.plot(0, 0, 'ko', markersize=10, label='Black Hole')
            
            # Event horizon circle
            r_s = rays_2d.r_s
            circle = plt.Circle((0, 0), r_s, color='black', alpha=0.3, label='Event Horizon')
            ax1.add_patch(circle)
            
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            ax1.set_title('2D Ray Trajectory')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.axis('equal')
        
        # Plot 3D ray trajectory (projected onto XY plane)
        if rays_3d:
            # Create a simple trail for 3D ray
            trail_3d = []
            ray_copy = Ray3D((5e11, 0.0, 0.0), (C * 0.8, 0.0, C * 0.6), SAGITTARIUS_A_MASS)
            steps = 0
            while steps < 500:
                success = ray_copy.step(1e6)
                steps += 1
                trail_3d.append((ray_copy.x, ray_copy.y))
                if not success or ray_copy.is_inside_event_horizon() or ray_copy.has_escaped():
                    break
            
            if trail_3d:
                trail_3d = np.array(trail_3d)
                ax2.plot(trail_3d[:, 0], trail_3d[:, 1], 'r-', linewidth=2, label='3D Ray (XY projection)')
                ax2.plot(0, 0, 'ko', markersize=10, label='Black Hole')
                
                # Event horizon circle
                r_s = ray_copy.r_s
                circle = plt.Circle((0, 0), r_s, color='black', alpha=0.3, label='Event Horizon')
                ax2.add_patch(circle)
                
                ax2.set_xlabel('X (m)')
                ax2.set_ylabel('Y (m)')
                ax2.set_title('3D Ray Trajectory (XY Projection)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.axis('equal')
        
        # Plot multiple rays
        if multiple_rays:
            for i, ray in enumerate(multiple_rays):
                if len(ray.trail) > 1:
                    trail = np.array(ray.trail)
                    ax3.plot(trail[:, 0], trail[:, 1], label=f'Ray {i+1}', linewidth=2)
            
            ax3.plot(0, 0, 'ko', markersize=10, label='Black Hole')
            
            # Event horizon circle
            if multiple_rays:
                r_s = multiple_rays[0].r_s
                circle = plt.Circle((0, 0), r_s, color='black', alpha=0.3, label='Event Horizon')
                ax3.add_patch(circle)
            
            ax3.set_xlabel('X (m)')
            ax3.set_ylabel('Y (m)')
            ax3.set_title('Multiple Ray Trajectories')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.axis('equal')
        
        # Plot radius vs step
        if rays_2d and len(rays_2d.trail) > 1:
            trail = np.array(rays_2d.trail)
            radii = np.sqrt(trail[:, 0]**2 + trail[:, 1]**2)
            steps = np.arange(len(radii))
            
            ax4.plot(steps, radii, 'b-', linewidth=2, label='2D Ray radius')
            ax4.axhline(y=rays_2d.r_s, color='r', linestyle='--', label='Event Horizon')
            ax4.set_xlabel('Step')
            ax4.set_ylabel('Radius (m)')
            ax4.set_title('Radius vs Step')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('geodesic_test_results.png', dpi=150, bbox_inches='tight')
        print("\n📊 Ray trajectory plots saved as 'geodesic_test_results.png'")
        
        # Show plot if possible
        try:
            plt.show()
        except:
            print("(Plot display not available in this environment)")
        
    except ImportError:
        print("Matplotlib not available - skipping plotting")

def main():
    """Main test function."""
    print("🌌 Geodesic Integration Test - Improved Stability")
    print("=" * 60)
    
    # Test 2D geodesic
    ray_2d = test_2d_geodesic()
    
    # Test 3D geodesic
    ray_3d = test_3d_geodesic()
    
    # Test multiple rays
    multiple_rays = test_multiple_rays()
    
    # Plot results
    plot_ray_trajectories(ray_2d, ray_3d, multiple_rays)
    
    print("\n" + "=" * 60)
    print("✅ Geodesic Integration Test Completed!")
    print("The improved integration should now be numerically stable.")
    print("Rays should trace realistic trajectories without overflow errors.")

if __name__ == "__main__":
    main()
