#!/usr/bin/env python3
"""
Test geodesic integration with very small step sizes.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from physics.constants import C, G, SAGITTARIUS_A_MASS, schwarzschild_radius
from physics.geodesic2d import Ray2D

def test_small_steps():
    """Test with very small step sizes."""
    print("🔬 Testing Geodesic Integration with Small Steps")
    print("=" * 60)
    
    # Create a ray at a safe distance
    position = (1e12, 0.0)  # 1 trillion km from center
    direction = (C * 0.5, C * 0.5)  # 50% speed of light
    
    ray = Ray2D(position, direction, SAGITTARIUS_A_MASS)
    print(f"Initial position: ({ray.x:.2e}, {ray.y:.2e})")
    print(f"Initial radius: {ray.r:.2e} m")
    print(f"Schwarzschild radius: {ray.r_s:.2e} m")
    print(f"Distance ratio: {ray.r / ray.r_s:.2f}")
    print(f"Energy: {ray.E:.2e}")
    print(f"Angular momentum: {ray.L:.2e}")
    
    # Trace ray with very small steps
    print("\nTracing ray with very small steps...")
    steps = 0
    max_steps = 10000
    
    while steps < max_steps:
        # Use very small step size
        success = ray.step(1e4)  # 10 km steps
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
        
        # Print progress every 1000 steps
        if steps % 1000 == 0:
            print(f"Step {steps}: r = {ray.r:.2e} m, phi = {ray.phi:.3f} rad, pos = ({ray.x:.2e}, {ray.y:.2e})")
    
    print(f"\nFinal results:")
    print(f"Total steps: {steps}")
    print(f"Final position: ({ray.x:.2e}, {ray.y:.2e})")
    print(f"Final radius: {ray.r:.2e} m")
    print(f"Trail length: {len(ray.trail)} points")
    
    return ray

def test_conservation():
    """Test conservation of energy and angular momentum."""
    print("\n🔬 Testing Conservation Laws")
    print("=" * 40)
    
    position = (1e12, 0.0)
    direction = (C * 0.3, C * 0.7)
    
    ray = Ray2D(position, direction, SAGITTARIUS_A_MASS)
    initial_E = ray.E
    initial_L = ray.L
    
    print(f"Initial E: {initial_E:.2e}")
    print(f"Initial L: {initial_L:.2e}")
    
    # Take many small steps
    for i in range(1000):
        success = ray.step(1e4)
        if not success:
            break
        
        if i % 100 == 0:
            current_E, current_L = ray.get_conserved_quantities()
            E_error = abs(current_E - initial_E) / initial_E
            L_error = abs(current_L - initial_L) / initial_L
            
            print(f"Step {i}: E_error = {E_error:.2e}, L_error = {L_error:.2e}")
    
    final_E, final_L = ray.get_conserved_quantities()
    final_E_error = abs(final_E - initial_E) / initial_E
    final_L_error = abs(final_L - initial_L) / initial_L
    
    print(f"\nFinal conservation errors:")
    print(f"Energy error: {final_E_error:.2e}")
    print(f"Angular momentum error: {final_L_error:.2e}")
    
    return ray

if __name__ == "__main__":
    ray1 = test_small_steps()
    ray2 = test_conservation()
    
    print("\n" + "=" * 60)
    print("✅ Small Step Test Completed!")
    print("The geodesic integration should now work with small steps.")
