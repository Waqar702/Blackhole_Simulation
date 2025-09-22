#!/usr/bin/env python3
"""
Debug the geodesic integration to see what's happening.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from physics.constants import C, G, SAGITTARIUS_A_MASS, schwarzschild_radius
from physics.geodesic2d import Ray2D

def debug_geodesic_step():
    """Debug a single geodesic step."""
    print("🔍 Debugging Geodesic Integration")
    print("=" * 50)
    
    # Create a ray
    position = (5e11, 0.0)  # 500 million km
    direction = (C * 0.8, C * 0.6)
    
    ray = Ray2D(position, direction, SAGITTARIUS_A_MASS)
    print(f"Initial state:")
    print(f"  r = {ray.r:.2e}")
    print(f"  phi = {ray.phi:.3f}")
    print(f"  dr = {ray.dr:.2e}")
    print(f"  dphi = {ray.dphi:.2e}")
    print(f"  r_s = {ray.r_s:.2e}")
    
    # Check initial state validity
    y0 = np.array([ray.r, ray.phi, ray.dr, ray.dphi], dtype=np.float64)
    print(f"\nInitial state validity: {ray._is_valid_state(y0)}")
    
    # Calculate geodesic RHS
    rhs = ray._geodesic_rhs(y0)
    print(f"\nGeodesic RHS:")
    print(f"  dr_dlambda = {rhs[0]:.2e}")
    print(f"  dphi_dlambda = {rhs[1]:.2e}")
    print(f"  d2r_dlambda2 = {rhs[2]:.2e}")
    print(f"  d2phi_dlambda2 = {rhs[3]:.2e}")
    
    # Test RK4 step
    dlambda = 1e6  # 1 million meters
    min_step = ray.r_s * 1e-6
    max_step = dlambda
    adaptive_step = min(max_step, max(min_step, ray.r * 0.001))
    
    print(f"\nStep calculation:")
    print(f"  dlambda = {dlambda:.2e}")
    print(f"  min_step = {min_step:.2e}")
    print(f"  max_step = {max_step:.2e}")
    print(f"  adaptive_step = {adaptive_step:.2e}")
    
    # RK4 stages
    k1 = ray._geodesic_rhs(y0)
    print(f"\nRK4 Stage 1:")
    print(f"  k1 = {k1}")
    
    y1 = y0 + 0.5 * adaptive_step * k1
    print(f"  y1 = {y1}")
    print(f"  y1 validity: {ray._is_valid_state(y1)}")
    
    if ray._is_valid_state(y1):
        k2 = ray._geodesic_rhs(y1)
        print(f"\nRK4 Stage 2:")
        print(f"  k2 = {k2}")
        
        y2 = y0 + 0.5 * adaptive_step * k2
        print(f"  y2 = {y2}")
        print(f"  y2 validity: {ray._is_valid_state(y2)}")
        
        if ray._is_valid_state(y2):
            k3 = ray._geodesic_rhs(y2)
            print(f"\nRK4 Stage 3:")
            print(f"  k3 = {k3}")
            
            y3 = y0 + adaptive_step * k3
            print(f"  y3 = {y3}")
            print(f"  y3 validity: {ray._is_valid_state(y3)}")
            
            if ray._is_valid_state(y3):
                k4 = ray._geodesic_rhs(y3)
                print(f"\nRK4 Stage 4:")
                print(f"  k4 = {k4}")
                
                y_new = y0 + (adaptive_step / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                print(f"\nFinal state:")
                print(f"  y_new = {y_new}")
                print(f"  y_new validity: {ray._is_valid_state(y_new)}")
                
                print(f"\nNew values:")
                print(f"  r_new = {y_new[0]:.2e}")
                print(f"  phi_new = {y_new[1]:.3f}")
                print(f"  dr_new = {y_new[2]:.2e}")
                print(f"  dphi_new = {y_new[3]:.2e}")
            else:
                print("Stage 3 failed validation")
        else:
            print("Stage 2 failed validation")
    else:
        print("Stage 1 failed validation")

if __name__ == "__main__":
    debug_geodesic_step()
