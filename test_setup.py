#!/usr/bin/env python3
"""
Test script to verify the black hole simulation setup.

This script tests basic imports and functionality to ensure
the project structure is correct.
"""

import sys
import traceback

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test physics modules
        import blackhole_python.physics.constants as constants
        import blackhole_python.physics.geodesic2d as geodesic2d
        import blackhole_python.physics.geodesic3d as geodesic3d
        import blackhole_python.physics.integrators as integrators
        print("✓ Physics modules imported successfully")
        
        # Test engine modules
        import blackhole_python.engine.window as window
        import blackhole_python.engine.shaders as shaders
        import blackhole_python.engine.camera as camera
        import blackhole_python.engine.grid as grid
        import blackhole_python.engine.gpu as gpu
        print("✓ Engine modules imported successfully")
        
        # Test scene modules
        import blackhole_python.scene.objects as objects
        import blackhole_python.scene.disk as disk
        import blackhole_python.scene.ubos as ubos
        print("✓ Scene modules imported successfully")
        
        # Test demo modules
        import blackhole_python.demos.demo_2d as demo_2d
        import blackhole_python.demos.demo_3d as demo_3d
        print("✓ Demo modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        traceback.print_exc()
        return False

def test_physics():
    """Test basic physics functionality."""
    print("\nTesting physics...")
    
    try:
        from blackhole_python.physics.constants import C, G, SAGITTARIUS_A_MASS, schwarzschild_radius
        from blackhole_python.physics.geodesic2d import Ray2D
        from blackhole_python.physics.geodesic3d import Ray3D
        
        # Test constants
        assert C == 299792458.0
        assert abs(G - 6.67430e-11) < 1e-16
        print("✓ Physical constants are correct")
        
        # Test Schwarzschild radius calculation
        mass = 1.0
        r_s = schwarzschild_radius(mass)
        expected = 2.0 * G * mass / (C * C)
        assert abs(r_s - expected) < 1e-15
        print("✓ Schwarzschild radius calculation works")
        
        # Test 2D ray creation
        ray_2d = Ray2D((1e11, 0.0), (C, 0.0), SAGITTARIUS_A_MASS)
        assert ray_2d.r > 0
        assert abs(ray_2d.E) > 0
        print("✓ 2D ray creation works")
        
        # Test 3D ray creation
        ray_3d = Ray3D((1e11, 0.0, 0.0), (C, 0.0, 0.0), SAGITTARIUS_A_MASS)
        assert ray_3d.r > 0
        assert abs(ray_3d.E) > 0
        print("✓ 3D ray creation works")
        
        return True
        
    except Exception as e:
        print(f"✗ Physics test failed: {e}")
        traceback.print_exc()
        return False

def test_scene():
    """Test scene management functionality."""
    print("\nTesting scene management...")
    
    try:
        from blackhole_python.scene.objects import SceneObject, SceneManager, create_default_scene
        
        # Test scene object creation
        obj = SceneObject(
            position=(1e11, 0.0, 0.0),
            radius=1e10,
            color=(1.0, 0.0, 0.0, 1.0),
            mass=1e30,
            name="Test Object",
        )
        assert obj.name == "Test Object"
        assert obj.is_black_hole() == False
        print("✓ Scene object creation works")
        
        # Test scene manager
        scene = SceneManager()
        scene.add_object(obj)
        assert len(scene.objects) == 1
        assert scene.get_object("Test Object") is obj
        print("✓ Scene management works")
        
        # Test default scene
        default_scene = create_default_scene()
        assert len(default_scene.objects) == 3
        assert default_scene.get_primary_black_hole() is not None
        print("✓ Default scene creation works")
        
        return True
        
    except Exception as e:
        print(f"✗ Scene test failed: {e}")
        traceback.print_exc()
        return False

def test_integrators():
    """Test numerical integration methods."""
    print("\nTesting integrators...")
    
    try:
        import numpy as np
        from blackhole_python.physics.integrators import RK4Integrator, RK45Integrator
        
        # Test RK4 integrator
        integrator = RK4Integrator(step_size=0.1)
        
        def rhs(y):
            return np.array([-y[0]])
        
        y0 = np.array([1.0])
        y_final = integrator.integrate(rhs, y0, steps=10)
        assert len(y_final) == 1
        assert y_final[0] < y0[0]  # Should decrease
        print("✓ RK4 integrator works")
        
        # Test RK45 integrator
        rk45 = RK45Integrator(initial_step_size=0.1)
        y_final, t_final, success = rk45.integrate(rhs, y0, t_end=1.0)
        assert success
        assert t_final > 0
        print("✓ RK45 integrator works")
        
        return True
        
    except Exception as e:
        print(f"✗ Integrator test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Black Hole Simulation - Setup Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_physics,
        test_scene,
        test_integrators,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed! Setup is working correctly.")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
