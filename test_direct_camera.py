#!/usr/bin/env python3
"""
Test direct camera positioning to see the black hole.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import moderngl
import glfw
import time
from engine.gpu import GPUComputePipeline
from engine.shaders import ShaderManager
from scene.objects import create_default_scene
from scene.disk import AccretionDisk
from physics.constants import SAGITTARIUS_A_MASS, schwarzschild_radius

def test_direct_camera():
    """Test direct camera positioning."""
    print("🔧 Testing Direct Camera Positioning")
    print("=" * 40)
    
    # Create GLFW context
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")
    
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    
    window = glfw.create_window(800, 600, "Direct Camera Test", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")
    
    glfw.make_context_current(window)
    glfw.set_key_callback(window, lambda w, k, s, a, m: glfw.set_window_should_close(w, True) if k == glfw.KEY_ESCAPE else None)
    
    # Create ModernGL context
    ctx = moderngl.create_context()
    
    # Create GPU pipeline
    gpu_pipeline = GPUComputePipeline(ctx)
    
    # Load debug compute shader
    compute_source = open('assets/geodesic_debug_detailed.comp', 'r').read()
    gpu_pipeline.create_compute_program(compute_source)
    
    # Load simple quad shaders
    vertex_source = open('assets/quad.vert', 'r').read()
    fragment_source = open('assets/quad_simple.frag', 'r').read()
    gpu_pipeline.create_quad_program(vertex_source, fragment_source)
    
    # Create buffers
    gpu_pipeline.create_buffers()
    gpu_pipeline.create_output_texture(400, 300)
    
    # Test direct camera positions
    camera_positions = [
        np.array([5e10, 0, 0]),      # X-axis
        np.array([0, 5e10, 0]),      # Y-axis  
        np.array([0, 0, 5e10]),      # Z-axis
        np.array([3e10, 3e10, 0]),   # XY plane
        np.array([1e10, 0, 0]),      # Closer X
        np.array([0, 0, 1e10]),      # Closer Z
    ]
    
    for i, pos in enumerate(camera_positions):
        print(f"\nTesting direct camera position {i+1}: {pos}")
        
        # Create scene
        scene = create_default_scene()
        
        # Create accretion disk
        disk = AccretionDisk(
            black_hole_mass=SAGITTARIUS_A_MASS,
            inner_radius_factor=2.2,
            outer_radius_factor=5.2,
        )
        
        print(f"  Distance to origin: {np.linalg.norm(pos):.2e}")
        print(f"  Black hole radius: {schwarzschild_radius(SAGITTARIUS_A_MASS):.2e}")
        print(f"  Disk inner radius: {disk.inner_radius:.2e}")
        print(f"  Disk outer radius: {disk.outer_radius:.2e}")
        
        # Update buffers with direct position
        right = np.array([1.0, 0.0, 0.0])
        up = np.array([0.0, 1.0, 0.0])
        forward = np.array([0.0, 0.0, 1.0])
        
        gpu_pipeline.update_camera_ubo(
            position=pos,
            right=right,
            up=up,
            forward=forward,
            tan_half_fov=0.577,
            aspect=4.0/3.0,
            moving=False,
        )
        
        gpu_pipeline.update_disk_ubo(
            inner_radius=disk.inner_radius,
            outer_radius=disk.outer_radius,
            num_rays=100.0,
            thickness=disk.thickness,
        )
        
        # Update objects buffer
        objects_data = []
        for obj in scene.objects:
            objects_data.append({
                'position': [obj.position[0], obj.position[1], obj.position[2]],
                'radius': obj.radius,
                'color': [obj.color[0], obj.color[1], obj.color[2], 1.0],
                'mass': obj.mass,
            })
        
        gpu_pipeline.update_objects_ubo(objects_data)
        
        # Dispatch compute shader
        gpu_pipeline.dispatch_compute(400, 300)
        
        # Read back texture data
        texture_data = gpu_pipeline.output_texture.read()
        texture_array = np.frombuffer(texture_data, dtype=np.uint8)
        texture_array = texture_array.reshape((300, 400, 4))
        
        # Count red pixels (black hole direction)
        red_pixels = np.all(texture_array[:,:,0] > 200, axis=1)
        num_red = np.sum(red_pixels)
        print(f"  Red pixels (black hole direction): {num_red}")
        
        # Count green pixels (disk region)
        green_pixels = np.all(texture_array[:,:,1] > 200, axis=1)
        num_green = np.sum(green_pixels)
        print(f"  Green pixels (disk region): {num_green}")
        
        if num_red > 0 or num_green > 0:
            print(f"  ✅ Position {i+1} shows black hole or disk!")
            
            # Render to screen
            ctx.viewport = (0, 0, 800, 600)
            ctx.clear(0.0, 0.0, 0.0, 1.0)
            ctx.disable(moderngl.DEPTH_TEST)
            gpu_pipeline.render_quad()
            glfw.swap_buffers(window)
            
            print(f"  Rendering position {i+1}... Press any key to continue")
            start_time = time.time()
            while not glfw.window_should_close(window):
                glfw.poll_events()
                if time.time() - start_time > 3.0:  # Show for 3 seconds
                    break
        else:
            print(f"  ❌ Position {i+1} shows no black hole or disk")
    
    # Cleanup
    gpu_pipeline.cleanup()
    glfw.destroy_window(window)
    glfw.terminate()
    print("✅ Cleanup completed")

if __name__ == "__main__":
    test_direct_camera()
