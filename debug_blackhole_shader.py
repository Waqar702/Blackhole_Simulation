#!/usr/bin/env python3
"""
Debug the black hole shader to see what's happening.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import moderngl
import glfw
import time
from engine.gpu import GPUComputePipeline
from engine.shaders import ShaderManager
from engine.camera import Camera
from scene.objects import create_default_scene
from scene.disk import AccretionDisk
from physics.constants import SAGITTARIUS_A_MASS, schwarzschild_radius

def debug_blackhole_shader():
    """Debug the black hole shader step by step."""
    print("🔍 Debugging Black Hole Shader")
    print("=" * 40)
    
    # Create GLFW context
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")
    
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    
    window = glfw.create_window(800, 600, "Black Hole Shader Debug", None, None)
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
    
    # Create camera
    camera = Camera(
        radius=5e10,
        azimuth=0.0,
        elevation=np.pi/2.0,
    )
    
    # Create scene
    scene = create_default_scene()
    
    # Create accretion disk
    disk = AccretionDisk(
        black_hole_mass=SAGITTARIUS_A_MASS,
        inner_radius_factor=2.2,
        outer_radius_factor=5.2,
    )
    
    print(f"Camera position: {camera.get_position()}")
    print(f"Black hole radius: {schwarzschild_radius(SAGITTARIUS_A_MASS):.2e}")
    print(f"Disk inner radius: {disk.inner_radius:.2e}")
    print(f"Disk outer radius: {disk.outer_radius:.2e}")
    
    # Update buffers
    pos = camera.get_position()
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
    
    # Analyze the output
    non_black_pixels = np.any(texture_array != 0, axis=2)
    num_non_black = np.sum(non_black_pixels)
    
    print(f"Non-black pixels: {num_non_black} / {400*300}")
    
    if num_non_black > 0:
        print("✅ Debug shader is producing output!")
        # Show some sample values
        center_region = texture_array[150:155, 200:205]
        print(f"Sample pixels (center): {center_region}")
        
        # Count red pixels (black hole direction)
        red_pixels = np.all(texture_array[:,:,0] > 200, axis=1)
        num_red = np.sum(red_pixels)
        print(f"Red pixels (black hole direction): {num_red}")
        
        # Count green pixels (disk region)
        green_pixels = np.all(texture_array[:,:,1] > 200, axis=1)
        num_green = np.sum(green_pixels)
        print(f"Green pixels (disk region): {num_green}")
        
    else:
        print("❌ Debug shader is not producing output")
        return False
    
    # Render to screen
    print("Rendering debug output...")
    ctx.viewport = (0, 0, 800, 600)
    ctx.clear(0.0, 0.0, 0.0, 1.0)
    ctx.disable(moderngl.DEPTH_TEST)
    gpu_pipeline.render_quad()
    glfw.swap_buffers(window)
    
    print("You should see:")
    print("- Colorful gradient based on ray direction")
    print("- Red areas where rays point toward black hole")
    print("- Green areas in disk region")
    print("- Press ESC to exit")
    
    # Keep window open
    start_time = time.time()
    while not glfw.window_should_close(window):
        glfw.poll_events()
        if time.time() - start_time > 10.0:  # Auto-close after 10 seconds
            break
    
    # Cleanup
    gpu_pipeline.cleanup()
    glfw.destroy_window(window)
    glfw.terminate()
    print("✅ Cleanup completed")
    return True

if __name__ == "__main__":
    debug_blackhole_shader()
