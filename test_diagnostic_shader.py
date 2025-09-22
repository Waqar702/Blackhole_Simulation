#!/usr/bin/env python3
"""
Test the diagnostic shader to ensure we can see output.
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

def test_diagnostic_shader():
    """Test the diagnostic shader that should definitely show something."""
    print("🔍 Testing Diagnostic Shader")
    print("=" * 40)
    
    # Create GLFW context
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")
    
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    
    window = glfw.create_window(800, 600, "Diagnostic Shader Test", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")
    
    glfw.make_context_current(window)
    
    # Create ModernGL context
    ctx = moderngl.create_context()
    
    # Create GPU pipeline
    gpu_pipeline = GPUComputePipeline(ctx)
    
    # Load diagnostic compute shader
    compute_source = open('assets/geodesic_diagnostic.comp', 'r').read()
    gpu_pipeline.create_compute_program(compute_source)
    
    # Load quad shaders
    vertex_source = open('assets/quad.vert', 'r').read()
    fragment_source = open('assets/quad.frag', 'r').read()
    gpu_pipeline.create_quad_program(vertex_source, fragment_source)
    
    # Create buffers
    gpu_pipeline.create_buffers()
    
    # Create output texture
    gpu_pipeline.create_output_texture(400, 300)
    
    # Create camera
    camera = Camera(
        radius=5e10,
        azimuth=0.0,
        elevation=np.pi/2.0,
    )
    
    print(f"Camera position: {camera.get_position()}")
    
    # Update camera buffer
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
        print("✅ Diagnostic shader is producing output!")
        # Show some sample values
        center_region = texture_array[150:155, 200:205]
        print(f"Sample pixels (center): {center_region}")
        
        # Show the range of values
        max_values = np.max(texture_array, axis=2)
        min_values = np.min(texture_array, axis=2)
        print(f"Pixel value range: {np.min(min_values)} - {np.max(max_values)}")
        
        # Check for specific patterns
        red_pixels = np.all(texture_array == [255, 0, 0, 255], axis=2)
        num_red = np.sum(red_pixels)
        print(f"Red pixels (circle): {num_red}")
        
    else:
        print("❌ Diagnostic shader is not producing output")
        return False
    
    # Render to screen
    print("Rendering to screen...")
    ctx.viewport = (0, 0, 800, 600)
    ctx.clear(0.0, 0.0, 0.0, 1.0)
    gpu_pipeline.render_quad()
    glfw.swap_buffers(window)
    
    print("You should see:")
    print("- A gradient background (red to yellow)")
    print("- A red circle in the center")
    print("- Some noise/texture")
    print("Press ESC to exit...")
    
    # Keep window open for a few seconds
    start_time = time.time()
    while not glfw.window_should_close(window):
        glfw.poll_events()
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break
        if time.time() - start_time > 5.0:  # Auto-close after 5 seconds
            break
    
    # Cleanup
    gpu_pipeline.cleanup()
    glfw.destroy_window(window)
    glfw.terminate()
    print("✅ Cleanup completed")
    return True

if __name__ == "__main__":
    test_diagnostic_shader()
