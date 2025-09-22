#!/usr/bin/env python3
"""
Test ray directions to see what's happening.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import moderngl
import glfw
import time
from engine.gpu import GPUComputePipeline
from engine.shaders import ShaderManager

def test_ray_directions():
    """Test ray directions."""
    print("🔧 Testing Ray Directions")
    print("=" * 40)
    
    # Create GLFW context
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")
    
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    
    window = glfw.create_window(800, 600, "Ray Directions Test", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")
    
    glfw.make_context_current(window)
    glfw.set_key_callback(window, lambda w, k, s, a, m: glfw.set_window_should_close(w, True) if k == glfw.KEY_ESCAPE else None)
    
    # Create ModernGL context
    ctx = moderngl.create_context()
    
    # Create GPU pipeline
    gpu_pipeline = GPUComputePipeline(ctx)
    
    # Load ray debug compute shader
    compute_source = open('assets/geodesic_ray_debug.comp', 'r').read()
    gpu_pipeline.create_compute_program(compute_source)
    
    # Load simple quad shaders
    vertex_source = open('assets/quad.vert', 'r').read()
    fragment_source = open('assets/quad_simple.frag', 'r').read()
    gpu_pipeline.create_quad_program(vertex_source, fragment_source)
    
    # Create buffers
    gpu_pipeline.create_buffers()
    gpu_pipeline.create_output_texture(400, 300)
    
    # Test with simple camera setup
    pos = np.array([5e10, 0, 0])
    right = np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])
    forward = np.array([0.0, 0.0, 1.0])
    
    print(f"Camera position: {pos}")
    print(f"Camera right: {right}")
    print(f"Camera up: {up}")
    print(f"Camera forward: {forward}")
    
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
    
    # Check for specific patterns
    red_pixels = np.all(texture_array == [255, 0, 0, 255], axis=2)
    num_red = np.sum(red_pixels)
    print(f"Red pixels (top-left corner): {num_red}")
    
    green_pixels = np.all(texture_array == [0, 255, 0, 255], axis=2)
    num_green = np.sum(green_pixels)
    print(f"Green pixels (center): {num_green}")
    
    yellow_pixels = np.all(texture_array == [255, 255, 0, 255], axis=2)
    num_yellow = np.sum(yellow_pixels)
    print(f"Yellow pixels (invalid directions): {num_yellow}")
    
    # Show some sample values
    center_region = texture_array[150:155, 200:205]
    print(f"Sample pixels (center): {center_region}")
    
    top_left_region = texture_array[0:5, 0:5]
    print(f"Sample pixels (top-left): {top_left_region}")
    
    # Render to screen
    ctx.viewport = (0, 0, 800, 600)
    ctx.clear(0.0, 0.0, 0.0, 1.0)
    ctx.disable(moderngl.DEPTH_TEST)
    gpu_pipeline.render_quad()
    glfw.swap_buffers(window)
    
    print("You should see:")
    print("- Colorful gradient based on ray directions")
    print("- Red square in top-left corner")
    print("- Green pixel in center")
    print("- Yellow pixels if ray directions are invalid")
    print("Press ESC to exit")
    
    # Keep window open
    start_time = time.time()
    while not glfw.window_should_close(window):
        glfw.poll_events()
        if time.time() - start_time > 5.0:  # Auto-close after 5 seconds
            break
    
    # Cleanup
    gpu_pipeline.cleanup()
    glfw.destroy_window(window)
    glfw.terminate()
    print("✅ Cleanup completed")

if __name__ == "__main__":
    test_ray_directions()
