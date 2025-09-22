#!/usr/bin/env python3
"""
Debug GPU output to see what the compute shader is producing.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import moderngl
import glfw
from engine.gpu import GPUComputePipeline
from engine.shaders import ShaderManager
from engine.camera import Camera
from scene.objects import create_default_scene
from scene.disk import AccretionDisk
from physics.constants import SAGITTARIUS_A_MASS, schwarzschild_radius

def debug_gpu_output():
    """Debug what the GPU is actually producing."""
    print("🔍 Debugging GPU Output")
    print("=" * 40)
    
    # Create GLFW context
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")
    
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    
    window = glfw.create_window(800, 600, "GPU Debug", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")
    
    glfw.make_context_current(window)
    
    # Create ModernGL context
    ctx = moderngl.create_context()
    
    # Create GPU pipeline
    gpu_pipeline = GPUComputePipeline(ctx)
    
    # Load compute shader
    compute_source = open('assets/geodesic.comp', 'r').read()
    gpu_pipeline.create_compute_program(compute_source)
    
    # Load quad shaders
    vertex_source = open('assets/quad.vert', 'r').read()
    fragment_source = open('assets/quad.frag', 'r').read()
    gpu_pipeline.create_quad_program(vertex_source, fragment_source)
    
    # Create buffers
    gpu_pipeline.create_buffers()
    
    # Create output texture
    gpu_pipeline.create_output_texture(400, 300)
    
    # Create camera - try different positions
    print("Testing different camera positions...")
    
    positions = [
        (6.34194e10, 0.0, 0.0),      # Default position
        (1e11, 0.0, 0.0),            # Closer to black hole
        (1e12, 0.0, 0.0),            # Far from black hole
        (5e10, 5e10, 0.0),           # Diagonal position
    ]
    
    for i, pos in enumerate(positions):
        print(f"\nPosition {i+1}: {pos}")
        
        # Create camera
        camera = Camera(
            radius=np.linalg.norm(pos),
            azimuth=np.arctan2(pos[1], pos[0]),
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
        
        print(f"  Camera position: {camera.get_position()}")
        print(f"  Black hole radius: {schwarzschild_radius(SAGITTARIUS_A_MASS):.2e}")
        print(f"  Disk inner radius: {disk.inner_radius:.2e}")
        print(f"  Disk outer radius: {disk.outer_radius:.2e}")
        
        # Update buffers
        camera_pos = camera.get_position()
        right = np.array([1.0, 0.0, 0.0])
        up = np.array([0.0, 1.0, 0.0])
        forward = np.array([0.0, 0.0, 1.0])
        
        gpu_pipeline.update_camera_ubo(
            position=camera_pos,
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
        
        # Read back texture data to see what was produced
        texture_data = gpu_pipeline.output_texture.read()
        texture_array = np.frombuffer(texture_data, dtype=np.uint8)
        texture_array = texture_array.reshape((300, 400, 4))  # RGBA
        
        # Analyze the output
        non_black_pixels = np.any(texture_array != 0, axis=2)
        num_non_black = np.sum(non_black_pixels)
        
        print(f"  Non-black pixels: {num_non_black} / {400*300}")
        
        if num_non_black > 0:
            # Find the range of values
            max_values = np.max(texture_array, axis=2)
            min_values = np.min(texture_array, axis=2)
            print(f"  Pixel value range: {np.min(min_values)} - {np.max(max_values)}")
            
            # Show some sample pixel values
            sample_pixels = texture_array[150:155, 200:205]  # Center region
            print(f"  Sample pixels (center): {sample_pixels}")
        
        # Render to screen
        ctx.viewport = (0, 0, 800, 600)
        ctx.clear(0.1, 0.1, 0.1, 1.0)
        gpu_pipeline.render_quad()
        glfw.swap_buffers(window)
        
        # Wait for key press
        print("  Press any key to continue...")
        while not glfw.window_should_close(window):
            glfw.poll_events()
            if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
                break
            if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS:
                break
    
    # Cleanup
    gpu_pipeline.cleanup()
    glfw.destroy_window(window)
    glfw.terminate()

if __name__ == "__main__":
    debug_gpu_output()
