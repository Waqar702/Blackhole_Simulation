#!/usr/bin/env python3
"""
Test the GPU compute pipeline for ray tracing.

This script tests the complete GPU pipeline including:
- Compute shader compilation
- Uniform buffer creation and updates
- Texture rendering
- Screen quad display
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

def create_glfw_context():
    """Create GLFW context and window."""
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")
    
    # Set OpenGL version and profile
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    
    # Create window
    window = glfw.create_window(800, 600, "Black Hole GPU Test", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")
    
    glfw.make_context_current(window)
    return window

def test_gpu_pipeline():
    """Test the complete GPU pipeline."""
    print("🚀 Testing GPU Compute Pipeline")
    print("=" * 50)
    
    try:
        # Create GLFW context
        window = create_glfw_context()
        
        # Create ModernGL context
        ctx = moderngl.create_context()
        print("✅ OpenGL context created")
        
        # Create shader manager
        shader_manager = ShaderManager(ctx)
        print("✅ Shader manager created")
        
        # Create GPU pipeline
        gpu_pipeline = GPUComputePipeline(ctx)
        print("✅ GPU pipeline created")
        
        # Load compute shader
        compute_source = open('assets/geodesic.comp', 'r').read()
        gpu_pipeline.create_compute_program(compute_source)
        print("✅ Compute shader loaded")
        
        # Load quad shaders
        vertex_source = open('assets/quad.vert', 'r').read()
        fragment_source = open('assets/quad.frag', 'r').read()
        gpu_pipeline.create_quad_program(vertex_source, fragment_source)
        print("✅ Quad shaders loaded")
        
        # Create buffers
        gpu_pipeline.create_buffers()
        print("✅ Uniform buffers created")
        
        # Create output texture
        gpu_pipeline.create_output_texture(400, 300)
        print("✅ Output texture created")
        
        # Create camera
        camera = Camera(
            radius=6.34194e10,  # Default distance
            azimuth=0.0,
            elevation=np.pi/2.0,
        )
        print("✅ Camera created")
        
        # Create scene
        scene = create_default_scene()
        print("✅ Scene created")
        
        # Create accretion disk
        disk = AccretionDisk(
            black_hole_mass=SAGITTARIUS_A_MASS,
            inner_radius_factor=2.2,
            outer_radius_factor=5.2,
        )
        print("✅ Accretion disk created")
        
        # Test buffer updates
        print("\n🔧 Testing Buffer Updates")
        print("-" * 30)
        
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
            tan_half_fov=0.577,  # 60 degrees
            aspect=4.0/3.0,
            moving=False,
        )
        print("✅ Camera buffer updated")
        
        # Update disk buffer
        gpu_pipeline.update_disk_ubo(
            inner_radius=disk.inner_radius,
            outer_radius=disk.outer_radius,
            num_rays=100.0,
            thickness=disk.thickness,
        )
        print("✅ Disk buffer updated")
        
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
        print("✅ Objects buffer updated")
        
        # Test compute shader dispatch
        print("\n🎯 Testing Compute Shader Dispatch")
        print("-" * 40)
        
        # Dispatch compute shader
        gpu_pipeline.dispatch_compute(400, 300)
        print("✅ Compute shader dispatched")
        
        # Test rendering
        print("\n🖼️  Testing Rendering")
        print("-" * 25)
        
        # Set viewport
        ctx.viewport = (0, 0, 800, 600)
        
        # Clear screen
        ctx.clear(0.1, 0.1, 0.1, 1.0)
        
        # Render quad
        gpu_pipeline.render_quad()
        print("✅ Screen quad rendered")
        
        # Swap buffers
        glfw.swap_buffers(window)
        print("✅ Buffers swapped")
        
        # Wait for user input
        print("\n🎉 GPU Pipeline Test Successful!")
        print("Press any key to close the window...")
        
        while not glfw.window_should_close(window):
            glfw.poll_events()
            
            # Check for key press
            if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
                break
        
        # Cleanup
        gpu_pipeline.cleanup()
        shader_manager.cleanup()
        glfw.destroy_window(window)
        glfw.terminate()
        
        print("✅ Cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU Pipeline Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_shader_compilation():
    """Test shader compilation separately."""
    print("\n🔧 Testing Shader Compilation")
    print("=" * 40)
    
    try:
        # Create GLFW context
        window = create_glfw_context()
        
        # Create ModernGL context
        ctx = moderngl.create_context()
        
        # Create shader manager
        shader_manager = ShaderManager(ctx)
        
        # Test compute shader
        compute_source = open('assets/geodesic.comp', 'r').read()
        compute_program = shader_manager.create_inline_shader(
            "geodesic_compute",
            compute_source=compute_source
        )
        print("✅ Compute shader compiled")
        
        # Test quad shaders
        vertex_source = open('assets/quad.vert', 'r').read()
        fragment_source = open('assets/quad.frag', 'r').read()
        quad_program = shader_manager.create_inline_shader(
            "quad_render",
            vertex_source=vertex_source,
            fragment_source=fragment_source
        )
        print("✅ Quad shaders compiled")
        
        # Cleanup
        shader_manager.cleanup()
        glfw.destroy_window(window)
        glfw.terminate()
        
        return True
        
    except Exception as e:
        print(f"❌ Shader Compilation Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🌌 GPU Compute Pipeline Test")
    print("=" * 60)
    
    # Test shader compilation first
    if not test_shader_compilation():
        print("❌ Shader compilation test failed")
        return
    
    # Test full pipeline
    if test_gpu_pipeline():
        print("\n🎉 All GPU Pipeline Tests Passed!")
        print("The GPU compute pipeline is working correctly.")
    else:
        print("\n❌ GPU Pipeline Tests Failed!")
        print("Check the error messages above for details.")

if __name__ == "__main__":
    main()
