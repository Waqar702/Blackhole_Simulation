#!/usr/bin/env python3
"""
Minimal compute shader test to isolate the issue.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import moderngl
import glfw

def test_minimal_compute():
    """Test the most minimal compute shader possible."""
    print("🔧 Minimal Compute Shader Test")
    print("=" * 40)
    
    # Create GLFW context
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")
    
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    
    window = glfw.create_window(800, 600, "Minimal Compute Test", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")
    
    glfw.make_context_current(window)
    
    # Create ModernGL context
    ctx = moderngl.create_context()
    
    # Test 1: Absolute minimal compute shader
    minimal_shader = """
#version 430
layout(local_size_x = 16, local_size_y = 16) in;
layout(binding = 0, rgba8) writeonly uniform image2D outImage;

void main() {
    ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
    if (pix.x >= 400 || pix.y >= 300) return;
    
    // Just set every pixel to red
    imageStore(outImage, pix, vec4(1.0, 0.0, 0.0, 1.0));
}
"""
    
    try:
        compute_program = ctx.compute_shader(minimal_shader)
        print("✅ Minimal compute shader created")
    except Exception as e:
        print(f"❌ Failed to create minimal compute shader: {e}")
        return
    
    # Create output texture
    output_texture = ctx.texture((400, 300), 4, dtype='f1')
    print("✅ Output texture created")
    
    # Dispatch compute shader
    try:
        output_texture.bind_to_image(0, 0, write=True)
        compute_program.run(25, 19, 1)  # 400/16 = 25, 300/16 = 19
        ctx.finish()
        print("✅ Minimal compute shader dispatched")
    except Exception as e:
        print(f"❌ Failed to dispatch compute shader: {e}")
        return
    
    # Read back texture data
    try:
        texture_data = output_texture.read()
        texture_array = np.frombuffer(texture_data, dtype=np.uint8)
        texture_array = texture_array.reshape((300, 400, 4))
        
        # Check if we got any non-zero pixels
        non_zero_pixels = np.any(texture_array != 0, axis=2)
        num_non_zero = np.sum(non_zero_pixels)
        
        print(f"Non-zero pixels: {num_non_zero} / {400*300}")
        
        if num_non_zero > 0:
            print("✅ Minimal compute shader works!")
            # Show some sample values
            sample_pixels = texture_array[150:155, 200:205]
            print(f"Sample pixels: {sample_pixels}")
        else:
            print("❌ Minimal compute shader is not working")
            
    except Exception as e:
        print(f"❌ Failed to read texture data: {e}")
        return
    
    # Test 2: Try with uniform buffers
    print("\n🔧 Testing with uniform buffers...")
    
    # Create a simple uniform buffer
    try:
        ubo = ctx.buffer(reserve=16)  # 4 floats
        ubo_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        ubo.write(ubo_data.tobytes())
        print("✅ Uniform buffer created")
    except Exception as e:
        print(f"❌ Failed to create uniform buffer: {e}")
        return
    
    # Test 3: Try the original black hole shader with minimal changes
    print("\n🔧 Testing original shader with minimal changes...")
    
    # Create a very simple version of the black hole shader
    simple_bh_shader = """
#version 430
layout(local_size_x = 16, local_size_y = 16) in;
layout(binding = 0, rgba8) writeonly uniform image2D outImage;

layout(std140, binding = 1) uniform Camera {
    vec3 camPos;     float _pad0;
    vec3 camRight;   float _pad1;
    vec3 camUp;      float _pad2;
    vec3 camForward; float _pad3;
    float tanHalfFov;
    float aspect;
    bool moving;
    int   _pad4;
} cam;

void main() {
    ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
    if (pix.x >= 400 || pix.y >= 300) return;
    
    // Just show camera position as color
    vec3 pos = cam.camPos;
    vec4 color = vec4(abs(pos) / 1e11, 1.0);  // Normalize position
    
    imageStore(outImage, pix, color);
}
"""
    
    try:
        bh_compute_program = ctx.compute_shader(simple_bh_shader)
        print("✅ Simple black hole compute shader created")
    except Exception as e:
        print(f"❌ Failed to create simple BH compute shader: {e}")
        return
    
    # Dispatch with uniform buffer
    try:
        output_texture.bind_to_image(0, 0, write=True)
        ubo.bind_to_uniform_block(1)
        bh_compute_program.run(25, 19, 1)
        ctx.finish()
        print("✅ Simple BH compute shader dispatched")
    except Exception as e:
        print(f"❌ Failed to dispatch simple BH compute shader: {e}")
        return
    
    # Read back texture data
    try:
        texture_data = output_texture.read()
        texture_array = np.frombuffer(texture_data, dtype=np.uint8)
        texture_array = texture_array.reshape((300, 400, 4))
        
        # Check if we got any non-zero pixels
        non_zero_pixels = np.any(texture_array != 0, axis=2)
        num_non_zero = np.sum(non_zero_pixels)
        
        print(f"Non-zero pixels: {num_non_zero} / {400*300}")
        
        if num_non_zero > 0:
            print("✅ Simple BH compute shader works!")
            # Show some sample values
            sample_pixels = texture_array[150:155, 200:205]
            print(f"Sample pixels: {sample_pixels}")
        else:
            print("❌ Simple BH compute shader is not working")
            
    except Exception as e:
        print(f"❌ Failed to read texture data: {e}")
        return
    
    # Cleanup
    glfw.destroy_window(window)
    glfw.terminate()
    print("✅ Cleanup completed")

if __name__ == "__main__":
    test_minimal_compute()
