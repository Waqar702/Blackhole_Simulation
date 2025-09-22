#!/usr/bin/env python3
"""
Fix the camera UBO data packing issue.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import moderngl
import glfw
from engine.camera import Camera

def test_camera_ubo_packing():
    """Test camera UBO data packing."""
    print("🔧 Testing Camera UBO Data Packing")
    print("=" * 40)
    
    # Create GLFW context
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")
    
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    
    window = glfw.create_window(800, 600, "Camera UBO Test", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")
    
    glfw.make_context_current(window)
    
    # Create ModernGL context
    ctx = moderngl.create_context()
    
    # Create camera
    camera = Camera(
        radius=5e10,
        azimuth=0.0,
        elevation=np.pi/2.0,
    )
    
    pos = camera.get_position()
    print(f"Camera position: {pos}")
    print(f"Position type: {type(pos)}")
    print(f"Position shape: {pos.shape}")
    
    # Test the current UBO packing method
    data = np.zeros(32, dtype=np.float32)  # 32 floats = 128 bytes
    
    # Position (vec3 + padding)
    data[0:3] = pos.astype(np.float32)
    
    # Right vector (vec3 + padding)
    data[4:7] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    
    # Up vector (vec3 + padding)
    data[8:11] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    
    # Forward vector (vec3 + padding)
    data[12:15] = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    
    # tanHalfFov, aspect, moving, padding
    data[16] = 0.577
    data[17] = 4.0/3.0
    data[18] = 1.0  # moving = true
    data[19] = 0.0  # padding
    
    print(f"UBO data first 20 elements: {data[:20]}")
    print(f"Position in UBO: {data[0:3]}")
    print(f"Right in UBO: {data[4:7]}")
    print(f"Up in UBO: {data[8:11]}")
    print(f"Forward in UBO: {data[12:15]}")
    print(f"tanHalfFov: {data[16]}")
    print(f"aspect: {data[17]}")
    print(f"moving: {data[18]}")
    
    # Create uniform buffer
    ubo = ctx.buffer(reserve=128)
    ubo.write(data.tobytes())
    
    # Test shader that reads the camera data
    test_shader = """
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
    
    // Show camera position as color
    vec3 pos = cam.camPos;
    vec4 color = vec4(abs(pos) / 1e11, 1.0);  // Normalize position
    
    // Make sure we see something
    if (length(pos) < 1e9) {
        color = vec4(1.0, 0.0, 0.0, 1.0);  // Red if position is too small
    }
    
    imageStore(outImage, pix, color);
}
"""
    
    try:
        compute_program = ctx.compute_shader(test_shader)
        print("✅ Test compute shader created")
    except Exception as e:
        print(f"❌ Failed to create test compute shader: {e}")
        return
    
    # Create output texture
    output_texture = ctx.texture((400, 300), 4, dtype='f1')
    
    # Dispatch compute shader
    output_texture.bind_to_image(0, 0, write=True)
    ubo.bind_to_uniform_block(1)
    compute_program.run(25, 19, 1)
    ctx.finish()
    
    # Read back texture data
    texture_data = output_texture.read()
    texture_array = np.frombuffer(texture_data, dtype=np.uint8)
    texture_array = texture_array.reshape((300, 400, 4))
    
    # Check if we got any non-zero pixels
    non_zero_pixels = np.any(texture_array != 0, axis=2)
    num_non_zero = np.sum(non_zero_pixels)
    
    print(f"Non-zero pixels: {num_non_zero} / {400*300}")
    
    if num_non_zero > 0:
        print("✅ Camera UBO is working!")
        # Show some sample values
        sample_pixels = texture_array[150:155, 200:205]
        print(f"Sample pixels: {sample_pixels}")
        
        # Check if we got red pixels (indicating position was too small)
        red_pixels = np.all(texture_array == [255, 0, 0, 255], axis=2)
        num_red = np.sum(red_pixels)
        if num_red > 0:
            print(f"Got {num_red} red pixels - camera position was too small")
        else:
            print("Got colored pixels - camera position is being read correctly")
    else:
        print("❌ Camera UBO is not working")
    
    # Cleanup
    glfw.destroy_window(window)
    glfw.terminate()
    print("✅ Cleanup completed")

if __name__ == "__main__":
    test_camera_ubo_packing()
