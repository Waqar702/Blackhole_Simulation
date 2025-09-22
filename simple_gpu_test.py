#!/usr/bin/env python3
"""
Simple GPU test to verify the compute shader is working.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import moderngl
import glfw

def create_simple_compute_shader():
    """Create a simple compute shader that just draws a pattern."""
    return """
#version 430
layout(local_size_x = 16, local_size_y = 16) in;

layout(binding = 0, rgba8) writeonly uniform image2D outImage;

void main() {
    ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
    
    if (pix.x >= 400 || pix.y >= 300) return;
    
    // Create a simple pattern
    float x = float(pix.x) / 400.0;
    float y = float(pix.y) / 300.0;
    
    // Draw a gradient
    vec4 color = vec4(x, y, 0.5, 1.0);
    
    // Draw a circle in the center
    vec2 center = vec2(0.5, 0.5);
    float dist = distance(vec2(x, y), center);
    if (dist < 0.2) {
        color = vec4(1.0, 0.0, 0.0, 1.0);  // Red circle
    }
    
    imageStore(outImage, pix, color);
}
"""

def create_simple_quad_shaders():
    """Create simple quad shaders."""
    vertex_source = """
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;
out vec2 TexCoord;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoord = aTexCoord;
}
"""

    fragment_source = """
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D screenTexture;

void main() {
    FragColor = texture(screenTexture, TexCoord);
}
"""
    
    return vertex_source, fragment_source

def test_simple_gpu():
    """Test simple GPU rendering."""
    print("🔧 Testing Simple GPU Rendering")
    print("=" * 40)
    
    # Create GLFW context
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")
    
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    
    window = glfw.create_window(800, 600, "Simple GPU Test", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")
    
    glfw.make_context_current(window)
    
    # Create ModernGL context
    ctx = moderngl.create_context()
    
    # Create compute shader
    compute_source = create_simple_compute_shader()
    compute_program = ctx.compute_shader(compute_source)
    print("✅ Simple compute shader created")
    
    # Create quad shaders
    vertex_source, fragment_source = create_simple_quad_shaders()
    quad_program = ctx.program(
        vertex_shader=vertex_source,
        fragment_shader=fragment_source,
    )
    print("✅ Simple quad shaders created")
    
    # Create output texture
    output_texture = ctx.texture((400, 300), 4, dtype='f1')
    output_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
    print("✅ Output texture created")
    
    # Create screen quad
    quad_vertices = np.array([
        # positions   # texCoords
        -1.0,  1.0,  0.0, 1.0,  # top left
        -1.0, -1.0,  0.0, 0.0,  # bottom left
         1.0, -1.0,  1.0, 0.0,  # bottom right
        -1.0,  1.0,  0.0, 1.0,  # top left
         1.0, -1.0,  1.0, 0.0,  # bottom right
         1.0,  1.0,  1.0, 1.0   # top right
    ], dtype=np.float32)
    
    vbo = ctx.buffer(quad_vertices.tobytes())
    quad_vao = ctx.vertex_array(
        quad_program,
        [(vbo, '2f 2f', 'aPos', 'aTexCoord')],
    )
    print("✅ Screen quad created")
    
    # Dispatch compute shader
    output_texture.bind_to_image(0, 0, write=True)
    compute_program.run(25, 19, 1)  # 400/16 = 25, 300/16 = 19
    ctx.finish()
    print("✅ Compute shader dispatched")
    
    # Read back texture data
    texture_data = output_texture.read()
    texture_array = np.frombuffer(texture_data, dtype=np.uint8)
    texture_array = texture_array.reshape((300, 400, 4))
    
    # Check if we got any non-zero pixels
    non_zero_pixels = np.any(texture_array != 0, axis=2)
    num_non_zero = np.sum(non_zero_pixels)
    
    print(f"Non-zero pixels: {num_non_zero} / {400*300}")
    
    if num_non_zero > 0:
        print("✅ Compute shader is producing output!")
        # Show some sample values
        center_region = texture_array[150:155, 200:205]
        print(f"Sample pixels (center): {center_region}")
    else:
        print("❌ Compute shader is not producing output")
    
    # Render to screen
    print("Rendering to screen...")
    ctx.viewport = (0, 0, 800, 600)
    ctx.clear(0.0, 0.0, 0.0, 1.0)
    
    # Disable depth test
    ctx.disable(moderngl.DEPTH_TEST)
    
    # Bind texture and render
    output_texture.use(0)
    quad_program['screenTexture'].value = 0
    quad_vao.render()
    
    # Swap buffers
    glfw.swap_buffers(window)
    
    print("Press ESC to exit...")
    while not glfw.window_should_close(window):
        glfw.poll_events()
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            break
    
    # Cleanup
    glfw.destroy_window(window)
    glfw.terminate()
    print("✅ Cleanup completed")

if __name__ == "__main__":
    test_simple_gpu()
