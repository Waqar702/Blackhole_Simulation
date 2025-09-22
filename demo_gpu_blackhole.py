#!/usr/bin/env python3
"""
GPU Black Hole Visualization Demo

This demo shows the actual black hole simulation running on GPU
with proper camera controls and real-time rendering.
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

class BlackHoleDemo:
    """Black hole visualization demo with GPU rendering."""
    
    def __init__(self, width=800, height=600):
        """Initialize the demo."""
        self.width = width
        self.height = height
        self.window = None
        self.ctx = None
        self.gpu_pipeline = None
        self.shader_manager = None
        self.camera = None
        self.scene = None
        self.disk = None
        
        # Demo parameters
        self.running = True
        self.last_time = time.time()
        self.frame_count = 0
        self.fps = 0.0
        
        # Camera movement
        self.camera_moving = False
        self.mouse_pos = (0.0, 0.0)
        self.mouse_pressed = False
        
    def create_window(self):
        """Create GLFW window and OpenGL context."""
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        # Set OpenGL version and profile
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        
        # Create window
        self.window = glfw.create_window(
            self.width, self.height, 
            "Black Hole GPU Visualization", None, None
        )
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        
        glfw.make_context_current(self.window)
        
        # Set callbacks
        glfw.set_key_callback(self.window, self.key_callback)
        glfw.set_cursor_pos_callback(self.window, self.cursor_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_callback)
        glfw.set_scroll_callback(self.window, self.scroll_callback)
        
        # Create ModernGL context
        self.ctx = moderngl.create_context()
        
        print("✅ Window and OpenGL context created")
    
    def key_callback(self, window, key, scancode, action, mods):
        """Handle keyboard input."""
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                self.running = False
            elif key == glfw.KEY_R:
                # Reset camera
                self.camera = Camera(
                    radius=6.34194e10,
                    azimuth=0.0,
                    elevation=np.pi/2.0,
                )
            elif key == glfw.KEY_SPACE:
                # Toggle camera movement
                self.camera_moving = not self.camera_moving
                print(f"Camera moving: {self.camera_moving}")
    
    def cursor_callback(self, window, xpos, ypos):
        """Handle mouse movement."""
        self.mouse_pos = (xpos, ypos)
        
        if self.mouse_pressed:
            # Convert mouse position to camera angles
            sensitivity = 0.01
            delta_x = (xpos - self.width/2) * sensitivity
            delta_y = (ypos - self.height/2) * sensitivity
            
            # Update camera
            self.camera.set_angles(
                self.camera.azimuth + delta_x,
                self.camera.elevation + delta_y
            )
    
    def mouse_callback(self, window, button, action, mods):
        """Handle mouse button presses."""
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.mouse_pressed = (action == glfw.PRESS)
    
    def scroll_callback(self, window, xoffset, yoffset):
        """Handle mouse scroll for zoom."""
        zoom_factor = 1.1
        if yoffset > 0:
            self.camera.radius *= zoom_factor
        else:
            self.camera.radius /= zoom_factor
        
        # Clamp radius
        self.camera.radius = max(1e10, min(1e12, self.camera.radius))
    
    def setup_gpu_pipeline(self):
        """Set up the GPU compute pipeline."""
        # Create shader manager
        self.shader_manager = ShaderManager(self.ctx)
        
        # Create GPU pipeline
        self.gpu_pipeline = GPUComputePipeline(self.ctx)
        
        # Load compute shader
        compute_source = open('assets/geodesic.comp', 'r').read()
        self.gpu_pipeline.create_compute_program(compute_source)
        
        # Load quad shaders
        vertex_source = open('assets/quad.vert', 'r').read()
        fragment_source = open('assets/quad.frag', 'r').read()
        self.gpu_pipeline.create_quad_program(vertex_source, fragment_source)
        
        # Create buffers
        self.gpu_pipeline.create_buffers()
        
        # Create output texture
        self.gpu_pipeline.create_output_texture(400, 300)
        
        print("✅ GPU pipeline set up")
    
    def setup_scene(self):
        """Set up the black hole scene."""
        # Create camera
        self.camera = Camera(
            radius=6.34194e10,  # Default distance
            azimuth=0.0,
            elevation=np.pi/2.0,
        )
        
        # Create scene
        self.scene = create_default_scene()
        
        # Create accretion disk
        self.disk = AccretionDisk(
            black_hole_mass=SAGITTARIUS_A_MASS,
            inner_radius_factor=2.2,
            outer_radius_factor=5.2,
        )
        
        print("✅ Scene set up")
    
    def update_buffers(self):
        """Update GPU buffers with current scene data."""
        # Update camera buffer
        pos = self.camera.get_position()
        right = np.array([1.0, 0.0, 0.0])
        up = np.array([0.0, 1.0, 0.0])
        forward = np.array([0.0, 0.0, 1.0])
        
        self.gpu_pipeline.update_camera_ubo(
            position=pos,
            right=right,
            up=up,
            forward=forward,
            tan_half_fov=0.577,  # 60 degrees
            aspect=float(self.width) / float(self.height),
            moving=self.camera_moving,
        )
        
        # Update disk buffer
        self.gpu_pipeline.update_disk_ubo(
            inner_radius=self.disk.inner_radius,
            outer_radius=self.disk.outer_radius,
            num_rays=100.0,
            thickness=self.disk.thickness,
        )
        
        # Update objects buffer
        objects_data = []
        for obj in self.scene.objects:
            objects_data.append({
                'position': [obj.position[0], obj.position[1], obj.position[2]],
                'radius': obj.radius,
                'color': [obj.color[0], obj.color[1], obj.color[2], 1.0],
                'mass': obj.mass,
            })
        
        self.gpu_pipeline.update_objects_ubo(objects_data)
    
    def render(self):
        """Render one frame."""
        # Update buffers
        self.update_buffers()
        
        # Dispatch compute shader
        self.gpu_pipeline.dispatch_compute(400, 300)
        
        # Set viewport
        self.ctx.viewport = (0, 0, self.width, self.height)
        
        # Clear screen
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        
        # Render quad
        self.gpu_pipeline.render_quad()
        
        # Swap buffers
        glfw.swap_buffers(self.window)
    
    def update_fps(self):
        """Update FPS counter."""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_time)
            self.frame_count = 0
            self.last_time = current_time
            
            # Print FPS and camera info
            print(f"FPS: {self.fps:.1f}, Camera: r={self.camera.radius:.2e}, "
                  f"az={self.camera.azimuth:.2f}, el={self.camera.elevation:.2f}")
    
    def run(self):
        """Run the demo."""
        try:
            # Setup
            self.create_window()
            self.setup_gpu_pipeline()
            self.setup_scene()
            
            print("\n🌌 Black Hole GPU Visualization Demo")
            print("=" * 50)
            print("Controls:")
            print("  ESC - Exit")
            print("  R - Reset camera")
            print("  SPACE - Toggle camera movement")
            print("  Mouse drag - Rotate camera")
            print("  Mouse wheel - Zoom in/out")
            print("=" * 50)
            
            # Main loop
            while not glfw.window_should_close(self.window) and self.running:
                # Handle events
                glfw.poll_events()
                
                # Render frame
                self.render()
                
                # Update FPS
                self.update_fps()
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.016)  # ~60 FPS limit
            
            print("\n✅ Demo completed successfully")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if self.gpu_pipeline:
            self.gpu_pipeline.cleanup()
        if self.shader_manager:
            self.shader_manager.cleanup()
        if self.window:
            glfw.destroy_window(self.window)
        glfw.terminate()
        print("✅ Cleanup completed")

def main():
    """Main demo function."""
    demo = BlackHoleDemo(800, 600)
    demo.run()

if __name__ == "__main__":
    main()
