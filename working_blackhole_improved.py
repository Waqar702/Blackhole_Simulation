#!/usr/bin/env python3
"""
Working Black Hole Demo with Improved Shader

This demo positions the camera further away to see the full black hole scene.
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

class WorkingBlackHoleImproved:
    """Working black hole demo with improved shader."""
    
    def __init__(self, width=800, height=600):
        """Initialize the demo."""
        self.width = width
        self.height = height
        self.window = None
        self.ctx = None
        self.gpu_pipeline = None
        self.scene = None
        self.disk = None
        self.running = True
        
    def create_window(self):
        """Create GLFW window and OpenGL context."""
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        
        self.window = glfw.create_window(
            self.width, self.height, 
            "Working Black Hole - Improved Shader", None, None
        )
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        
        glfw.make_context_current(self.window)
        glfw.set_key_callback(self.window, self.key_callback)
        
        self.ctx = moderngl.create_context()
        print("✅ Window created")
    
    def key_callback(self, window, key, scancode, action, mods):
        """Handle keyboard input."""
        if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
            self.running = False
    
    def setup_gpu_pipeline(self):
        """Set up the GPU compute pipeline."""
        self.gpu_pipeline = GPUComputePipeline(self.ctx)
        
        # Load IMPROVED compute shader
        compute_source = open('assets/geodesic_improved.comp', 'r').read()
        self.gpu_pipeline.create_compute_program(compute_source)
        
        # Load SIMPLE quad shaders (no tone mapping)
        vertex_source = open('assets/quad.vert', 'r').read()
        fragment_source = open('assets/quad_simple.frag', 'r').read()
        self.gpu_pipeline.create_quad_program(vertex_source, fragment_source)
        
        # Create buffers and texture
        self.gpu_pipeline.create_buffers()
        self.gpu_pipeline.create_output_texture(400, 300)
        
        print("✅ GPU pipeline set up")
    
    def setup_scene(self):
        """Set up the black hole scene."""
        # Create scene
        self.scene = create_default_scene()
        
        # Create accretion disk
        self.disk = AccretionDisk(
            black_hole_mass=SAGITTARIUS_A_MASS,
            inner_radius_factor=2.2,
            outer_radius_factor=5.2,
        )
        
        print("✅ Scene set up")
        print(f"Black hole radius: {schwarzschild_radius(SAGITTARIUS_A_MASS):.2e}")
        print(f"Disk inner radius: {self.disk.inner_radius:.2e}")
        print(f"Disk outer radius: {self.disk.outer_radius:.2e}")
    
    def update_buffers(self):
        """Update GPU buffers with current scene data."""
        # Position camera FURTHER away to see the full scene
        # Camera at (0, 0, 2e11) looking toward (0, 0, 0) - much further away
        pos = np.array([0.0, 0.0, 2e11])  # 10x further than before
        right = np.array([1.0, 0.0, 0.0])
        up = np.array([0.0, 1.0, 0.0])
        forward = np.array([0.0, 0.0, -1.0])  # Looking toward origin
        
        self.gpu_pipeline.update_camera_ubo(
            position=pos,
            right=right,
            up=up,
            forward=forward,
            tan_half_fov=0.577,  # 60 degrees
            aspect=float(self.width) / float(self.height),
            moving=False,
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
        
        # Set viewport and clear
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        
        # Disable depth test
        self.ctx.disable(moderngl.DEPTH_TEST)
        
        # Render quad
        self.gpu_pipeline.render_quad()
        
        # Swap buffers
        glfw.swap_buffers(self.window)
    
    def run(self):
        """Run the demo."""
        try:
            # Setup
            self.create_window()
            self.setup_gpu_pipeline()
            self.setup_scene()
            
            print("\n🌌 Working Black Hole Simulation - Improved Shader")
            print("=" * 60)
            print("You should see:")
            print("- Dark blue starfield background with white stars")
            print("- Black hole (black circle in center)")
            print("- Orange/red accretion disk around black hole")
            print("- Press ESC to exit")
            print("=" * 60)
            
            # Main loop
            while not glfw.window_should_close(self.window) and self.running:
                glfw.poll_events()
                self.render()
                time.sleep(0.016)  # ~60 FPS limit
            
            print("\n✅ Demo completed successfully!")
            
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
        if self.window:
            glfw.destroy_window(self.window)
        glfw.terminate()
        print("✅ Cleanup completed")

def main():
    """Main demo function."""
    demo = WorkingBlackHoleImproved(800, 600)
    demo.run()

if __name__ == "__main__":
    main()
