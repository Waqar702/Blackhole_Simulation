"""
3D black hole simulation demo.

This demo showcases 3D ray tracing with GPU acceleration,
matching the functionality of the original black_hole.cpp.
"""

import numpy as np
import math
import logging
from typing import List, Tuple

from ..engine import Window, Camera, ShaderManager, GridRenderer, GPUComputePipeline
from ..physics import Ray3D, SAGITTARIUS_A_MASS, schwarzschild_radius
from ..scene import SceneManager, AccretionDisk, create_default_scene

logger = logging.getLogger(__name__)


class Demo3D:
    """3D black hole simulation demonstration."""
    
    def __init__(self, width: int = 800, height: int = 600):
        """
        Initialize 3D demo.
        
        Args:
            width: Window width
            height: Window height
        """
        self.width = width
        self.height = height
        
        # Initialize systems
        self.window = None
        self.camera = None
        self.shader_manager = None
        self.grid_renderer = None
        self.gpu_pipeline = None
        self.scene = None
        self.accretion_disk = None
        
        # Demo state
        self.running = False
        self.time = 0.0
        self.fps = 0.0
        self.frame_count = 0
        self.last_fps_time = 0.0
        
        # Physical parameters
        self.black_hole_mass = SAGITTARIUS_A_MASS
        self.r_s = schwarzschild_radius(self.black_hole_mass)
        
        # Rendering parameters
        self.compute_width = 200
        self.compute_height = 150
        self.adaptive_resolution = True
    
    def initialize(self) -> bool:
        """
        Initialize the demo.
        
        Returns:
            True if successful
        """
        try:
            # Create window
            self.window = Window(
                width=self.width,
                height=self.height,
                title="Black Hole 3D Simulation",
            )
            
            if not self.window.create():
                logger.error("Failed to create window")
                return False
            
            # Set up input callbacks
            self._setup_callbacks()
            
            # Create camera
            self.camera = Camera(
                radius=6.34194e10,
                azimuth=0.0,
                elevation=math.pi / 2.0,
            )
            
            # Create scene
            self.scene = create_default_scene()
            
            # Create accretion disk
            self.accretion_disk = AccretionDisk(
                black_hole_mass=self.black_hole_mass,
                inner_radius_factor=2.2,
                outer_radius_factor=5.2,
            )
            
            # Create shader manager
            self.shader_manager = ShaderManager(self.window.ctx)
            
            # Create GPU pipeline
            self.gpu_pipeline = GPUComputePipeline(self.window.ctx)
            self._setup_gpu_pipeline()
            
            # Create grid renderer
            self.grid_renderer = GridRenderer(self.window.ctx)
            self._setup_grid_renderer()
            
            logger.info("3D demo initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize 3D demo: {e}")
            return False
    
    def _setup_callbacks(self) -> None:
        """Set up input callbacks."""
        if not self.window:
            return
        
        # Key callback
        self.window.key_callback = self._on_key
        
        # Mouse callbacks
        self.window.mouse_button_callback = self._on_mouse_button
        self.window.cursor_pos_callback = self._on_mouse_move
        self.window.scroll_callback = self._on_scroll
        self.window.framebuffer_size_callback = self._on_framebuffer_resize
    
    def _setup_gpu_pipeline(self) -> None:
        """Set up GPU compute pipeline."""
        if not self.gpu_pipeline:
            return
        
        # Load compute shader
        compute_source = self._get_compute_shader_source()
        self.gpu_pipeline.create_compute_program(compute_source)
        
        # Load quad rendering shaders
        vertex_source, fragment_source = self._get_quad_shader_sources()
        self.gpu_pipeline.create_quad_program(vertex_source, fragment_source)
        
        # Create buffers
        self.gpu_pipeline.create_buffers()
        
        # Create output texture
        self.gpu_pipeline.create_output_texture(self.compute_width, self.compute_height)
    
    def _setup_grid_renderer(self) -> None:
        """Set up grid renderer."""
        if not self.grid_renderer:
            return
        
        # Load grid shaders
        vertex_source, fragment_source = self._get_grid_shader_sources()
        self.grid_renderer.create_program(vertex_source, fragment_source)
        
        # Generate initial grid
        objects = self.scene.get_objects_for_gpu()
        self.grid_renderer.generate_grid(objects)
    
    def _get_compute_shader_source(self) -> str:
        """Get compute shader source code."""
        return """
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
        
        layout(std140, binding = 2) uniform Disk {
            float disk_r1;
            float disk_r2;
            float disk_num;
            float thickness;
        };
        
        layout(std140, binding = 3) uniform Objects {
            int numObjects;
            vec4 objPosRadius[16];
            vec4 objColor[16];
            float  mass[16]; 
        };
        
        const float SagA_rs = 1.269e10;
        const float D_LAMBDA = 1e7;
        const double ESCAPE_R = 1e30;
        
        void main() {
            int WIDTH = cam.moving ? 200 : 200;
            int HEIGHT = cam.moving ? 150 : 150;
            
            ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
            if (pix.x >= WIDTH || pix.y >= HEIGHT) return;
            
            // Init Ray
            float u = (2.0 * (pix.x + 0.5) / WIDTH - 1.0) * cam.aspect * cam.tanHalfFov;
            float v = (1.0 - 2.0 * (pix.y + 0.5) / HEIGHT) * cam.tanHalfFov;
            vec3 dir = normalize(u * cam.camRight - v * cam.camUp + cam.camForward);
            
            vec4 color = vec4(0.0);
            
            // Simple ray-sphere intersection for now
            vec3 rayStart = cam.camPos;
            float t = 0.0;
            
            // Check intersection with black hole
            vec3 center = vec3(0.0);
            float radius = SagA_rs;
            
            vec3 oc = rayStart - center;
            float a = dot(dir, dir);
            float b = 2.0 * dot(oc, dir);
            float c = dot(oc, oc) - radius * radius;
            float discriminant = b * b - 4.0 * a * c;
            
            if (discriminant > 0.0) {
                float t1 = (-b - sqrt(discriminant)) / (2.0 * a);
                float t2 = (-b + sqrt(discriminant)) / (2.0 * a);
                
                if (t1 > 0.0 || t2 > 0.0) {
                    color = vec4(0.0, 0.0, 0.0, 1.0);  // Black hole
                }
            }
            
            // Check disk intersection
            if (color.a == 0.0) {
                float t_disk = -rayStart.y / dir.y;
                if (t_disk > 0.0) {
                    vec3 disk_point = rayStart + t_disk * dir;
                    float r = length(disk_point.xz);
                    
                    if (r >= disk_r1 && r <= disk_r2) {
                        float r_norm = (r - disk_r1) / (disk_r2 - disk_r1);
                        color = vec4(1.0, r_norm, 0.2, 0.8);
                    }
                }
            }
            
            imageStore(outImage, pix, color);
        }
        """
    
    def _get_quad_shader_sources(self) -> Tuple[str, str]:
        """Get quad rendering shader sources."""
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
    
    def _get_grid_shader_sources(self) -> Tuple[str, str]:
        """Get grid rendering shader sources."""
        vertex_source = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        
        uniform mat4 viewProj;
        
        void main() {
            gl_Position = viewProj * vec4(aPos, 1.0);
        }
        """
        
        fragment_source = """
        #version 330 core
        out vec4 FragColor;
        
        uniform vec4 gridColor;
        
        void main() {
            FragColor = gridColor;
        }
        """
        
        return vertex_source, fragment_source
    
    def _on_key(self, key: int, scancode: int, action: int, modifiers: int) -> None:
        """Handle keyboard input."""
        if action == 1:  # Press
            if key == 71:  # G key - toggle gravity
                gravity_enabled = not self.scene.gravity_enabled
                self.scene.set_gravity_enabled(gravity_enabled)
                logger.info(f"Gravity {'enabled' if gravity_enabled else 'disabled'}")
            elif key == 82:  # R key - reset camera
                self.camera.set_angles(0.0, math.pi / 2.0)
                self.camera.set_radius(6.34194e10)
                logger.info("Camera reset")
            elif key == 83:  # S key - take screenshot
                self._take_screenshot()
            elif key == 70:  # F key - toggle fullscreen
                self._toggle_fullscreen()
    
    def _on_mouse_button(self, button: int, action: int, modifiers: int) -> None:
        """Handle mouse button events."""
        if self.camera:
            self.camera.process_mouse_button(button, action, modifiers)
    
    def _on_mouse_move(self, x: float, y: float) -> None:
        """Handle mouse movement."""
        if self.camera:
            self.camera.process_mouse_move(x, y)
    
    def _on_scroll(self, x_offset: float, y_offset: float) -> None:
        """Handle scroll wheel."""
        if self.camera:
            self.camera.process_scroll(x_offset, y_offset)
    
    def _on_framebuffer_resize(self, width: int, height: int) -> None:
        """Handle framebuffer resize."""
        self.width = width
        self.height = height
        logger.info(f"Framebuffer resized to {width}x{height}")
    
    def _take_screenshot(self) -> None:
        """Take a screenshot."""
        # This would be implemented with proper screenshot functionality
        logger.info("Screenshot requested")
    
    def _toggle_fullscreen(self) -> None:
        """Toggle fullscreen mode."""
        # This would be implemented with proper fullscreen functionality
        logger.info("Fullscreen toggle requested")
    
    def update(self, dt: float) -> None:
        """
        Update demo state.
        
        Args:
            dt: Time step in seconds
        """
        self.time += dt
        
        # Update scene gravity
        if self.scene.gravity_enabled:
            self.scene.update_gravity(dt)
        
        # Update grid if objects moved
        if self.scene.gravity_enabled:
            objects = self.scene.get_objects_for_gpu()
            self.grid_renderer.generate_grid(objects)
        
        # Update FPS counter
        self.frame_count += 1
        if self.time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (self.time - self.last_fps_time)
            self.frame_count = 0
            self.last_fps_time = self.time
            logger.debug(f"FPS: {self.fps:.1f}")
    
    def render(self) -> None:
        """Render the demo."""
        if not self.window or not self.window.ctx:
            return
        
        # Clear screen
        self.window.ctx.clear(0.0, 0.0, 0.0, 1.0)
        
        # Determine compute resolution
        compute_w = self.compute_width if not self.camera.moving else 200
        compute_h = self.compute_height if not self.camera.moving else 150
        
        # Update GPU buffers
        self._update_gpu_buffers()
        
        # Dispatch compute shader
        self.gpu_pipeline.dispatch_compute(compute_w, compute_h)
        
        # Render compute results
        self.gpu_pipeline.render_quad()
        
        # Render grid overlay
        self._render_grid()
    
    def _update_gpu_buffers(self) -> None:
        """Update GPU uniform buffers."""
        if not self.gpu_pipeline:
            return
        
        # Get camera basis vectors
        right, up, forward = self.camera.get_basis_vectors()
        position = self.camera.get_position()
        
        # Calculate camera parameters
        tan_half_fov = math.tan(math.radians(60.0) / 2.0)
        aspect = self.width / self.height
        
        # Update camera buffer
        self.gpu_pipeline.update_camera_ubo(
            position=position,
            right=right,
            up=up,
            forward=forward,
            tan_half_fov=tan_half_fov,
            aspect=aspect,
            moving=self.camera.moving,
        )
        
        # Update disk buffer
        disk_params = self.accretion_disk.get_disk_parameters()
        self.gpu_pipeline.update_disk_buffer(
            inner_radius=disk_params['inner_radius'],
            outer_radius=disk_params['outer_radius'],
            num_rays=2.0,
            thickness=disk_params['thickness'],
        )
        
        # Update objects buffer
        objects = self.scene.get_objects_for_gpu()
        self.gpu_pipeline.update_objects_buffer(objects)
    
    def _render_grid(self) -> None:
        """Render spacetime grid."""
        if not self.grid_renderer:
            return
        
        # Get view-projection matrix
        aspect = self.width / self.height
        view_proj = self.camera.get_view_projection_matrix(aspect)
        
        # Render grid
        self.grid_renderer.render(view_proj)
    
    def run(self) -> None:
        """Run the demo main loop."""
        if not self.initialize():
            logger.error("Failed to initialize demo")
            return
        
        self.running = True
        last_time = self.window.get_time()
        
        logger.info("Starting 3D demo main loop")
        
        while self.running and not self.window.should_close():
            current_time = self.window.get_time()
            dt = current_time - last_time
            last_time = current_time
            
            # Handle events
            self.window.poll_events()
            
            # Update
            self.update(dt)
            
            # Render
            self.render()
            
            # Swap buffers
            self.window.swap_buffers()
        
        self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up demo resources."""
        if self.grid_renderer:
            self.grid_renderer.cleanup()
        
        if self.gpu_pipeline:
            self.gpu_pipeline.cleanup()
        
        if self.shader_manager:
            self.shader_manager.cleanup()
        
        if self.window:
            self.window.destroy()
        
        logger.info("3D demo cleaned up")


def main():
    """Main entry point for 3D demo."""
    logging.basicConfig(level=logging.INFO)
    
    demo = Demo3D()
    demo.run()


if __name__ == "__main__":
    main()
