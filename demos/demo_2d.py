"""
2D black hole lensing demo.

This demo showcases 2D ray tracing with gravitational lensing effects,
matching the functionality of the original 2D_lensing.cpp.
"""

import numpy as np
import math
import logging
from typing import List, Tuple

from ..engine import Window, Camera, ShaderManager
from ..physics import Ray2D, SAGITTARIUS_A_MASS, schwarzschild_radius
from ..scene import SceneManager, create_default_scene

logger = logging.getLogger(__name__)


class Demo2D:
    """2D black hole lensing demonstration."""
    
    def __init__(self, width: int = 800, height: int = 600):
        """
        Initialize 2D demo.
        
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
        self.scene = None
        
        # Demo state
        self.running = False
        self.rays: List[Ray2D] = []
        self.time = 0.0
        
        # Physical parameters
        self.black_hole_mass = SAGITTARIUS_A_MASS
        self.r_s = schwarzschild_radius(self.black_hole_mass)
        
        # Viewport parameters
        self.viewport_width = 100000000000.0  # Width in meters
        self.viewport_height = 75000000000.0  # Height in meters
        
        # Navigation state
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.zoom = 1.0
        self.dragging = False
        self.last_mouse_x = 0.0
        self.last_mouse_y = 0.0
    
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
                title="Black Hole 2D Lensing Demo",
            )
            
            if not self.window.create():
                logger.error("Failed to create window")
                return False
            
            # Set up input callbacks
            self._setup_callbacks()
            
            # Create scene
            self.scene = create_default_scene()
            
            # Create shader manager
            self.shader_manager = ShaderManager(self.window.ctx)
            
            logger.info("2D demo initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize 2D demo: {e}")
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
    
    def _on_key(self, key: int, scancode: int, action: int, modifiers: int) -> None:
        """Handle keyboard input."""
        if action == 1:  # Press
            if key == 71:  # G key - toggle gravity
                gravity_enabled = not self.scene.gravity_enabled
                self.scene.set_gravity_enabled(gravity_enabled)
                logger.info(f"Gravity {'enabled' if gravity_enabled else 'disabled'}")
            elif key == 82:  # R key - reset view
                self._reset_view()
            elif key == 67:  # C key - clear rays
                self.rays.clear()
                logger.info("Rays cleared")
            elif key == 83:  # S key - shoot test ray
                self._shoot_test_ray()
    
    def _on_mouse_button(self, button: int, action: int, modifiers: int) -> None:
        """Handle mouse button events."""
        if button == 2:  # Middle mouse button
            if action == 1:  # Press
                self.dragging = True
                self.last_mouse_x, self.last_mouse_y = self.window.get_cursor_pos()
            elif action == 0:  # Release
                self.dragging = False
        elif button == 0:  # Left mouse button
            if action == 1:  # Press
                self._shoot_ray_at_mouse()
    
    def _on_mouse_move(self, x: float, y: float) -> None:
        """Handle mouse movement."""
        if self.dragging:
            dx = x - self.last_mouse_x
            dy = y - self.last_mouse_y
            
            # Pan view
            scale = self.viewport_width / self.width * self.zoom
            self.offset_x -= dx * scale
            self.offset_y += dy * scale
            
            self.last_mouse_x = x
            self.last_mouse_y = y
    
    def _on_scroll(self, x_offset: float, y_offset: float) -> None:
        """Handle scroll wheel."""
        zoom_factor = 1.1
        if y_offset > 0:
            self.zoom /= zoom_factor
        else:
            self.zoom *= zoom_factor
        
        self.zoom = max(0.1, min(10.0, self.zoom))
    
    def _reset_view(self) -> None:
        """Reset view to default."""
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.zoom = 1.0
        logger.info("View reset")
    
    def _shoot_test_ray(self) -> None:
        """Shoot a test ray from the left side."""
        # Position ray on left side of screen
        start_x = -self.viewport_width * 0.4 + self.offset_x
        start_y = self.offset_y
        
        # Direction towards center
        direction = (C * 0.8, 0.0)  # 80% speed of light
        
        ray = Ray2D((start_x, start_y), direction, self.black_hole_mass)
        self.rays.append(ray)
        
        logger.info("Test ray shot")
    
    def _shoot_ray_at_mouse(self) -> None:
        """Shoot a ray from camera position towards mouse."""
        mouse_x, mouse_y = self.window.get_cursor_pos()
        
        # Convert screen coordinates to world coordinates
        world_x = (mouse_x / self.width - 0.5) * self.viewport_width / self.zoom + self.offset_x
        world_y = (0.5 - mouse_y / self.height) * self.viewport_height / self.zoom + self.offset_y
        
        # Ray starts from camera position (offset)
        start_x = self.offset_x
        start_y = self.offset_y
        
        # Direction towards mouse position
        dx = world_x - start_x
        dy = world_y - start_y
        length = math.sqrt(dx*dx + dy*dy)
        
        if length > 0:
            direction = (dx / length * C, dy / length * C)
            
            ray = Ray2D((start_x, start_y), direction, self.black_hole_mass)
            self.rays.append(ray)
            
            logger.info(f"Ray shot towards ({world_x:.2e}, {world_y:.2e})")
    
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
        
        # Update rays
        for ray in self.rays[:]:  # Copy list to allow removal
            ray.step(1e7)  # Integration step size
            
            # Remove rays that have escaped or hit black hole
            if ray.has_escaped() or ray.is_inside_event_horizon():
                self.rays.remove(ray)
    
    def render(self) -> None:
        """Render the demo."""
        if not self.window or not self.window.ctx:
            return
        
        # Clear screen
        self.window.ctx.clear(0.0, 0.0, 0.1, 1.0)
        
        # Set up orthographic projection for 2D rendering
        self._setup_2d_projection()
        
        # Draw black hole
        self._draw_black_hole()
        
        # Draw rays
        self._draw_rays()
        
        # Draw UI
        self._draw_ui()
    
    def _setup_2d_projection(self) -> None:
        """Set up 2D orthographic projection."""
        # Calculate view bounds
        left = -self.viewport_width + self.offset_x
        right = self.viewport_width + self.offset_x
        bottom = -self.viewport_height + self.offset_y
        top = self.viewport_height + self.offset_y
        
        # Apply zoom
        center_x = (left + right) / 2
        center_y = (bottom + top) / 2
        width = (right - left) / self.zoom
        height = (top - bottom) / self.zoom
        
        left = center_x - width / 2
        right = center_x + width / 2
        bottom = center_y - height / 2
        top = center_y + height / 2
        
        # Set up orthographic projection matrix
        # For now, we'll use immediate mode rendering
        # In a full implementation, this would use proper matrix setup
    
    def _draw_black_hole(self) -> None:
        """Draw the black hole."""
        # Draw black hole as a filled circle
        # This is a simplified version - in a full implementation,
        # this would use proper OpenGL rendering
        logger.debug("Drawing black hole")
    
    def _draw_rays(self) -> None:
        """Draw ray trails."""
        for ray in self.rays:
            trail = ray.get_trail()
            if len(trail) > 1:
                # Draw trail as lines
                # This is a simplified version
                logger.debug(f"Drawing ray trail with {len(trail)} points")
    
    def _draw_ui(self) -> None:
        """Draw user interface."""
        # Draw instructions and status
        # This would be implemented with proper text rendering
        logger.debug("Drawing UI")
    
    def run(self) -> None:
        """Run the demo main loop."""
        if not self.initialize():
            logger.error("Failed to initialize demo")
            return
        
        self.running = True
        last_time = self.window.get_time()
        
        logger.info("Starting 2D demo main loop")
        
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
        if self.shader_manager:
            self.shader_manager.cleanup()
        
        if self.window:
            self.window.destroy()
        
        logger.info("2D demo cleaned up")


def main():
    """Main entry point for 2D demo."""
    logging.basicConfig(level=logging.INFO)
    
    demo = Demo2D()
    demo.run()


if __name__ == "__main__":
    main()
