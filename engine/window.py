"""
Window management and OpenGL context handling.

This module provides the Window class for managing GLFW windows,
OpenGL context, and basic input handling.
"""

import glfw
import moderngl
from typing import Optional, Callable, Any
import logging

logger = logging.getLogger(__name__)


class Window:
    """GLFW window wrapper with OpenGL context management."""
    
    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        title: str = "Black Hole Simulation",
        fullscreen: bool = False,
        vsync: bool = True,
    ):
        """
        Initialize the window.
        
        Args:
            width: Window width in pixels
            height: Window height in pixels  
            title: Window title
            fullscreen: Whether to start in fullscreen mode
            vsync: Whether to enable vertical synchronization
        """
        self.width = width
        self.height = height
        self.title = title
        self.fullscreen = fullscreen
        self.vsync = vsync
        
        self.window: Optional[glfw._GLFWwindow] = None
        self.ctx: Optional[moderngl.Context] = None
        
        # Callbacks
        self.key_callback: Optional[Callable[[int, int, int, int], None]] = None
        self.mouse_button_callback: Optional[Callable[[int, int, int], None]] = None
        self.cursor_pos_callback: Optional[Callable[[float, float], None]] = None
        self.scroll_callback: Optional[Callable[[float, float], None]] = None
        self.framebuffer_size_callback: Optional[Callable[[int, int], None]] = None
        
        self._should_close = False
        
    def create(self) -> bool:
        """
        Create the window and OpenGL context.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Initialize GLFW
            if not glfw.init():
                logger.error("Failed to initialize GLFW")
                return False
            
            # Configure OpenGL context
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
            
            # Create window
            monitor = glfw.get_primary_monitor() if self.fullscreen else None
            self.window = glfw.create_window(
                self.width, self.height, self.title, monitor, None
            )
            
            if not self.window:
                logger.error("Failed to create GLFW window")
                glfw.terminate()
                return False
            
            # Set OpenGL context
            glfw.make_context_current(self.window)
            
            # Create ModernGL context
            self.ctx = moderngl.create_context()
            
            # Set window callbacks
            self._setup_callbacks()
            
            # Enable vsync
            glfw.swap_interval(1 if self.vsync else 0)
            
            # Set viewport
            self.ctx.viewport = (0, 0, self.width, self.height)
            
            logger.info(f"Window created: {self.width}x{self.height}")
            logger.info(f"OpenGL version: {self.ctx.version}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create window: {e}")
            return False
    
    def _setup_callbacks(self) -> None:
        """Set up GLFW callbacks."""
        if not self.window:
            return
            
        # Store window pointer for callbacks
        glfw.set_window_user_pointer(self.window, self)
        
        # Key callback
        glfw.set_key_callback(self.window, self._key_callback)
        
        # Mouse button callback  
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        
        # Cursor position callback
        glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
        
        # Scroll callback
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        
        # Framebuffer size callback
        glfw.set_framebuffer_size_callback(self.window, self._framebuffer_size_callback)
    
    @staticmethod
    def _key_callback(window: glfw._GLFWwindow, key: int, scancode: int, action: int, mods: int) -> None:
        """GLFW key callback wrapper."""
        win = glfw.get_window_user_pointer(window)
        if win.key_callback:
            win.key_callback(key, scancode, action, mods)
    
    @staticmethod
    def _mouse_button_callback(window: glfw._GLFWwindow, button: int, action: int, mods: int) -> None:
        """GLFW mouse button callback wrapper."""
        win = glfw.get_window_user_pointer(window)
        if win.mouse_button_callback:
            win.mouse_button_callback(button, action, mods)
    
    @staticmethod
    def _cursor_pos_callback(window: glfw._GLFWwindow, xpos: float, ypos: float) -> None:
        """GLFW cursor position callback wrapper."""
        win = glfw.get_window_user_pointer(window)
        if win.cursor_pos_callback:
            win.cursor_pos_callback(xpos, ypos)
    
    @staticmethod
    def _scroll_callback(window: glfw._GLFWwindow, xoffset: float, yoffset: float) -> None:
        """GLFW scroll callback wrapper."""
        win = glfw.get_window_user_pointer(window)
        if win.scroll_callback:
            win.scroll_callback(xoffset, yoffset)
    
    @staticmethod
    def _framebuffer_size_callback(window: glfw._GLFWwindow, width: int, height: int) -> None:
        """GLFW framebuffer size callback wrapper."""
        win = glfw.get_window_user_pointer(window)
        if win.ctx:
            win.ctx.viewport = (0, 0, width, height)
        win.width = width
        win.height = height
        if win.framebuffer_size_callback:
            win.framebuffer_size_callback(width, height)
    
    def poll_events(self) -> None:
        """Poll for events."""
        glfw.poll_events()
    
    def swap_buffers(self) -> None:
        """Swap front and back buffers."""
        if self.window:
            glfw.swap_buffers(self.window)
    
    def should_close(self) -> bool:
        """Check if window should close."""
        if self.window:
            return glfw.window_should_close(self.window) or self._should_close
        return True
    
    def close(self) -> None:
        """Mark window for closing."""
        self._should_close = True
    
    def get_time(self) -> float:
        """Get current time in seconds."""
        return glfw.get_time()
    
    def set_title(self, title: str) -> None:
        """Set window title."""
        self.title = title
        if self.window:
            glfw.set_window_title(self.window, title)
    
    def set_cursor_pos(self, x: float, y: float) -> None:
        """Set cursor position."""
        if self.window:
            glfw.set_cursor_pos(self.window, x, y)
    
    def get_cursor_pos(self) -> tuple[float, float]:
        """Get cursor position."""
        if self.window:
            return glfw.get_cursor_pos(self.window)
        return (0.0, 0.0)
    
    def destroy(self) -> None:
        """Destroy the window and cleanup resources."""
        if self.window:
            glfw.destroy_window(self.window)
        glfw.terminate()
        self.window = None
        self.ctx = None
        logger.info("Window destroyed")
