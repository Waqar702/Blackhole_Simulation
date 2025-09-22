"""
Shader management and compilation.

This module provides the ShaderManager class for loading, compiling,
and managing OpenGL shaders with error handling and hot-reload support.
"""

import os
import moderngl
from typing import Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class ShaderManager:
    """Manages OpenGL shader compilation and caching."""
    
    def __init__(self, ctx: moderngl.Context):
        """
        Initialize shader manager.
        
        Args:
            ctx: ModernGL context
        """
        self.ctx = ctx
        self.shaders: Dict[str, moderngl.Program] = {}
        self.shader_paths: Dict[str, str] = {}
        
    def load_shader(
        self,
        name: str,
        vertex_path: Optional[str] = None,
        fragment_path: Optional[str] = None,
        geometry_path: Optional[str] = None,
        compute_path: Optional[str] = None,
    ) -> moderngl.Program:
        """
        Load and compile a shader program.
        
        Args:
            name: Unique name for the shader program
            vertex_path: Path to vertex shader source
            fragment_path: Path to fragment shader source
            geometry_path: Path to geometry shader source
            compute_path: Path to compute shader source
            
        Returns:
            Compiled shader program
            
        Raises:
            FileNotFoundError: If shader file not found
            moderngl.Error: If shader compilation fails
        """
        try:
            # Read shader sources
            sources = {}
            
            if vertex_path and os.path.exists(vertex_path):
                sources['vertex'] = self._read_file(vertex_path)
                self.shader_paths[f"{name}_vertex"] = vertex_path
                
            if fragment_path and os.path.exists(fragment_path):
                sources['fragment'] = self._read_file(fragment_path)
                self.shader_paths[f"{name}_fragment"] = fragment_path
                
            if geometry_path and os.path.exists(geometry_path):
                sources['geometry'] = self._read_file(geometry_path)
                self.shader_paths[f"{name}_geometry"] = geometry_path
                
            if compute_path and os.path.exists(compute_path):
                sources['compute'] = self._read_file(compute_path)
                self.shader_paths[f"{name}_compute"] = compute_path
            
            if not sources:
                raise ValueError(f"No valid shader sources provided for {name}")
            
            # Compile shader program
            if 'compute' in sources:
                # Create compute shader
                program = self.ctx.compute_shader(sources['compute'])
            else:
                # Create regular shader program
                program = self.ctx.program(
                    vertex_shader=sources.get('vertex'),
                    fragment_shader=sources.get('fragment'),
                    geometry_shader=sources.get('geometry'),
                )
            
            self.shaders[name] = program
            
            logger.info(f"Loaded shader: {name}")
            return program
            
        except Exception as e:
            logger.error(f"Failed to load shader {name}: {e}")
            raise
    
    def load_compute_shader(self, name: str, compute_path: str) -> moderngl.Program:
        """
        Load a compute shader.
        
        Args:
            name: Unique name for the compute shader
            compute_path: Path to compute shader source
            
        Returns:
            Compiled compute shader program
        """
        return self.load_shader(name, compute_path=compute_path)
    
    def create_inline_shader(
        self,
        name: str,
        vertex_source: Optional[str] = None,
        fragment_source: Optional[str] = None,
        compute_source: Optional[str] = None,
    ) -> moderngl.Program:
        """
        Create a shader program from inline source code.
        
        Args:
            name: Unique name for the shader program
            vertex_source: Vertex shader source code
            fragment_source: Fragment shader source code
            compute_source: Compute shader source code
            
        Returns:
            Compiled shader program
        """
        try:
            if compute_source:
                # Create compute shader
                program = self.ctx.compute_shader(compute_source)
            else:
                # Create regular shader program
                if not vertex_source and not fragment_source:
                    raise ValueError(f"No shader sources provided for {name}")
                
                program = self.ctx.program(
                    vertex_shader=vertex_source,
                    fragment_shader=fragment_source,
                )
            
            self.shaders[name] = program
            
            logger.info(f"Created inline shader: {name}")
            return program
            
        except Exception as e:
            logger.error(f"Failed to create inline shader {name}: {e}")
            raise
    
    def get_shader(self, name: str) -> Optional[moderngl.Program]:
        """
        Get a compiled shader program by name.
        
        Args:
            name: Shader program name
            
        Returns:
            Shader program or None if not found
        """
        return self.shaders.get(name)
    
    def reload_shader(self, name: str) -> bool:
        """
        Reload a shader from disk.
        
        Args:
            name: Shader program name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if name not in self.shaders:
                logger.warning(f"Shader {name} not found for reload")
                return False
            
            # Find the shader paths
            vertex_path = self.shader_paths.get(f"{name}_vertex")
            fragment_path = self.shader_paths.get(f"{name}_fragment")
            geometry_path = self.shader_paths.get(f"{name}_geometry")
            compute_path = self.shader_paths.get(f"{name}_compute")
            
            if not any([vertex_path, fragment_path, geometry_path, compute_path]):
                logger.warning(f"No paths found for shader {name}")
                return False
            
            # Reload the shader
            self.load_shader(
                name,
                vertex_path=vertex_path,
                fragment_path=fragment_path,
                geometry_path=geometry_path,
                compute_path=compute_path,
            )
            
            logger.info(f"Reloaded shader: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reload shader {name}: {e}")
            return False
    
    def _read_file(self, path: str) -> str:
        """
        Read a file and return its contents.
        
        Args:
            path: File path
            
        Returns:
            File contents as string
        """
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def cleanup(self) -> None:
        """Clean up shader resources."""
        for shader in self.shaders.values():
            if hasattr(shader, 'release'):
                shader.release()
        self.shaders.clear()
        self.shader_paths.clear()
        logger.info("Shader manager cleaned up")
