# Black Hole Simulation - Python Implementation Plan

## Project Overview
This document outlines the step-by-step plan to reimplement the C++ black hole simulation in Python, based on the existing codebase in `../black_hole/`.

## Architecture Analysis
Based on the C++ codebase analysis:

### Core Components
1. **Physics Engine**: Schwarzschild null geodesics with RK4 integration
2. **GPU Compute Pipeline**: GLSL compute shader for ray tracing
3. **Rendering Pipeline**: OpenGL with moderngl wrapper
4. **Camera System**: Orbit camera with adaptive resolution
5. **Scene Management**: Objects, accretion disk, spacetime grid

### Key Files to Port
- `black_hole.cpp` → Main 3D GPU simulation
- `CPU-geodesic.cpp` → CPU ray tracing implementation
- `2D_lensing.cpp` → 2D demo with trails
- `geodesic.comp` → GPU compute shader
- `grid.vert/frag` → Grid rendering shaders

## Implementation Steps

### Phase 1: Project Setup & Infrastructure
**Goal**: Establish Python project structure and dependencies

1. **Project Structure**
   ```
   blackhole_python/
   ├── pyproject.toml
   ├── requirements.txt
   ├── README.md
   ├── app.py                    # Main entry point
   ├── engine/
   │   ├── __init__.py
   │   ├── window.py            # GLFW window management
   │   ├── shaders.py           # Shader compilation
   │   ├── camera.py            # Orbit camera system
   │   ├── grid.py              # Spacetime grid rendering
   │   └── gpu.py               # GPU compute pipeline
   ├── physics/
   │   ├── __init__.py
   │   ├── constants.py         # Physical constants
   │   ├── geodesic2d.py        # 2D null geodesics
   │   ├── geodesic3d.py        # 3D null geodesics
   │   └── integrators.py       # RK4/RK45 integration
   ├── scene/
   │   ├── __init__.py
   │   ├── objects.py           # Scene objects
   │   ├── disk.py              # Accretion disk
   │   └── ubos.py              # Uniform buffer objects
   ├── demos/
   │   ├── __init__.py
   │   ├── demo_2d.py           # 2D lensing demo
   │   └── demo_3d.py           # 3D GPU demo
   ├── assets/
   │   ├── geodesic.comp        # Compute shader
   │   ├── grid.vert            # Grid vertex shader
   │   ├── grid.frag            # Grid fragment shader
   │   ├── quad.vert            # Screen quad vertex shader
   │   └── quad.frag            # Screen quad fragment shader
   └── tests/
       ├── __init__.py
       ├── test_physics.py      # Physics tests
       └── test_integration.py  # Integration tests
   ```

2. **Dependencies**
   ```toml
   # pyproject.toml
   [project]
   dependencies = [
       "numpy>=1.24.0",
       "PyOpenGL>=3.1.6",
       "glfw>=2.5.0",
       "moderngl>=5.8.0",
       "moderngl-window>=2.4.0",
       "numba>=0.57.0",
       "pillow>=9.5.0",
       "pytest>=7.2.0",
       "scipy>=1.10.0",  # Optional for RK45
   ]
   ```

### Phase 2: Physics Core Implementation
**Goal**: Implement accurate Schwarzschild null geodesics

1. **Constants & Units** (`physics/constants.py`)
   ```python
   # Physical constants from C++ code
   C = 299792458.0  # Speed of light (m/s)
   G = 6.67430e-11  # Gravitational constant
   SAGITTARIUS_A_MASS = 8.54e36  # kg
   ```

2. **2D Null Geodesics** (`physics/geodesic2d.py`)
   - Port `Ray` struct from `2D_lensing.cpp`
   - Implement `geodesicRHS()` for 2D Schwarzschild
   - Implement `rk4Step()` with same algorithm
   - Maintain conserved quantities E, L

3. **3D Null Geodesics** (`physics/geodesic3d.py`)
   - Port `Ray` struct from `CPU-geodesic.cpp`
   - Implement spherical coordinate conversion
   - Implement 3D geodesic equations
   - Handle coordinate transformations

4. **Integration Methods** (`physics/integrators.py`)
   - RK4 implementation (primary)
   - Optional RK45 adaptive step
   - Error estimation and step control

### Phase 3: GPU Compute Pipeline
**Goal**: Port GLSL compute shader to work with Python

1. **Shader Management** (`engine/shaders.py`)
   - Load and compile GLSL shaders
   - Handle shader errors and debugging
   - Hot-reload during development

2. **GPU Pipeline** (`engine/gpu.py`)
   - Create compute shader program
   - Set up output texture (RGBA8)
   - Manage uniform buffer objects (UBOs)
   - Dispatch compute workgroups
   - Render results to screen quad

3. **Uniform Buffers** (`scene/ubos.py`)
   - Camera UBO (position, basis vectors, FOV)
   - Disk UBO (inner/outer radius, thickness)
   - Objects UBO (positions, colors, masses)

### Phase 4: Camera & Input System
**Goal**: Implement orbit camera with controls

1. **Camera Implementation** (`engine/camera.py`)
   - Orbit camera around black hole center
   - Azimuth/elevation/radius controls
   - Mouse drag for orbit, scroll for zoom
   - Adaptive resolution based on movement
   - Key bindings (G for gravity toggle)

2. **Input Handling** (`engine/window.py`)
   - GLFW window creation and management
   - Mouse button and cursor callbacks
   - Keyboard input handling
   - Window resize handling

### Phase 5: Rendering System
**Goal**: Implement spacetime grid and screen rendering

1. **Grid Rendering** (`engine/grid.py`)
   - CPU generation of warped grid vertices
   - Schwarzschild geometry warping
   - Line rendering with transparency
   - Dynamic grid updates

2. **Screen Rendering**
   - Full-screen quad for compute results
   - Proper viewport and aspect ratio handling
   - Depth testing and blending

### Phase 6: Scene Management
**Goal**: Model scene objects and accretion disk

1. **Scene Objects** (`scene/objects.py`)
   - ObjectData equivalent with position, color, mass
   - Black hole representation
   - Multiple object support

2. **Accretion Disk** (`scene/disk.py`)
   - Disk geometry and properties
   - Color gradients based on radius
   - Intersection testing with rays

### Phase 7: Main Application Loop
**Goal**: Compose all components into working simulation

1. **Main Loop** (`app.py`)
   - Initialize all systems
   - Update camera and scene
   - Dispatch compute shader
   - Render grid overlay
   - Handle input and timing

2. **Performance Optimization**
   - Adaptive resolution scaling
   - Frame rate monitoring
   - Memory management

### Phase 8: Demo Applications
**Goal**: Create working demos matching C++ functionality

1. **2D Demo** (`demos/demo_2d.py`)
   - Match `2D_lensing.cpp` functionality
   - Ray trails and visualization
   - Interactive ray shooting

2. **3D Demo** (`demos/demo_3d.py`)
   - Match `black_hole.cpp` functionality
   - GPU compute rendering
   - Camera controls and grid

### Phase 9: Testing & Validation
**Goal**: Ensure correctness and reliability

1. **Physics Tests** (`tests/test_physics.py`)
   - Test geodesic invariants (E, L conservation)
   - Test interception logic
   - Test escape conditions
   - Compare CPU vs GPU results

2. **Integration Tests** (`tests/test_integration.py`)
   - End-to-end rendering tests
   - Performance benchmarks
   - Memory leak detection

### Phase 10: Documentation & Polish
**Goal**: Complete user experience

1. **Documentation**
   - README with usage instructions
   - API documentation
   - Physics explanations
   - Performance tips

2. **User Interface**
   - Control instructions
   - Parameter adjustments
   - Visual feedback

## Technical Specifications

### Physics Accuracy
- Use double precision for geodesic calculations
- Maintain conserved quantities within numerical precision
- Match C++ integration parameters exactly

### Performance Targets
- 60 FPS for 800x600 resolution
- Adaptive resolution during camera movement
- Efficient GPU memory usage

### Compatibility
- Python 3.9+
- Modern OpenGL (4.3+ for compute shaders)
- Cross-platform (Windows, Linux, macOS)

## Success Criteria
1. ✅ 2D demo matches C++ functionality
2. ✅ 3D demo matches C++ functionality  
3. ✅ Physics calculations are numerically identical
4. ✅ Performance is comparable to C++ version
5. ✅ Code is well-documented and maintainable
6. ✅ Tests validate correctness

## Risk Mitigation
- **Complexity**: Start with 2D, then extend to 3D
- **Performance**: Profile early, optimize GPU usage
- **Compatibility**: Test on multiple platforms
- **Physics**: Validate against known solutions

---

*This document should be updated as implementation progresses and requirements evolve.*
