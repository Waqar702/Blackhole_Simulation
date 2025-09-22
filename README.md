# Black Hole Simulation - Python Implementation

A Python implementation of a black hole simulation featuring ray tracing, gravitational lensing, and spacetime visualization. This project ports the C++ black hole simulation to Python while maintaining physical accuracy and performance.

## Features

- **2D & 3D Null Geodesics**: Accurate Schwarzschild metric ray tracing
- **GPU Acceleration**: OpenGL compute shaders for real-time rendering
- **Interactive Camera**: Orbit controls around the black hole
- **Spacetime Visualization**: Warped grid showing curvature effects
- **Accretion Disk**: Realistic disk rendering with gravitational lensing
- **Educational**: Perfect for understanding general relativity concepts

## Quick Start

### Prerequisites

- Python 3.9 or higher
- OpenGL 4.3+ compatible graphics card
- Modern graphics drivers

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd blackhole_python

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Running the Demos

```bash
# 2D lensing demo (CPU-based ray tracing)
python -m blackhole_python.demos.demo_2d

# 3D GPU-accelerated simulation
python -m blackhole_python.demos.demo_3d

# Or use the command line scripts
blackhole-2d
blackhole-3d
```

## Controls

### Camera Controls
- **Left Mouse Drag**: Orbit around the black hole
- **Scroll Wheel**: Zoom in/out
- **G Key**: Toggle gravity effects on/off
- **R Key**: Reset camera position

### Simulation Controls
- **Space**: Pause/resume simulation
- **S**: Take screenshot
- **F**: Toggle fullscreen
- **ESC**: Exit application

## Project Structure

```
blackhole_python/
├── engine/           # Core rendering engine
├── physics/          # Physics simulation modules
├── scene/            # Scene objects and data
├── demos/            # Demo applications
├── assets/           # Shaders and resources
└── tests/            # Test suite
```

## Physics

This simulation implements:

- **Schwarzschild Metric**: Static, spherically symmetric black hole
- **Null Geodesics**: Light ray paths in curved spacetime
- **Runge-Kutta Integration**: Accurate numerical integration
- **Conserved Quantities**: Energy and angular momentum conservation

## Performance

- **Target**: 60 FPS at 800x600 resolution
- **GPU**: OpenGL compute shaders for parallel ray tracing
- **CPU**: Optimized numpy operations with optional numba acceleration
- **Adaptive**: Dynamic resolution scaling during camera movement

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=blackhole_python

# Run specific test categories
pytest -m physics    # Physics tests only
pytest -m integration # Integration tests only
```

### Code Quality

```bash
# Format code
black .

# Lint code
flake8 .

# Type checking
mypy .
```

## Educational Use

This simulation is excellent for:
- Understanding general relativity concepts
- Visualizing gravitational lensing effects
- Learning about black hole physics
- Exploring numerical methods in physics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

## References

- Original C++ implementation: [GitHub Repository]
- Schwarzschild Metric: General Relativity textbooks
- Numerical Methods: Numerical Recipes in Physics
- OpenGL Compute Shaders: OpenGL documentation

## Acknowledgments

- Based on the original C++ black hole simulation
- Inspired by educational physics simulations
- Built with the Python scientific computing ecosystem
