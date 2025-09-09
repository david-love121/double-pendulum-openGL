# Double Pendulum OpenGL Visualizer

A high-performance OpenGL-based visualization of double pendulum dynamics with chaos analysis capabilities.

## Features

- **Real-time Simulation**: 60fps visualization using Lagrangian mechanics
- **Interactive Controls**: Adjust pendulum parameters in real-time
- **Chaos Analysis**: Visualize chaotic behavior patterns across different initial conditions
- **Modern Graphics**: OpenGL 4.6 core profile with smooth rendering
- **Cross-platform**: Built for Linux with Wayland/X11 support

## System Requirements

- **OS**: Arch Linux (or similar) with Wayland/X11
- **GPU**: OpenGL 4.6 compatible graphics card
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Dependencies**: See installation section below

## Installation

### System Dependencies

Install required packages on Arch Linux:

```bash
sudo pacman -S base-devel cmake ninja
sudo pacman -S glfw-wayland glew glm 
sudo pacman -S gtest
```

### Building the Project

1. Clone the repository:
```bash
git clone <repository-url>
cd three-js-pendulum
git checkout 001-build-an-opengl
```

2. Initialize submodules:
```bash
git submodule update --init --recursive
```

3. Build the project:
```bash
./build.sh
```

Or manually:
```bash
mkdir build && cd build
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF
ninja
```

## Usage

### Basic Usage

Run the application:
```bash
cd build
./pendulum-visualizer
```

### Command Line Options

```bash
./pendulum-visualizer [options]

Options:
  --width WIDTH        Window width (default: 1200)
  --height HEIGHT      Window height (default: 800)
  --fullscreen         Start in fullscreen mode
  --no-vsync           Disable vertical sync
  --config FILE        Load configuration file
  --help               Show help message
```

### Controls

**Keyboard:**
- `Space` - Play/Pause simulation
- `R` - Reset simulation to initial conditions
- `1` - Switch to Simulation view
- `2` - Switch to Analysis view
- `Esc` - Exit application

**Mouse:**
- `Scroll` - Zoom in/out
- `Click + Drag` - Pan camera (future feature)

**UI Controls:**
- Use the parameter sliders to adjust pendulum properties
- Click "Start Analysis" to begin chaos analysis computation
- Toggle display options like labels and trails

## Project Structure

```
src/
├── core/           # Application framework
├── physics/        # Lagrangian mechanics simulation
├── rendering/      # OpenGL graphics rendering
├── ui/            # ImGui user interface
└── analysis/      # Chaos analysis algorithms

include/           # Public header files
shaders/          # GLSL shader files
config/           # Configuration files
tests/            # Unit and integration tests
```

## Technical Details

### Physics Implementation

- **Lagrangian Mechanics**: Accurate double pendulum equations of motion
- **RK4 Integration**: 4th-order Runge-Kutta for numerical stability
- **Energy Conservation**: Monitors energy drift for accuracy validation
- **Timestep**: 1000Hz physics with 60fps rendering

### Rendering Pipeline

- **OpenGL 4.6 Core**: Modern graphics pipeline
- **Custom Shaders**: GLSL shaders for primitives and effects
- **Immediate Mode**: Direct geometry submission for simple scenes
- **Text Rendering**: ImGui overlays for labels and UI

### Chaos Analysis

- **Lyapunov Exponents**: Quantify chaotic behavior
- **Color Mapping**: Visual representation from stable (blue) to chaotic (red)
- **Grid Analysis**: Systematic exploration of initial condition space
- **GPU Acceleration**: CUDA support for large-scale analysis (optional)

## Configuration

The application loads configuration from `config/default.json`:

```json
{
  "pendulum": {
    "l1": 1.0,
    "l2": 1.0,
    "m1": 1.0,
    "m2": 1.0,
    "g": 9.81
  },
  "simulation": {
    "timestep": 0.001,
    "initial_theta1": 1.0,
    "initial_theta2": 0.0
  }
}
```

## Development

### Building Tests

```bash
cd build
ninja test
```

### Debug Build

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
ninja
```

### Adding Features

1. Follow the library-based architecture
2. Add tests before implementation (TDD)
3. Update documentation and configuration
4. Ensure 60fps performance target

## Troubleshooting

### Common Issues

**OpenGL Context Creation Fails:**
```bash
glxinfo | grep "OpenGL version"  # Check OpenGL support
```

**Missing Dependencies:**
```bash
ldd ./pendulum-visualizer  # Check linked libraries
```

**Performance Issues:**
- Enable GPU performance mode
- Check for thermal throttling
- Verify OpenGL hardware acceleration

### Debug Output

Run with debug output:
```bash
MESA_DEBUG=1 ./pendulum-visualizer
```

## License

This project is developed as part of the spec-kit-testing framework for educational and research purposes.

## Contributing

This implementation follows the constitutional principles outlined in the project documentation:
- Library-first architecture
- Test-driven development
- Performance-focused design
- Clear separation of concerns

See the `specs/001-build-an-opengl/` directory for detailed technical specifications.
