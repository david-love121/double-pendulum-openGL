# Quickstart Guide: Double Pendulum OpenGL Visualization

**Date**: September 6, 2025  
**Purpose**: Validate the implementation matches the specification requirements  

## Prerequisites

### System Requirements
- **OS**: Arch Linux with Hyprland window manager
- **GPU**: NVIDIA graphics card (GTX 1060 or newer recommended)
- **Memory**: 8GB RAM minimum, 16GB recommended for large chaos analysis
- **Disk**: 500MB available space

### Software Dependencies
```bash
# Install system packages
sudo pacman -S base-devel cmake ninja
sudo pacman -S glfw-wayland glew glm 
sudo pacman -S gtest

# Install NVIDIA CUDA (from NVIDIA website or AUR)
yay -S cuda
# OR download from https://developer.nvidia.com/cuda-downloads

# Verify CUDA installation
nvcc --version
nvidia-smi
```

## Build Instructions

### 1. Clone and Setup
```bash
git clone <repository-url>
cd three-js-pendulum
git checkout 001-build-an-opengl

# Initialize git submodules for ImGui
git submodule update --init --recursive
```

### 2. Configure Build
```bash
mkdir build && cd build

# Configure with CMake
cmake .. -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCUDA_ARCHITECTURES=86 \
  -DUSE_CUDA=ON \
  -DUSE_TESTS=ON

# For debug builds (development)
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Debug -DUSE_CUDA=ON -DUSE_TESTS=ON
```

### 3. Compile
```bash
# Build all targets
ninja

# Build specific targets
ninja pendulum-visualizer      # Main application
ninja physics-lib-tests       # Physics library tests
ninja rendering-lib-tests     # Rendering library tests
ninja ui-lib-tests            # UI library tests
```

### 4. Run Tests (TDD Validation)
```bash
# Run all tests
ninja test

# Run specific test suites
./tests/physics_tests
./tests/rendering_tests
./tests/ui_tests
./tests/integration_tests

# Expected output: All tests pass
# - Physics: Energy conservation, numerical stability
# - Rendering: OpenGL context, shader compilation
# - UI: ImGui initialization, parameter validation
# - Integration: CUDA-OpenGL interop, end-to-end workflow
```

## Application Launch

### 1. Basic Launch
```bash
# Run from build directory
./pendulum-visualizer

# Run with configuration file
./pendulum-visualizer --config ../config/default.json

# Run with specific parameters
./pendulum-visualizer --l1 1.0 --l2 1.5 --theta1 0.5 --theta2 0.0
```

### 2. Command Line Options
```bash
./pendulum-visualizer --help

Options:
  --config FILE       Load configuration from JSON file
  --l1 FLOAT         First pendulum length (default: 1.0)
  --l2 FLOAT         Second pendulum length (default: 1.0)
  --m1 FLOAT         First pendulum mass (default: 1.0)
  --m2 FLOAT         Second pendulum mass (default: 1.0)
  --theta1 FLOAT     Initial angle 1 in radians (default: 1.0)
  --theta2 FLOAT     Initial angle 2 in radians (default: 0.0)
  --fullscreen       Launch in fullscreen mode
  --vsync            Enable vertical sync
  --gpu-device INT   Select CUDA device (multi-GPU systems)
  --version          Show version information
  --help             Show this help message
```

## Verification Steps

### Step 1: Application Startup
**Expected Behavior**:
- Application window opens without errors
- OpenGL context created successfully
- CUDA device detected and initialized
- UI displays with simulation view active

**Success Criteria**:
- No console error messages
- Window title shows "Double Pendulum Visualizer v0.1.0"
- FPS counter shows ~60fps in top-right corner

### Step 2: Simulation View Validation
**Test Scenario**:
1. Launch application with default parameters
2. Verify pendulum animation starts automatically
3. Check that labels are visible and correct

**Expected Results**:
- ✅ Double pendulum animates smoothly at 60fps
- ✅ Arms labeled "L1" and "L2" visible near pendulum arms
- ✅ Angle indicators "θ1" and "θ2" visible with arc lines from vertical
- ✅ Pendulum motion follows realistic physics (energy conservation)
- ✅ UI controls respond to parameter changes in real-time

**Validation Commands**:
```bash
# Test energy conservation (should remain constant with damping=0)
./pendulum-visualizer --damping 0.0 --theta1 1.5 --theta2 0.0

# Test extreme parameters (should remain stable)
./pendulum-visualizer --l1 0.1 --l2 2.0 --theta1 3.0 --theta2 -1.5
```

### Step 3: Analysis View Validation
**Test Scenario**:
1. Switch to analysis view using UI button or Tab key
2. Configure analysis parameters (256x256 grid)
3. Start chaos analysis computation

**Expected Results**:
- ✅ View switches to analysis mode showing empty grid
- ✅ Grid resolution controls allow values from 64x64 to 1024x1024
- ✅ Angle ranges default to [-π, π] for both axes
- ✅ "Start Analysis" button triggers CUDA computation
- ✅ Progress bar shows completion percentage and time estimate
- ✅ Color grid appears with blue (stable) to red (chaotic) gradient
- ✅ Analysis completes in < 5 seconds for 256x256 grid

**Validation Commands**:
```bash
# Test analysis with specific parameters
./pendulum-visualizer --analysis-mode \
  --grid-resolution 128 \
  --theta1-range -2.0,2.0 \
  --theta2-range -2.0,2.0
```

### Step 4: Performance Validation
**Test Scenario**:
1. Run continuous simulation for 5 minutes
2. Monitor GPU/CPU usage and memory consumption
3. Switch between views during computation

**Expected Results**:
- ✅ Maintains 60fps during simulation
- ✅ GPU memory usage < 2GB
- ✅ CPU usage < 50% on single core
- ✅ No memory leaks (constant memory usage after warmup)
- ✅ View switching doesn't interrupt ongoing computations

**Performance Testing**:
```bash
# Monitor system resources
nvidia-smi -l 1  # GPU usage monitoring
htop             # CPU/memory monitoring

# Stress test with large analysis
./pendulum-visualizer --grid-resolution 512 --integration-time 10.0
```

### Step 5: Error Handling Validation
**Test Scenario**:
1. Provide invalid parameters
2. Test OpenGL context loss simulation
3. Test CUDA device unavailability

**Expected Results**:
- ✅ Invalid parameters rejected with clear error messages
- ✅ Application recovers gracefully from OpenGL errors
- ✅ Fallback to CPU computation if CUDA unavailable
- ✅ No crashes or undefined behavior

**Error Testing**:
```bash
# Test invalid parameters
./pendulum-visualizer --l1 -1.0    # Should show error
./pendulum-visualizer --l1 0.0     # Should show error
./pendulum-visualizer --theta1 10.0 # Should clamp to valid range

# Test GPU memory limits
./pendulum-visualizer --grid-resolution 2048  # May exceed GPU memory
```

## Troubleshooting

### Common Issues

**OpenGL Context Creation Fails**:
```bash
# Check OpenGL support
glxinfo | grep "OpenGL version"
# Should show OpenGL 4.6 or higher

# Verify Hyprland OpenGL integration
echo $WAYLAND_DISPLAY
echo $XDG_SESSION_TYPE
```

**CUDA Initialization Fails**:
```bash
# Verify CUDA installation
nvcc --version
nvidia-smi

# Check device availability
./pendulum-visualizer --list-cuda-devices
```

**Performance Issues**:
```bash
# Enable GPU performance mode
sudo nvidia-smi -pm 1
sudo nvidia-smi -pl 300  # Set power limit to 300W (adjust for your GPU)

# Check thermal throttling
nvidia-smi -q -d TEMPERATURE
```

**Build Failures**:
```bash
# Clean build
rm -rf build && mkdir build && cd build

# Verbose build output
cmake .. -DCMAKE_VERBOSE_MAKEFILE=ON
ninja -v
```

## Success Criteria Summary

The implementation is validated when:
- [x] All tests pass without errors
- [x] Application launches successfully on Arch Linux with Hyprland
- [x] Simulation view shows labeled, animated double pendulum
- [x] Analysis view generates and displays chaos visualization
- [x] Performance targets met (60fps simulation, <5s analysis)
- [x] UI controls work correctly for parameter adjustment
- [x] CUDA acceleration functional for chaos analysis
- [x] No memory leaks or stability issues during extended use

**Final Validation**: Run the application for 30 minutes with mixed usage (simulation + analysis) to confirm stability and performance consistency.
