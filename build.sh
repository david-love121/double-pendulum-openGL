#!/bin/bash

# Build script for Double Pendulum Visualizer
# This script sets up the build environment and compiles the project

set -e

PROJECT_ROOT="/home/david/repos/spec-kit-testing/three-js-pendulum"
BUILD_DIR="$PROJECT_ROOT/build"

echo "Double Pendulum Visualizer Build Script"
echo "========================================"

# Check if we're in the right directory
if [[ ! -f "$PROJECT_ROOT/CMakeLists.txt" ]]; then
    echo "Error: CMakeLists.txt not found. Are you in the project root?"
    exit 1
fi

# Check for required system packages
echo "Checking for required system packages..."

check_package() {
    if ! pacman -Q "$1" >/dev/null 2>&1; then
        echo "Error: Package '$1' is not installed"
        echo "Please install it with: sudo pacman -S $1"
        exit 1
    else
        echo "✓ $1 is installed"
    fi
}

# Check required packages
check_package "cmake"
check_package "ninja"
check_package "glfw-wayland"
check_package "glew"
check_package "glm"
check_package "gtest"
check_package "pkg-config"

echo "All required packages are installed!"

# Check for Wayland session
if [[ -z "$WAYLAND_DISPLAY" ]]; then
    echo "Warning: WAYLAND_DISPLAY not set. Make sure you're running in a Wayland session."
    echo "Current session type: ${XDG_SESSION_TYPE:-unknown}"
else
    echo "✓ Wayland session detected: $WAYLAND_DISPLAY"
fi

# Create build directory
echo "Setting up build directory..."
if [[ -d "$BUILD_DIR" ]]; then
    echo "Cleaning existing build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure project
echo "Configuring project with CMake..."
cmake .. -GNinja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DUSE_CUDA=ON \
    -DUSE_TESTS=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Build the project
echo "Building project..."
ninja

echo "Build completed successfully!"
echo "You can now run the application with: ./pendulum-visualizer"
