cmake_minimum_required(VERSION 3.18)

# Options
option(USE_CUDA "Enable CUDA acceleration" OFF)
option(USE_TESTS "Build tests" ON)

# Project declaration with conditional CUDA
if(USE_CUDA)
    project(DoublePendulumVisualizer VERSION 0.1.0 LANGUAGES CXX CUDA)
else()
    project(DoublePendulumVisualizer VERSION 0.1.0 LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(USE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
endif()

# Find required packages
find_package(OpenGL REQUIRED)
find_package(PkgConfig REQUIRED)

# Use pkg-config to find GLFW with Wayland support
pkg_check_modules(GLFW3 REQUIRED glfw3)

find_package(GLEW REQUIRED)
find_package(glm REQUIRED)

if(USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
    # Enable extended lambda for device code and suppress GLM warnings
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda -diag-suppress=20012")
endif()

if(USE_TESTS)
    find_package(GTest REQUIRED)
    enable_testing()
endif()

# Include directories
include_directories(include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/imgui)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/imgui/backends)

# ImGui setup
set(IMGUI_SOURCES
    external/imgui/imgui.cpp
    external/imgui/imgui_demo.cpp
    external/imgui/imgui_draw.cpp
    external/imgui/imgui_tables.cpp
    external/imgui/imgui_widgets.cpp
    external/imgui/backends/imgui_impl_glfw.cpp
    external/imgui/backends/imgui_impl_opengl3.cpp
)

# Physics Library
set(PHYSICS_SOURCES
    src/physics/PendulumState.cpp
    src/physics/LagrangianSolver.cpp
)

if(USE_CUDA)
    set(PHYSICS_CUDA_SOURCES
        src/physics/cuda/KernelLauncher.cpp
        src/physics/cuda/pendulum_kernels.cu
        src/physics/cuda/CudaPhysicsSolver.cu
        src/physics/cuda/CudaPhysicsEngine.cpp
    )
    list(APPEND PHYSICS_SOURCES ${PHYSICS_CUDA_SOURCES})
endif()

add_library(physics-lib ${PHYSICS_SOURCES})
target_include_directories(physics-lib PUBLIC include)
if(USE_CUDA)
    target_compile_definitions(physics-lib PUBLIC USE_CUDA)
    target_link_libraries(physics-lib CUDA::cudart)
    set_property(TARGET physics-lib PROPERTY CUDA_RUNTIME_LIBRARY Static)
endif()

# Rendering Library
set(RENDERING_SOURCES
    src/rendering/Renderer.cpp
    src/rendering/Shader.cpp
    src/rendering/Camera.cpp
    src/rendering/Primitives.cpp
    src/rendering/TextRenderer.cpp
    src/rendering/TrailRenderer.cpp
)

if(USE_CUDA)
    set(RENDERING_CUDA_SOURCES
        src/rendering/cuda/CudaTrailRenderer.cu
    )
    list(APPEND RENDERING_SOURCES ${RENDERING_CUDA_SOURCES})
endif()

add_library(rendering-lib ${RENDERING_SOURCES})
target_include_directories(rendering-lib PUBLIC include)
target_link_libraries(rendering-lib 
    OpenGL::GL 
    GLEW::GLEW 
    glm::glm
)

if(USE_CUDA)
    target_compile_definitions(rendering-lib PUBLIC USE_CUDA)
    target_link_libraries(rendering-lib CUDA::cudart CUDA::cuda_driver)
    set_property(TARGET rendering-lib PROPERTY CUDA_RUNTIME_LIBRARY Static)
endif()

# UI Library
set(UI_SOURCES
    src/ui/UIManager.cpp
    src/ui/ViewManager.cpp
    src/ui/SimulationView.cpp
    src/ui/AnalysisView.cpp
    src/ui/ImGuiWrapper.cpp
    ${IMGUI_SOURCES}
)

add_library(ui-lib ${UI_SOURCES})
target_include_directories(ui-lib PUBLIC include)
target_link_libraries(ui-lib 
    ${GLFW3_LIBRARIES}
    OpenGL::GL 
    GLEW::GLEW
)
target_include_directories(ui-lib PRIVATE ${GLFW3_INCLUDE_DIRS})
target_compile_options(ui-lib PRIVATE ${GLFW3_CFLAGS_OTHER})

# Analysis Library
set(ANALYSIS_SOURCES
    src/analysis/ChaosAnalyzer.cpp
    src/analysis/ColorMapper.cpp
    src/analysis/InitialConditions.cpp
)

if(USE_CUDA)
    set(ANALYSIS_CUDA_SOURCES
        src/physics/cuda/KernelLauncher.cpp
        src/physics/cuda/pendulum_kernels.cu
    )
    list(APPEND ANALYSIS_SOURCES ${ANALYSIS_CUDA_SOURCES})
endif()

add_library(analysis-lib ${ANALYSIS_SOURCES})
target_include_directories(analysis-lib PUBLIC include)
target_link_libraries(analysis-lib physics-lib)
if(USE_CUDA)
    target_compile_definitions(analysis-lib PUBLIC USE_CUDA)
    target_link_libraries(analysis-lib CUDA::cudart)
    set_property(TARGET analysis-lib PROPERTY CUDA_RUNTIME_LIBRARY Static)
endif()

# Core Application
set(CORE_SOURCES
    src/core/Application.cpp
    src/core/Window.cpp
    src/core/Input.cpp
)

add_library(core-lib ${CORE_SOURCES})
target_include_directories(core-lib PUBLIC include)
target_link_libraries(core-lib 
    physics-lib 
    rendering-lib 
    ui-lib 
    analysis-lib
    ${GLFW3_LIBRARIES}
    OpenGL::GL 
    GLEW::GLEW
)
target_include_directories(core-lib PRIVATE ${GLFW3_INCLUDE_DIRS})
target_compile_options(core-lib PRIVATE ${GLFW3_CFLAGS_OTHER})

# Main Application
set(MAIN_SOURCES src/main.cpp)

if(USE_CUDA)
    list(APPEND MAIN_SOURCES src/cuda_demo.cpp)
endif()

add_executable(pendulum-visualizer ${MAIN_SOURCES})
target_link_libraries(pendulum-visualizer core-lib)

# Tests
if(USE_TESTS)
    # Tests will be added later
    # add_subdirectory(tests)
endif()

# Configuration files
configure_file(config/default.json ${CMAKE_BINARY_DIR}/config/default.json COPYONLY)

# Shader files
file(COPY shaders/ DESTINATION ${CMAKE_BINARY_DIR}/shaders/)
