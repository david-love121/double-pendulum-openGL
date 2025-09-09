# Rendering Library Interface Contract

**Library**: `rendering-lib`  
**Purpose**: OpenGL-based visualization for pendulum simulation and chaos analysis  

## Renderer Interface

### render_simulation()
**Description**: Render the pendulum simulation view with labels and indicators
**Input**:
```cpp
struct RenderContext {
    int viewport_width, viewport_height;
    mat4 view_matrix;
    mat4 projection_matrix;
};

struct PendulumRenderData {
    vec2 pivot_position;        // Pendulum mount point
    vec2 bob1_position;         // First pendulum bob position
    vec2 bob2_position;         // Second pendulum bob position
    double l1, l2;              // Arm lengths (for labels)
    double theta1, theta2;      // Angles (for arc indicators)
    bool show_labels;           // Display L1, L2 text
    bool show_angles;           // Display theta arcs
    bool show_trail;            // Display trajectory trail
};
```

**Output**: OpenGL draw calls completed, framebuffer updated

**Behavior**:
- MUST render pendulum arms as lines with appropriate thickness
- MUST render pendulum bobs as circles with physics-accurate positions
- MUST render text labels "L1", "L2" near respective arms
- MUST render angle arcs from vertical axis to pendulum arms
- MUST render theta1, theta2 labels near arc indicators
- MUST maintain 60fps performance during animation
- MUST handle viewport resize gracefully

**Error Conditions**:
- `OPENGL_ERROR`: OpenGL state error or shader compilation failure
- `INVALID_VIEWPORT`: Viewport dimensions <= 0
- `SHADER_ERROR`: Vertex/fragment shader compilation failure

### render_analysis()
**Description**: Render the chaos analysis view as color-coded grid
**Input**:
```cpp
struct AnalysisRenderData {
    int grid_width, grid_height;
    float* color_data;          // RGB values, row-major order
    double theta1_min, theta1_max;
    double theta2_min, theta2_max;
    bool show_grid_lines;
    bool show_axis_labels;
};
```

**Output**: OpenGL draw calls completed, framebuffer updated

**Behavior**:
- MUST render color grid using OpenGL texture/quad rendering
- MUST render axis labels showing angle ranges
- MUST render coordinate grid lines if requested
- MUST handle zooming and panning interactions
- MUST maintain performance for up to 1024x1024 grids
- MUST use appropriate color interpolation (bilinear)

**Error Conditions**:
- `TEXTURE_ERROR`: GPU texture allocation failure
- `INVALID_DATA`: Null or malformed color data
- `MEMORY_ERROR`: Insufficient GPU memory for texture

## Shader Interface

### compile_shader()
**Description**: Compile and link GLSL shader programs
**Input**:
```cpp
struct ShaderSource {
    std::string vertex_source;
    std::string fragment_source;
    std::string geometry_source;    // Optional
};
```

**Output**:
```cpp
struct ShaderProgram {
    GLuint program_id;
    bool compilation_success;
    std::string error_log;
    std::map<std::string, GLint> uniform_locations;
};
```

**Behavior**:
- MUST validate shader source before compilation
- MUST provide detailed error messages for compilation failures
- MUST cache uniform locations for performance
- MUST handle OpenGL context loss gracefully

### set_uniform()
**Description**: Set shader uniform values with type safety
**Input**:
```cpp
template<typename T>
void set_uniform(const std::string& name, const T& value);

// Supported types: float, vec2, vec3, vec4, mat3, mat4, int, bool
```

**Behavior**:
- MUST validate uniform name exists in shader
- MUST check OpenGL uniform type matches template parameter
- MUST handle inactive uniforms gracefully (no error)

## TextRenderer Interface

### render_text()
**Description**: Render text strings with specified font and positioning
**Input**:
```cpp
struct TextRenderData {
    std::string text;
    vec2 position;              // Screen coordinates or world coordinates
    float font_size;            // In pixels or world units
    vec4 color;                 // RGBA color
    enum CoordinateSystem { SCREEN, WORLD } coord_system;
    enum Alignment { LEFT, CENTER, RIGHT } alignment;
};
```

**Output**: Text rendered to current framebuffer

**Behavior**:
- MUST support basic ASCII character set (32-126)
- MUST handle UTF-8 encoding for mathematical symbols (θ, π)
- MUST maintain text readability at different zoom levels
- MUST batch text rendering for performance
- Performance target: > 1000 characters at 60fps

**Error Conditions**:
- `FONT_ERROR`: Font loading or texture atlas creation failure
- `INVALID_CHARACTER`: Unsupported character code
- `RENDER_ERROR`: OpenGL text rendering failure

## Camera Interface

### update_view()
**Description**: Update view matrix based on camera parameters
**Input**:
```cpp
struct CameraParams {
    vec2 position;              // World space camera center
    float zoom_level;           // Zoom factor (1.0 = default)
    float aspect_ratio;         // Viewport width/height
    vec2 world_bounds_min;      // Minimum world coordinates
    vec2 world_bounds_max;      // Maximum world coordinates
};
```

**Output**:
```cpp
struct ViewMatrices {
    mat4 view_matrix;
    mat4 projection_matrix;
    mat4 view_projection_matrix;
};
```

**Behavior**:
- MUST compute orthographic projection for 2D visualization
- MUST clamp camera position to valid world bounds
- MUST maintain aspect ratio during viewport changes
- MUST provide smooth interpolation for camera movements

### screen_to_world()
**Description**: Convert screen coordinates to world coordinates
**Input**:
```cpp
vec2 screen_position;           // Screen coordinates (pixels)
ViewMatrices matrices;
int viewport_width, viewport_height;
```

**Output**:
```cpp
vec2 world_position;            // World coordinates
```

**Behavior**:
- MUST handle viewport coordinate system correctly
- MUST account for current zoom and camera position
- MUST provide inverse transformation accuracy

## Contract Validation Tests

### Renderer Tests
1. **Framerate Test**: Verify 60fps during complex pendulum animation
2. **Accuracy Test**: Verify pendulum positions match physics simulation
3. **Label Test**: Verify text labels appear correctly positioned and readable
4. **Viewport Test**: Verify correct rendering after window resize

### Shader Tests
1. **Compilation Test**: Verify all required shaders compile successfully
2. **Uniform Test**: Verify uniform setting/getting works correctly
3. **Error Handling Test**: Verify shader compilation errors reported properly
4. **Performance Test**: Verify shader execution meets performance targets

### Text Rendering Tests
1. **Character Test**: Verify all required characters render correctly
2. **Unicode Test**: Verify mathematical symbols (θ, π) display properly
3. **Performance Test**: Verify text rendering doesn't impact framerate
4. **Positioning Test**: Verify text positioning accurate in world/screen space

### Camera Tests
1. **Transformation Test**: Verify world↔screen coordinate transformations
2. **Bounds Test**: Verify camera clamping to world bounds
3. **Zoom Test**: Verify zoom operations maintain center position
4. **Aspect Ratio Test**: Verify correct rendering at different aspect ratios

## OpenGL State Dependencies

**Required OpenGL Version**: 4.6 Core Profile
**Required Extensions**: None (using core functionality only)

**OpenGL State Assumptions**:
- Depth testing disabled (2D rendering)
- Blending enabled for text/UI transparency
- Viewport configured correctly by application
- Active shader program set before render calls

**Resource Management**:
- Vertex Array Objects (VAOs) for geometry
- Buffer Objects (VBOs) for vertex data
- Texture objects for color grids and font atlas
- Shader program objects with uniform caching
