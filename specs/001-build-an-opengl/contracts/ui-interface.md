# UI Library Interface Contract

**Library**: `ui-lib`  
**Purpose**: ImGui-based user interface for view switching and parameter controls  

## ViewManager Interface

### update()
**Description**: Update UI state and handle user interactions
**Input**:
```cpp
struct UIContext {
    bool simulation_running;
    bool analysis_computing;
    double computation_progress;    // [0.0, 1.0] for analysis
    int current_view;              // 0=simulation, 1=analysis
};

struct SimulationParams {
    double l1, l2;                 // Pendulum lengths
    double m1, m2;                 // Pendulum masses
    double theta1_0, theta2_0;     // Initial angles
    double omega1_0, omega2_0;     // Initial velocities
    double damping;                // Damping coefficient
    bool show_labels;              // Display labels
    bool show_trails;              // Display trajectory
};

struct AnalysisParams {
    int grid_resolution_x, grid_resolution_y;
    double theta1_min, theta1_max;
    double theta2_min, theta2_max;
    int color_scheme;              // Color mapping scheme
    double integration_time;       // Analysis duration
};
```

**Output**:
```cpp
struct UIEvents {
    bool view_changed;             // User switched views
    bool simulation_reset;         // User requested reset
    bool simulation_play_pause;    // User toggled playback
    bool analysis_start;           // User started analysis
    bool parameters_changed;       // User modified parameters
    int new_view_id;              // Target view if changed
};
```

**Behavior**:
- MUST provide intuitive controls for all simulation parameters
- MUST show real-time parameter values during simulation
- MUST provide progress bar during chaos analysis computation
- MUST handle view switching without interrupting ongoing computations
- MUST validate parameter ranges and show appropriate warnings
- Response time: < 16ms for UI updates (60fps)

**Error Conditions**:
- `IMGUI_ERROR`: ImGui context initialization failure
- `INVALID_RANGE`: Parameter values outside valid ranges
- `UI_OVERFLOW`: Too many UI elements for screen space

### render_simulation_ui()
**Description**: Render controls specific to simulation view
**Input**:
```cpp
struct SimulationUIState {
    SimulationParams current_params;
    bool simulation_running;
    double current_time;
    double current_energy;
    int fps_counter;
};
```

**Output**: ImGui draw commands submitted

**Behavior**:
- MUST show real-time simulation status (time, energy, FPS)
- MUST provide sliders for pendulum parameters (L1, L2, masses)
- MUST provide angle input for initial conditions
- MUST show play/pause/reset buttons
- MUST display current pendulum angles and velocities
- MUST provide checkbox controls for visual options

### render_analysis_ui()
**Description**: Render controls specific to chaos analysis view
**Input**:
```cpp
struct AnalysisUIState {
    AnalysisParams current_params;
    bool computing;
    double progress;               // [0.0, 1.0]
    double estimated_time_remaining;
    int completed_points;
    int total_points;
};
```

**Output**: ImGui draw commands submitted

**Behavior**:
- MUST show analysis progress with percentage and time estimates
- MUST provide controls for grid resolution and angle ranges
- MUST show start/stop/cancel buttons for analysis
- MUST display computation statistics (points/second)
- MUST provide color scheme selection dropdown
- MUST show interactive zoom/pan controls for result viewing

## ImGuiWrapper Interface

### initialize()
**Description**: Initialize ImGui with OpenGL backend
**Input**:
```cpp
struct ImGuiConfig {
    GLFWwindow* window_handle;
    bool enable_docking;
    bool enable_viewports;
    std::string font_path;         // Path to font file
    float font_size_pixels;
};
```

**Output**: `SUCCESS` or error code

**Behavior**:
- MUST initialize ImGui with OpenGL3 backend
- MUST load custom font with mathematical symbols
- MUST configure ImGui style for scientific application
- MUST handle high-DPI displays correctly

**Error Conditions**:
- `INIT_ERROR`: ImGui initialization failure
- `FONT_ERROR`: Font loading failure
- `BACKEND_ERROR`: OpenGL backend setup failure

### begin_frame()
**Description**: Start new ImGui frame for rendering
**Input**: None

**Output**: Frame ready for UI rendering

**Behavior**:
- MUST call ImGui::NewFrame()
- MUST handle input events from GLFW
- MUST update display scaling for high-DPI

### end_frame()
**Description**: Finish ImGui frame and render to screen
**Input**: None

**Output**: UI rendered to current framebuffer

**Behavior**:
- MUST call ImGui::Render()
- MUST execute ImGui draw commands via OpenGL
- MUST maintain UI performance (no frame drops)

## ParameterValidator Interface

### validate_pendulum_params()
**Description**: Validate pendulum physics parameters
**Input**:
```cpp
struct PendulumParams {
    double l1, l2, m1, m2, g, damping;
};
```

**Output**:
```cpp
struct ValidationResult {
    bool valid;
    std::vector<std::string> warnings;
    std::vector<std::string> errors;
};
```

**Behavior**:
- MUST check all parameters are positive and finite
- MUST warn about extreme parameter ratios (e.g., l1/l2 > 100)
- MUST validate damping coefficient range [0.0, 1.0]
- MUST check for numerical stability conditions

### validate_analysis_params()
**Description**: Validate chaos analysis parameters
**Input**:
```cpp
struct AnalysisParams {
    int resolution_x, resolution_y;
    double theta_ranges[4];        // min1, max1, min2, max2
    double integration_time;
};
```

**Output**: `ValidationResult` (same structure as above)

**Behavior**:
- MUST check resolution limits based on available GPU memory
- MUST validate angle ranges are within [-π, π]
- MUST warn about excessive computation time estimates
- MUST check integration time is reasonable (0.1s to 100s)

## Contract Validation Tests

### ViewManager Tests
1. **State Management Test**: Verify UI state persists across view switches
2. **Parameter Validation Test**: Verify invalid inputs are rejected with clear messages
3. **Event Handling Test**: Verify UI events trigger correct application responses
4. **Performance Test**: Verify UI updates don't impact simulation framerate

### ImGui Integration Tests
1. **Initialization Test**: Verify ImGui initializes successfully with OpenGL backend
2. **Font Test**: Verify mathematical symbols render correctly in UI
3. **Input Test**: Verify mouse/keyboard input handled correctly
4. **Rendering Test**: Verify UI renders without visual artifacts

### Parameter Validation Tests
1. **Range Test**: Verify parameter ranges enforced correctly
2. **Stability Test**: Verify numerical stability warnings appear appropriately
3. **Memory Test**: Verify grid resolution limits based on available GPU memory
4. **Performance Test**: Verify validation doesn't introduce UI lag

## UI Layout Specification

### Simulation View Layout
```
┌─────────────────┬─────────────────┐
│                 │ Controls Panel  │
│   Pendulum      │ ┌─────────────┐ │
│   Animation     │ │ Parameters  │ │
│                 │ │ L1: [1.0  ] │ │
│                 │ │ L2: [1.0  ] │ │
│                 │ │ θ1: [0.5  ] │ │
│                 │ │ θ2: [0.0  ] │ │
│                 │ └─────────────┘ │
│                 │ ┌─────────────┐ │
│                 │ │ Status      │ │
│                 │ │ Time: 12.3s │ │
│                 │ │ FPS:  60    │ │
│                 │ │ Energy: 9.8 │ │
│                 │ └─────────────┘ │
└─────────────────┴─────────────────┘
```

### Analysis View Layout
```
┌─────────────────┬─────────────────┐
│                 │ Analysis Panel  │
│   Chaos Grid    │ ┌─────────────┐ │
│   Visualization │ │ Grid Setup  │ │
│                 │ │ Res: 256x256│ │
│                 │ │ θ1: [-π, π] │ │
│                 │ │ θ2: [-π, π] │ │
│                 │ └─────────────┘ │
│                 │ ┌─────────────┐ │
│                 │ │ Progress    │ │
│                 │ │ [████  ] 67%│ │
│                 │ │ ETA: 12s    │ │
│                 │ └─────────────┘ │
└─────────────────┴─────────────────┘
```

## UI Style Guidelines

- **Colors**: Dark theme with blue accents for scientific application
- **Fonts**: Monospace for numerical values, sans-serif for labels
- **Controls**: Immediate feedback on parameter changes
- **Responsiveness**: All controls respond within 16ms
- **Accessibility**: High contrast, readable at different DPI settings
