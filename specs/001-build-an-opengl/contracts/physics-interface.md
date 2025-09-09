# Physics Library Interface Contract

**Library**: `physics-lib`  
**Purpose**: Lagrangian mechanics simulation with CUDA acceleration  

## PhysicsEngine Interface

### simulate_step()
**Description**: Advance pendulum simulation by one timestep using Lagrangian mechanics
**Input**:
```cpp
struct SimulationState {
    double theta1, theta2;      // Current angles (radians)
    double omega1, omega2;      // Current angular velocities (rad/s)
    double timestamp;           // Current time (seconds)
};

struct PendulumConfiguration {
    double l1, l2;              // Arm lengths (meters)
    double m1, m2;              // Masses (kg)
    double g;                   // Gravity (m/s²)
    double damping;             // Damping coefficient
};

double timestep;                // Integration step size (seconds)
```

**Output**:
```cpp
struct SimulationState {
    double theta1, theta2;      // Updated angles
    double omega1, omega2;      // Updated velocities
    double timestamp;           // Updated time
};
```

**Behavior**:
- MUST use Runge-Kutta 4th order integration for accuracy
- MUST preserve energy within 0.1% tolerance (with damping=0)
- MUST handle angle wrap-around at ±π boundaries
- MUST detect and prevent numerical instability
- Response time: < 1ms for single step

**Error Conditions**:
- `INVALID_TIMESTEP`: timestep <= 0 or > 0.01
- `UNSTABLE_STATE`: NaN or infinite values detected
- `CONFIGURATION_ERROR`: Invalid pendulum parameters

### reset()
**Description**: Reset simulation to initial conditions
**Input**:
```cpp
struct InitialConditions {
    double theta1_0, theta2_0;  // Initial angles
    double omega1_0, omega2_0;  // Initial velocities
};
```

**Output**: `SUCCESS` or error code

## ChaosAnalyzer Interface (CUDA)

### analyze_grid()
**Description**: Compute Lyapunov exponents for grid of initial conditions
**Input**:
```cpp
struct AnalysisGrid {
    int resolution_x, resolution_y;
    double theta1_min, theta1_max;
    double theta2_min, theta2_max;
    double integration_time;     // Total time to analyze (seconds)
    int sample_points;           // Number of sample points for divergence
};
```

**Output**:
```cpp
struct AnalysisResult {
    std::vector<double> lyapunov_exponents;  // Row-major grid order
    double computation_time;                 // Wall-clock time (seconds)
    int successful_computations;             // Count of valid results
};
```

**Behavior**:
- MUST use parallel CUDA kernels (one thread per grid point)
- MUST handle divergent trajectories gracefully
- MUST provide progress callbacks for UI updates
- Performance target: < 1 second for 256x256 grid
- Memory limit: Must fit in available GPU memory

**Error Conditions**:
- `GPU_MEMORY_ERROR`: Insufficient GPU memory for grid
- `CUDA_ERROR`: CUDA runtime failure
- `COMPUTATION_TIMEOUT`: Analysis exceeds maximum time limit

## KernelLauncher Interface (CUDA)

### launch_pendulum_kernel()
**Description**: Launch CUDA kernel for parallel pendulum simulation
**Input**:
```cpp
struct KernelConfig {
    dim3 grid_size;
    dim3 block_size;
    size_t shared_memory_bytes;
    cudaStream_t stream;
};

// Device memory pointers
double* d_initial_conditions;   // Input: [theta1, theta2, omega1, omega2] * N
double* d_results;              // Output: Lyapunov exponents * N
int num_points;                 // Number of simulation points
double integration_time;        // Total simulation time
```

**Output**: `cudaError_t` status code

**Behavior**:
- MUST use asynchronous execution with streams
- MUST validate device memory pointers
- MUST handle CUDA errors gracefully
- MUST provide kernel completion checking

## Contract Validation Tests

### Physics Engine Tests
1. **Energy Conservation Test**: Verify energy stays constant (±0.1%) with damping=0
2. **Numerical Stability Test**: Run 1000s simulation without overflow/NaN
3. **Angle Normalization Test**: Verify angles stay in [-π, π] range
4. **Performance Test**: Verify < 1ms per timestep on target hardware

### Chaos Analyzer Tests  
1. **Memory Allocation Test**: Verify successful GPU memory allocation for max grid
2. **Parallel Execution Test**: Verify all grid points computed correctly
3. **Progress Callback Test**: Verify progress updates during computation
4. **Error Handling Test**: Verify graceful handling of GPU memory exhaustion

### CUDA Kernel Tests
1. **Stream Synchronization Test**: Verify asynchronous execution works correctly
2. **Memory Transfer Test**: Verify correct host-device data transfers
3. **Error Propagation Test**: Verify CUDA errors properly reported to host
4. **Performance Test**: Verify meets throughput targets

## Interface Dependencies

```
Application
    ↓
PhysicsEngine ←→ ChaosAnalyzer
    ↓                ↓
KernelLauncher ←-----+
    ↓
CUDA Runtime
```

**External Dependencies**:
- CUDA Runtime API 12.x
- cuRAND for random number generation (if needed)
- Standard C++ math library
- OpenGL interop for visualization data sharing
