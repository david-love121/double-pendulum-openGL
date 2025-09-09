#ifdef USE_CUDA

#include "CudaPhysicsSolver.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdexcept>
#include <cmath>

namespace pendulum {
namespace cuda {

// Device function for computing derivatives (shared with existing kernels)
__device__ void computeDerivatives(double theta1, double theta2, double omega1, double omega2,
                                  double l1, double l2, double m1, double m2, double g, double damping,
                                  double& dtheta1, double& dtheta2, double& domega1, double& domega2) {
    dtheta1 = omega1;
    dtheta2 = omega2;
    
    double delta = theta2 - theta1;
    double cos_delta = cos(delta);
    double sin_delta = sin(delta);
    
    double den1 = (m1 + m2) * l1 - m2 * l1 * cos_delta * cos_delta;
    double den2 = (l2 / l1) * den1;
    
    double num1 = -m2 * l1 * omega1 * omega1 * sin_delta * cos_delta +
                  m2 * g * sin(theta2) * cos_delta +
                  m2 * l2 * omega2 * omega2 * sin_delta -
                  (m1 + m2) * g * sin(theta1) -
                  damping * omega1;
    
    double num2 = -m2 * l2 * omega2 * omega2 * sin_delta * cos_delta +
                  (m1 + m2) * g * sin(theta1) * cos_delta +
                  (m1 + m2) * l1 * omega1 * omega1 * sin_delta -
                  (m1 + m2) * g * sin(theta2) -
                  damping * omega2;
    
    domega1 = num1 / den1;
    domega2 = num2 / den2;
}

// Device function for normalizing angles
__device__ void normalizeAngles(double& theta1, double& theta2) {
    auto normalize = [](double& angle) {
        while (angle > M_PI) angle -= 2.0 * M_PI;
        while (angle < -M_PI) angle += 2.0 * M_PI;
    };
    
    normalize(theta1);
    normalize(theta2);
}

// Device function for computing energy
__device__ double computeEnergy(double theta1, double theta2, double omega1, double omega2,
                               double l1, double l2, double m1, double m2, double g) {
    // Kinetic energy
    double v1_sq = l1 * l1 * omega1 * omega1;
    double v2_sq = l2 * l2 * omega2 * omega2 + 
                   l1 * l1 * omega1 * omega1 + 
                   2.0 * l1 * l2 * omega1 * omega2 * cos(theta1 - theta2);
    
    double kinetic = 0.5 * m1 * v1_sq + 0.5 * m2 * v2_sq;
    
    // Potential energy
    double h1 = -l1 * cos(theta1);
    double h2 = -l1 * cos(theta1) - l2 * cos(theta2);
    double potential = m1 * g * h1 + m2 * g * h2;
    
    return kinetic + potential;
}

// CUDA kernel for single RK4 integration step
__global__ void rk4StepKernel(SimulationState* states,
                              const PendulumConfiguration* config,
                              double timestep,
                              int num_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_states) return;
    
    SimulationState& state = states[idx];
    const PendulumConfiguration& cfg = *config;
    
    // RK4 integration
    double k1_th1, k1_th2, k1_om1, k1_om2;
    computeDerivatives(state.theta1, state.theta2, state.omega1, state.omega2,
                      cfg.l1, cfg.l2, cfg.m1, cfg.m2, cfg.g, cfg.damping,
                      k1_th1, k1_th2, k1_om1, k1_om2);
    
    double state2_th1 = state.theta1 + k1_th1 * timestep * 0.5;
    double state2_th2 = state.theta2 + k1_th2 * timestep * 0.5;
    double state2_om1 = state.omega1 + k1_om1 * timestep * 0.5;
    double state2_om2 = state.omega2 + k1_om2 * timestep * 0.5;
    
    double k2_th1, k2_th2, k2_om1, k2_om2;
    computeDerivatives(state2_th1, state2_th2, state2_om1, state2_om2,
                      cfg.l1, cfg.l2, cfg.m1, cfg.m2, cfg.g, cfg.damping,
                      k2_th1, k2_th2, k2_om1, k2_om2);
    
    double state3_th1 = state.theta1 + k2_th1 * timestep * 0.5;
    double state3_th2 = state.theta2 + k2_th2 * timestep * 0.5;
    double state3_om1 = state.omega1 + k2_om1 * timestep * 0.5;
    double state3_om2 = state.omega2 + k2_om2 * timestep * 0.5;
    
    double k3_th1, k3_th2, k3_om1, k3_om2;
    computeDerivatives(state3_th1, state3_th2, state3_om1, state3_om2,
                      cfg.l1, cfg.l2, cfg.m1, cfg.m2, cfg.g, cfg.damping,
                      k3_th1, k3_th2, k3_om1, k3_om2);
    
    double state4_th1 = state.theta1 + k3_th1 * timestep;
    double state4_th2 = state.theta2 + k3_th2 * timestep;
    double state4_om1 = state.omega1 + k3_om1 * timestep;
    double state4_om2 = state.omega2 + k3_om2 * timestep;
    
    double k4_th1, k4_th2, k4_om1, k4_om2;
    computeDerivatives(state4_th1, state4_th2, state4_om1, state4_om2,
                      cfg.l1, cfg.l2, cfg.m1, cfg.m2, cfg.g, cfg.damping,
                      k4_th1, k4_th2, k4_om1, k4_om2);
    
    // Update state
    state.theta1 += (k1_th1 + 2*k2_th1 + 2*k3_th1 + k4_th1) * timestep / 6.0;
    state.theta2 += (k1_th2 + 2*k2_th2 + 2*k3_th2 + k4_th2) * timestep / 6.0;
    state.omega1 += (k1_om1 + 2*k2_om1 + 2*k3_om1 + k4_om1) * timestep / 6.0;
    state.omega2 += (k1_om2 + 2*k2_om2 + 2*k3_om2 + k4_om2) * timestep / 6.0;
    
    // Normalize angles
    normalizeAngles(state.theta1, state.theta2);
    
    // Update timestamp and energy
    state.timestamp += timestep;
    state.energy = computeEnergy(state.theta1, state.theta2, state.omega1, state.omega2,
                                cfg.l1, cfg.l2, cfg.m1, cfg.m2, cfg.g);
}

// CUDA kernel for batch integration (multiple timesteps)
__global__ void batchIntegrationKernel(SimulationState* states,
                                      SimulationState* temp_states,
                                      const PendulumConfiguration* config,
                                      double timestep,
                                      int num_steps,
                                      int num_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_states) return;
    
    SimulationState current_state = states[idx];
    
    for (int step = 0; step < num_steps; ++step) {
        // Perform RK4 step (reusing the logic from rk4StepKernel)
        const PendulumConfiguration& cfg = *config;
        
        double k1_th1, k1_th2, k1_om1, k1_om2;
        computeDerivatives(current_state.theta1, current_state.theta2, 
                          current_state.omega1, current_state.omega2,
                          cfg.l1, cfg.l2, cfg.m1, cfg.m2, cfg.g, cfg.damping,
                          k1_th1, k1_th2, k1_om1, k1_om2);
        
        double state2_th1 = current_state.theta1 + k1_th1 * timestep * 0.5;
        double state2_th2 = current_state.theta2 + k1_th2 * timestep * 0.5;
        double state2_om1 = current_state.omega1 + k1_om1 * timestep * 0.5;
        double state2_om2 = current_state.omega2 + k1_om2 * timestep * 0.5;
        
        double k2_th1, k2_th2, k2_om1, k2_om2;
        computeDerivatives(state2_th1, state2_th2, state2_om1, state2_om2,
                          cfg.l1, cfg.l2, cfg.m1, cfg.m2, cfg.g, cfg.damping,
                          k2_th1, k2_th2, k2_om1, k2_om2);
        
        double state3_th1 = current_state.theta1 + k2_th1 * timestep * 0.5;
        double state3_th2 = current_state.theta2 + k2_th2 * timestep * 0.5;
        double state3_om1 = current_state.omega1 + k2_om1 * timestep * 0.5;
        double state3_om2 = current_state.omega2 + k2_om2 * timestep * 0.5;
        
        double k3_th1, k3_th2, k3_om1, k3_om2;
        computeDerivatives(state3_th1, state3_th2, state3_om1, state3_om2,
                          cfg.l1, cfg.l2, cfg.m1, cfg.m2, cfg.g, cfg.damping,
                          k3_th1, k3_th2, k3_om1, k3_om2);
        
        double state4_th1 = current_state.theta1 + k3_th1 * timestep;
        double state4_th2 = current_state.theta2 + k3_th2 * timestep;
        double state4_om1 = current_state.omega1 + k3_om1 * timestep;
        double state4_om2 = current_state.omega2 + k3_om2 * timestep;
        
        double k4_th1, k4_th2, k4_om1, k4_om2;
        computeDerivatives(state4_th1, state4_th2, state4_om1, state4_om2,
                          cfg.l1, cfg.l2, cfg.m1, cfg.m2, cfg.g, cfg.damping,
                          k4_th1, k4_th2, k4_om1, k4_om2);
        
        // Update state
        current_state.theta1 += (k1_th1 + 2*k2_th1 + 2*k3_th1 + k4_th1) * timestep / 6.0;
        current_state.theta2 += (k1_th2 + 2*k2_th2 + 2*k3_th2 + k4_th2) * timestep / 6.0;
        current_state.omega1 += (k1_om1 + 2*k2_om1 + 2*k3_om1 + k4_om1) * timestep / 6.0;
        current_state.omega2 += (k1_om2 + 2*k2_om2 + 2*k3_om2 + k4_om2) * timestep / 6.0;
        
        // Normalize angles
        normalizeAngles(current_state.theta1, current_state.theta2);
        
        // Update timestamp and energy
        current_state.timestamp += timestep;
        current_state.energy = computeEnergy(current_state.theta1, current_state.theta2,
                                           current_state.omega1, current_state.omega2,
                                           cfg.l1, cfg.l2, cfg.m1, cfg.m2, cfg.g);
    }
    
    // Write final state back
    states[idx] = current_state;
}

// Host wrapper functions
extern "C" {
    void launchRK4Step(SimulationState* states,
                       const PendulumConfiguration* config,
                       double timestep,
                       int num_states,
                       cudaStream_t stream) {
        const int blockSize = 256;
        const int gridSize = (num_states + blockSize - 1) / blockSize;
        
        rk4StepKernel<<<gridSize, blockSize, 0, stream>>>(
            states, config, timestep, num_states);
    }
    
    void launchBatchIntegration(SimulationState* states,
                               SimulationState* temp_states,
                               const PendulumConfiguration* config,
                               double timestep,
                               int num_steps,
                               int num_states,
                               cudaStream_t stream) {
        const int blockSize = 256;
        const int gridSize = (num_states + blockSize - 1) / blockSize;
        
        batchIntegrationKernel<<<gridSize, blockSize, 0, stream>>>(
            states, temp_states, config, timestep, num_steps, num_states);
    }
}

// Host class implementation
CudaPhysicsSolver::CudaPhysicsSolver() {
    // Constructor
}

CudaPhysicsSolver::~CudaPhysicsSolver() {
    cleanup();
}

bool CudaPhysicsSolver::initialize(int device_id) {
    if (m_initialized) {
        return true;
    }
    
    // Check CUDA availability
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        std::cerr << "No CUDA devices available: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    if (device_id >= deviceCount) {
        std::cerr << "Invalid device ID: " << device_id << " (max: " << deviceCount - 1 << ")" << std::endl;
        return false;
    }
    
    // Set device
    err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        std::cerr << "Failed to set CUDA device: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    m_deviceId = device_id;
    
    // Create CUDA stream
    err = cudaStreamCreate(&m_stream);
    if (err != cudaSuccess) {
        std::cerr << "Failed to create CUDA stream: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Create CUDA events for timing
    err = cudaEventCreate(&m_startEvent);
    if (err != cudaSuccess) {
        std::cerr << "Failed to create CUDA start event: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return false;
    }
    
    err = cudaEventCreate(&m_stopEvent);
    if (err != cudaSuccess) {
        std::cerr << "Failed to create CUDA stop event: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return false;
    }
    
    // Allocate initial device memory for single state
    allocateDeviceMemory(1024);  // Start with support for 1024 parallel states
    
    m_initialized = true;
    std::cout << "CUDA Physics Solver initialized on device " << device_id << std::endl;
    
    return true;
}

void CudaPhysicsSolver::cleanup() {
    freeDeviceMemory();
    
    if (m_stream) {
        cudaStreamDestroy(m_stream);
        m_stream = nullptr;
    }
    
    if (m_startEvent) {
        cudaEventDestroy(m_startEvent);
        m_startEvent = nullptr;
    }
    
    if (m_stopEvent) {
        cudaEventDestroy(m_stopEvent);
        m_stopEvent = nullptr;
    }
    
    m_initialized = false;
}

void CudaPhysicsSolver::allocateDeviceMemory(size_t max_states) {
    if (max_states <= m_maxStates && d_states != nullptr) {
        return;  // Already have enough memory
    }
    
    freeDeviceMemory();
    
    cudaError_t err;
    
    // Allocate device memory
    err = cudaMalloc(&d_states, max_states * sizeof(SimulationState));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory for states: " + 
                                std::string(cudaGetErrorString(err)));
    }
    
    err = cudaMalloc(&d_temp_states, max_states * sizeof(SimulationState));
    if (err != cudaSuccess) {
        cudaFree(d_states);
        d_states = nullptr;
        throw std::runtime_error("Failed to allocate device memory for temp states: " + 
                                std::string(cudaGetErrorString(err)));
    }
    
    err = cudaMalloc(&d_config, sizeof(PendulumConfiguration));
    if (err != cudaSuccess) {
        cudaFree(d_states);
        cudaFree(d_temp_states);
        d_states = nullptr;
        d_temp_states = nullptr;
        throw std::runtime_error("Failed to allocate device memory for config: " + 
                                std::string(cudaGetErrorString(err)));
    }
    
    m_maxStates = max_states;
}

void CudaPhysicsSolver::freeDeviceMemory() {
    if (d_states) {
        cudaFree(d_states);
        d_states = nullptr;
    }
    
    if (d_temp_states) {
        cudaFree(d_temp_states);
        d_temp_states = nullptr;
    }
    
    if (d_config) {
        cudaFree(d_config);
        d_config = nullptr;
    }
    
    m_maxStates = 0;
}

SimulationState CudaPhysicsSolver::step(const SimulationState& state, 
                                       const PendulumConfiguration& config, 
                                       double timestep) {
    if (!m_initialized) {
        throw std::runtime_error("CUDA Physics Solver not initialized");
    }
    
    // Ensure we have device memory
    allocateDeviceMemory(1);
    
    // Copy data to device
    cudaError_t err = cudaMemcpy(d_states, &state, sizeof(SimulationState), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy state to device: " + std::string(cudaGetErrorString(err)));
    }
    
    err = cudaMemcpy(d_config, &config, sizeof(PendulumConfiguration), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy config to device: " + std::string(cudaGetErrorString(err)));
    }
    
    // Record start time
    cudaEventRecord(m_startEvent, m_stream);
    
    // Launch kernel
    launchRK4Step(d_states, d_config, timestep, 1, m_stream);
    
    // Record stop time
    cudaEventRecord(m_stopEvent, m_stream);
    
    // Copy result back
    SimulationState result;
    err = cudaMemcpy(&result, d_states, sizeof(SimulationState), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy result from device: " + std::string(cudaGetErrorString(err)));
    }
    
    // Calculate kernel time
    cudaEventSynchronize(m_stopEvent);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, m_startEvent, m_stopEvent);
    m_lastKernelTime = milliseconds / 1000.0;  // Convert to seconds
    
    return result;
}

std::vector<SimulationState> CudaPhysicsSolver::stepBatch(const SimulationState& initial_state,
                                                         const PendulumConfiguration& config,
                                                         double timestep,
                                                         int num_steps) {
    if (!m_initialized) {
        throw std::runtime_error("CUDA Physics Solver not initialized");
    }
    
    // For batch processing, we simulate one trajectory but capture intermediate states
    allocateDeviceMemory(num_steps + 1);
    
    // Set up initial state on device
    std::vector<SimulationState> host_states(num_steps + 1);
    host_states[0] = initial_state;
    
    cudaError_t err = cudaMemcpy(d_states, &initial_state, sizeof(SimulationState), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy initial state to device: " + std::string(cudaGetErrorString(err)));
    }
    
    err = cudaMemcpy(d_config, &config, sizeof(PendulumConfiguration), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy config to device: " + std::string(cudaGetErrorString(err)));
    }
    
    // Record start time
    cudaEventRecord(m_startEvent, m_stream);
    
    // Perform simulation steps one by one, capturing each state
    for (int i = 0; i < num_steps; ++i) {
        launchRK4Step(d_states, d_config, timestep, 1, m_stream);
        
        // Copy current state back to host
        err = cudaMemcpy(&host_states[i + 1], d_states, sizeof(SimulationState), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy intermediate state from device: " + std::string(cudaGetErrorString(err)));
        }
    }
    
    // Record stop time
    cudaEventRecord(m_stopEvent, m_stream);
    
    // Calculate kernel time
    cudaEventSynchronize(m_stopEvent);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, m_startEvent, m_stopEvent);
    m_lastKernelTime = milliseconds / 1000.0;
    
    return host_states;
}

std::vector<SimulationState> CudaPhysicsSolver::stepParallel(const std::vector<SimulationState>& states,
                                                            const PendulumConfiguration& config,
                                                            double timestep) {
    if (!m_initialized) {
        throw std::runtime_error("CUDA Physics Solver not initialized");
    }
    
    if (states.empty()) {
        return {};
    }
    
    // Ensure device memory is sufficient
    allocateDeviceMemory(states.size());
    
    // Copy states to device
    copyToDevice(states);
    
    cudaError_t err = cudaMemcpy(d_config, &config, sizeof(PendulumConfiguration), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy config to device: " + std::string(cudaGetErrorString(err)));
    }
    
    // Record start time
    cudaEventRecord(m_startEvent, m_stream);
    
    // Launch parallel integration
    launchRK4Step(d_states, d_config, timestep, static_cast<int>(states.size()), m_stream);
    
    // Record stop time
    cudaEventRecord(m_stopEvent, m_stream);
    
    // Copy results back
    std::vector<SimulationState> results(states.size());
    copyFromDevice(results, states.size());
    
    // Calculate kernel time
    cudaEventSynchronize(m_stopEvent);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, m_startEvent, m_stopEvent);
    m_lastKernelTime = milliseconds / 1000.0;
    
    return results;
}

void CudaPhysicsSolver::copyToDevice(const std::vector<SimulationState>& states) {
    cudaError_t err = cudaMemcpyAsync(d_states, states.data(), 
                                     states.size() * sizeof(SimulationState), 
                                     cudaMemcpyHostToDevice, m_stream);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy states to device: " + std::string(cudaGetErrorString(err)));
    }
}

void CudaPhysicsSolver::copyFromDevice(std::vector<SimulationState>& states, size_t count) {
    cudaError_t err = cudaMemcpyAsync(states.data(), d_states,
                                     count * sizeof(SimulationState),
                                     cudaMemcpyDeviceToHost, m_stream);
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy states from device: " + std::string(cudaGetErrorString(err)));
    }
    
    // Synchronize to ensure copy is complete
    cudaStreamSynchronize(m_stream);
}

} // namespace cuda
} // namespace pendulum

#endif // USE_CUDA
