#pragma once

#ifdef USE_CUDA

#include "PendulumState.h"
#include <vector>
#include <memory>
#include <cuda_runtime.h>

namespace pendulum {
namespace cuda {

/**
 * CUDA-accelerated physics solver for double pendulum simulation
 * Handles both single simulation steps and batch processing
 */
class CudaPhysicsSolver {
public:
    CudaPhysicsSolver();
    ~CudaPhysicsSolver();
    
    bool initialize(int device_id = 0);
    void cleanup();
    
    // Single simulation step
    SimulationState step(const SimulationState& state, 
                        const PendulumConfiguration& config, 
                        double timestep);
    
    // Batch simulation for multiple timesteps
    std::vector<SimulationState> stepBatch(const SimulationState& initial_state,
                                          const PendulumConfiguration& config,
                                          double timestep,
                                          int num_steps);
    
    // Parallel simulation for multiple initial conditions
    std::vector<SimulationState> stepParallel(const std::vector<SimulationState>& states,
                                             const PendulumConfiguration& config,
                                             double timestep);
    
    // Check if CUDA is initialized and working
    bool isInitialized() const { return m_initialized; }
    
    // Performance monitoring
    double getLastKernelTime() const { return m_lastKernelTime; }
    
private:
    void allocateDeviceMemory(size_t max_states);
    void freeDeviceMemory();
    void copyToDevice(const std::vector<SimulationState>& states);
    void copyFromDevice(std::vector<SimulationState>& states, size_t count);
    
    // CUDA state
    bool m_initialized = false;
    int m_deviceId = 0;
    size_t m_maxStates = 0;
    double m_lastKernelTime = 0.0;
    
    // Device memory pointers
    SimulationState* d_states = nullptr;
    SimulationState* d_temp_states = nullptr;
    PendulumConfiguration* d_config = nullptr;
    
    // CUDA streams for asynchronous execution
    cudaStream_t m_stream = nullptr;
    
    // CUDA events for timing
    cudaEvent_t m_startEvent = nullptr;
    cudaEvent_t m_stopEvent = nullptr;
};

// Device kernels (implemented in .cu file)
extern "C" {
    void launchRK4Step(SimulationState* states,
                       const PendulumConfiguration* config,
                       double timestep,
                       int num_states,
                       cudaStream_t stream);
    
    void launchBatchIntegration(SimulationState* states,
                               SimulationState* temp_states,
                               const PendulumConfiguration* config,
                               double timestep,
                               int num_steps,
                               int num_states,
                               cudaStream_t stream);
}

} // namespace cuda
} // namespace pendulum

#endif // USE_CUDA
