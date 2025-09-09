#pragma once

#ifdef USE_CUDA

#include "CudaPhysicsSolver.h"
#include "PendulumState.h"
#include <memory>
#include <vector>
#include <glm/glm.hpp>

namespace pendulum {

// Forward declarations
namespace cuda {
    class CudaTrailRenderer;
}

namespace cuda {

/**
 * Integrated CUDA-accelerated physics engine that combines
 * simulation and trail rendering with optimized data flow
 */
class CudaPhysicsEngine {
public:
    CudaPhysicsEngine();
    ~CudaPhysicsEngine();
    
    bool initialize(int device_id = 0);
    void cleanup();
    
    // Simulation control
    void setConfiguration(const PendulumConfiguration& config);
    void setState(const SimulationState& state);
    SimulationState getState() const { return m_currentState; }
    
    // Integrated simulation and trail generation
    void step(double timestep);
    void stepBatch(double timestep, int num_steps);
    
    // Trail management
    void setTrailColor(float r, float g, float b);
    void setTrailFading(bool enabled, float duration = 5.0f);
    void setTrailMaxPoints(size_t max_points);
    void clearTrails();
    
    // Rendering
    void renderTrails(const glm::mat4& view, const glm::mat4& projection);
    
    // Performance monitoring
    double getPhysicsTime() const;
    double getTrailTime() const;
    double getTotalTime() const;
    
    // Multi-pendulum simulation (for chaos analysis)
    void simulateMultiple(const std::vector<SimulationState>& initial_states,
                         double timestep,
                         int num_steps,
                         std::vector<std::vector<SimulationState>>& trajectories);
    
    bool isInitialized() const { return m_initialized; }
    
private:
    void updateTrailsFromSimulation();
    void synchronizeGPUData();
    
    // Components
    std::unique_ptr<CudaPhysicsSolver> m_physicsSolver;
    std::unique_ptr<CudaTrailRenderer> m_trailRenderer;
    
    // State
    bool m_initialized = false;
    SimulationState m_currentState;
    PendulumConfiguration m_config;
    
    // Trail configuration
    bool m_trailFadingEnabled = true;
    float m_trailFadeDuration = 5.0f;
    size_t m_maxTrailPoints = 50000;
    
    // Performance tracking
    mutable double m_lastPhysicsTime = 0.0;
    mutable double m_lastTrailTime = 0.0;
    mutable double m_lastTotalTime = 0.0;
    
    // CUDA synchronization
    cudaEvent_t m_startEvent = nullptr;
    cudaEvent_t m_stopEvent = nullptr;
};

} // namespace cuda
} // namespace pendulum

#endif // USE_CUDA
