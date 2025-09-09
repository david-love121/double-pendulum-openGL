#ifdef USE_CUDA

#include "CudaPhysicsEngine.h"
#include "CudaTrailRenderer.h"
#include <iostream>
#include <chrono>

namespace pendulum {
namespace cuda {

CudaPhysicsEngine::CudaPhysicsEngine() {
    // Initialize default configuration
    m_config.l1 = 1.0;
    m_config.l2 = 1.0;
    m_config.m1 = 1.0;
    m_config.m2 = 1.0;
    m_config.g = 9.81;
    m_config.damping = 0.0;
    
    // Initialize default state
    m_currentState.theta1 = 1.0;
    m_currentState.theta2 = 0.0;
    m_currentState.omega1 = 0.0;
    m_currentState.omega2 = 0.0;
    m_currentState.timestamp = 0.0;
    m_currentState.energy = 0.0;
}

CudaPhysicsEngine::~CudaPhysicsEngine() {
    cleanup();
}

bool CudaPhysicsEngine::initialize(int device_id) {
    if (m_initialized) {
        return true;
    }
    
    // Create CUDA events for timing
    cudaError_t err = cudaEventCreate(&m_startEvent);
    if (err != cudaSuccess) {
        std::cerr << "Failed to create CUDA start event: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    err = cudaEventCreate(&m_stopEvent);
    if (err != cudaSuccess) {
        std::cerr << "Failed to create CUDA stop event: " << cudaGetErrorString(err) << std::endl;
        cleanup();
        return false;
    }
    
    // Initialize physics solver
    m_physicsSolver = std::make_unique<CudaPhysicsSolver>();
    if (!m_physicsSolver->initialize(device_id)) {
        std::cerr << "Failed to initialize CUDA physics solver" << std::endl;
        cleanup();
        return false;
    }
    
    // Initialize trail renderer
    m_trailRenderer = std::make_unique<CudaTrailRenderer>();
    if (!m_trailRenderer->initialize()) {
        std::cerr << "Failed to initialize CUDA trail renderer" << std::endl;
        cleanup();
        return false;
    }
    
    // Configure trail renderer
    m_trailRenderer->setMaxPoints(m_maxTrailPoints);
    m_trailRenderer->setColor(1.0f, 1.0f, 1.0f);  // Default white trails
    
    m_initialized = true;
    std::cout << "CUDA Physics Engine initialized successfully" << std::endl;
    
    return true;
}

void CudaPhysicsEngine::cleanup() {
    m_physicsSolver.reset();
    m_trailRenderer.reset();
    
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

void CudaPhysicsEngine::setConfiguration(const PendulumConfiguration& config) {
    if (!config.isValid()) {
        throw std::invalid_argument("Invalid pendulum configuration");
    }
    
    m_config = config;
    
    // Update energy for current state
    m_currentState.energy = m_currentState.computeEnergy(m_config);
}

void CudaPhysicsEngine::setState(const SimulationState& state) {
    m_currentState = state;
    m_currentState.energy = m_currentState.computeEnergy(m_config);
}

void CudaPhysicsEngine::step(double timestep) {
    if (!m_initialized) {
        throw std::runtime_error("CUDA Physics Engine not initialized");
    }
    
    // Record total timing start
    cudaEventRecord(m_startEvent);
    
    // Perform physics simulation
    auto physics_start = std::chrono::high_resolution_clock::now();
    m_currentState = m_physicsSolver->step(m_currentState, m_config, timestep);
    m_lastPhysicsTime = m_physicsSolver->getLastKernelTime();
    
    // Update trails with new position
    updateTrailsFromSimulation();
    
    // Record total timing end
    cudaEventRecord(m_stopEvent);
    cudaEventSynchronize(m_stopEvent);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, m_startEvent, m_stopEvent);
    m_lastTotalTime = milliseconds / 1000.0;
}

void CudaPhysicsEngine::stepBatch(double timestep, int num_steps) {
    if (!m_initialized) {
        throw std::runtime_error("CUDA Physics Engine not initialized");
    }
    
    // Record total timing start
    cudaEventRecord(m_startEvent);
    
    // Perform batch physics simulation
    std::vector<SimulationState> trajectory = m_physicsSolver->stepBatch(
        m_currentState, m_config, timestep, num_steps);
    
    m_lastPhysicsTime = m_physicsSolver->getLastKernelTime();
    
    // Update current state to the final state
    if (!trajectory.empty()) {
        m_currentState = trajectory.back();
        
        // Add all trajectory points to trails
        std::vector<glm::vec2> positions;
        std::vector<float> timestamps;
        
        positions.reserve(trajectory.size());
        timestamps.reserve(trajectory.size());
        
        for (const auto& state : trajectory) {
            glm::vec2 bob2_pos = state.getBob2Position(m_config);
            positions.push_back(bob2_pos);
            timestamps.push_back(static_cast<float>(state.timestamp));
        }
        
        // Use batch add for efficiency
        if (m_trailRenderer->isCudaEnabled()) {
            m_trailRenderer->addPointBatch(positions, timestamps);
        } else {
            // Fallback to individual additions
            for (size_t i = 0; i < positions.size(); ++i) {
                m_trailRenderer->addPoint(positions[i].x, positions[i].y, timestamps[i]);
            }
        }
        
        m_lastTrailTime = m_trailRenderer->getLastCudaTime();
    }
    
    // Record total timing end
    cudaEventRecord(m_stopEvent);
    cudaEventSynchronize(m_stopEvent);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, m_startEvent, m_stopEvent);
    m_lastTotalTime = milliseconds / 1000.0;
}

void CudaPhysicsEngine::updateTrailsFromSimulation() {
    // Get bob 2 position (end of second pendulum)
    glm::vec2 bob2_pos = m_currentState.getBob2Position(m_config);
    
    // Add point to trail
    auto trail_start = std::chrono::high_resolution_clock::now();
    m_trailRenderer->addPoint(bob2_pos.x, bob2_pos.y, 
                             static_cast<float>(m_currentState.timestamp));
    
    // Update trail fading
    if (m_trailFadingEnabled) {
        m_trailRenderer->updateFading(static_cast<float>(m_currentState.timestamp),
                                     m_trailFadeDuration, true);
    }
    
    m_lastTrailTime = m_trailRenderer->getLastCudaTime();
}

void CudaPhysicsEngine::setTrailColor(float r, float g, float b) {
    if (m_trailRenderer) {
        m_trailRenderer->setColor(r, g, b);
    }
}

void CudaPhysicsEngine::setTrailFading(bool enabled, float duration) {
    m_trailFadingEnabled = enabled;
    m_trailFadeDuration = duration;
}

void CudaPhysicsEngine::setTrailMaxPoints(size_t max_points) {
    m_maxTrailPoints = max_points;
    if (m_trailRenderer) {
        m_trailRenderer->setMaxPoints(max_points);
    }
}

void CudaPhysicsEngine::clearTrails() {
    if (m_trailRenderer) {
        m_trailRenderer->clear();
    }
}

void CudaPhysicsEngine::renderTrails(const glm::mat4& view, const glm::mat4& projection) {
    if (m_trailRenderer) {
        m_trailRenderer->render(view, projection);
    }
}

void CudaPhysicsEngine::simulateMultiple(const std::vector<SimulationState>& initial_states,
                                        double timestep,
                                        int num_steps,
                                        std::vector<std::vector<SimulationState>>& trajectories) {
    if (!m_initialized) {
        throw std::runtime_error("CUDA Physics Engine not initialized");
    }
    
    trajectories.clear();
    trajectories.resize(initial_states.size());
    
    // Record timing start
    cudaEventRecord(m_startEvent);
    
    // Simulate all initial conditions in parallel
    std::vector<SimulationState> current_states = initial_states;
    
    for (int step = 0; step < num_steps; ++step) {
        // Parallel step for all states
        current_states = m_physicsSolver->stepParallel(current_states, m_config, timestep);
        
        // Store intermediate states
        for (size_t i = 0; i < current_states.size(); ++i) {
            trajectories[i].push_back(current_states[i]);
        }
    }
    
    // Record timing end
    cudaEventRecord(m_stopEvent);
    cudaEventSynchronize(m_stopEvent);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, m_startEvent, m_stopEvent);
    m_lastTotalTime = milliseconds / 1000.0;
    m_lastPhysicsTime = m_physicsSolver->getLastKernelTime();
}

double CudaPhysicsEngine::getPhysicsTime() const {
    return m_lastPhysicsTime;
}

double CudaPhysicsEngine::getTrailTime() const {
    return m_lastTrailTime;
}

double CudaPhysicsEngine::getTotalTime() const {
    return m_lastTotalTime;
}

} // namespace cuda
} // namespace pendulum

#endif // USE_CUDA
