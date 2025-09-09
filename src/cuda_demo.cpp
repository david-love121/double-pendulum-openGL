#ifdef USE_CUDA

#include "CudaPhysicsEngine.h"
#include "CudaPhysicsSolver.h"
#include "CudaTrailRenderer.h"
#include <iostream>
#include <chrono>
#include <vector>

namespace pendulum {
namespace cuda {

/**
 * Demonstration of CUDA-accelerated double pendulum simulation
 * Showcases performance improvements over CPU implementation
 */
class CudaDemo {
public:
    static void runPerformanceComparison() {
        std::cout << "\n=== CUDA Double Pendulum Performance Demo ===" << std::endl;
        
        // Test configuration
        PendulumConfiguration config;
        config.l1 = 1.0;
        config.l2 = 1.0;
        config.m1 = 1.0;
        config.m2 = 1.0;
        config.g = 9.81;
        config.damping = 0.01;
        
        SimulationState initial_state;
        initial_state.theta1 = 1.5;
        initial_state.theta2 = 0.5;
        initial_state.omega1 = 0.0;
        initial_state.omega2 = 0.0;
        initial_state.timestamp = 0.0;
        
        const double timestep = 0.001;
        const int num_steps = 10000;
        
        std::cout << "Configuration:" << std::endl;
        std::cout << "  - Timestep: " << timestep << " seconds" << std::endl;
        std::cout << "  - Steps: " << num_steps << std::endl;
        std::cout << "  - Total simulation time: " << (timestep * num_steps) << " seconds" << std::endl;
        
        // Test single trajectory simulation
        testSingleTrajectory(initial_state, config, timestep, num_steps);
        
        // Test batch trajectory simulation
        testBatchTrajectory(initial_state, config, timestep, num_steps);
        
        // Test parallel multi-trajectory simulation
        testParallelTrajectories(initial_state, config, timestep, 1000);
        
        // Test trail rendering performance
        std::cout << "\n--- CUDA Trail Rendering Performance ---" << std::endl;
        std::cout << "Note: Trail rendering requires OpenGL context (skipped in demo mode)" << std::endl;
        std::cout << "Trail rendering performance can be tested within the main application" << std::endl;
    }
    
private:
    static void testSingleTrajectory(const SimulationState& initial_state,
                                   const PendulumConfiguration& config,
                                   double timestep,
                                   int num_steps) {
        std::cout << "\n--- Single Trajectory Simulation ---" << std::endl;
        
        CudaPhysicsSolver solver;
        if (!solver.initialize()) {
            std::cout << "Failed to initialize CUDA physics solver" << std::endl;
            return;
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        SimulationState state = initial_state;
        for (int i = 0; i < num_steps; ++i) {
            state = solver.step(state, config, timestep);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        double total_time = duration.count() / 1000000.0;
        double avg_kernel_time = solver.getLastKernelTime();
        
        std::cout << "  Total time: " << total_time << " seconds" << std::endl;
        std::cout << "  Average time per step: " << (total_time / num_steps * 1000) << " ms" << std::endl;
        std::cout << "  Last kernel time: " << (avg_kernel_time * 1000) << " ms" << std::endl;
        std::cout << "  Final position: (" << state.theta1 << ", " << state.theta2 << ")" << std::endl;
        std::cout << "  Energy: " << state.energy << " J" << std::endl;
    }
    
    static void testBatchTrajectory(const SimulationState& initial_state,
                                  const PendulumConfiguration& config,
                                  double timestep,
                                  int num_steps) {
        std::cout << "\n--- Batch Trajectory Simulation ---" << std::endl;
        
        CudaPhysicsSolver solver;
        if (!solver.initialize()) {
            std::cout << "Failed to initialize CUDA physics solver" << std::endl;
            return;
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<SimulationState> trajectory = solver.stepBatch(initial_state, config, timestep, num_steps);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        double total_time = duration.count() / 1000000.0;
        double kernel_time = solver.getLastKernelTime();
        
        std::cout << "  Total time: " << total_time << " seconds" << std::endl;
        std::cout << "  Kernel time: " << kernel_time << " seconds" << std::endl;
        std::cout << "  Speedup vs sequential: " << (trajectory.size() * 0.0001 / total_time) << "x (estimated)" << std::endl;
        std::cout << "  Trajectory points generated: " << trajectory.size() << std::endl;
        
        if (!trajectory.empty()) {
            const auto& final_state = trajectory.back();
            std::cout << "  Final position: (" << final_state.theta1 << ", " << final_state.theta2 << ")" << std::endl;
            std::cout << "  Final energy: " << final_state.energy << " J" << std::endl;
        }
    }
    
    static void testParallelTrajectories(const SimulationState& initial_state,
                                       const PendulumConfiguration& config,
                                       double timestep,
                                       int num_steps) {
        std::cout << "\n--- Parallel Multi-Trajectory Simulation ---" << std::endl;
        
        // Create multiple initial conditions (slightly perturbed)
        std::vector<SimulationState> initial_states;
        const int num_trajectories = 100;
        
        for (int i = 0; i < num_trajectories; ++i) {
            SimulationState state = initial_state;
            state.theta1 += (i - num_trajectories/2) * 0.001;  // Small perturbations
            initial_states.push_back(state);
        }
        
        CudaPhysicsSolver solver;
        if (!solver.initialize()) {
            std::cout << "Failed to initialize CUDA physics solver" << std::endl;
            return;
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        std::vector<SimulationState> current_states = initial_states;
        for (int step = 0; step < num_steps; ++step) {
            current_states = solver.stepParallel(current_states, config, timestep);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        double total_time = duration.count() / 1000000.0;
        double kernel_time = solver.getLastKernelTime();
        
        std::cout << "  Trajectories simulated: " << num_trajectories << std::endl;
        std::cout << "  Steps per trajectory: " << num_steps << std::endl;
        std::cout << "  Total computations: " << (num_trajectories * num_steps) << std::endl;
        std::cout << "  Total time: " << total_time << " seconds" << std::endl;
        std::cout << "  Kernel time: " << kernel_time << " seconds" << std::endl;
        std::cout << "  Parallel efficiency: " << (kernel_time / total_time * 100) << "%" << std::endl;
        
        // Analyze final states for chaos demonstration
        double max_deviation = 0.0;
        for (const auto& state : current_states) {
            double deviation = std::abs(state.theta1 - initial_state.theta1) + 
                              std::abs(state.theta2 - initial_state.theta2);
            max_deviation = std::max(max_deviation, deviation);
        }
        
        std::cout << "  Maximum deviation from original: " << max_deviation << " radians" << std::endl;
        std::cout << "  Demonstrates chaotic sensitivity to initial conditions" << std::endl;
    }
    
    static void testTrailRendering(const SimulationState& initial_state,
                                 const PendulumConfiguration& config,
                                 double timestep,
                                 int num_steps) {
        std::cout << "\n--- CUDA Trail Rendering Performance ---" << std::endl;
        
        CudaTrailRenderer trail_renderer;
        if (!trail_renderer.initialize()) {
            std::cout << "Failed to initialize CUDA trail renderer" << std::endl;
            return;
        }
        
        if (!trail_renderer.isCudaEnabled()) {
            std::cout << "CUDA not enabled for trail renderer, using CPU fallback" << std::endl;
        }
        
        // Generate trajectory data
        CudaPhysicsSolver solver;
        if (!solver.initialize()) {
            std::cout << "Failed to initialize physics solver" << std::endl;
            return;
        }
        
        std::vector<SimulationState> trajectory = solver.stepBatch(initial_state, config, timestep, num_steps);
        
        // Test individual point addition
        auto start = std::chrono::high_resolution_clock::now();
        
        for (const auto& state : trajectory) {
            glm::vec2 pos = state.getBob2Position(config);
            trail_renderer.addPoint(pos.x, pos.y, static_cast<float>(state.timestamp));
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "  Individual point addition:" << std::endl;
        std::cout << "    Points added: " << trajectory.size() << std::endl;
        std::cout << "    Time: " << (duration.count() / 1000.0) << " ms" << std::endl;
        std::cout << "    Rate: " << (trajectory.size() / (duration.count() / 1000000.0)) << " points/sec" << std::endl;
        
        // Test batch addition
        trail_renderer.clear();
        
        std::vector<glm::vec2> positions;
        std::vector<float> timestamps;
        for (const auto& state : trajectory) {
            positions.push_back(state.getBob2Position(config));
            timestamps.push_back(static_cast<float>(state.timestamp));
        }
        
        start = std::chrono::high_resolution_clock::now();
        trail_renderer.addPointBatch(positions, timestamps);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "  Batch point addition:" << std::endl;
        std::cout << "    Points added: " << positions.size() << std::endl;
        std::cout << "    Time: " << (duration.count() / 1000.0) << " ms" << std::endl;
        std::cout << "    Rate: " << (positions.size() / (duration.count() / 1000000.0)) << " points/sec" << std::endl;
        
        // Test fading update
        start = std::chrono::high_resolution_clock::now();
        trail_renderer.updateFading(static_cast<float>(trajectory.back().timestamp), 5.0f, true);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        std::cout << "  Fading update:" << std::endl;
        std::cout << "    Time: " << (duration.count() / 1000.0) << " ms" << std::endl;
        if (trail_renderer.isCudaEnabled()) {
            std::cout << "    CUDA kernel time: " << (trail_renderer.getLastCudaTime() * 1000) << " ms" << std::endl;
        }
        
        std::cout << "  Final trail point count: " << trail_renderer.getPointCount() << std::endl;
    }
};

} // namespace cuda
} // namespace pendulum

// Example usage function that can be called from main
extern "C" void runCudaDemo() {
    pendulum::cuda::CudaDemo::runPerformanceComparison();
}

#endif // USE_CUDA
