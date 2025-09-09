#include "ChaosAnalysis.h"
#include "LagrangianSolver.h"
#include "PendulumState.h"
#include <iostream>
#include <cmath>
#include <algorithm>

#ifdef USE_CUDA
#include "CudaChaosAnalysis.h"
#endif

namespace pendulum {

ChaosAnalysisGrid::ChaosAnalysisGrid(int resolution_x, int resolution_y,
                                   double theta1_min, double theta1_max,
                                   double theta2_min, double theta2_max)
    : m_resolutionX(resolution_x), m_resolutionY(resolution_y),
      m_theta1Min(theta1_min), m_theta1Max(theta1_max),
      m_theta2Min(theta2_min), m_theta2Max(theta2_max) {
    
    m_points.resize(resolution_x * resolution_y);
    initialize();
}

ChaosAnalysisPoint& ChaosAnalysisGrid::getPoint(int x, int y) {
    return m_points[y * m_resolutionX + x];
}

const ChaosAnalysisPoint& ChaosAnalysisGrid::getPoint(int x, int y) const {
    return m_points[y * m_resolutionX + x];
}

void ChaosAnalysisGrid::initialize() {
    std::cout << "Initializing chaos analysis grid: " << m_resolutionX << "x" << m_resolutionY << std::endl;
    
    double dx = (m_theta1Max - m_theta1Min) / (m_resolutionX - 1);
    double dy = (m_theta2Max - m_theta2Min) / (m_resolutionY - 1);
    
    for (int y = 0; y < m_resolutionY; ++y) {
        for (int x = 0; x < m_resolutionX; ++x) {
            ChaosAnalysisPoint& point = getPoint(x, y);
            point.initial_theta1 = m_theta1Min + x * dx;
            point.initial_theta2 = m_theta2Min + y * dy;
            point.lyapunov_exponent = 0.0;
            point.color = glm::vec3(0.0, 0.0, 0.0);
            point.computation_status = ChaosAnalysisPoint::PENDING;
        }
    }
}

bool ChaosAnalysisGrid::isValid() const {
    return m_resolutionX > 0 && m_resolutionY > 0 &&
           m_theta1Min < m_theta1Max && m_theta2Min < m_theta2Max;
}

// Helper class for computing Lyapunov exponents
class LyapunovCalculator {
public:
    static double computeLyapunovExponent(double theta1_0, double theta2_0, 
                                        const PendulumConfiguration& config,
                                        double integration_time = 5.0) {
        // Small perturbation for derivative calculation
        const double eps = 1e-8;
        
        // Initial conditions
        SimulationState state1, state2;
        state1.theta1 = theta1_0;
        state1.theta2 = theta2_0;
        state1.omega1 = 0.0;
        state1.omega2 = 0.0;
        state1.timestamp = 0.0;
        
        // Perturbed initial conditions
        state2 = state1;
        state2.theta1 += eps;
        
        LagrangianSolver solver;
        double dt = 0.01;
        int steps = static_cast<int>(integration_time / dt);
        
        double lyapunov_sum = 0.0;
        int count = 0;
        
        for (int i = 0; i < steps; ++i) {
            // Evolve both trajectories
            try {
                state1 = solver.step(state1, config, dt);
                state2 = solver.step(state2, config, dt);
                
                // Calculate separation
                double dx1 = state2.theta1 - state1.theta1;
                double dx2 = state2.theta2 - state1.theta2;
                double dx3 = state2.omega1 - state1.omega1;
                double dx4 = state2.omega2 - state1.omega2;
                
                double separation = sqrt(dx1*dx1 + dx2*dx2 + dx3*dx3 + dx4*dx4);
                
                if (separation > 0 && separation < 1e10) {
                    lyapunov_sum += log(separation / eps);
                    count++;
                    
                    // Renormalize to prevent overflow
                    if (separation > 1e-6) {
                        double scale = eps / separation;
                        state2.theta1 = state1.theta1 + dx1 * scale;
                        state2.theta2 = state1.theta2 + dx2 * scale;
                        state2.omega1 = state1.omega1 + dx3 * scale;
                        state2.omega2 = state1.omega2 + dx4 * scale;
                    }
                }
            } catch (const std::exception& e) {
                // Trajectory became unstable, return high positive value
                return 10.0;
            }
        }
        
        return count > 0 ? lyapunov_sum / (count * dt) : 0.0;
    }
};

// Color mapping functions
namespace {
    glm::vec3 mapToColor(double lyapunov, int color_scheme) {
        // Normalize lyapunov exponent to [0,1] range
        // Typical range is roughly [-2, 2] for double pendulum
        double normalized = std::max(0.0, std::min(1.0, (lyapunov + 2.0) / 4.0));
        
        switch (color_scheme) {
            case 0: // Blue-Red
                return glm::vec3(normalized, 0.0, 1.0 - normalized);
            
            case 1: // Viridis-like
                if (normalized < 0.25) {
                    double t = normalized * 4.0;
                    return glm::vec3(0.267 * t, 0.005 + 0.352 * t, 0.329 + 0.315 * t);
                } else if (normalized < 0.5) {
                    double t = (normalized - 0.25) * 4.0;
                    return glm::vec3(0.267 + 0.126 * t, 0.357 + 0.369 * t, 0.644 - 0.041 * t);
                } else if (normalized < 0.75) {
                    double t = (normalized - 0.5) * 4.0;
                    return glm::vec3(0.393 + 0.349 * t, 0.726 + 0.132 * t, 0.603 - 0.313 * t);
                } else {
                    double t = (normalized - 0.75) * 4.0;
                    return glm::vec3(0.742 + 0.191 * t, 0.858 + 0.095 * t, 0.290 + 0.710 * t);
                }
            
            case 2: // Plasma-like
                if (normalized < 0.33) {
                    double t = normalized * 3.0;
                    return glm::vec3(0.050 + 0.485 * t, 0.030 + 0.147 * t, 0.527 + 0.134 * t);
                } else if (normalized < 0.66) {
                    double t = (normalized - 0.33) * 3.0;
                    return glm::vec3(0.535 + 0.309 * t, 0.177 + 0.407 * t, 0.661 - 0.229 * t);
                } else {
                    double t = (normalized - 0.66) * 3.0;
                    return glm::vec3(0.844 + 0.130 * t, 0.584 + 0.378 * t, 0.432 - 0.412 * t);
                }
            
            default:
                return glm::vec3(normalized, normalized, normalized);
        }
    }
}

// Analysis computation function
void computeChaosAnalysis(ChaosAnalysisGrid* grid, const PendulumConfiguration& config,
                         double integration_time, int color_scheme,
                         std::function<void(double)> progress_callback) {
    
    if (!grid || !grid->isValid()) {
        std::cerr << "Invalid chaos analysis grid" << std::endl;
        return;
    }
    
    std::cout << "Starting chaos analysis computation..." << std::endl;
    
#ifdef USE_CUDA
    // Try CUDA acceleration first
    if (cuda::isCudaAvailable()) {
        std::cout << "CUDA is available, using GPU acceleration..." << std::endl;
        
        bool cuda_success = cuda::computeChaosAnalysisGPU(grid, config, integration_time, color_scheme, 0);
        
        if (cuda_success) {
            std::cout << "CUDA chaos analysis completed successfully!" << std::endl;
            grid->updateProgress(1.0);
            if (progress_callback) {
                progress_callback(1.0);
            }
            return;
        } else {
            std::cout << "CUDA computation failed, falling back to CPU..." << std::endl;
        }
    } else {
        std::cout << "CUDA not available, using CPU computation..." << std::endl;
    }
#else
    std::cout << "CUDA support not compiled, using CPU computation..." << std::endl;
#endif
    
    // CPU fallback implementation
    std::cout << "Using CPU for chaos analysis computation..." << std::endl;
    std::cout << "Grid size: " << grid->getResolutionX() << "x" << grid->getResolutionY() << std::endl;
    std::cout << "Integration time: " << integration_time << "s" << std::endl;
    
    int total_points = grid->getResolutionX() * grid->getResolutionY();
    int completed = 0;
    
    for (int y = 0; y < grid->getResolutionY(); ++y) {
        for (int x = 0; x < grid->getResolutionX(); ++x) {
            ChaosAnalysisPoint& point = grid->getPoint(x, y);
            
            point.computation_status = ChaosAnalysisPoint::COMPUTING;
            
            // Compute Lyapunov exponent
            try {
                point.lyapunov_exponent = LyapunovCalculator::computeLyapunovExponent(
                    point.initial_theta1, point.initial_theta2, config, integration_time);
                
                // Map to color
                point.color = mapToColor(point.lyapunov_exponent, color_scheme);
                
                point.computation_status = ChaosAnalysisPoint::COMPLETE;
                
            } catch (const std::exception& e) {
                std::cerr << "Error computing point (" << x << "," << y << "): " << e.what() << std::endl;
                point.computation_status = ChaosAnalysisPoint::ERROR;
                point.color = glm::vec3(0.5, 0.5, 0.5); // Gray for errors
            }
            
            completed++;
            if (progress_callback && completed % 10 == 0) {
                double progress = static_cast<double>(completed) / total_points;
                progress_callback(progress);
            }
        }
    }
    
    std::cout << "Chaos analysis computation completed!" << std::endl;
}

} // namespace pendulum
