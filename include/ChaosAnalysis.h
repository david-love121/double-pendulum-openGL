#pragma once

#include <vector>
#include <functional>
#include <glm/glm.hpp>

namespace pendulum {

// Forward declarations
struct PendulumConfiguration;

/**
 * Represents one point in the chaos analysis visualization
 */
struct ChaosAnalysisPoint {
    double initial_theta1;     // Initial angle of first pendulum
    double initial_theta2;     // Initial angle of second pendulum
    double lyapunov_exponent;  // Computed Lyapunov exponent
    glm::vec3 color;          // RGB color [0,1] based on chaos characteristics
    
    enum Status {
        PENDING,
        COMPUTING,
        COMPLETE,
        ERROR
    } computation_status = PENDING;
    
    bool isValid() const {
        return initial_theta1 >= -M_PI && initial_theta1 <= M_PI &&
               initial_theta2 >= -M_PI && initial_theta2 <= M_PI &&
               std::isfinite(lyapunov_exponent);
    }
};

/**
 * Collection of analysis points forming the chaos visualization
 */
class ChaosAnalysisGrid {
public:
    ChaosAnalysisGrid(int resolution_x, int resolution_y,
                     double theta1_min, double theta1_max,
                     double theta2_min, double theta2_max);

    // Getters
    int getResolutionX() const { return m_resolutionX; }
    int getResolutionY() const { return m_resolutionY; }
    double getTheta1Min() const { return m_theta1Min; }
    double getTheta1Max() const { return m_theta1Max; }
    double getTheta2Min() const { return m_theta2Min; }
    double getTheta2Max() const { return m_theta2Max; }
    double getComputationProgress() const { return m_computationProgress; }
    
    // Access points
    ChaosAnalysisPoint& getPoint(int x, int y);
    const ChaosAnalysisPoint& getPoint(int x, int y) const;
    
    // Get all points as contiguous array for GPU processing
    std::vector<ChaosAnalysisPoint>& getPoints() { return m_points; }
    const std::vector<ChaosAnalysisPoint>& getPoints() const { return m_points; }
    
    // Update computation progress
    void updateProgress(double progress) { m_computationProgress = progress; }
    
    // Initialize grid with initial conditions
    void initialize();
    
    // Validate grid parameters
    bool isValid() const;

private:
    int m_resolutionX, m_resolutionY;
    double m_theta1Min, m_theta1Max;
    double m_theta2Min, m_theta2Max;
    std::vector<ChaosAnalysisPoint> m_points;
    double m_computationProgress = 0.0;
};

/**
 * Compute chaos analysis for the given grid
 */
void computeChaosAnalysis(ChaosAnalysisGrid* grid, const PendulumConfiguration& config,
                         double integration_time, int color_scheme,
                         std::function<void(double)> progress_callback = nullptr);

} // namespace pendulum
