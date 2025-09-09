#include "PendulumState.h"
#include <cmath>

namespace pendulum {

void SimulationState::normalizeAngles() {
    auto normalize = [](double& angle) {
        while (angle > M_PI) angle -= 2.0 * M_PI;
        while (angle < -M_PI) angle += 2.0 * M_PI;
    };
    
    normalize(theta1);
    normalize(theta2);
}

double SimulationState::computeEnergy(const PendulumConfiguration& config) const {
    // Kinetic energy
    double v1_sq = config.l1 * config.l1 * omega1 * omega1;
    double v2_sq = config.l2 * config.l2 * omega2 * omega2 + 
                   config.l1 * config.l1 * omega1 * omega1 + 
                   2.0 * config.l1 * config.l2 * omega1 * omega2 * cos(theta1 - theta2);
    
    double kinetic = 0.5 * config.m1 * v1_sq + 0.5 * config.m2 * v2_sq;
    
    // Potential energy (taking lowest point as zero reference)
    double h1 = -config.l1 * cos(theta1);
    double h2 = -config.l1 * cos(theta1) - config.l2 * cos(theta2);
    double potential = config.m1 * config.g * h1 + config.m2 * config.g * h2;
    
    return kinetic + potential;
}

glm::vec2 SimulationState::getBob1Position(const PendulumConfiguration& config) const {
    return glm::vec2(config.l1 * sin(theta1), -config.l1 * cos(theta1));
}

glm::vec2 SimulationState::getBob2Position(const PendulumConfiguration& config) const {
    glm::vec2 bob1_pos = getBob1Position(config);
    return bob1_pos + glm::vec2(config.l2 * sin(theta2), -config.l2 * cos(theta2));
}

} // namespace pendulum
