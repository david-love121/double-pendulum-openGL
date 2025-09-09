#pragma once

#include <glm/glm.hpp>

namespace pendulum {

/**
 * Represents the physical parameters of a double pendulum system
 */
struct PendulumConfiguration {
    double l1 = 1.0;        // Length of first pendulum arm (meters)
    double l2 = 1.0;        // Length of second pendulum arm (meters)
    double m1 = 1.0;        // Mass of first pendulum bob (kg)
    double m2 = 1.0;        // Mass of second pendulum bob (kg)
    double g = 9.81;        // Gravitational acceleration (m/s²)
    double damping = 0.0;   // Air resistance coefficient

    // Validation
    bool isValid() const {
        return l1 > 0.0 && l2 > 0.0 && m1 > 0.0 && m2 > 0.0 && g > 0.0 && damping >= 0.0;
    }
};

/**
 * Represents the instantaneous state of a double pendulum during simulation
 */
struct SimulationState {
    double theta1 = 1.0;    // Angle of first pendulum from vertical (radians)
    double theta2 = 0.0;    // Angle of second pendulum from vertical (radians)
    double omega1 = 0.0;    // Angular velocity of first pendulum (rad/s)
    double omega2 = 0.0;    // Angular velocity of second pendulum (rad/s)
    double timestamp = 0.0; // Simulation time (seconds)
    double energy = 0.0;    // Total system energy (computed, cached)

    // Normalize angles to [-π, π] range
    void normalizeAngles();
    
    // Compute system energy given configuration
    double computeEnergy(const PendulumConfiguration& config) const;
    
    // Get positions of pendulum bobs in world coordinates
    glm::vec2 getBob1Position(const PendulumConfiguration& config) const;
    glm::vec2 getBob2Position(const PendulumConfiguration& config) const;
};

enum class SimulationStatus {
    INITIAL,
    RUNNING,
    PAUSED,
    ERROR
};

} // namespace pendulum
