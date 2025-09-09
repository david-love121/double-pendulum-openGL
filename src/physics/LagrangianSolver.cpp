#include "LagrangianSolver.h"
#include <cmath>
#include <stdexcept>

namespace pendulum {

LagrangianSolver::LagrangianSolver() {
    // Constructor
}

LagrangianSolver::~LagrangianSolver() {
    // Destructor
}

SimulationState LagrangianSolver::step(const SimulationState& state, 
                                     const PendulumConfiguration& config, 
                                     double timestep) {
    if (timestep <= 0.0 || timestep > 0.01) {
        throw std::invalid_argument("Invalid timestep");
    }
    
    if (!config.isValid()) {
        throw std::invalid_argument("Invalid pendulum configuration");
    }
    
    // RK4 integration
    StateDerivative k1 = computeDerivative(state, config);
    
    SimulationState state2 = state;
    state2.theta1 += k1.dtheta1 * timestep * 0.5;
    state2.theta2 += k1.dtheta2 * timestep * 0.5;
    state2.omega1 += k1.domega1 * timestep * 0.5;
    state2.omega2 += k1.domega2 * timestep * 0.5;
    StateDerivative k2 = computeDerivative(state2, config);
    
    SimulationState state3 = state;
    state3.theta1 += k2.dtheta1 * timestep * 0.5;
    state3.theta2 += k2.dtheta2 * timestep * 0.5;
    state3.omega1 += k2.domega1 * timestep * 0.5;
    state3.omega2 += k2.domega2 * timestep * 0.5;
    StateDerivative k3 = computeDerivative(state3, config);
    
    SimulationState state4 = state;
    state4.theta1 += k3.dtheta1 * timestep;
    state4.theta2 += k3.dtheta2 * timestep;
    state4.omega1 += k3.domega1 * timestep;
    state4.omega2 += k3.domega2 * timestep;
    StateDerivative k4 = computeDerivative(state4, config);
    
    SimulationState newState = state;
    newState.theta1 += (k1.dtheta1 + 2*k2.dtheta1 + 2*k3.dtheta1 + k4.dtheta1) * timestep / 6.0;
    newState.theta2 += (k1.dtheta2 + 2*k2.dtheta2 + 2*k3.dtheta2 + k4.dtheta2) * timestep / 6.0;
    newState.omega1 += (k1.domega1 + 2*k2.domega1 + 2*k3.domega1 + k4.domega1) * timestep / 6.0;
    newState.omega2 += (k1.domega2 + 2*k2.domega2 + 2*k3.domega2 + k4.domega2) * timestep / 6.0;
    newState.timestamp += timestep;
    
    // Normalize angles
    newState.normalizeAngles();
    
    // Update energy
    newState.energy = newState.computeEnergy(config);
    
    // Check numerical stability
    if (!std::isfinite(newState.theta1) || !std::isfinite(newState.theta2) ||
        !std::isfinite(newState.omega1) || !std::isfinite(newState.omega2)) {
        m_isStable = false;
        throw std::runtime_error("Numerical instability detected");
    }
    
    return newState;
}

void LagrangianSolver::reset() {
    m_isStable = true;
}

bool LagrangianSolver::isStable() const {
    return m_isStable;
}

LagrangianSolver::StateDerivative LagrangianSolver::computeDerivative(
    const SimulationState& state, const PendulumConfiguration& config) const {
    
    StateDerivative deriv;
    
    // Angular velocities are derivatives of angles
    deriv.dtheta1 = state.omega1;
    deriv.dtheta2 = state.omega2;
    
    // Compute angular accelerations using Lagrangian mechanics
    deriv.domega1 = computeAlpha1(state, config);
    deriv.domega2 = computeAlpha2(state, config);
    
    return deriv;
}

double LagrangianSolver::computeAlpha1(const SimulationState& state, 
                                     const PendulumConfiguration& config) const {
    double theta1 = state.theta1, theta2 = state.theta2;
    double omega1 = state.omega1, omega2 = state.omega2;
    double l1 = config.l1, l2 = config.l2;
    double m1 = config.m1, m2 = config.m2;
    double g = config.g;
    
    double delta = theta2 - theta1;
    double cos_delta = cos(delta);
    double sin_delta = sin(delta);
    
    double den1 = (m1 + m2) * l1 - m2 * l1 * cos_delta * cos_delta;
    
    double num1 = -m2 * l1 * omega1 * omega1 * sin_delta * cos_delta +
                  m2 * g * sin(theta2) * cos_delta +
                  m2 * l2 * omega2 * omega2 * sin_delta -
                  (m1 + m2) * g * sin(theta1);
    
    // Add damping term
    num1 -= config.damping * omega1;
    
    return num1 / den1;
}

double LagrangianSolver::computeAlpha2(const SimulationState& state, 
                                     const PendulumConfiguration& config) const {
    double theta1 = state.theta1, theta2 = state.theta2;
    double omega1 = state.omega1, omega2 = state.omega2;
    double l1 = config.l1, l2 = config.l2;
    double m1 = config.m1, m2 = config.m2;
    double g = config.g;
    
    double delta = theta2 - theta1;
    double cos_delta = cos(delta);
    double sin_delta = sin(delta);
    
    double den2 = l2 / l1 * ((m1 + m2) * l1 - m2 * l1 * cos_delta * cos_delta);
    
    double num2 = -m2 * l2 * omega2 * omega2 * sin_delta * cos_delta +
                  (m1 + m2) * g * sin(theta1) * cos_delta +
                  (m1 + m2) * l1 * omega1 * omega1 * sin_delta -
                  (m1 + m2) * g * sin(theta2);
    
    // Add damping term
    num2 -= config.damping * omega2;
    
    return num2 / den2;
}

} // namespace pendulum
