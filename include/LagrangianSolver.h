#pragma once

#include "PendulumState.h"

namespace pendulum {

/**
 * Lagrangian mechanics solver for double pendulum system
 */
class LagrangianSolver {
public:
    LagrangianSolver();
    ~LagrangianSolver();

    /**
     * Advance simulation by one timestep using RK4 integration
     * @param state Current simulation state
     * @param config Pendulum configuration
     * @param timestep Integration step size (seconds)
     * @return Updated simulation state
     */
    SimulationState step(const SimulationState& state, 
                        const PendulumConfiguration& config, 
                        double timestep);

    /**
     * Reset solver to initial conditions
     */
    void reset();

    /**
     * Check if solver is in stable numerical state
     */
    bool isStable() const;

private:
    // RK4 integration helper functions
    struct StateDerivative {
        double dtheta1, dtheta2, domega1, domega2;
    };

    StateDerivative computeDerivative(const SimulationState& state, 
                                    const PendulumConfiguration& config) const;
    
    // Lagrangian mechanics equations
    double computeAlpha1(const SimulationState& state, const PendulumConfiguration& config) const;
    double computeAlpha2(const SimulationState& state, const PendulumConfiguration& config) const;

    bool m_isStable = true;
    double m_energyTolerance = 0.001; // 0.1% energy conservation tolerance
};

} // namespace pendulum
