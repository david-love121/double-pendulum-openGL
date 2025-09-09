#include "CudaChaosAnalysis.h"
#include "PendulumState.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>

namespace pendulum {
namespace cuda {

// Device function for RK4 integration step
__device__ void rk4_step(double& theta1, double& theta2, double& omega1, double& omega2,
                        double l1, double l2, double m1, double m2, double g, double damping, double dt) {
    // Compute derivatives at current state
    auto compute_derivatives = [=] __device__ (double th1, double th2, double om1, double om2,
                                              double& dth1, double& dth2, double& dom1, double& dom2) {
        dth1 = om1;
        dth2 = om2;
        
        double delta = th2 - th1;
        double den1 = (m1 + m2) * l1 - m2 * l1 * cos(delta) * cos(delta);
        double den2 = (l2 / l1) * den1;
        
        double num1 = -m2 * l1 * om1 * om1 * sin(delta) * cos(delta) +
                      m2 * g * sin(th2) * cos(delta) +
                      m2 * l2 * om2 * om2 * sin(delta) -
                      (m1 + m2) * g * sin(th1) -
                      damping * om1;
        
        double num2 = -m2 * l2 * om2 * om2 * sin(delta) * cos(delta) +
                      (m1 + m2) * g * sin(th1) * cos(delta) +
                      (m1 + m2) * l1 * om1 * om1 * sin(delta) -
                      (m1 + m2) * g * sin(th2) -
                      damping * om2;
        
        dom1 = num1 / den1;
        dom2 = num2 / den2;
    };
    
    // RK4 integration
    double k1_th1, k1_th2, k1_om1, k1_om2;
    double k2_th1, k2_th2, k2_om1, k2_om2;
    double k3_th1, k3_th2, k3_om1, k3_om2;
    double k4_th1, k4_th2, k4_om1, k4_om2;
    
    // k1
    compute_derivatives(theta1, theta2, omega1, omega2, k1_th1, k1_th2, k1_om1, k1_om2);
    
    // k2
    compute_derivatives(theta1 + dt*k1_th1/2, theta2 + dt*k1_th2/2, 
                       omega1 + dt*k1_om1/2, omega2 + dt*k1_om2/2,
                       k2_th1, k2_th2, k2_om1, k2_om2);
    
    // k3
    compute_derivatives(theta1 + dt*k2_th1/2, theta2 + dt*k2_th2/2,
                       omega1 + dt*k2_om1/2, omega2 + dt*k2_om2/2,
                       k3_th1, k3_th2, k3_om1, k3_om2);
    
    // k4
    compute_derivatives(theta1 + dt*k3_th1, theta2 + dt*k3_th2,
                       omega1 + dt*k3_om1, omega2 + dt*k3_om2,
                       k4_th1, k4_th2, k4_om1, k4_om2);
    
    // Update state
    theta1 += dt * (k1_th1 + 2*k2_th1 + 2*k3_th1 + k4_th1) / 6.0;
    theta2 += dt * (k1_th2 + 2*k2_th2 + 2*k3_th2 + k4_th2) / 6.0;
    omega1 += dt * (k1_om1 + 2*k2_om1 + 2*k3_om1 + k4_om1) / 6.0;
    omega2 += dt * (k1_om2 + 2*k2_om2 + 2*k3_om2 + k4_om2) / 6.0;
}

// Device function to compute Lyapunov exponent
__device__ double compute_lyapunov_exponent(double initial_theta1, double initial_theta2,
                                           double l1, double l2, double m1, double m2, 
                                           double g, double damping, double integration_time) {
    const double dt = 0.001;  // Integration timestep
    const int num_steps = (int)(integration_time / dt);
    const double epsilon = 1e-8;  // Small perturbation for Lyapunov calculation
    
    // Main trajectory
    double theta1 = initial_theta1;
    double theta2 = initial_theta2;
    double omega1 = 0.0;
    double omega2 = 0.0;
    
    // Perturbed trajectory
    double theta1_p = initial_theta1 + epsilon;
    double theta2_p = initial_theta2;
    double omega1_p = 0.0;
    double omega2_p = 0.0;
    
    double sum_log = 0.0;
    int count = 0;
    
    for (int i = 0; i < num_steps; i++) {
        // Integrate main trajectory
        rk4_step(theta1, theta2, omega1, omega2, l1, l2, m1, m2, g, damping, dt);
        
        // Integrate perturbed trajectory
        rk4_step(theta1_p, theta2_p, omega1_p, omega2_p, l1, l2, m1, m2, g, damping, dt);
        
        // Calculate separation every 100 steps to avoid numerical issues
        if (i % 100 == 0 && i > 0) {
            double dx1 = theta1_p - theta1;
            double dx2 = theta2_p - theta2;
            double dv1 = omega1_p - omega1;
            double dv2 = omega2_p - omega2;
            
            double separation = sqrt(dx1*dx1 + dx2*dx2 + dv1*dv1 + dv2*dv2);
            
            if (separation > 1e-12 && separation < 1e12) {
                sum_log += log(separation / epsilon);
                count++;
                
                // Renormalize perturbation
                double norm_factor = epsilon / separation;
                theta1_p = theta1 + dx1 * norm_factor;
                theta2_p = theta2 + dx2 * norm_factor;
                omega1_p = omega1 + dv1 * norm_factor;
                omega2_p = omega2 + dv2 * norm_factor;
            }
        }
    }
    
    return count > 0 ? sum_log / (count * dt * 100) : 0.0;
}

// Device function to map Lyapunov exponent to color
__device__ void map_to_color(double lyapunov, int color_scheme, float* r, float* g_out, float* b) {
    // Normalize Lyapunov exponent to [0, 1] range
    double normalized = tanh(lyapunov / 2.0) * 0.5 + 0.5;
    normalized = fmax(0.0, fmin(1.0, normalized));
    
    switch (color_scheme) {
        case 0: // Blue-Red
            *r = (float)normalized;
            *g_out = 0.0f;
            *b = (float)(1.0 - normalized);
            break;
            
        case 1: // Viridis-like
            {
                double t = normalized;
                *r = (float)(0.267004 + t * (0.004874 + t * (0.991834 - 0.267004)));
                *g_out = (float)(0.004874 + t * (0.622550 + t * (0.991834 - 0.004874)));
                *b = (float)(0.329415 + t * (0.883019 + t * (0.644875 - 0.329415)));
            }
            break;
            
        case 2: // Plasma-like
            {
                double t = normalized;
                *r = (float)(0.050383 + t * (0.895498 + t * (0.987053 - 0.050383)));
                *g_out = (float)(0.029803 + t * (0.020330 + t * (0.987053 - 0.029803)));
                *b = (float)(0.527975 + t * (0.421549 + t * (0.987053 - 0.527975)));
            }
            break;
            
        default:
            *r = *g_out = *b = (float)normalized;
            break;
    }
}

// CUDA kernel for chaos analysis computation
__global__ void chaos_analysis_kernel(
    double* initial_theta1_array,
    double* initial_theta2_array,
    double* lyapunov_array,
    float* color_array,
    int* status_array,
    int grid_size,
    double l1, double l2, double m1, double m2, double g, double damping,
    double integration_time, int color_scheme) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= grid_size) return;
    
    // Mark as computing
    status_array[idx] = 1; // COMPUTING
    
    // Get initial conditions for this point
    double theta1_0 = initial_theta1_array[idx];
    double theta2_0 = initial_theta2_array[idx];
    
    // Compute Lyapunov exponent
    double lyapunov = compute_lyapunov_exponent(theta1_0, theta2_0, l1, l2, m1, m2, g, damping, integration_time);
    
    // Store result
    lyapunov_array[idx] = lyapunov;
    
    // Compute color
    float r, green, b;
    map_to_color(lyapunov, color_scheme, &r, &green, &b);
    color_array[idx * 3 + 0] = r;
    color_array[idx * 3 + 1] = green;
    color_array[idx * 3 + 2] = b;
    
    // Mark as complete
    status_array[idx] = 2; // COMPLETE
}

// Kernel launcher function
void chaos_analysis_kernel_launch(
    double* initial_theta1_array,
    double* initial_theta2_array,
    double* lyapunov_array,
    float* color_array,
    int* status_array,
    int grid_size,
    double l1, double l2, double m1, double m2, double g, double damping,
    double integration_time, int color_scheme,
    int block_size) {
    
    int grid_blocks = (grid_size + block_size - 1) / block_size;
    
    chaos_analysis_kernel<<<grid_blocks, block_size>>>(
        initial_theta1_array, initial_theta2_array, lyapunov_array,
        color_array, status_array, grid_size,
        l1, l2, m1, m2, g, damping,
        integration_time, color_scheme
    );
}

} // namespace cuda
} // namespace pendulum
