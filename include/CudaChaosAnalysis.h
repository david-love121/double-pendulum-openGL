#pragma once

#ifdef USE_CUDA

#include "ChaosAnalysis.h"
#include "PendulumState.h"

namespace pendulum {
namespace cuda {

/**
 * CUDA-accelerated chaos analysis computation
 */
bool computeChaosAnalysisGPU(ChaosAnalysisGrid* grid, 
                            const PendulumConfiguration& config,
                            double integration_time, 
                            int color_scheme,
                            int device_id = 0);

/**
 * Check if CUDA is available and working
 */
bool isCudaAvailable();

/**
 * Get CUDA device information
 */
void printCudaDeviceInfo(int device_id = 0);

} // namespace cuda
} // namespace pendulum

#endif // USE_CUDA
