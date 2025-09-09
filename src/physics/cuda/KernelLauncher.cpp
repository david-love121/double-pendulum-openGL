#include "CudaChaosAnalysis.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

namespace pendulum {
namespace cuda {

// Declare the CUDA kernel that's defined in .cu file
void chaos_analysis_kernel_launch(
    double* initial_theta1_array,
    double* initial_theta2_array,
    double* lyapunov_array,
    float* color_array,
    int* status_array,
    int grid_size,
    double l1, double l2, double m1, double m2, double g, double damping,
    double integration_time, int color_scheme,
    int block_size);

bool isCudaAvailable() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error != cudaSuccess || device_count == 0) {
        return false;
    }
    
    // Test basic CUDA functionality
    cudaDeviceProp prop;
    error = cudaGetDeviceProperties(&prop, 0);
    return error == cudaSuccess;
}

void printCudaDeviceInfo(int device_id) {
    cudaDeviceProp prop;
    cudaError_t error = cudaGetDeviceProperties(&prop, device_id);
    
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        return;
    }
    
    std::cout << "CUDA Device " << device_id << ": " << prop.name << std::endl;
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "  Total Global Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
}

bool computeChaosAnalysisGPU(ChaosAnalysisGrid* grid, 
                            const PendulumConfiguration& config,
                            double integration_time, 
                            int color_scheme,
                            int device_id) {
    
    if (!grid || !grid->isValid()) {
        std::cerr << "Invalid chaos analysis grid" << std::endl;
        return false;
    }
    
    // Set CUDA device
    cudaError_t error = cudaSetDevice(device_id);
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error setting device: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    std::cout << "Using CUDA device " << device_id << " for chaos analysis" << std::endl;
    printCudaDeviceInfo(device_id);
    
    int grid_size = grid->getResolutionX() * grid->getResolutionY();
    std::cout << "Computing " << grid_size << " points in parallel..." << std::endl;
    
    // Prepare host data
    std::vector<double> h_theta1(grid_size);
    std::vector<double> h_theta2(grid_size);
    std::vector<double> h_lyapunov(grid_size);
    std::vector<float> h_colors(grid_size * 3);
    std::vector<int> h_status(grid_size);
    
    // Fill initial conditions
    for (int y = 0; y < grid->getResolutionY(); ++y) {
        for (int x = 0; x < grid->getResolutionX(); ++x) {
            int idx = y * grid->getResolutionX() + x;
            
            double theta1_range = grid->getTheta1Max() - grid->getTheta1Min();
            double theta2_range = grid->getTheta2Max() - grid->getTheta2Min();
            
            h_theta1[idx] = grid->getTheta1Min() + (x + 0.5) * theta1_range / grid->getResolutionX();
            h_theta2[idx] = grid->getTheta2Min() + (y + 0.5) * theta2_range / grid->getResolutionY();
            h_status[idx] = 0; // PENDING
        }
    }
    
    // Allocate GPU memory
    double *d_theta1, *d_theta2, *d_lyapunov;
    float *d_colors;
    int *d_status;
    
    size_t size_double = grid_size * sizeof(double);
    size_t size_float3 = grid_size * 3 * sizeof(float);
    size_t size_int = grid_size * sizeof(int);
    
    error = cudaMalloc(&d_theta1, size_double);
    if (error != cudaSuccess) {
        std::cerr << "CUDA malloc error: " << cudaGetErrorString(error) << std::endl;
        return false;
    }
    
    cudaMalloc(&d_theta2, size_double);
    cudaMalloc(&d_lyapunov, size_double);
    cudaMalloc(&d_colors, size_float3);
    cudaMalloc(&d_status, size_int);
    
    // Copy data to GPU
    cudaMemcpy(d_theta1, h_theta1.data(), size_double, cudaMemcpyHostToDevice);
    cudaMemcpy(d_theta2, h_theta2.data(), size_double, cudaMemcpyHostToDevice);
    cudaMemcpy(d_status, h_status.data(), size_int, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int block_size = 256;
    int grid_blocks = (grid_size + block_size - 1) / block_size;
    
    std::cout << "Launching kernel with " << grid_blocks << " blocks of " << block_size << " threads" << std::endl;
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    // Launch the kernel via wrapper function
    chaos_analysis_kernel_launch(
        d_theta1, d_theta2, d_lyapunov, d_colors, d_status,
        grid_size,
        config.l1, config.l2, config.m1, config.m2, config.g, config.damping,
        integration_time, color_scheme,
        block_size
    );
    
    cudaEventRecord(stop);
    
    // Wait for completion and check for errors
    error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        std::cerr << "CUDA kernel execution error: " << cudaGetErrorString(error) << std::endl;
        // Cleanup and return false
        cudaFree(d_theta1);
        cudaFree(d_theta2);
        cudaFree(d_lyapunov);
        cudaFree(d_colors);
        cudaFree(d_status);
        return false;
    }
    
    // Get timing information
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "CUDA kernel completed in " << milliseconds << " ms" << std::endl;
    
    // Copy results back to host
    cudaMemcpy(h_lyapunov.data(), d_lyapunov, size_double, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_colors.data(), d_colors, size_float3, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_status.data(), d_status, size_int, cudaMemcpyDeviceToHost);
    
    // Update grid with results
    for (int y = 0; y < grid->getResolutionY(); ++y) {
        for (int x = 0; x < grid->getResolutionX(); ++x) {
            int idx = y * grid->getResolutionX() + x;
            ChaosAnalysisPoint& point = grid->getPoint(x, y);
            
            point.lyapunov_exponent = h_lyapunov[idx];
            point.color.r = h_colors[idx * 3 + 0];
            point.color.g = h_colors[idx * 3 + 1];
            point.color.b = h_colors[idx * 3 + 2];
            point.computation_status = static_cast<ChaosAnalysisPoint::Status>(h_status[idx]);
        }
    }
    
    // Cleanup GPU memory
    cudaFree(d_theta1);
    cudaFree(d_theta2);
    cudaFree(d_lyapunov);
    cudaFree(d_colors);
    cudaFree(d_status);
    
    // Cleanup events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    std::cout << "CUDA chaos analysis completed successfully!" << std::endl;
    
    return true;
}

} // namespace cuda
} // namespace pendulum
