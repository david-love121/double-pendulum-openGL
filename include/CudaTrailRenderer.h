#pragma once

#ifdef USE_CUDA

#include <GL/glew.h>  // Include GLEW first
#include "TrailRenderer.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

namespace pendulum {
namespace cuda {

// Forward declaration for CUDA-CPU interface
struct CudaTrailPoint {
    float x, y;
    float timestamp;
    float alpha;
};

/**
 * CUDA-accelerated trail rendering system with GPU-based data preparation
 * and OpenGL-CUDA interoperability for zero-copy rendering
 */
class CudaTrailRenderer : public TrailRenderer {
public:
    CudaTrailRenderer();
    ~CudaTrailRenderer();
    
    bool initialize() override;
    void cleanup() override;
    
    // Enhanced CUDA-accelerated trail management
    void addPoint(float x, float y, float timestamp) override;
    void addPointBatch(const std::vector<glm::vec2>& positions, 
                      const std::vector<float>& timestamps);
    void updateFading(float current_time, float fade_duration, bool fade_enabled) override;
    void clear() override;
    
    // CUDA-specific functionality
    void updateFadingGPU(float current_time, float fade_duration, bool fade_enabled);
    void removeOldPointsGPU(float cutoff_time);
    void generateVertexDataGPU();
    
    // Rendering with GPU-prepared data
    void render(const glm::mat4& view, const glm::mat4& projection) const override;
    
    // Configuration
    void setMaxPoints(size_t max_points) override;
    
    // Performance monitoring
    double getLastCudaTime() const { return m_lastCudaTime; }
    bool isCudaEnabled() const { return m_cudaInitialized; }
    
private:
    bool initializeCuda();
    void cleanupCuda();
    void allocateDeviceMemory(size_t max_points);
    void freeDeviceMemory();
    void mapGLBuffers();
    void unmapGLBuffers();
    
    // CUDA state
    bool m_cudaInitialized = false;
    int m_deviceId = 0;
    size_t m_maxDevicePoints = 0;
    double m_lastCudaTime = 0.0;
    
    // Device memory pointers
    CudaTrailPoint* d_points = nullptr;
    float* d_vertices = nullptr;  // Interleaved x,y positions
    float* d_colors = nullptr;    // Interleaved r,g,b,a values
    int* d_point_count = nullptr; // Device-side point counter
    
    // OpenGL-CUDA interop resources
    cudaGraphicsResource_t m_vboResource = nullptr;
    cudaGraphicsResource_t m_colorVboResource = nullptr;
    bool m_buffersMapped = false;
    
    // CUDA streams and events
    cudaStream_t m_stream = nullptr;
    cudaEvent_t m_startEvent = nullptr;
    cudaEvent_t m_stopEvent = nullptr;
    
    // Host-side point counter for synchronization
    int m_hostPointCount = 0;
};

// CUDA kernel launchers (implemented in .cu file)
extern "C" {
    void launchUpdateFading(CudaTrailPoint* points,
                           int point_count,
                           float current_time,
                           float fade_duration,
                           bool fade_enabled,
                           cudaStream_t stream);
    
    void launchRemoveOldPoints(CudaTrailPoint* points,
                              int* point_count,
                              float cutoff_time,
                              cudaStream_t stream);
    
    void launchGenerateVertexData(const CudaTrailPoint* points,
                                 int point_count,
                                 float* vertices,
                                 float* colors,
                                 float base_color_r,
                                 float base_color_g,
                                 float base_color_b,
                                 cudaStream_t stream);
}

} // namespace cuda
} // namespace pendulum

#endif // USE_CUDA
