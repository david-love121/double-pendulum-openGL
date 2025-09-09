#ifdef USE_CUDA

#include <GL/glew.h>  // Include GLEW first
#include "CudaTrailRenderer.h"
#include "Rendering.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <stdexcept>
#include <algorithm>

namespace pendulum {
namespace cuda {

// CUDA kernels for trail processing

__global__ void updateFadingKernel(CudaTrailPoint* points,
                                  int point_count,
                                  float current_time,
                                  float fade_duration,
                                  bool fade_enabled) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= point_count) return;
    
    CudaTrailPoint& point = points[idx];
    
    if (!fade_enabled) {
        point.alpha = 0.8f;
    } else {
        float age = current_time - point.timestamp;
        point.alpha = fmaxf(0.0f, 1.0f - (age / fade_duration));
    }
}

__global__ void removeOldPointsKernel(CudaTrailPoint* points,
                                     int* point_count,
                                     float cutoff_time) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_points = *point_count;
    
    if (idx >= total_points) return;
    
    // Use shared memory for efficient compaction
    extern __shared__ int shared_data[];
    int* write_indices = shared_data;
    int* valid_flags = &shared_data[blockDim.x];
    
    int local_idx = threadIdx.x;
    
    // Check if this point should be kept
    bool is_valid = (points[idx].timestamp >= cutoff_time);
    valid_flags[local_idx] = is_valid ? 1 : 0;
    
    __syncthreads();
    
    // Compute prefix sum to get write indices
    int write_pos = 0;
    for (int i = 0; i < local_idx; ++i) {
        write_pos += valid_flags[i];
    }
    write_indices[local_idx] = write_pos;
    
    __syncthreads();
    
    // Global write position
    int global_write_pos = 0;
    if (blockIdx.x > 0) {
        // In a real implementation, we'd need inter-block communication
        // For simplicity, we'll use atomic operations
    }
    
    if (is_valid) {
        int write_index = atomicAdd(point_count, 0) - total_points + write_pos;
        if (write_index < total_points && write_index != idx) {
            points[write_index] = points[idx];
        }
    }
    
    __syncthreads();
    
    // Update the count (only one thread per block)
    if (threadIdx.x == 0) {
        int valid_count = 0;
        for (int i = 0; i < blockDim.x && blockIdx.x * blockDim.x + i < total_points; ++i) {
            if (valid_flags[i]) valid_count++;
        }
        atomicAdd(point_count, -valid_count);
    }
}

__global__ void generateVertexDataKernel(const CudaTrailPoint* points,
                                        int point_count,
                                        float* vertices,
                                        float* colors,
                                        float base_color_r,
                                        float base_color_g,
                                        float base_color_b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= point_count) return;
    
    const CudaTrailPoint& point = points[idx];
    
    // Position data (interleaved x,y)
    vertices[idx * 2] = point.x;
    vertices[idx * 2 + 1] = point.y;
    
    // Color data (interleaved r,g,b,a)
    colors[idx * 4] = base_color_r;
    colors[idx * 4 + 1] = base_color_g;
    colors[idx * 4 + 2] = base_color_b;
    colors[idx * 4 + 3] = point.alpha;
}

// Simplified remove old points kernel (more efficient implementation)
__global__ void removeOldPointsSimpleKernel(CudaTrailPoint* points,
                                           int* point_count,
                                           float cutoff_time) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_points = *point_count;
    
    if (idx >= total_points) return;
    
    // Simple approach: mark invalid points with negative timestamp
    if (points[idx].timestamp < cutoff_time) {
        points[idx].timestamp = -1.0f;  // Mark for removal
    }
}

__global__ void compactPointsKernel(CudaTrailPoint* points,
                                   int* point_count) {
    int total_points = *point_count;
    int write_pos = 0;
    
    // Single-threaded compaction (can be optimized with parallel scan)
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int i = 0; i < total_points; ++i) {
            if (points[i].timestamp >= 0.0f) {  // Valid point
                if (write_pos != i) {
                    points[write_pos] = points[i];
                }
                write_pos++;
            }
        }
        *point_count = write_pos;
    }
}

// Host wrapper functions
extern "C" {
    void launchUpdateFading(CudaTrailPoint* points,
                           int point_count,
                           float current_time,
                           float fade_duration,
                           bool fade_enabled,
                           cudaStream_t stream) {
        const int blockSize = 256;
        const int gridSize = (point_count + blockSize - 1) / blockSize;
        
        updateFadingKernel<<<gridSize, blockSize, 0, stream>>>(
            points, point_count, current_time, fade_duration, fade_enabled);
    }
    
    void launchRemoveOldPoints(CudaTrailPoint* points,
                              int* point_count,
                              float cutoff_time,
                              cudaStream_t stream) {
        int host_point_count;
        cudaMemcpy(&host_point_count, point_count, sizeof(int), cudaMemcpyDeviceToHost);
        
        if (host_point_count <= 0) return;
        
        const int blockSize = 256;
        const int gridSize = (host_point_count + blockSize - 1) / blockSize;
        
        // Mark invalid points
        removeOldPointsSimpleKernel<<<gridSize, blockSize, 0, stream>>>(
            points, point_count, cutoff_time);
        
        // Compact array
        compactPointsKernel<<<1, 1, 0, stream>>>(points, point_count);
    }
    
    void launchGenerateVertexData(const CudaTrailPoint* points,
                                 int point_count,
                                 float* vertices,
                                 float* colors,
                                 float base_color_r,
                                 float base_color_g,
                                 float base_color_b,
                                 cudaStream_t stream) {
        const int blockSize = 256;
        const int gridSize = (point_count + blockSize - 1) / blockSize;
        
        generateVertexDataKernel<<<gridSize, blockSize, 0, stream>>>(
            points, point_count, vertices, colors, 
            base_color_r, base_color_g, base_color_b);
    }
}

// Host class implementation
CudaTrailRenderer::CudaTrailRenderer() : TrailRenderer() {
    // Constructor
}

CudaTrailRenderer::~CudaTrailRenderer() {
    cleanup();
}

bool CudaTrailRenderer::initialize() {
    // Initialize base TrailRenderer first
    if (!TrailRenderer::initialize()) {
        return false;
    }
    
    // Initialize CUDA components
    if (!initializeCuda()) {
        std::cerr << "Failed to initialize CUDA for trail renderer, falling back to CPU" << std::endl;
        return true;  // Still return true to allow fallback to CPU rendering
    }
    
    std::cout << "CUDA Trail Renderer initialized successfully" << std::endl;
    return true;
}

void CudaTrailRenderer::cleanup() {
    cleanupCuda();
    TrailRenderer::cleanup();
}

bool CudaTrailRenderer::initializeCuda() {
    // Check CUDA availability
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        return false;
    }
    
    // Use device 0
    m_deviceId = 0;
    err = cudaSetDevice(m_deviceId);
    if (err != cudaSuccess) {
        return false;
    }
    
    // Create CUDA stream
    err = cudaStreamCreate(&m_stream);
    if (err != cudaSuccess) {
        return false;
    }
    
    // Create CUDA events for timing
    err = cudaEventCreate(&m_startEvent);
    if (err != cudaSuccess) {
        cleanupCuda();
        return false;
    }
    
    err = cudaEventCreate(&m_stopEvent);
    if (err != cudaSuccess) {
        cleanupCuda();
        return false;
    }
    
    // Allocate initial device memory
    allocateDeviceMemory(m_maxPoints);
    
    // Register OpenGL buffers with CUDA
    mapGLBuffers();
    
    m_cudaInitialized = true;
    return true;
}

void CudaTrailRenderer::cleanupCuda() {
    if (m_buffersMapped) {
        unmapGLBuffers();
    }
    
    freeDeviceMemory();
    
    if (m_stream) {
        cudaStreamDestroy(m_stream);
        m_stream = nullptr;
    }
    
    if (m_startEvent) {
        cudaEventDestroy(m_startEvent);
        m_startEvent = nullptr;
    }
    
    if (m_stopEvent) {
        cudaEventDestroy(m_stopEvent);
        m_stopEvent = nullptr;
    }
    
    m_cudaInitialized = false;
}

void CudaTrailRenderer::allocateDeviceMemory(size_t max_points) {
    if (max_points <= m_maxDevicePoints && d_points != nullptr) {
        return;  // Already have enough memory
    }
    
    freeDeviceMemory();
    
    cudaError_t err;
    
    // Allocate device memory for trail points
    err = cudaMalloc(&d_points, max_points * sizeof(CudaTrailPoint));
    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory for trail points: " + 
                                std::string(cudaGetErrorString(err)));
    }
    
    // Allocate device memory for vertex data (will be mapped to OpenGL buffers)
    err = cudaMalloc(&d_vertices, max_points * 2 * sizeof(float));
    if (err != cudaSuccess) {
        freeDeviceMemory();
        throw std::runtime_error("Failed to allocate device memory for vertices: " + 
                                std::string(cudaGetErrorString(err)));
    }
    
    err = cudaMalloc(&d_colors, max_points * 4 * sizeof(float));
    if (err != cudaSuccess) {
        freeDeviceMemory();
        throw std::runtime_error("Failed to allocate device memory for colors: " + 
                                std::string(cudaGetErrorString(err)));
    }
    
    // Allocate device memory for point counter
    err = cudaMalloc(&d_point_count, sizeof(int));
    if (err != cudaSuccess) {
        freeDeviceMemory();
        throw std::runtime_error("Failed to allocate device memory for point count: " + 
                                std::string(cudaGetErrorString(err)));
    }
    
    // Initialize point count to 0
    int zero = 0;
    cudaMemcpy(d_point_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
    
    m_maxDevicePoints = max_points;
}

void CudaTrailRenderer::freeDeviceMemory() {
    if (d_points) {
        cudaFree(d_points);
        d_points = nullptr;
    }
    
    if (d_vertices) {
        cudaFree(d_vertices);
        d_vertices = nullptr;
    }
    
    if (d_colors) {
        cudaFree(d_colors);
        d_colors = nullptr;
    }
    
    if (d_point_count) {
        cudaFree(d_point_count);
        d_point_count = nullptr;
    }
    
    m_maxDevicePoints = 0;
}

void CudaTrailRenderer::mapGLBuffers() {
    if (!m_cudaInitialized || m_buffersMapped) {
        return;
    }
    
    // Register OpenGL VBOs with CUDA for direct access
    cudaError_t err = cudaGraphicsGLRegisterBuffer(&m_vboResource, m_vbo, 
                                                  cudaGraphicsMapFlagsWriteDiscard);
    if (err != cudaSuccess) {
        std::cerr << "Failed to register vertex buffer with CUDA: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    err = cudaGraphicsGLRegisterBuffer(&m_colorVboResource, m_colorVbo, 
                                      cudaGraphicsMapFlagsWriteDiscard);
    if (err != cudaSuccess) {
        std::cerr << "Failed to register color buffer with CUDA: " << cudaGetErrorString(err) << std::endl;
        cudaGraphicsUnregisterResource(m_vboResource);
        m_vboResource = nullptr;
        return;
    }
    
    m_buffersMapped = true;
}

void CudaTrailRenderer::unmapGLBuffers() {
    if (!m_buffersMapped) {
        return;
    }
    
    if (m_vboResource) {
        cudaGraphicsUnregisterResource(m_vboResource);
        m_vboResource = nullptr;
    }
    
    if (m_colorVboResource) {
        cudaGraphicsUnregisterResource(m_colorVboResource);
        m_colorVboResource = nullptr;
    }
    
    m_buffersMapped = false;
}

void CudaTrailRenderer::addPoint(float x, float y, float timestamp) {
    if (!m_cudaInitialized) {
        // Fallback to CPU implementation
        TrailRenderer::addPoint(x, y, timestamp);
        return;
    }
    
    // Ensure device memory is sufficient
    if (m_hostPointCount >= static_cast<int>(m_maxDevicePoints)) {
        // Remove some old points or resize
        allocateDeviceMemory(m_maxDevicePoints * 2);
    }
    
    // Add point directly to device memory
    CudaTrailPoint new_point = { x, y, timestamp, 1.0f };
    
    cudaError_t err = cudaMemcpyAsync(&d_points[m_hostPointCount], &new_point, 
                                     sizeof(CudaTrailPoint), 
                                     cudaMemcpyHostToDevice, m_stream);
    if (err != cudaSuccess) {
        std::cerr << "Failed to add point to device: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    m_hostPointCount++;
    
    // Update device counter
    cudaMemcpyAsync(d_point_count, &m_hostPointCount, sizeof(int), 
                   cudaMemcpyHostToDevice, m_stream);
    
    m_buffersDirty = true;
}

void CudaTrailRenderer::addPointBatch(const std::vector<glm::vec2>& positions, 
                                     const std::vector<float>& timestamps) {
    if (!m_cudaInitialized || positions.size() != timestamps.size()) {
        // Fallback to individual CPU additions
        for (size_t i = 0; i < positions.size() && i < timestamps.size(); ++i) {
            TrailRenderer::addPoint(positions[i].x, positions[i].y, timestamps[i]);
        }
        return;
    }
    
    size_t batch_size = positions.size();
    if (m_hostPointCount + batch_size > m_maxDevicePoints) {
        allocateDeviceMemory(std::max(m_maxDevicePoints * 2, m_hostPointCount + batch_size));
    }
    
    // Prepare batch data on host
    std::vector<CudaTrailPoint> host_points(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        host_points[i] = { positions[i].x, positions[i].y, timestamps[i], 1.0f };
    }
    
    // Copy batch to device
    cudaError_t err = cudaMemcpyAsync(&d_points[m_hostPointCount], host_points.data(),
                                     batch_size * sizeof(CudaTrailPoint),
                                     cudaMemcpyHostToDevice, m_stream);
    if (err != cudaSuccess) {
        std::cerr << "Failed to add point batch to device: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    m_hostPointCount += static_cast<int>(batch_size);
    
    // Update device counter
    cudaMemcpyAsync(d_point_count, &m_hostPointCount, sizeof(int), 
                   cudaMemcpyHostToDevice, m_stream);
    
    m_buffersDirty = true;
}

void CudaTrailRenderer::updateFading(float current_time, float fade_duration, bool fade_enabled) {
    if (!m_cudaInitialized || m_hostPointCount == 0) {
        TrailRenderer::updateFading(current_time, fade_duration, fade_enabled);
        return;
    }
    
    updateFadingGPU(current_time, fade_duration, fade_enabled);
}

void CudaTrailRenderer::updateFadingGPU(float current_time, float fade_duration, bool fade_enabled) {
    if (m_hostPointCount == 0) return;
    
    // Record start time
    cudaEventRecord(m_startEvent, m_stream);
    
    // Remove old points first if fading is enabled
    if (fade_enabled) {
        removeOldPointsGPU(current_time - fade_duration);
    }
    
    // Update fading for remaining points
    launchUpdateFading(d_points, m_hostPointCount, current_time, fade_duration, fade_enabled, m_stream);
    
    // Record stop time
    cudaEventRecord(m_stopEvent, m_stream);
    
    // Calculate kernel time
    cudaEventSynchronize(m_stopEvent);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, m_startEvent, m_stopEvent);
    m_lastCudaTime = milliseconds / 1000.0;
    
    m_buffersDirty = true;
}

void CudaTrailRenderer::removeOldPointsGPU(float cutoff_time) {
    if (m_hostPointCount == 0) return;
    
    launchRemoveOldPoints(d_points, d_point_count, cutoff_time, m_stream);
    
    // Update host point count
    cudaMemcpyAsync(&m_hostPointCount, d_point_count, sizeof(int), 
                   cudaMemcpyDeviceToHost, m_stream);
    cudaStreamSynchronize(m_stream);
}

void CudaTrailRenderer::generateVertexDataGPU() {
    if (m_hostPointCount == 0) return;
    
    // Map OpenGL buffers for CUDA access
    float* mapped_vertices = nullptr;
    float* mapped_colors = nullptr;
    size_t vertex_size, color_size;
    
    if (m_buffersMapped) {
        cudaGraphicsMapResources(1, &m_vboResource, m_stream);
        cudaGraphicsMapResources(1, &m_colorVboResource, m_stream);
        
        cudaGraphicsResourceGetMappedPointer((void**)&mapped_vertices, &vertex_size, m_vboResource);
        cudaGraphicsResourceGetMappedPointer((void**)&mapped_colors, &color_size, m_colorVboResource);
        
        // Generate vertex data directly into OpenGL buffers
        launchGenerateVertexData(d_points, m_hostPointCount, 
                                mapped_vertices, mapped_colors,
                                m_color[0], m_color[1], m_color[2], m_stream);
        
        cudaGraphicsUnmapResources(1, &m_vboResource, m_stream);
        cudaGraphicsUnmapResources(1, &m_colorVboResource, m_stream);
    } else {
        // Fallback: generate in device memory and copy to OpenGL
        launchGenerateVertexData(d_points, m_hostPointCount,
                                d_vertices, d_colors,
                                m_color[0], m_color[1], m_color[2], m_stream);
        
        // Copy to OpenGL buffers
        std::vector<float> host_vertices(m_hostPointCount * 2);
        std::vector<float> host_colors(m_hostPointCount * 4);
        
        cudaMemcpyAsync(host_vertices.data(), d_vertices, 
                       m_hostPointCount * 2 * sizeof(float), 
                       cudaMemcpyDeviceToHost, m_stream);
        cudaMemcpyAsync(host_colors.data(), d_colors,
                       m_hostPointCount * 4 * sizeof(float),
                       cudaMemcpyDeviceToHost, m_stream);
        
        cudaStreamSynchronize(m_stream);
        
        // Upload to OpenGL
        glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
        glBufferData(GL_ARRAY_BUFFER, host_vertices.size() * sizeof(float), 
                     host_vertices.data(), GL_DYNAMIC_DRAW);
        
        glBindBuffer(GL_ARRAY_BUFFER, m_colorVbo);
        glBufferData(GL_ARRAY_BUFFER, host_colors.size() * sizeof(float), 
                     host_colors.data(), GL_DYNAMIC_DRAW);
        
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
}

void CudaTrailRenderer::clear() {
    if (m_cudaInitialized) {
        int zero = 0;
        cudaMemcpy(d_point_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
        m_hostPointCount = 0;
    }
    
    TrailRenderer::clear();
}

void CudaTrailRenderer::setMaxPoints(size_t max_points) {
    TrailRenderer::setMaxPoints(max_points);
    
    if (m_cudaInitialized) {
        allocateDeviceMemory(max_points);
    }
}

void CudaTrailRenderer::render(const glm::mat4& view, const glm::mat4& projection) const {
    if (m_hostPointCount == 0 || !m_shader || !m_shader->isValid()) {
        return;
    }
    
    // Update GPU buffers if needed
    if (m_buffersDirty && m_cudaInitialized) {
        const_cast<CudaTrailRenderer*>(this)->generateVertexDataGPU();
        const_cast<CudaTrailRenderer*>(this)->m_buffersDirty = false;
    } else if (m_buffersDirty) {
        // Fallback to CPU buffer update
        const_cast<CudaTrailRenderer*>(this)->updateGPUBuffers();
    }
    
    // Use shader and set uniforms
    m_shader->use();
    m_shader->setUniform("view", view);
    m_shader->setUniform("projection", projection);
    
    // Enable blending for trail transparency
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // Render as line strip
    glBindVertexArray(m_vao);
    glDrawArrays(GL_LINE_STRIP, 0, m_hostPointCount);
    glBindVertexArray(0);
    
    glDisable(GL_BLEND);
    m_shader->unbind();
}

} // namespace cuda
} // namespace pendulum

#endif // USE_CUDA
