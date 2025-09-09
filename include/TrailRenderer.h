#pragma once

#include <vector>
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <memory>

namespace pendulum {

// Forward declaration
class Shader;

struct TrailPoint {
    float x, y;
    float timestamp;
    float alpha;
    
    TrailPoint(float x_, float y_, float timestamp_) 
        : x(x_), y(y_), timestamp(timestamp_), alpha(1.0f) {}
};

/**
 * GPU-accelerated trail rendering system using VAOs/VBOs
 */
class TrailRenderer {
public:
    TrailRenderer();
    virtual ~TrailRenderer();
    
    virtual bool initialize();
    virtual void cleanup();
    
    // Trail management
    virtual void addPoint(float x, float y, float timestamp);
    virtual void updateFading(float current_time, float fade_duration, bool fade_enabled);
    virtual void clear();
    
    // Rendering
    virtual void render(const glm::mat4& view, const glm::mat4& projection) const;
    
    // Configuration
    void setColor(float r, float g, float b) { m_color[0] = r; m_color[1] = g; m_color[2] = b; }
    virtual void setMaxPoints(size_t max_points) { m_maxPoints = max_points; }
    
    size_t getPointCount() const { return m_points.size(); }

protected:
    void updateGPUBuffers();
    void removeOldPoints(float cutoff_time);
    
    // OpenGL objects
    GLuint m_vao = 0;
    GLuint m_vbo = 0;      // Vertex positions
    GLuint m_colorVbo = 0; // Per-vertex colors
    std::unique_ptr<Shader> m_shader;
    
    // Trail data
    std::vector<TrailPoint> m_points;
    std::vector<float> m_vertices;     // Interleaved x,y positions
    std::vector<float> m_colors;       // Interleaved r,g,b,a values
    
    // Configuration
    float m_color[3] = {1.0f, 1.0f, 1.0f}; // Base color
    size_t m_maxPoints = 50000;
    bool m_buffersDirty = true;
};

} // namespace pendulum
