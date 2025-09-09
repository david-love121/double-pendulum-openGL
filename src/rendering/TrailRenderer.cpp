#include "TrailRenderer.h"
#include "Rendering.h"
#include <algorithm>
#include <iostream>

namespace pendulum {

TrailRenderer::TrailRenderer() {
    m_points.reserve(10000);
    m_vertices.reserve(20000);  // 2 floats per point
    m_colors.reserve(40000);    // 4 floats per point
}

TrailRenderer::~TrailRenderer() {
    cleanup();
}

bool TrailRenderer::initialize() {
    // Generate VAO and VBOs
    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);
    glGenBuffers(1, &m_colorVbo);
    
    if (m_vao == 0 || m_vbo == 0 || m_colorVbo == 0) {
        std::cerr << "Failed to generate OpenGL buffers for trail renderer" << std::endl;
        return false;
    }
    
    // Initialize shader
    m_shader = std::make_unique<Shader>();
    
    std::string vertexSource = R"(
        #version 460 core
        layout (location = 0) in vec2 position;
        layout (location = 1) in vec4 color;
        
        uniform mat4 view;
        uniform mat4 projection;
        
        out vec4 fragColor;
        
        void main() {
            gl_Position = projection * view * vec4(position, 0.0, 1.0);
            fragColor = color;
        }
    )";
    
    std::string fragmentSource = R"(
        #version 460 core
        in vec4 fragColor;
        out vec4 outputColor;
        
        void main() {
            outputColor = fragColor;
        }
    )";
    
    if (!m_shader->loadFromSource(vertexSource, fragmentSource)) {
        std::cerr << "Failed to load trail renderer shader" << std::endl;
        return false;
    }
    
    // Setup VAO
    glBindVertexArray(m_vao);
    
    // Position attribute (location 0)
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);
    
    // Color attribute (location 1)
    glBindBuffer(GL_ARRAY_BUFFER, m_colorVbo);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(1);
    
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    return true;
}

void TrailRenderer::cleanup() {
    if (m_vao != 0) {
        glDeleteVertexArrays(1, &m_vao);
        m_vao = 0;
    }
    if (m_vbo != 0) {
        glDeleteBuffers(1, &m_vbo);
        m_vbo = 0;
    }
    if (m_colorVbo != 0) {
        glDeleteBuffers(1, &m_colorVbo);
        m_colorVbo = 0;
    }
    
    // Clean up shader
    m_shader.reset();
}

void TrailRenderer::addPoint(float x, float y, float timestamp) {
    m_points.emplace_back(x, y, timestamp);
    
    // Enforce max points limit for memory management
    if (m_points.size() > m_maxPoints) {
        size_t remove_count = m_points.size() - m_maxPoints;
        m_points.erase(m_points.begin(), m_points.begin() + remove_count);
    }
    
    m_buffersDirty = true;
}

void TrailRenderer::updateFading(float current_time, float fade_duration, bool fade_enabled) {
    if (!fade_enabled) {
        // Set all points to full alpha for infinite trails
        for (auto& point : m_points) {
            point.alpha = 0.8f;
        }
    } else {
        // Remove old points first
        removeOldPoints(current_time - fade_duration);
        
        // Update alpha for remaining points based on age
        for (auto& point : m_points) {
            float age = current_time - point.timestamp;
            point.alpha = std::max(0.0f, 1.0f - (age / fade_duration));
        }
    }
    
    m_buffersDirty = true;
}

void TrailRenderer::removeOldPoints(float cutoff_time) {
    auto new_end = std::remove_if(m_points.begin(), m_points.end(),
        [cutoff_time](const TrailPoint& point) {
            return point.timestamp < cutoff_time;
        });
    
    if (new_end != m_points.end()) {
        m_points.erase(new_end, m_points.end());
        m_buffersDirty = true;
    }
}

void TrailRenderer::clear() {
    m_points.clear();
    m_vertices.clear();
    m_colors.clear();
    m_buffersDirty = true;
}

void TrailRenderer::updateGPUBuffers() {
    if (!m_buffersDirty || m_points.empty()) {
        return;
    }
    
    const size_t point_count = m_points.size();
    
    // Prepare vertex data
    m_vertices.resize(point_count * 2);
    m_colors.resize(point_count * 4);
    
    for (size_t i = 0; i < point_count; ++i) {
        const TrailPoint& point = m_points[i];
        
        // Position data
        m_vertices[i * 2] = point.x;
        m_vertices[i * 2 + 1] = point.y;
        
        // Color data (base color with computed alpha)
        m_colors[i * 4] = m_color[0];     // R
        m_colors[i * 4 + 1] = m_color[1]; // G
        m_colors[i * 4 + 2] = m_color[2]; // B
        m_colors[i * 4 + 3] = point.alpha; // A
    }
    
    // Upload to GPU
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, m_vertices.size() * sizeof(float), 
                 m_vertices.data(), GL_DYNAMIC_DRAW);
    
    glBindBuffer(GL_ARRAY_BUFFER, m_colorVbo);
    glBufferData(GL_ARRAY_BUFFER, m_colors.size() * sizeof(float), 
                 m_colors.data(), GL_DYNAMIC_DRAW);
    
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    m_buffersDirty = false;
}

void TrailRenderer::render(const glm::mat4& view, const glm::mat4& projection) const {
    if (m_points.empty() || !m_shader || !m_shader->isValid()) {
        return;
    }
    
    // Update GPU buffers if needed (const_cast for lazy update)
    const_cast<TrailRenderer*>(this)->updateGPUBuffers();
    
    // Use shader and set uniforms
    m_shader->use();
    m_shader->setUniform("view", view);
    m_shader->setUniform("projection", projection);
    
    // Enable blending for trail transparency
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    // Render as line strip
    glBindVertexArray(m_vao);
    glDrawArrays(GL_LINE_STRIP, 0, static_cast<GLsizei>(m_points.size()));
    glBindVertexArray(0);
    
    glDisable(GL_BLEND);
    m_shader->unbind();
}

} // namespace pendulum
