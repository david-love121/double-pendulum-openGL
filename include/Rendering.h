#pragma once

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <string>
#include <memory>

namespace pendulum {

/**
 * OpenGL shader management
 */
class Shader {
public:
    Shader();
    ~Shader();

    bool loadFromFiles(const std::string& vertexPath, const std::string& fragmentPath);
    bool loadFromSource(const std::string& vertexSource, const std::string& fragmentSource);
    
    void use() const;
    void unbind() const;
    
    // Uniform setters
    void setUniform(const std::string& name, int value) const;
    void setUniform(const std::string& name, float value) const;
    void setUniform(const std::string& name, const glm::vec2& value) const;
    void setUniform(const std::string& name, const glm::vec3& value) const;
    void setUniform(const std::string& name, const glm::vec4& value) const;
    void setUniform(const std::string& name, const glm::mat4& value) const;
    
    GLuint getID() const { return m_programID; }
    bool isValid() const { return m_programID != 0; }

private:
    GLuint compileShader(const std::string& source, GLenum type);
    GLuint linkProgram(GLuint vertexShader, GLuint fragmentShader);
    GLint getUniformLocation(const std::string& name) const;
    
    GLuint m_programID = 0;
};

/**
 * Camera for view transformations
 */
class Camera {
public:
    Camera();
    
    void setPosition(const glm::vec2& position) { m_position = position; }
    void setZoom(float zoom) { m_zoom = zoom; }
    void setViewport(int width, int height);
    
    glm::vec2 getPosition() const { return m_position; }
    float getZoom() const { return m_zoom; }
    
    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix() const;
    
    // Screen to world coordinate conversion
    glm::vec2 screenToWorld(const glm::vec2& screenPos) const;
    glm::vec2 worldToScreen(const glm::vec2& worldPos) const;

private:
    glm::vec2 m_position = glm::vec2(0.0f);
    float m_zoom = 1.0f;
    int m_viewportWidth = 800;
    int m_viewportHeight = 600;
};

/**
 * Rendering primitives for pendulum visualization
 */
class Primitives {
public:
    static void initialize();
    static void cleanup();
    
    // Set view and projection matrices for all primitives
    static void setMatrices(const glm::mat4& view, const glm::mat4& projection);
    
    // Basic shape rendering
    static void drawLine(const glm::vec2& start, const glm::vec2& end, 
                        const glm::vec4& color, float thickness = 1.0f);
    static void drawCircle(const glm::vec2& center, float radius, 
                          const glm::vec4& color, bool filled = true);
    static void drawArc(const glm::vec2& center, float radius, 
                       float startAngle, float endAngle, 
                       const glm::vec4& color, float thickness = 1.0f);
    
    // Text rendering
    static void drawText(const std::string& text, const glm::vec2& position, 
                        const glm::vec4& color, float scale = 1.0f);

private:
    static void initializeLineRenderer();
    static void initializeCircleRenderer();
    static void initializeTextRenderer();
    
    static bool s_initialized;
    static GLuint s_lineVAO, s_lineVBO;
    static GLuint s_circleVAO, s_circleVBO, s_circleEBO;
    static std::unique_ptr<Shader> s_lineShader;
    static std::unique_ptr<Shader> s_circleShader;
};

} // namespace pendulum
