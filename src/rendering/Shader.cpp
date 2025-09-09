#include "Rendering.h"
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

namespace pendulum {

Shader::Shader() {
    // Constructor
}

Shader::~Shader() {
    if (m_programID != 0) {
        glDeleteProgram(m_programID);
    }
}

bool Shader::loadFromFiles(const std::string& vertexPath, const std::string& fragmentPath) {
    // Read vertex shader
    std::ifstream vertexFile(vertexPath);
    if (!vertexFile.is_open()) {
        std::cerr << "Failed to open vertex shader: " << vertexPath << std::endl;
        return false;
    }
    std::stringstream vertexStream;
    vertexStream << vertexFile.rdbuf();
    std::string vertexSource = vertexStream.str();
    
    // Read fragment shader
    std::ifstream fragmentFile(fragmentPath);
    if (!fragmentFile.is_open()) {
        std::cerr << "Failed to open fragment shader: " << fragmentPath << std::endl;
        return false;
    }
    std::stringstream fragmentStream;
    fragmentStream << fragmentFile.rdbuf();
    std::string fragmentSource = fragmentStream.str();
    
    return loadFromSource(vertexSource, fragmentSource);
}

bool Shader::loadFromSource(const std::string& vertexSource, const std::string& fragmentSource) {
    GLuint vertexShader = compileShader(vertexSource, GL_VERTEX_SHADER);
    if (vertexShader == 0) return false;
    
    GLuint fragmentShader = compileShader(fragmentSource, GL_FRAGMENT_SHADER);
    if (fragmentShader == 0) {
        glDeleteShader(vertexShader);
        return false;
    }
    
    m_programID = linkProgram(vertexShader, fragmentShader);
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    return m_programID != 0;
}

void Shader::use() const {
    glUseProgram(m_programID);
}

void Shader::unbind() const {
    glUseProgram(0);
}

void Shader::setUniform(const std::string& name, int value) const {
    glUniform1i(getUniformLocation(name), value);
}

void Shader::setUniform(const std::string& name, float value) const {
    glUniform1f(getUniformLocation(name), value);
}

void Shader::setUniform(const std::string& name, const glm::vec2& value) const {
    glUniform2fv(getUniformLocation(name), 1, &value[0]);
}

void Shader::setUniform(const std::string& name, const glm::vec3& value) const {
    glUniform3fv(getUniformLocation(name), 1, &value[0]);
}

void Shader::setUniform(const std::string& name, const glm::vec4& value) const {
    glUniform4fv(getUniformLocation(name), 1, &value[0]);
}

void Shader::setUniform(const std::string& name, const glm::mat4& value) const {
    glUniformMatrix4fv(getUniformLocation(name), 1, GL_FALSE, &value[0][0]);
}

GLuint Shader::compileShader(const std::string& source, GLenum type) {
    GLuint shader = glCreateShader(type);
    const char* src = source.c_str();
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);
    
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLchar infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Shader compilation failed: " << infoLog << std::endl;
        glDeleteShader(shader);
        return 0;
    }
    
    return shader;
}

GLuint Shader::linkProgram(GLuint vertexShader, GLuint fragmentShader) {
    GLuint program = glCreateProgram();
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    
    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        GLchar infoLog[512];
        glGetProgramInfoLog(program, 512, nullptr, infoLog);
        std::cerr << "Program linking failed: " << infoLog << std::endl;
        glDeleteProgram(program);
        return 0;
    }
    
    return program;
}

GLint Shader::getUniformLocation(const std::string& name) const {
    return glGetUniformLocation(m_programID, name.c_str());
}

// Camera implementation
Camera::Camera() {
    // Constructor
}

void Camera::setViewport(int width, int height) {
    m_viewportWidth = width;
    m_viewportHeight = height;
}

glm::mat4 Camera::getViewMatrix() const {
    return glm::translate(glm::mat4(1.0f), glm::vec3(-m_position, 0.0f));
}

glm::mat4 Camera::getProjectionMatrix() const {
    float aspect = static_cast<float>(m_viewportWidth) / static_cast<float>(m_viewportHeight);
    float scale = 1.0f / m_zoom;
    return glm::ortho(-scale * aspect, scale * aspect, -scale, scale, -1.0f, 1.0f);
}

glm::vec2 Camera::screenToWorld(const glm::vec2& screenPos) const {
    // Convert from screen coordinates to normalized device coordinates
    float x = (2.0f * screenPos.x) / m_viewportWidth - 1.0f;
    float y = 1.0f - (2.0f * screenPos.y) / m_viewportHeight;
    
    // Apply inverse transformations
    float aspect = static_cast<float>(m_viewportWidth) / static_cast<float>(m_viewportHeight);
    float scale = 1.0f / m_zoom;
    
    glm::vec2 worldPos;
    worldPos.x = x * scale * aspect + m_position.x;
    worldPos.y = y * scale + m_position.y;
    
    return worldPos;
}

glm::vec2 Camera::worldToScreen(const glm::vec2& worldPos) const {
    float aspect = static_cast<float>(m_viewportWidth) / static_cast<float>(m_viewportHeight);
    float scale = 1.0f / m_zoom;
    
    // Apply transformations
    float x = (worldPos.x - m_position.x) / (scale * aspect);
    float y = (worldPos.y - m_position.y) / scale;
    
    // Convert to screen coordinates
    glm::vec2 screenPos;
    screenPos.x = (x + 1.0f) * m_viewportWidth * 0.5f;
    screenPos.y = (1.0f - y) * m_viewportHeight * 0.5f;
    
    return screenPos;
}

} // namespace pendulum
