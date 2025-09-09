#include "Rendering.h"
#include <iostream>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>

namespace pendulum {

// Static member definitions
bool Primitives::s_initialized = false;
GLuint Primitives::s_lineVAO = 0;
GLuint Primitives::s_lineVBO = 0;
GLuint Primitives::s_circleVAO = 0;
GLuint Primitives::s_circleVBO = 0;
GLuint Primitives::s_circleEBO = 0;
std::unique_ptr<Shader> Primitives::s_lineShader;
std::unique_ptr<Shader> Primitives::s_circleShader;

// Current matrices for rendering
static glm::mat4 s_currentView = glm::mat4(1.0f);
static glm::mat4 s_currentProjection = glm::mat4(1.0f);

void Primitives::initialize() {
    if (s_initialized) return;
    
    initializeLineRenderer();
    initializeCircleRenderer();
    initializeTextRenderer();
    
    s_initialized = true;
}

void Primitives::cleanup() {
    if (!s_initialized) return;
    
    if (s_lineVAO) {
        glDeleteVertexArrays(1, &s_lineVAO);
        glDeleteBuffers(1, &s_lineVBO);
    }
    
    if (s_circleVAO) {
        glDeleteVertexArrays(1, &s_circleVAO);
        glDeleteBuffers(1, &s_circleVBO);
        glDeleteBuffers(1, &s_circleEBO);
    }
    
    s_lineShader.reset();
    s_circleShader.reset();
    
    s_initialized = false;
}

void Primitives::setMatrices(const glm::mat4& view, const glm::mat4& projection) {
    s_currentView = view;
    s_currentProjection = projection;
}

void Primitives::initializeLineRenderer() {
    // Line shader
    s_lineShader = std::make_unique<Shader>();
    
    std::string vertexSource = R"(
        #version 460 core
        layout (location = 0) in vec2 position;
        uniform mat4 view;
        uniform mat4 projection;
        void main() {
            gl_Position = projection * view * vec4(position, 0.0, 1.0);
        }
    )";
    
    std::string fragmentSource = R"(
        #version 460 core
        uniform vec4 color;
        out vec4 FragColor;
        void main() {
            FragColor = color;
        }
    )";
    
    if (!s_lineShader->loadFromSource(vertexSource, fragmentSource)) {
        std::cerr << "Failed to load line shader" << std::endl;
    }
    
    // Line VAO/VBO
    glGenVertexArrays(1, &s_lineVAO);
    glGenBuffers(1, &s_lineVBO);
    
    glBindVertexArray(s_lineVAO);
    glBindBuffer(GL_ARRAY_BUFFER, s_lineVBO);
    glBufferData(GL_ARRAY_BUFFER, 4 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    glBindVertexArray(0);
}

void Primitives::initializeCircleRenderer() {
    // Circle shader with texture coordinates
    s_circleShader = std::make_unique<Shader>();
    
    std::string vertexSource = R"(
        #version 460 core
        layout (location = 0) in vec2 position;
        layout (location = 1) in vec2 texCoord;
        uniform mat4 view;
        uniform mat4 projection;
        out vec2 TexCoord;
        void main() {
            TexCoord = texCoord;
            gl_Position = projection * view * vec4(position, 0.0, 1.0);
        }
    )";
    
    std::string fragmentSource = R"(
        #version 460 core
        in vec2 TexCoord;
        uniform vec4 color;
        out vec4 FragColor;
        void main() {
            vec2 center = vec2(0.5, 0.5);
            float dist = distance(TexCoord, center);
            if (dist > 0.5) discard;
            FragColor = color;
        }
    )";
    
    if (!s_circleShader->loadFromSource(vertexSource, fragmentSource)) {
        std::cerr << "Failed to load circle shader" << std::endl;
    }
    
    // Circle quad vertices (position + texcoord)
    float vertices[] = {
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f,
        -1.0f,  1.0f,  0.0f, 1.0f
    };
    
    unsigned int indices[] = {
        0, 1, 2,
        2, 3, 0
    };
    
    glGenVertexArrays(1, &s_circleVAO);
    glGenBuffers(1, &s_circleVBO);
    glGenBuffers(1, &s_circleEBO);
    
    glBindVertexArray(s_circleVAO);
    
    glBindBuffer(GL_ARRAY_BUFFER, s_circleVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, s_circleEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    
    // Position
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Texture coordinates
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    glBindVertexArray(0);
}

void Primitives::initializeTextRenderer() {
    // Text rendering will be handled by ImGui for now
}

void Primitives::drawLine(const glm::vec2& start, const glm::vec2& end, 
                         const glm::vec4& color, float thickness) {
    if (!s_initialized || !s_lineShader) return;
    
    float vertices[] = {
        start.x, start.y,
        end.x, end.y
    };
    
    glBindVertexArray(s_lineVAO);
    glBindBuffer(GL_ARRAY_BUFFER, s_lineVBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
    
    s_lineShader->use();
    s_lineShader->setUniform("view", s_currentView);
    s_lineShader->setUniform("projection", s_currentProjection);
    s_lineShader->setUniform("color", color);
    
    glLineWidth(thickness);
    glDrawArrays(GL_LINES, 0, 2);
    
    glBindVertexArray(0);
}

void Primitives::drawCircle(const glm::vec2& center, float radius, 
                           const glm::vec4& color, bool filled) {
    if (!s_initialized || !s_circleShader) return;
    
    // Create transformation matrix for circle positioning and scaling
    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3(center, 0.0f));
    model = glm::scale(model, glm::vec3(radius, radius, 1.0f));
    
    glm::mat4 mvp = s_currentProjection * s_currentView * model;
    
    s_circleShader->use();
    s_circleShader->setUniform("view", glm::mat4(1.0f));
    s_circleShader->setUniform("projection", mvp);
    s_circleShader->setUniform("color", color);
    
    glBindVertexArray(s_circleVAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
}

void Primitives::drawArc(const glm::vec2& center, float radius, 
                        float startAngle, float endAngle, 
                        const glm::vec4& color, float thickness) {
    if (!s_initialized || !s_lineShader) return;
    
    // Draw arc as series of line segments
    int segments = 32;
    float angleStep = (endAngle - startAngle) / segments;
    
    s_lineShader->use();
    s_lineShader->setUniform("view", s_currentView);
    s_lineShader->setUniform("projection", s_currentProjection);
    s_lineShader->setUniform("color", color);
    
    glLineWidth(thickness);
    glBindVertexArray(s_lineVAO);
    
    for (int i = 0; i < segments; ++i) {
        float angle1 = startAngle + i * angleStep;
        float angle2 = startAngle + (i + 1) * angleStep;
        
        glm::vec2 p1 = center + radius * glm::vec2(sin(angle1), -cos(angle1));
        glm::vec2 p2 = center + radius * glm::vec2(sin(angle2), -cos(angle2));
        
        float vertices[] = { p1.x, p1.y, p2.x, p2.y };
        
        glBindBuffer(GL_ARRAY_BUFFER, s_lineVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
        glDrawArrays(GL_LINES, 0, 2);
    }
    
    glBindVertexArray(0);
}

void Primitives::drawText(const std::string& text, const glm::vec2& position, 
                         const glm::vec4& color, float scale) {
    // Text rendering will be handled by ImGui overlays for now
    // In a full implementation, this would use a font atlas and text shaders
}

} // namespace pendulum
