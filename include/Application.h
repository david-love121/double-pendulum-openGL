#pragma once

#include "PendulumState.h"
#include "LagrangianSolver.h"
#include "Rendering.h"
#include "UI.h"
#include "ChaosAnalysis.h"
#include "TrailRenderer.h"
#include <memory>
#include <GLFW/glfw3.h>

namespace pendulum {

/**
 * Main application class coordinating all subsystems
 */
class Application {
public:
    Application();
    ~Application();
    
    bool initialize(int argc, char* argv[]);
    void run();
    void cleanup();
    
    // Configuration
    struct Config {
        int window_width = 1200;
        int window_height = 800;
        bool fullscreen = false;
        bool vsync = true;
        int cuda_device = 0;
        std::string config_file = "config/default.json";
    };

private:
    bool initializeWindow();
    bool initializeOpenGL();
    bool initializeSubsystems();
    void parseCommandLine(int argc, char* argv[]);
    bool loadConfiguration();
    
    void update(double deltaTime);
    void render();
    void handleUIEvents(const UIEvents& events);
    
    // GLFW callbacks
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    
    // Application state
    Config m_config;
    GLFWwindow* m_window = nullptr;
    bool m_running = false;
    double m_lastFrameTime = 0.0;
    
    // Subsystems
    std::unique_ptr<LagrangianSolver> m_physicsSolver;
    std::unique_ptr<Camera> m_camera;
    std::unique_ptr<UIManager> m_uiManager;
    std::unique_ptr<ChaosAnalysisGrid> m_chaosGrid;
    
    // Trail rendering
    TrailRenderer m_bob1Trail;  // First pendulum bob trail
    TrailRenderer m_bob2Trail;  // Second pendulum bob trail
    
    // Current state
    PendulumConfiguration m_pendulumConfig;
    SimulationState m_currentState;
    SimulationStatus m_simulationStatus = SimulationStatus::INITIAL;
    int m_currentView = 0; // 0=simulation, 1=analysis
    
    // Timing
    double m_physicsAccumulator = 0.0;
    const double m_physicsTimestep = 0.001; // 1000Hz physics
    const double m_maxFrameTime = 0.05; // Cap at 20fps minimum
};

} // namespace pendulum
