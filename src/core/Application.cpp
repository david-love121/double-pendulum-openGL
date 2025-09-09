#include "Application.h"
#include "Rendering.h"
#include <GL/glew.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <iostream>
#include <cstring>
#include <chrono>
#include <algorithm>

namespace pendulum {

Application::Application() {
    // Constructor
}

Application::~Application() {
    cleanup();
}

bool Application::initialize(int argc, char* argv[]) {
    parseCommandLine(argc, argv);
    
    if (!initializeWindow()) {
        return false;
    }
    
    if (!initializeOpenGL()) {
        return false;
    }
    
    if (!initializeSubsystems()) {
        return false;
    }
    
    if (!loadConfiguration()) {
        return false;
    }
    
    m_running = true;
    return true;
}

void Application::parseCommandLine(int argc, char* argv[]) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--fullscreen") == 0) {
            m_config.fullscreen = true;
        } else if (strcmp(argv[i], "--no-vsync") == 0) {
            m_config.vsync = false;
        } else if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
            m_config.window_width = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc) {
            m_config.window_height = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            m_config.config_file = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0) {
            std::cout << "Double Pendulum Visualizer v0.1.0\n";
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --width WIDTH        Window width (default: 1200)\n";
            std::cout << "  --height HEIGHT      Window height (default: 800)\n";
            std::cout << "  --fullscreen         Start in fullscreen mode\n";
            std::cout << "  --no-vsync           Disable vertical sync\n";
            std::cout << "  --config FILE        Load configuration file\n";
            std::cout << "  --help               Show this help\n";
            exit(0);
        }
    }
}

bool Application::initializeWindow() {
    // Force Wayland platform - no X11 fallback
    glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_WAYLAND);
    
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW with Wayland" << std::endl;
        std::cerr << "Make sure you're running in a Wayland session" << std::endl;
        return false;
    }
    
    // Verify we're using Wayland
    if (glfwGetPlatform() != GLFW_PLATFORM_WAYLAND) {
        std::cerr << "Error: GLFW did not initialize with Wayland platform" << std::endl;
        glfwTerminate();
        return false;
    }
    
    std::cout << "✓ GLFW initialized with Wayland platform" << std::endl;
    
    // OpenGL context hints
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    
    // Request discrete GPU (NVIDIA) for better performance
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_FALSE);
    glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
    
    // Create window
    GLFWmonitor* monitor = m_config.fullscreen ? glfwGetPrimaryMonitor() : nullptr;
    m_window = glfwCreateWindow(m_config.window_width, m_config.window_height, 
                               "Double Pendulum Visualizer v0.1.0", monitor, nullptr);
    
    if (!m_window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }
    
    glfwMakeContextCurrent(m_window);
    
    // Set callbacks
    glfwSetWindowUserPointer(m_window, this);
    glfwSetKeyCallback(m_window, keyCallback);
    glfwSetMouseButtonCallback(m_window, mouseButtonCallback);
    glfwSetScrollCallback(m_window, scrollCallback);
    glfwSetFramebufferSizeCallback(m_window, framebufferSizeCallback);
    
    // Enable/disable vsync
    glfwSwapInterval(m_config.vsync ? 1 : 0);
    
    return true;
}

bool Application::initializeOpenGL() {
    // Initialize GLEW
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        std::cerr << "GLEW initialization failed: " << glewGetErrorString(err) << std::endl;
        return false;
    }
    
    // Check OpenGL version and GPU info
    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLSL Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
    std::cout << "Vendor: " << glGetString(GL_VENDOR) << std::endl;
    std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;
    
    // Check if we're using NVIDIA GPU
    const char* renderer = (const char*)glGetString(GL_RENDERER);
    if (strstr(renderer, "NVIDIA") != nullptr) {
        std::cout << "✓ NVIDIA GPU detected for rendering!" << std::endl;
    } else if (strstr(renderer, "Intel") != nullptr) {
        std::cout << "⚠ Using Intel integrated graphics. For better performance, use:" << std::endl;
        std::cout << "  prime-run ./pendulum-visualizer" << std::endl;
    } else {
        std::cout << "? Unknown GPU vendor" << std::endl;
    }
    
    // Enable OpenGL features
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    
    // Set clear color
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    
    // Initialize rendering primitives
    Primitives::initialize();
    
    return true;
}

bool Application::initializeSubsystems() {
    // Initialize physics solver
    m_physicsSolver = std::make_unique<LagrangianSolver>();
    
    // Initialize camera
    m_camera = std::make_unique<Camera>();
    int width, height;
    glfwGetFramebufferSize(m_window, &width, &height);
    m_camera->setViewport(width, height);
    m_camera->setZoom(0.3f); // Zoom out to see the pendulum
    
    // Initialize UI
    m_uiManager = std::make_unique<UIManager>();
    if (!m_uiManager->initialize()) {
        std::cerr << "Failed to initialize UI" << std::endl;
        return false;
    }
    
    // Initialize ImGui backends
    if (!ImGui_ImplGlfw_InitForOpenGL(m_window, true)) {
        std::cerr << "Failed to initialize ImGui GLFW backend" << std::endl;
        return false;
    }
    
    if (!ImGui_ImplOpenGL3_Init("#version 460")) {
        std::cerr << "Failed to initialize ImGui OpenGL backend" << std::endl;
        return false;
    }
    
    // Set default pendulum configuration
    m_pendulumConfig.l1 = 1.0;
    m_pendulumConfig.l2 = 1.0;
    m_pendulumConfig.m1 = 1.0;
    m_pendulumConfig.m2 = 1.0;
    m_pendulumConfig.g = 9.81;
    m_pendulumConfig.damping = 0.0;
    
    // Set initial state
    m_currentState.theta1 = 1.0;
    m_currentState.theta2 = 0.0;
    m_currentState.omega1 = 0.0;
    m_currentState.omega2 = 0.0;
    m_currentState.timestamp = 0.0;
    m_currentState.energy = m_currentState.computeEnergy(m_pendulumConfig);
    
    m_simulationStatus = SimulationStatus::INITIAL;
    
    // Initialize trail renderers
    if (!m_bob1Trail.initialize()) {
        std::cerr << "Failed to initialize trail renderer 1" << std::endl;
        return false;
    }
    
    if (!m_bob2Trail.initialize()) {
        std::cerr << "Failed to initialize trail renderer 2" << std::endl;
        return false;
    }
    
    // Configure trail colors
    m_bob1Trail.setColor(1.0f, 0.2f, 0.2f); // Red-ish for first pendulum
    m_bob2Trail.setColor(0.2f, 0.2f, 1.0f); // Blue-ish for second pendulum
    
    // Set reasonable max points for performance
    m_bob1Trail.setMaxPoints(25000);
    m_bob2Trail.setMaxPoints(25000);
    
    return true;
}

bool Application::loadConfiguration() {
    // TODO: Load configuration from JSON file
    return true;
}

void Application::run() {
    m_lastFrameTime = glfwGetTime();
    
    while (!glfwWindowShouldClose(m_window) && m_running) {
        double currentTime = glfwGetTime();
        double deltaTime = currentTime - m_lastFrameTime;
        m_lastFrameTime = currentTime;
        
        // Cap maximum frame time to prevent spiral of death
        if (deltaTime > m_maxFrameTime) {
            deltaTime = m_maxFrameTime;
        }
        
        glfwPollEvents();
        
        update(deltaTime);
        render();
        
        glfwSwapBuffers(m_window);
    }
}

void Application::update(double deltaTime) {
    // Update UI and handle events
    UIContext uiContext;
    uiContext.simulation_running = (m_simulationStatus == SimulationStatus::RUNNING);
    uiContext.analysis_computing = false; // TODO: implement analysis
    uiContext.computation_progress = 0.0;
    uiContext.current_view = m_currentView;
    
    UIEvents events = m_uiManager->update(uiContext);
    handleUIEvents(events);
    
    // Update physics if simulation is running
    if (m_simulationStatus == SimulationStatus::RUNNING) {
        m_physicsAccumulator += deltaTime;
        
        while (m_physicsAccumulator >= m_physicsTimestep) {
            try {
                m_currentState = m_physicsSolver->step(m_currentState, m_pendulumConfig, m_physicsTimestep);
                m_physicsAccumulator -= m_physicsTimestep;
                
                // Record trail points if enabled
                const SimulationParams& params = m_uiManager->getSimulationParams();
                if (params.show_trails) {
                    glm::vec2 bob1 = m_currentState.getBob1Position(m_pendulumConfig);
                    glm::vec2 bob2 = m_currentState.getBob2Position(m_pendulumConfig);
                    
                    // Add points to trail renderers
                    m_bob1Trail.addPoint(bob1.x, bob1.y, static_cast<float>(m_currentState.timestamp));
                    m_bob2Trail.addPoint(bob2.x, bob2.y, static_cast<float>(m_currentState.timestamp));
                    
                    // Update trail fading
                    m_bob1Trail.updateFading(static_cast<float>(m_currentState.timestamp), 
                                           static_cast<float>(params.trail_fade_time), 
                                           params.trail_fade_enabled);
                    m_bob2Trail.updateFading(static_cast<float>(m_currentState.timestamp), 
                                           static_cast<float>(params.trail_fade_time), 
                                           params.trail_fade_enabled);
                }
            } catch (const std::exception& e) {
                std::cerr << "Physics error: " << e.what() << std::endl;
                m_simulationStatus = SimulationStatus::ERROR;
                break;
            }
        }
    }
}

void Application::render() {
    glClear(GL_COLOR_BUFFER_BIT);
    
    if (m_currentView == 0) {
        // Render simulation view
        glm::mat4 viewMatrix = m_camera->getViewMatrix();
        glm::mat4 projMatrix = m_camera->getProjectionMatrix();
        
        // Set matrices for all primitives
        Primitives::setMatrices(viewMatrix, projMatrix);
        
        // Draw pendulum
        glm::vec2 pivot(0.0f, 0.0f);
        glm::vec2 bob1 = m_currentState.getBob1Position(m_pendulumConfig);
        glm::vec2 bob2 = m_currentState.getBob2Position(m_pendulumConfig);
        
        // Draw pendulum arms
        Primitives::drawLine(pivot, bob1, glm::vec4(1.0f, 1.0f, 1.0f, 1.0f), 3.0f);
        Primitives::drawLine(bob1, bob2, glm::vec4(1.0f, 1.0f, 1.0f, 1.0f), 3.0f);
        
        // Draw pivot point
        Primitives::drawCircle(pivot, 0.05f, glm::vec4(0.5f, 0.5f, 0.5f, 1.0f));
        
        // Draw pendulum bobs
        Primitives::drawCircle(bob1, 0.1f, glm::vec4(1.0f, 0.2f, 0.2f, 1.0f));
        Primitives::drawCircle(bob2, 0.1f, glm::vec4(0.2f, 0.2f, 1.0f, 1.0f));
        
        // Draw trails if enabled
        const SimulationParams& params = m_uiManager->getSimulationParams();
        if (params.show_trails) {
            // Render trails using GPU-accelerated renderers with proper transformations
            m_bob1Trail.render(viewMatrix, projMatrix);
            m_bob2Trail.render(viewMatrix, projMatrix);
        }
        
        // Draw angle arcs if labels are enabled
        if (params.show_labels) {
            // Draw theta1 arc
            Primitives::drawArc(pivot, 0.3f, -M_PI/2, -M_PI/2 + m_currentState.theta1, 
                               glm::vec4(1.0f, 0.8f, 0.2f, 0.8f), 2.0f);
            
            // Draw theta2 arc
            Primitives::drawArc(bob1, 0.2f, -M_PI/2, -M_PI/2 + m_currentState.theta2, 
                               glm::vec4(0.2f, 0.8f, 1.0f, 0.8f), 2.0f);
        }
    } else {
        // Render analysis view
        if (m_chaosGrid) {
            // Render chaos analysis as colored points
            glm::mat4 viewMatrix = m_camera->getViewMatrix();
            glm::mat4 projMatrix = m_camera->getProjectionMatrix();
            Primitives::setMatrices(viewMatrix, projMatrix);
            
            // Map angle ranges to screen coordinates
            double theta1_range = m_chaosGrid->getTheta1Max() - m_chaosGrid->getTheta1Min();
            double theta2_range = m_chaosGrid->getTheta2Max() - m_chaosGrid->getTheta2Min();
            
            // Draw grid points
            for (int y = 0; y < m_chaosGrid->getResolutionY(); ++y) {
                for (int x = 0; x < m_chaosGrid->getResolutionX(); ++x) {
                    const ChaosAnalysisPoint& point = m_chaosGrid->getPoint(x, y);
                    
                    if (point.computation_status == ChaosAnalysisPoint::COMPLETE) {
                        // Map initial conditions to screen position
                        float screen_x = -2.0f + 4.0f * (point.initial_theta1 - m_chaosGrid->getTheta1Min()) / theta1_range;
                        float screen_y = -2.0f + 4.0f * (point.initial_theta2 - m_chaosGrid->getTheta2Min()) / theta2_range;
                        
                        glm::vec2 position(screen_x, screen_y);
                        glm::vec4 color(point.color.r, point.color.g, point.color.b, 1.0f);
                        
                        // Draw small circle for each point
                        float point_size = 4.0f / std::max(m_chaosGrid->getResolutionX(), m_chaosGrid->getResolutionY());
                        Primitives::drawCircle(position, point_size, color);
                    }
                }
            }
            
            // Draw axis labels and grid
            // Draw border
            Primitives::drawLine(glm::vec2(-2.0f, -2.0f), glm::vec2(2.0f, -2.0f), 
                               glm::vec4(0.5f, 0.5f, 0.5f, 1.0f), 2.0f);
            Primitives::drawLine(glm::vec2(2.0f, -2.0f), glm::vec2(2.0f, 2.0f), 
                               glm::vec4(0.5f, 0.5f, 0.5f, 1.0f), 2.0f);
            Primitives::drawLine(glm::vec2(2.0f, 2.0f), glm::vec2(-2.0f, 2.0f), 
                               glm::vec4(0.5f, 0.5f, 0.5f, 1.0f), 2.0f);
            Primitives::drawLine(glm::vec2(-2.0f, 2.0f), glm::vec2(-2.0f, -2.0f), 
                               glm::vec4(0.5f, 0.5f, 0.5f, 1.0f), 2.0f);
        } else {
            // No analysis data available, show placeholder
            glm::mat4 viewMatrix = m_camera->getViewMatrix();
            glm::mat4 projMatrix = m_camera->getProjectionMatrix();
            Primitives::setMatrices(viewMatrix, projMatrix);
            
            // Draw placeholder text area
            Primitives::drawCircle(glm::vec2(0.0f, 0.0f), 0.1f, glm::vec4(0.5f, 0.5f, 0.5f, 1.0f));
        }
    }
    
    // Render UI
    m_uiManager->render();
}

void Application::handleUIEvents(const UIEvents& events) {
    if (events.view_changed) {
        m_currentView = events.new_view_id;
    }
    
    if (events.simulation_reset) {
        const SimulationParams& params = m_uiManager->getSimulationParams();
        m_pendulumConfig = params.toConfiguration();
        m_currentState = params.toInitialState();
        m_currentState.energy = m_currentState.computeEnergy(m_pendulumConfig);
        m_physicsSolver->reset();
        m_simulationStatus = SimulationStatus::INITIAL;
        
        // Clear trails
        m_bob1Trail.clear();
        m_bob2Trail.clear();
    }
    
    if (events.simulation_play_pause) {
        if (m_simulationStatus == SimulationStatus::RUNNING) {
            m_simulationStatus = SimulationStatus::PAUSED;
        } else if (m_simulationStatus == SimulationStatus::PAUSED || 
                   m_simulationStatus == SimulationStatus::INITIAL) {
            m_simulationStatus = SimulationStatus::RUNNING;
        }
    }
    
    if (events.parameters_changed) {
        const SimulationParams& params = m_uiManager->getSimulationParams();
        m_pendulumConfig = params.toConfiguration();
        // Don't reset state unless explicitly requested
    }
    
    if (events.analysis_start) {
        std::cout << "Starting chaos analysis..." << std::endl;
        
        // Get analysis parameters
        const AnalysisParams& params = m_uiManager->getAnalysisParams();
        
        // Create or recreate chaos grid
        m_chaosGrid = std::make_unique<ChaosAnalysisGrid>(
            params.grid_resolution_x, params.grid_resolution_y,
            params.theta1_min, params.theta1_max,
            params.theta2_min, params.theta2_max
        );
        
        // Start computation in a separate thread (simplified version)
        // For now, we'll compute it directly but add progress logging
        auto progress_callback = [this](double progress) {
            std::cout << "Chaos analysis progress: " << (progress * 100.0) << "%" << std::endl;
        };
        
        computeChaosAnalysis(m_chaosGrid.get(), m_pendulumConfig,
                           params.integration_time, params.color_scheme,
                           progress_callback);
        
        std::cout << "Chaos analysis completed!" << std::endl;
    }
}

void Application::cleanup() {
    // Cleanup ImGui backends first
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    
    if (m_uiManager) {
        m_uiManager->cleanup();
    }
    
    // Cleanup trail renderers
    m_bob1Trail.cleanup();
    m_bob2Trail.cleanup();
    
    Primitives::cleanup();
    
    if (m_window) {
        glfwDestroyWindow(m_window);
        glfwTerminate();
    }
}

// GLFW Callbacks
void Application::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    Application* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    
    if (action == GLFW_PRESS) {
        switch (key) {
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, GLFW_TRUE);
                break;
            case GLFW_KEY_SPACE:
                // Toggle simulation
                if (app->m_simulationStatus == SimulationStatus::RUNNING) {
                    app->m_simulationStatus = SimulationStatus::PAUSED;
                } else {
                    app->m_simulationStatus = SimulationStatus::RUNNING;
                }
                break;
            case GLFW_KEY_R:
                // Reset simulation
                {
                    app->m_simulationStatus = SimulationStatus::INITIAL;
                    const SimulationParams& params = app->m_uiManager->getSimulationParams();
                    app->m_currentState = params.toInitialState();
                    app->m_physicsSolver->reset();
                }
                break;
            case GLFW_KEY_1:
                app->m_currentView = 0;
                break;
            case GLFW_KEY_2:
                app->m_currentView = 1;
                break;
        }
    }
}

void Application::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    // Handle mouse input for camera control
}

void Application::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    Application* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    
    // Zoom camera
    float currentZoom = app->m_camera->getZoom();
    float newZoom = currentZoom * (1.0f + 0.1f * yoffset);
    newZoom = std::max(0.1f, std::min(5.0f, newZoom));
    app->m_camera->setZoom(newZoom);
}

void Application::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    Application* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    
    glViewport(0, 0, width, height);
    app->m_camera->setViewport(width, height);
    app->m_uiManager->onWindowResize(width, height);
}

} // namespace pendulum
