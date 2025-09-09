#include "UI.h"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>
#include <iostream>

namespace pendulum {

UIManager::UIManager() {
    // Constructor
}

UIManager::~UIManager() {
    cleanup();
}

bool UIManager::initialize() {
    // Setup ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    
    // Configure fonts for better rendering
    io.Fonts->Clear();
    
    // Try to load system font (Noto)
    ImFont* font = nullptr;
    const char* font_paths[] = {
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/TTF/NotoSans-Regular.ttf",
        "/system/fonts/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"
    };
    
    for (const char* path : font_paths) {
        font = io.Fonts->AddFontFromFileTTF(path, 16.0f);
        if (font) {
            std::cout << "✓ Loaded font: " << path << std::endl;
            break;
        }
    }
    
    if (!font) {
        std::cout << "⚠ Could not load system font, using default ImGui font" << std::endl;
        io.Fonts->AddFontDefault();
    }
    
    // Font atlas will be built automatically by ImGui backend
    // Don't call io.Fonts->Build() manually with newer backends
    
    // Setup style
    ImGui::StyleColorsDark();
    
    // Improve font rendering quality
    ImGuiStyle& style = ImGui::GetStyle();
    style.AntiAliasedLines = true;
    style.AntiAliasedLinesUseTex = true;
    style.AntiAliasedFill = true;
    
    return true;
}

void UIManager::cleanup() {
    // Don't shutdown ImGui backends here - they're managed by Application
    // Only cleanup the context
    ImGui::DestroyContext();
}

UIEvents UIManager::update(const UIContext& context) {
    UIEvents events;
    
    // Start new frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    
    // Main menu bar
    renderMenuBar(events, context);
    
    // View-specific controls
    if (context.current_view == 0) {
        renderSimulationControls(events, context);
    } else {
        renderAnalysisControls(events, context);
    }
    
    // Status bar
    renderStatusBar(context);
    
    // Demo window for debugging
    if (m_showDemo) {
        ImGui::ShowDemoWindow(&m_showDemo);
    }
    
    return events;
}

void UIManager::render() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void UIManager::renderMenuBar(UIEvents& events, const UIContext& context) {
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("View")) {
            if (ImGui::MenuItem("Simulation", "1", context.current_view == 0)) {
                events.view_changed = true;
                events.new_view_id = 0;
            }
            if (ImGui::MenuItem("Analysis", "2", context.current_view == 1)) {
                events.view_changed = true;
                events.new_view_id = 1;
            }
            ImGui::EndMenu();
        }
        
        if (ImGui::BeginMenu("Simulation")) {
            if (ImGui::MenuItem("Reset", "R")) {
                events.simulation_reset = true;
            }
            if (ImGui::MenuItem(context.simulation_running ? "Pause" : "Play", "Space")) {
                events.simulation_play_pause = true;
            }
            ImGui::EndMenu();
        }
        
        if (ImGui::BeginMenu("Help")) {
            if (ImGui::MenuItem("Demo Window")) {
                m_showDemo = !m_showDemo;
            }
            ImGui::EndMenu();
        }
        
        ImGui::EndMainMenuBar();
    }
}

void UIManager::renderSimulationControls(UIEvents& events, const UIContext& context) {
    ImGui::Begin("Simulation Controls");
    
    // Pendulum parameters
    ImGui::SeparatorText("Pendulum Parameters");
    
    float l1 = static_cast<float>(m_simulationParams.l1);
    if (ImGui::SliderFloat("Length L1", &l1, 0.1f, 3.0f, "%.2f m")) {
        m_simulationParams.l1 = l1;
        events.parameters_changed = true;
    }
    
    float l2 = static_cast<float>(m_simulationParams.l2);
    if (ImGui::SliderFloat("Length L2", &l2, 0.1f, 3.0f, "%.2f m")) {
        m_simulationParams.l2 = l2;
        events.parameters_changed = true;
    }
    
    float m1 = static_cast<float>(m_simulationParams.m1);
    if (ImGui::SliderFloat("Mass M1", &m1, 0.1f, 5.0f, "%.2f kg")) {
        m_simulationParams.m1 = m1;
        events.parameters_changed = true;
    }
    
    float m2 = static_cast<float>(m_simulationParams.m2);
    if (ImGui::SliderFloat("Mass M2", &m2, 0.1f, 5.0f, "%.2f kg")) {
        m_simulationParams.m2 = m2;
        events.parameters_changed = true;
    }
    
    // Initial conditions
    ImGui::SeparatorText("Initial Conditions");
    
    float theta1_0 = static_cast<float>(m_simulationParams.theta1_0);
    if (ImGui::SliderFloat("Theta1", &theta1_0, -M_PI, M_PI, "%.3f rad")) {
        m_simulationParams.theta1_0 = theta1_0;
        events.parameters_changed = true;
    }
    
    float theta2_0 = static_cast<float>(m_simulationParams.theta2_0);
    if (ImGui::SliderFloat("Theta2", &theta2_0, -M_PI, M_PI, "%.3f rad")) {
        m_simulationParams.theta2_0 = theta2_0;
        events.parameters_changed = true;
    }
    
    float omega1_0 = static_cast<float>(m_simulationParams.omega1_0);
    if (ImGui::SliderFloat("Omega1", &omega1_0, -5.0f, 5.0f, "%.3f rad/s")) {
        m_simulationParams.omega1_0 = omega1_0;
        events.parameters_changed = true;
    }
    
    float omega2_0 = static_cast<float>(m_simulationParams.omega2_0);
    if (ImGui::SliderFloat("Omega2", &omega2_0, -5.0f, 5.0f, "%.3f rad/s")) {
        m_simulationParams.omega2_0 = omega2_0;
        events.parameters_changed = true;
    }
    
    float damping = static_cast<float>(m_simulationParams.damping);
    if (ImGui::SliderFloat("Damping", &damping, 0.0f, 1.0f, "%.3f")) {
        m_simulationParams.damping = damping;
        events.parameters_changed = true;
    }
    
    // Display options
    ImGui::SeparatorText("Display Options");
    
    ImGui::Checkbox("Show Labels", &m_simulationParams.show_labels);
    ImGui::Checkbox("Show Trails", &m_simulationParams.show_trails);
    
    // Trail fade time control (only show when trails are enabled)
    if (m_simulationParams.show_trails) {
        ImGui::Indent();
        if (ImGui::Checkbox("Trail Fade", &m_simulationParams.trail_fade_enabled)) {
            // Toggle happened, trails will be affected on next frame
        }
        
        // Trail fade time control (only show when fade is enabled)
        if (m_simulationParams.trail_fade_enabled) {
            float trailFadeTime = static_cast<float>(m_simulationParams.trail_fade_time);
            if (ImGui::SliderFloat("Fade Time (s)", &trailFadeTime, 0.5f, 20.0f, "%.1f")) {
                m_simulationParams.trail_fade_time = static_cast<double>(trailFadeTime);
                // Clamp to reasonable bounds
                if (m_simulationParams.trail_fade_time < 0.1) {
                    m_simulationParams.trail_fade_time = 0.1;
                }
            }
        } else {
            ImGui::TextDisabled("Trail will persist indefinitely");
            ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Warning: May use significant memory for long simulations");
        }
        ImGui::Unindent();
    }
    
    // Control buttons
    ImGui::SeparatorText("Controls");
    
    if (ImGui::Button(context.simulation_running ? "Pause" : "Play")) {
        events.simulation_play_pause = true;
    }
    
    ImGui::SameLine();
    if (ImGui::Button("Reset")) {
        events.simulation_reset = true;
    }
    
    ImGui::End();
}

void UIManager::renderAnalysisControls(UIEvents& events, const UIContext& context) {
    ImGui::Begin("Analysis Controls");
    
    // Grid parameters
    ImGui::SeparatorText("Grid Parameters");
    
    if (ImGui::SliderInt("Resolution X", &m_analysisParams.grid_resolution_x, 64, 1024)) {
        // Grid resolution changed
    }
    
    if (ImGui::SliderInt("Resolution Y", &m_analysisParams.grid_resolution_y, 64, 1024)) {
        // Grid resolution changed
    }
    
    // Angle ranges
    ImGui::SeparatorText("Angle Ranges");
    
    float theta1_min = static_cast<float>(m_analysisParams.theta1_min);
    ImGui::SliderFloat("Theta1 Min", &theta1_min, -M_PI, M_PI, "%.3f");
    m_analysisParams.theta1_min = theta1_min;
    
    float theta1_max = static_cast<float>(m_analysisParams.theta1_max);
    ImGui::SliderFloat("Theta1 Max", &theta1_max, -M_PI, M_PI, "%.3f");
    m_analysisParams.theta1_max = theta1_max;
    
    float theta2_min = static_cast<float>(m_analysisParams.theta2_min);
    ImGui::SliderFloat("Theta2 Min", &theta2_min, -M_PI, M_PI, "%.3f");
    m_analysisParams.theta2_min = theta2_min;
    
    float theta2_max = static_cast<float>(m_analysisParams.theta2_max);
    ImGui::SliderFloat("Theta2 Max", &theta2_max, -M_PI, M_PI, "%.3f");
    m_analysisParams.theta2_max = theta2_max;
    
    // Analysis parameters
    ImGui::SeparatorText("Analysis Parameters");
    
    float integration_time = static_cast<float>(m_analysisParams.integration_time);
    ImGui::SliderFloat("Integration Time", &integration_time, 1.0f, 20.0f, "%.1f s");
    m_analysisParams.integration_time = integration_time;
    
    const char* colorSchemes[] = {"Blue-Red", "Viridis", "Plasma"};
    ImGui::Combo("Color Scheme", &m_analysisParams.color_scheme, colorSchemes, 3);
    
    // Control buttons
    ImGui::SeparatorText("Controls");
    
    if (context.analysis_computing) {
        ImGui::ProgressBar(context.computation_progress, ImVec2(-1, 0));
        if (ImGui::Button("Stop Analysis")) {
            // Stop analysis (not implemented yet)
        }
    } else {
        if (ImGui::Button("Start Analysis")) {
            events.analysis_start = true;
        }
    }
    
    ImGui::End();
}

void UIManager::renderStatusBar(const UIContext& context) {
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImVec2 work_pos = viewport->WorkPos;
    ImVec2 work_size = viewport->WorkSize;
    
    ImGui::SetNextWindowPos(ImVec2(work_pos.x, work_pos.y + work_size.y - 30));
    ImGui::SetNextWindowSize(ImVec2(work_size.x, 30));
    
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | 
                            ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings;
    
    if (ImGui::Begin("StatusBar", nullptr, flags)) {
        const char* viewName = (context.current_view == 0) ? "Simulation" : "Analysis";
        ImGui::Text("View: %s", viewName);
        
        ImGui::SameLine(200);
        if (context.simulation_running) {
            ImGui::Text("Status: Running");
        } else {
            ImGui::Text("Status: Paused");
        }
        
        if (context.analysis_computing) {
            ImGui::SameLine(400);
            ImGui::Text("Computing: %.1f%%", context.computation_progress * 100.0);
        }
    }
    ImGui::End();
}

void UIManager::onWindowResize(int width, int height) {
    m_windowWidth = width;
    m_windowHeight = height;
}

} // namespace pendulum
