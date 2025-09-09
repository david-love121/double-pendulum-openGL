# Tasks: Double Pendulum OpenGL Visualization

**Input**: Design documents from `/home/david/repos/spec-kit-testing/three-js-pendulum/specs/001-build-an-opengl/`
**Prerequisites**: plan.md (required), research.md, data-model.md, contracts/

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → Extracted: C++17/20, OpenGL 4.6, CUDA 12.x, 4 libraries architecture
   → Structure: Single project with src/, tests/, external/, shaders/
2. Load optional design documents:
   → data-model.md: 6 entities (PendulumConfiguration, SimulationState, etc.)
   → contracts/: 3 interface files (physics, rendering, ui)
   → research.md: Technology decisions, build system (CMake)
3. Generate tasks by category:
   → Setup: CMake, dependencies, CUDA setup, project structure
   → Tests: contract tests for 3 libraries, integration tests
   → Core: 6 data models, 4 libraries implementation
   → Integration: CUDA-OpenGL interop, application assembly
   → Polish: unit tests, performance validation, quickstart verification
4. Apply task rules:
   → Different files = mark [P] for parallel
   → Same file = sequential (no [P])
   → Tests before implementation (TDD)
5. Number tasks sequentially (T001, T002...)
6. Generate dependency graph: Setup → Tests → Models → Libraries → Integration → Polish
7. Create parallel execution examples for contract tests and data models
8. Validate task completeness:
   → All 3 contracts have tests? YES
   → All 6 entities have models? YES  
   → All 4 libraries implemented? YES
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
- **Single C++ project**: `src/`, `tests/`, `include/`, `external/` at repository root
- CMake build system with CUDA support
- Separate libraries: physics, rendering, ui, analysis

## Phase 3.1: Setup
- [ ] T001 Create project structure with src/{core,physics,rendering,ui,analysis}, tests/{contract,integration,unit}, include/, external/, shaders/, config/ directories
- [ ] T002 Initialize CMakeLists.txt with C++17, OpenGL 4.6, CUDA 12.x support and find_package for GLFW3, GLEW, GLM, ImGui, GTest
- [ ] T003 [P] Configure git submodules for ImGui in external/imgui/ and initialize
- [ ] T004 [P] Create default.json configuration file in config/ with pendulum parameters, grid resolution, and render settings
- [ ] T005 [P] Set up .gitignore for C++ project with build/, CMakeCache.txt, and IDE files

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3

**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**
- [ ] T006 [P] Physics contract test: simulate_step() energy conservation in tests/contract/test_physics_interface.cpp
- [ ] T007 [P] Physics contract test: analyze_grid() CUDA parallel computation in tests/contract/test_physics_interface.cpp  
- [ ] T008 [P] Rendering contract test: render_simulation() OpenGL drawing in tests/contract/test_rendering_interface.cpp
- [ ] T009 [P] Rendering contract test: render_analysis() texture/color grid in tests/contract/test_rendering_interface.cpp
- [ ] T010 [P] UI contract test: ViewManager update() and event handling in tests/contract/test_ui_interface.cpp
- [ ] T011 [P] UI contract test: ImGuiWrapper OpenGL backend initialization in tests/contract/test_ui_interface.cpp
- [ ] T012 [P] Integration test: Simulation view with pendulum animation and labels in tests/integration/test_simulation_view.cpp
- [ ] T013 [P] Integration test: Analysis view with chaos grid computation and display in tests/integration/test_analysis_view.cpp
- [ ] T014 [P] Integration test: CUDA-OpenGL interoperability data sharing in tests/integration/test_cuda_opengl_interop.cpp

## Phase 3.3: Core Data Models (ONLY after tests are failing)
- [ ] T015 [P] PendulumConfiguration struct with validation in src/physics/PendulumState.h
- [ ] T016 [P] SimulationState struct with energy computation in src/physics/PendulumState.h
- [ ] T017 [P] ChaosAnalysisPoint struct with color mapping in src/analysis/ChaosAnalyzer.h
- [ ] T018 [P] ChaosAnalysisGrid container class in src/analysis/ChaosAnalyzer.h
- [ ] T019 [P] RenderingPrimitive struct for OpenGL geometry in src/rendering/Primitives.h
- [ ] T020 [P] ViewConfiguration struct for UI state management in src/ui/ViewManager.h

## Phase 3.4: Core Library Implementation (after data models)
- [ ] T021 [P] PhysicsEngine class with Runge-Kutta integration in src/physics/LagrangianSolver.cpp
- [ ] T022 [P] CUDA kernel implementation for parallel pendulum simulation in src/physics/cuda/pendulum_kernels.cu
- [ ] T023 [P] KernelLauncher class for CUDA memory management in src/physics/cuda/KernelLauncher.cpp
- [ ] T024 [P] ChaosAnalyzer class with Lyapunov exponent computation in src/analysis/ChaosAnalyzer.cpp
- [ ] T025 [P] ColorMapper class for chaos visualization in src/analysis/ColorMapper.cpp
- [ ] T026 Renderer class with OpenGL state management in src/rendering/Renderer.cpp
- [ ] T027 Shader class for GLSL compilation and uniform handling in src/rendering/Shader.cpp
- [ ] T028 Camera class for 2D view transformations in src/rendering/Camera.cpp
- [ ] T029 TextRenderer class with font atlas and Unicode support in src/rendering/TextRenderer.cpp
- [ ] T030 ViewManager class for UI state and view switching in src/ui/ViewManager.cpp
- [ ] T031 ImGuiWrapper class with OpenGL3 backend setup in src/ui/ImGuiWrapper.cpp
- [ ] T032 SimulationView class for pendulum animation controls in src/ui/SimulationView.cpp
- [ ] T033 AnalysisView class for chaos analysis interface in src/ui/AnalysisView.cpp

## Phase 3.5: Application Integration
- [ ] T034 Application class with GLFW window management in src/core/Application.cpp
- [ ] T035 Window class with OpenGL context creation in src/core/Window.cpp
- [ ] T036 Input class for keyboard/mouse handling in src/core/Input.cpp
- [ ] T037 Main application entry point with library integration in src/main.cpp
- [ ] T038 GLSL vertex shader for pendulum rendering in shaders/pendulum.vert
- [ ] T039 GLSL fragment shader for pendulum coloring in shaders/pendulum.frag
- [ ] T040 GLSL shaders for UI text and widgets in shaders/ui.vert and shaders/ui.frag

## Phase 3.6: Performance and Integration Testing
- [ ] T041 CUDA-OpenGL buffer sharing for zero-copy chaos data in src/analysis/ChaosAnalyzer.cpp
- [ ] T042 Memory pool management for GPU allocations in src/physics/cuda/KernelLauncher.cpp
- [ ] T043 Asynchronous CUDA streams for non-blocking computation in src/analysis/ChaosAnalyzer.cpp
- [ ] T044 OpenGL error checking and validation throughout rendering pipeline
- [ ] T045 Performance benchmarking: 60fps simulation and <5s chaos analysis validation

## Phase 3.7: Polish and Validation
- [ ] T046 [P] Unit tests for PendulumConfiguration validation in tests/unit/test_pendulum_config.cpp
- [ ] T047 [P] Unit tests for chaos analysis algorithms in tests/unit/test_chaos_analyzer.cpp
- [ ] T048 [P] Unit tests for rendering primitive generation in tests/unit/test_primitives.cpp
- [ ] T049 [P] Unit tests for UI parameter validation in tests/unit/test_ui_validation.cpp
- [ ] T050 Execute quickstart.md build and validation steps
- [ ] T051 Performance optimization: GPU memory usage and framerate analysis
- [ ] T052 Update include/PendulumVisualizer.h with public API documentation
- [ ] T053 Code cleanup: remove duplication and optimize shader compilation

## Dependencies
```
Setup (T001-T005) → Tests (T006-T014) → Models (T015-T020) → Libraries (T021-T033) → Integration (T034-T045) → Polish (T046-T053)

Specific dependencies:
- T021-T024 (Physics) requires T015-T016 (PendulumConfiguration, SimulationState)
- T025 (ColorMapper) requires T017-T018 (ChaosAnalysisPoint, ChaosAnalysisGrid)
- T026-T029 (Rendering) requires T019 (RenderingPrimitive)
- T030-T033 (UI) requires T020 (ViewConfiguration)
- T034-T037 (Application) requires all libraries (T021-T033)
- T041-T043 (CUDA-OpenGL) requires T022-T024 (CUDA kernels) and T026 (Renderer)
- T044-T045 (Performance) requires complete integration (T034-T040)
```

## Parallel Execution Examples

### Phase 3.2: Contract Tests (T006-T014)
```bash
# All contract tests can run in parallel (different test files):
Task: "Physics contract test: simulate_step() energy conservation in tests/contract/test_physics_interface.cpp"
Task: "Physics contract test: analyze_grid() CUDA parallel computation in tests/contract/test_physics_interface.cpp"
Task: "Rendering contract test: render_simulation() OpenGL drawing in tests/contract/test_rendering_interface.cpp"
Task: "Rendering contract test: render_analysis() texture/color grid in tests/contract/test_rendering_interface.cpp"
Task: "UI contract test: ViewManager update() and event handling in tests/contract/test_ui_interface.cpp"
```

### Phase 3.3: Data Models (T015-T020)
```bash
# All data models can be created in parallel (different header files):
Task: "PendulumConfiguration struct with validation in src/physics/PendulumState.h"
Task: "ChaosAnalysisPoint struct with color mapping in src/analysis/ChaosAnalyzer.h"
Task: "RenderingPrimitive struct for OpenGL geometry in src/rendering/Primitives.h"
Task: "ViewConfiguration struct for UI state management in src/ui/ViewManager.h"
```

### Phase 3.4: Library Implementation (T021-T025, T026-T033 partially)
```bash
# Independent library components can run in parallel:
Task: "PhysicsEngine class with Runge-Kutta integration in src/physics/LagrangianSolver.cpp"
Task: "CUDA kernel implementation for parallel pendulum simulation in src/physics/cuda/pendulum_kernels.cu"
Task: "ChaosAnalyzer class with Lyapunov exponent computation in src/analysis/ChaosAnalyzer.cpp"
Task: "ColorMapper class for chaos visualization in src/analysis/ColorMapper.cpp"
```

## Notes
- [P] tasks = different files, no dependencies between them
- Verify all tests fail before implementing (RED phase of TDD)
- Each library contract test must validate the specific interface behavior
- CUDA tests require NVIDIA GPU - provide CPU fallback for CI/CD
- OpenGL tests require graphics context - use headless testing where possible
- Commit after each completed task for incremental progress tracking

## Task Generation Rules Applied

1. **From Contracts** (physics-interface.md, rendering-interface.md, ui-interface.md):
   - 3 contract files → 6 contract test tasks [P] (T006-T011)
   - Interface methods → library implementation tasks (T021-T033)
   
2. **From Data Model** (data-model.md):
   - 6 entities → 6 model creation tasks [P] (T015-T020)
   - Entity relationships → service integration tasks (T041-T043)
   
3. **From Quickstart** (quickstart.md):
   - Build instructions → setup tasks (T001-T005)
   - Validation steps → integration tests (T012-T014) and final validation (T050)
   - Performance targets → benchmark tasks (T045, T051)

4. **Ordering**:
   - Setup → Tests → Models → Libraries → Integration → Polish
   - TDD enforced: Tests (T006-T014) must complete before implementation (T021+)
   - Dependencies respected: Models before libraries, libraries before integration

## Validation Checklist
*GATE: Checked before task execution*

- [x] All 3 contracts have corresponding tests (T006-T011)
- [x] All 6 entities have model tasks (T015-T020)
- [x] All tests come before implementation (T006-T014 before T021+)
- [x] Parallel tasks truly independent (different files, no shared state)
- [x] Each task specifies exact file path for implementation
- [x] No task modifies same file as another [P] task
- [x] CUDA and OpenGL integration properly sequenced
- [x] Performance and validation tasks cover quickstart requirements
