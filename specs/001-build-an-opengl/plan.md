# Implementation Plan: Double Pendulum OpenGL Visualization

**Branch**: `001-build-an-opengl` | **Date**: September 6, 2025 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/home/david/repos/spec-kit-testing/three-js-pendulum/specs/001-build-an-opengl/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
4. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
5. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file (e.g., `CLAUDE.md` for Claude Code, `.github/copilot-instructions.md` for GitHub Copilot, or `GEMINI.md` for Gemini CLI).
6. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
7. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
8. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary
Native OpenGL application for visualizing double pendulum dynamics with two main views: (1) detailed real-time simulation with Lagrangian mechanics showing labeled pendulum arms and angles, and (2) chaos analysis visualization showing color-coded behavioral patterns across different initial conditions. Built for Arch Linux with Hyprland, leveraging NVIDIA GPU for performance-critical computations.

## Technical Context
**Language/Version**: C++17/20 with CUDA 12.x for GPU acceleration  
**Primary Dependencies**: OpenGL 4.6, GLFW3, GLEW, GLM (math), ImGui (UI), CUDA Runtime  
**Storage**: Configuration files (JSON/YAML), simulation data caching (binary format)  
**Testing**: Google Test (gtest) with OpenGL context mocking  
**Target Platform**: Arch Linux with Hyprland, NVIDIA GPU (RTX series)
**Project Type**: single (native desktop application)  
**Performance Goals**: 60 fps real-time simulation, sub-millisecond chaos analysis computation per point  
**Constraints**: GPU memory efficient, responsive UI during heavy computation, numerical stability  
**Scale/Scope**: 10k+ initial conditions for chaos analysis, real-time physics at 1000Hz internal timestep

**User Implementation Details**: OpenGL native graphical application, C++ core logic, CUDA for parallelized heavy computations, optimized for Arch Linux with Hyprland and NVIDIA graphics card.

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Simplicity**:
- Projects: 1 (single OpenGL application)
- Using framework directly? Yes (direct OpenGL, GLFW, ImGui usage)
- Single data model? Yes (unified pendulum state representation)
- Avoiding patterns? Yes (direct GPU memory management, no unnecessary abstractions)

**Architecture**:
- EVERY feature as library? Yes (physics-lib, rendering-lib, ui-lib, chaos-analysis-lib)
- Libraries listed: 
  - physics-lib: Lagrangian mechanics simulation, CUDA acceleration
  - rendering-lib: OpenGL primitives, shader management, text rendering
  - ui-lib: ImGui wrapper for view switching and controls
  - chaos-analysis-lib: parallel initial condition analysis
- CLI per library: --simulate, --analyze, --render-test, --benchmark
- Library docs: llms.txt format planned? Yes

**Testing (NON-NEGOTIABLE)**:
- RED-GREEN-Refactor cycle enforced? Yes (tests fail first, then implement)
- Git commits show tests before implementation? Yes (enforced via pre-commit)
- Order: Contract→Integration→E2E→Unit strictly followed? Yes
- Real dependencies used? Yes (actual OpenGL context, real GPU)
- Integration tests for: OpenGL context creation, CUDA-OpenGL interop, shader compilation
- FORBIDDEN: Implementation before test, skipping RED phase ✓

**Observability**:
- Structured logging included? Yes (JSON logs with GPU metrics)
- Frontend logs → backend? N/A (single application)
- Error context sufficient? Yes (OpenGL error checking, CUDA error handling)

**Versioning**:
- Version number assigned? 0.1.0 (MAJOR.MINOR.BUILD)
- BUILD increments on every change? Yes
- Breaking changes handled? Yes (shader compatibility, config migration)

## Project Structure

### Documentation (this feature)
```
specs/[###-feature]/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
# C++ OpenGL Application Structure
src/
├── core/                # Core application framework
│   ├── Application.h/cpp
│   ├── Window.h/cpp
│   └── Input.h/cpp
├── physics/             # Physics simulation library
│   ├── PendulumState.h/cpp
│   ├── LagrangianSolver.h/cpp
│   └── cuda/            # CUDA acceleration
│       ├── KernelLauncher.h/cpp
│       └── pendulum_kernels.cu
├── rendering/           # OpenGL rendering library
│   ├── Renderer.h/cpp
│   ├── Shader.h/cpp
│   ├── Camera.h/cpp
│   ├── Primitives.h/cpp
│   └── TextRenderer.h/cpp
├── ui/                  # User interface library
│   ├── ViewManager.h/cpp
│   ├── SimulationView.h/cpp
│   ├── AnalysisView.h/cpp
│   └── ImGuiWrapper.h/cpp
├── analysis/            # Chaos analysis library
│   ├── ChaosAnalyzer.h/cpp
│   ├── ColorMapper.h/cpp
│   └── InitialConditions.h/cpp
└── main.cpp            # Application entry point

include/                # Public headers
├── PendulumVisualizor.h
└── [library headers]

tests/
├── contract/           # Library interface tests
├── integration/        # OpenGL/CUDA integration tests
└── unit/              # Individual component tests

external/              # Third-party dependencies
├── glfw/
├── glew/
├── glm/
└── imgui/

shaders/               # GLSL shader files
├── pendulum.vert
├── pendulum.frag
├── ui.vert
└── ui.frag

CMakeLists.txt         # Build configuration
config/                # Application configuration
└── default.json
```

**Structure Decision**: Option 1 (Single C++ project) - native desktop application

## Phase 0: Outline & Research
1. **Extract unknowns from Technical Context** above:
   - For each NEEDS CLARIFICATION → research task
   - For each dependency → best practices task
   - For each integration → patterns task

2. **Generate and dispatch research agents**:
   ```
   For each unknown in Technical Context:
     Task: "Research {unknown} for {feature context}"
   For each technology choice:
     Task: "Find best practices for {tech} in {domain}"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all NEEDS CLARIFICATION resolved

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

1. **Extract entities from feature spec** → `data-model.md`:
   - Entity name, fields, relationships
   - Validation rules from requirements
   - State transitions if applicable

2. **Generate API contracts** from functional requirements:
   - For each user action → endpoint
   - Use standard REST/GraphQL patterns
   - Output OpenAPI/GraphQL schema to `/contracts/`

3. **Generate contract tests** from contracts:
   - One test file per endpoint
   - Assert request/response schemas
   - Tests must fail (no implementation yet)

4. **Extract test scenarios** from user stories:
   - Each story → integration test scenario
   - Quickstart test = story validation steps

5. **Update agent file incrementally** (O(1) operation):
   - Run `/scripts/update-agent-context.sh [claude|gemini|copilot]` for your AI assistant
   - If exists: Add only NEW tech from current plan
   - Preserve manual additions between markers
   - Update recent changes (keep last 3)
   - Keep under 150 lines for token efficiency
   - Output to repository root

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, agent-specific file

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Load `/templates/tasks-template.md` as base template
- Generate TDD-ordered tasks from Phase 1 design artifacts:
  - **Contract Tests**: One task per interface in `/contracts/` (physics, rendering, ui)
  - **Data Model Tasks**: Create C++ structs/classes from `data-model.md` entities
  - **Library Implementation Tasks**: Implement each library to pass contract tests
  - **Integration Tasks**: CUDA-OpenGL interop, end-to-end user scenarios
  - **Build System Tasks**: CMakeLists.txt, dependency management, test harness

**Specific Task Categories**:
1. **Setup Tasks** [P]: CMake configuration, dependency setup, project structure
2. **Contract Test Tasks** [P]: Physics engine tests, rendering tests, UI tests
3. **Core Library Tasks**: Physics simulation, CUDA kernels, OpenGL rendering
4. **UI Implementation Tasks**: ImGui integration, parameter validation, view management
5. **Integration Tasks**: Library integration, CUDA-OpenGL interop, performance optimization
6. **Validation Tasks**: Execute quickstart.md steps, performance benchmarks

**Ordering Strategy**:
- **Phase 2A** [TDD Setup]: Contract tests MUST fail before implementation
- **Phase 2B** [Core Libraries]: Physics → Rendering → UI (dependency order)
- **Phase 2C** [Integration]: Combine libraries into working application
- **Phase 2D** [Validation]: Performance tuning and quickstart verification
- Mark [P] for parallel execution (independent contract tests, data models)

**Estimated Task Breakdown**:
- Setup and Build System: 3-4 tasks
- Contract Tests (failing): 8-10 tasks [P]
- Data Models and Core Types: 5-6 tasks [P]
- Physics Library Implementation: 6-8 tasks
- Rendering Library Implementation: 6-8 tasks  
- UI Library Implementation: 4-5 tasks
- Integration and Validation: 4-5 tasks
- **Total**: 36-46 numbered, sequenced tasks in tasks.md

**Constitutional Compliance**:
- RED-GREEN-Refactor cycle enforced: Each contract test MUST fail before implementation
- Library-first architecture: No direct application code, all features in libraries
- Real dependencies: Actual OpenGL context, real CUDA devices for testing
- Performance gates: Benchmarks must pass before task completion

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)  
**Phase 4**: Implementation (execute tasks.md following constitutional principles)  
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking
*Fill ONLY if Constitution Check has violations that must be justified*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |


## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented (none required)

---
*Based on Constitution v2.1.1 - See `/memory/constitution.md`*