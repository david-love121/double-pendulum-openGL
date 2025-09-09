#include "Application.h"
#include <iostream>
#include <cstdlib>

#ifdef USE_CUDA
extern "C" void runCudaDemo();
#endif

int main(int argc, char* argv[]) {
    try {
        // Check for CUDA demo flag
        bool run_cuda_demo = false;
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--cuda-demo") {
                run_cuda_demo = true;
                break;
            }
        }
        
#ifdef USE_CUDA
        if (run_cuda_demo) {
            std::cout << "Running CUDA performance demonstration..." << std::endl;
            runCudaDemo();
            return EXIT_SUCCESS;
        }
#else
        if (run_cuda_demo) {
            std::cout << "CUDA support not compiled in. Use -DUSE_CUDA=ON when building." << std::endl;
            return EXIT_FAILURE;
        }
#endif
        
        pendulum::Application app;
        
        if (!app.initialize(argc, argv)) {
            std::cerr << "Failed to initialize application" << std::endl;
            return EXIT_FAILURE;
        }
        
        app.run();
        app.cleanup();
        
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Application error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (...) {
        std::cerr << "Unknown application error" << std::endl;
        return EXIT_FAILURE;
    }
}
