#include <iostream>

// CUDA benchmark functions
extern "C" float run_matmul_cuda();
extern "C" float run_vector_add_cuda();
extern "C" float run_conv1d_cuda();
extern "C" float run_reduction_cuda();
extern "C" float run_bandwidth_cuda();


int main() {
    std::cout << "========================\n";
    std::cout << " CUDA BENCHMARK RESULTS \n";
    std::cout << "========================\n";

    std::cout << "\n[CUDA] Running Matrix Multiplication...\n";
    float t1 = run_matmul_cuda();
    std::cout << "Duration: " << t1 << " ms\n";

    std::cout << "\n[CUDA] Running Vector Addition...\n";
    float t2 = run_vector_add_cuda();
    std::cout << "Duration: " << t2 << " ms\n";

    std::cout << "\n[CUDA] Running 1D Convolution...\n";
    float t3 = run_conv1d_cuda();
    std::cout << "Duration: " << t3 << " ms\n";

    std::cout << "\n[CUDA] Running Reduction (Sum)...\n";
    float t4 = run_reduction_cuda();
    std::cout << "Duration: " << t4 << " ms\n";

    std::cout << "\n[CUDA] Running Memory Bandwidth Test...\n";
    float t5 = run_bandwidth_cuda();
    std::cout << "Duration: " << t5 << " ms\n";

    return 0;
}

