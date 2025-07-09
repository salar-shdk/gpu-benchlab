#include <iostream>

// HIP benchmark functions
extern float run_matmul_hip();
extern float run_vector_add_hip();
extern float run_conv1d_hip();
extern float run_reduction_hip();
extern float run_bandwidth_hip();

int main() {
    std::cout << "=======================\n";
    std::cout << " HIP BENCHMARK RESULTS \n";
    std::cout << "=======================\n";

    std::cout << "\n[HIP] Running Matrix Multiplication...\n";
    float t1 = run_matmul_hip();
    std::cout << "Duration: " << t1 << " ms\n";

    std::cout << "\n[HIP] Running Vector Addition...\n";
    float t2 = run_vector_add_hip();
    std::cout << "Duration: " << t2 << " ms\n";

    std::cout << "\n[HIP] Running 1D Convolution...\n";
    float t3 = run_conv1d_hip();
    std::cout << "Duration: " << t3 << " ms\n";

    std::cout << "\n[HIP] Running Reduction (Sum)...\n";
    float t4 = run_reduction_hip();
    std::cout << "Duration: " << t4 << " ms\n";

    std::cout << "\n[HIP] Running Memory Bandwidth Test...\n";
    float t5 = run_bandwidth_hip();
    std::cout << "Duration: " << t5 << " ms\n";

    return 0;
}

