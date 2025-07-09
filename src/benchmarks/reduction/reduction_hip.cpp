#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>

__global__ void reduce_sum(const float* input, float* output, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (int i = tid; i < N; i += stride)
        sum += input[i];

    atomicAdd(output, sum);
}

extern "C" float run_reduction_hip() {
    const int N = 1 << 20;
    size_t size = N * sizeof(float);

    float* h_input = new float[N];
    for (int i = 0; i < N; ++i) h_input[i] = 1.0f;

    float* d_input;
    float* d_output;
    float h_output = 0.0f;

    hipMalloc(&d_input, size);
    hipMalloc(&d_output, sizeof(float));
    hipMemcpy(d_input, h_input, size, hipMemcpyHostToDevice);
    hipMemcpy(d_output, &h_output, sizeof(float), hipMemcpyHostToDevice);

    dim3 threads(256);
    dim3 blocks(256);

    auto start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(reduce_sum, blocks, threads, 0, 0, d_input, d_output, N);
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    hipMemcpy(&h_output, d_output, sizeof(float), hipMemcpyDeviceToHost);

    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "HIP Reduction Sum = " << h_output << "\n";

    hipFree(d_input);
    hipFree(d_output);
    delete[] h_input;

    return duration.count();
}

