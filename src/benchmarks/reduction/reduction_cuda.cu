#include <cuda_runtime.h>
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

extern "C" float run_reduction_cuda() {
    const int N = 1 << 20;
    size_t size = N * sizeof(float);

    float* h_input = new float[N];
    for (int i = 0; i < N; ++i) h_input[i] = 1.0f;

    float* d_input;
    float* d_output;
    float h_output = 0.0f;

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, sizeof(float));
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, &h_output, sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(256);
    dim3 blocks(256);

    auto start = std::chrono::high_resolution_clock::now();
    reduce_sum<<<blocks, threads>>>(d_input, d_output, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    std::chrono::duration<float, std::milli> duration = end - start;
    std::cout << "CUDA Reduction Sum = " << h_output << "\n";

    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_input;

    return duration.count();
}

