#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

__global__ void copy_kernel(const float* src, float* dst, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        dst[idx] = src[idx];
}

extern "C" float run_bandwidth_cuda() {
    const int N = 1 << 26; // ~64M floats = 256MB
    size_t size = N * sizeof(float);

    float* d_src;
    float* d_dst;

    cudaMalloc(&d_src, size);
    cudaMalloc(&d_dst, size);

    dim3 threads(256);
    dim3 blocks((N + threads.x - 1) / threads.x);

    auto start = std::chrono::high_resolution_clock::now();
    copy_kernel<<<blocks, threads>>>(d_src, d_dst, N);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> duration = end - start;

    cudaFree(d_src);
    cudaFree(d_dst);

    float seconds = duration.count() / 1000.0f;
    float bandwidthGBs = (2.0f * size / 1e9f) / seconds; // 2x because read + write

    std::cout << "CUDA Memory Bandwidth: " << bandwidthGBs << " GB/s\n";
    return duration.count();
}

