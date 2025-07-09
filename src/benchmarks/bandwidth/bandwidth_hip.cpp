#include <hip/hip_runtime.h>
#include <iostream>
#include <chrono>

__global__ void copy_kernel(const float* src, float* dst, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        dst[idx] = src[idx];
}

extern "C" float run_bandwidth_hip() {
    const int N = 1 << 26;
    size_t size = N * sizeof(float);

    float* d_src;
    float* d_dst;

    hipMalloc(&d_src, size);
    hipMalloc(&d_dst, size);

    dim3 threads(256);
    dim3 blocks((N + threads.x - 1) / threads.x);

    auto start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(copy_kernel, blocks, threads, 0, 0, d_src, d_dst, N);
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> duration = end - start;

    hipFree(d_src);
    hipFree(d_dst);

    float seconds = duration.count() / 1000.0f;
    float bandwidthGBs = (2.0f * size / 1e9f) / seconds;

    std::cout << "HIP Memory Bandwidth: " << bandwidthGBs << " GB/s\n";
    return duration.count();
}

