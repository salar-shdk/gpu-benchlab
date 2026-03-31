#include <hip/hip_runtime.h>
#include <iostream>

__global__ void copy_kernel(const float* src, float* dst, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        dst[idx] = src[idx];
}

extern "C" float run_bandwidth_hip() {
    const int N = 1 << 26;   // ~64M floats = 256MB
    size_t size = N * sizeof(float);

    float* d_src;
    float* d_dst;
    hipMalloc(&d_src, size);
    hipMalloc(&d_dst, size);

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    hipLaunchKernelGGL(copy_kernel, grid, block, 0, 0, d_src, d_dst, N);
    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float ms = 0.0f;
    hipEventElapsedTime(&ms, start, stop);

    float seconds      = ms / 1000.0f;
    float bandwidthGBs = (2.0f * size / 1e9f) / seconds;   // 2x: one read + one write
    std::cout << "HIP Memory Bandwidth: " << bandwidthGBs << " GB/s\n";

    hipEventDestroy(start);
    hipEventDestroy(stop);
    hipFree(d_src);
    hipFree(d_dst);

    return ms;
}
