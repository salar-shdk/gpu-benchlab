#include <cuda_runtime.h>
#include <iostream>

__global__ void copy_kernel(const float* src, float* dst, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        dst[idx] = src[idx];
}

extern "C" float run_bandwidth_cuda() {
    const int N = 1 << 26;   // ~64M floats = 256MB
    size_t size = N * sizeof(float);

    float* d_src;
    float* d_dst;
    cudaMalloc(&d_src, size);
    cudaMalloc(&d_dst, size);

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    copy_kernel<<<grid, block>>>(d_src, d_dst, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    float seconds      = ms / 1000.0f;
    float bandwidthGBs = (2.0f * size / 1e9f) / seconds;   // 2x: one read + one write
    std::cout << "CUDA Memory Bandwidth: " << bandwidthGBs << " GB/s\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_src);
    cudaFree(d_dst);

    return ms;
}
