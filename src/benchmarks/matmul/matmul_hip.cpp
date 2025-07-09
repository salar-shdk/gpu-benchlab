// benchmarks/matmul/matmul_hip.cpp
#include <hip/hip_runtime.h>
#include <chrono>
#include <iostream>

__global__ void matmul(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

extern "C" float run_matmul_hip() {
    const int N = 1024;
    size_t size = N * N * sizeof(float);

    float *h_A = new float[N * N], *h_B = new float[N * N], *h_C = new float[N * N];
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    hipMalloc(&d_A, size);
    hipMalloc(&d_B, size);
    hipMalloc(&d_C, size);
    hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (N + threads.y - 1) / threads.y);

    auto start = std::chrono::high_resolution_clock::now();
    hipLaunchKernelGGL(matmul, blocks, threads, 0, 0, d_A, d_B, d_C, N);
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<float, std::milli> elapsed = end - start;

    hipMemcpy(h_C, d_C, size, hipMemcpyDeviceToHost);
    hipFree(d_A); hipFree(d_B); hipFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C;

    return elapsed.count();
}

